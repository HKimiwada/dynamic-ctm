# models/ctm_bio_hebbian.py
"""
HebbianCTM: Extends BioInspiredCTM with Hebbian plastic synapses.
"""
import torch
import torch.nn as nn
import numpy as np
import math

from models.ctm_bio import BioInspiredCTM
from models.bio_hebbian_modules import HebbianPlasticity, HebbianSynapseModel


class HebbianCTM(BioInspiredCTM):
    """
    CTM with Hebbian meta-learning for synapse weights.
    
    Uses HYBRID approach: static synapses (backbone) + Hebbian modulation.
    This provides stability from the static path while allowing Hebbian
    plasticity to learn adaptive refinements.
    """
    
    def __init__(
        self,
        # Hebbian plasticity settings
        use_hebbian_synapses: bool = True,
        hebbian_init_scale: float = 0.01,
        hebbian_weight_decay: float = 0.01,  # Increased from 0.001
        n_plastic_layers: int = 2,
        hebbian_modulation_scale: float = 0.1,  # How much Hebbian contributes
        **kwargs
    ):
        # Store Hebbian settings before calling super().__init__
        self._use_hebbian_synapses = use_hebbian_synapses
        self._hebbian_init_scale = hebbian_init_scale
        self._hebbian_weight_decay = hebbian_weight_decay
        self._n_plastic_layers = n_plastic_layers
        
        # Call parent init (BioInspiredCTM -> ContinuousThoughtMachine)
        # NOTE: Do NOT delete self.synapses - we use it as backbone!
        super().__init__(**kwargs)
        
        # Add Hebbian module as modulation (not replacement)
        if use_hebbian_synapses:
            self.hebbian_synapses = HebbianSynapseModel(
                d_model=self.d_model,
                d_input=self.d_input,
                n_plastic_layers=n_plastic_layers,
                init_scale=hebbian_init_scale,
                weight_decay=hebbian_weight_decay,
            )
            
            # Learnable gate for Hebbian contribution (starts small)
            self.hebbian_gate = nn.Parameter(torch.tensor(hebbian_modulation_scale))
            
            # Layer norm to stabilize Hebbian output
            self.hebbian_norm = nn.LayerNorm(self.d_model)
    
    def _init_hebbian_states(self, batch_size: int, device: torch.device):
        """Initialize Hebbian plastic weights."""
        if self._use_hebbian_synapses:
            return {
                'plastic_weights': self.hebbian_synapses.init_plastic_weights(batch_size, device),
                'prev_activations': None,
            }
        return {}

    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []
        
        # Bio-inspired tracking
        bio_tracking = {
            'facilitation': [],
            'depression': [],
            'running_rates': [],
            'refractory': [],
            'plastic_weights': [],
            'hebbian_contribution': [],  # Track how much Hebbian adds
        }

        # --- Featurise Input Data ---
        kv = self.compute_features(x)

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        # --- Initialize Bio States (from parent) ---
        bio_states = self._init_bio_states(B, device)
        
        # --- Initialize Hebbian States ---
        hebbian_states = self._init_hebbian_states(B, device)

        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=torch.float32)

        # --- Initialise Recurrent Synch Values ---
        decay_alpha_action, decay_beta_action = None, None
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
            activated_state, None, None, r_out, synch_type='out'
        )

        # --- Recurrent Loop ---
        for stepi in range(self.iterations):

            # --- Calculate Synchronisation for Input Data Interaction ---
            synchronisation_action, decay_alpha_action, decay_beta_action = \
                self.compute_synchronisation(
                    activated_state, decay_alpha_action, decay_beta_action, 
                    r_action, synch_type='action'
                )

            # --- Interact with Data via Attention ---
            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(
                q, kv, kv, average_attn_weights=False, need_weights=True
            )
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)

            # --- Apply Synapses: HYBRID (Static backbone + Hebbian modulation) ---
            # Static backbone - always provides stable gradient path
            state = self.synapses(pre_synapse_input)
            
            # Hebbian modulation - adaptive refinement
            if self._use_hebbian_synapses:
                hebbian_out, new_plastic_weights, activations = self.hebbian_synapses(
                    pre_synapse_input,
                    hebbian_states['plastic_weights'],
                    hebbian_states['prev_activations']
                )
                hebbian_states['plastic_weights'] = new_plastic_weights
                hebbian_states['prev_activations'] = activations
                
                # Normalize and bound Hebbian contribution
                hebbian_out = self.hebbian_norm(hebbian_out)
                hebbian_contribution = torch.sigmoid(self.hebbian_gate) * torch.tanh(hebbian_out)
                
                # Combine: static + bounded Hebbian
                state = state + hebbian_contribution
                
                # Track contribution magnitude
                if track:
                    bio_tracking['hebbian_contribution'].append(
                        hebbian_contribution.abs().mean().item()
                    )
            
            # --- Bio: Short-Term Plasticity (modulates synapse output) ---
            if self.use_short_term_plasticity:
                modulation, bio_states['facilitation'], bio_states['depression'] = \
                    self.stp(
                        pre_synapse_input[:, :self.d_model],
                        state,
                        bio_states['facilitation'],
                        bio_states['depression']
                    )
                state = state * modulation
            
            # Update trace
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply Neuron-Level Models ---
            activated_state = self.trace_processor(state_trace)
            
            # --- Bio: Homeostatic Regulation ---
            if self.use_homeostasis:
                activated_state, bio_states['running_rates'] = \
                    self.homeostasis(activated_state, bio_states['running_rates'])
            
            # --- Bio: Lateral Inhibition ---
            if self.use_lateral_inhibition:
                activated_state = self.lateral_inhibition(activated_state)
            
            # --- Bio: Refractory Dynamics ---
            if self.use_refractory:
                activated_state, bio_states['refractory'] = \
                    self.refractory(activated_state, bio_states['refractory'])
            
            # --- Bio: Synaptic Noise ---
            if self.use_synaptic_noise:
                activated_state = self.synaptic_noise(activated_state, self.training)

            # --- Calculate Synchronisation for Output Predictions ---
            synchronisation_out, decay_alpha_out, decay_beta_out = \
                self.compute_synchronisation(
                    activated_state, decay_alpha_out, decay_beta_out, 
                    r_out, synch_type='out'
                )

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())
                
                # Bio tracking
                if self.use_short_term_plasticity:
                    bio_tracking['facilitation'].append(
                        bio_states['facilitation'].detach().cpu().numpy()
                    )
                    bio_tracking['depression'].append(
                        bio_states['depression'].detach().cpu().numpy()
                    )
                if self.use_homeostasis:
                    bio_tracking['running_rates'].append(
                        bio_states['running_rates'].detach().cpu().numpy()
                    )
                if self.use_refractory:
                    bio_tracking['refractory'].append(
                        bio_states['refractory'].detach().cpu().numpy()
                    )
                # Hebbian tracking
                if self._use_hebbian_synapses:
                    weight_stats = [w.abs().mean().item() for w in hebbian_states['plastic_weights']]
                    bio_tracking['plastic_weights'].append(weight_stats)

        # --- Return Values ---
        if track:
            return (
                predictions, 
                certainties, 
                (np.array(synch_out_tracking), np.array(synch_action_tracking)), 
                np.array(pre_activations_tracking), 
                np.array(post_activations_tracking), 
                np.array(attention_tracking),
                bio_tracking
            )
        return predictions, certainties, synchronisation_out