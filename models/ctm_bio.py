# models/ctm_bio.py
import torch
import torch.nn as nn
import numpy as np
import math

from models.ctm import ContinuousThoughtMachine
from models.bio_modules import (
    ShortTermPlasticity,
    HomeostaticRegulation, 
    LateralInhibition,
    RefractoryDynamics,
    SynapticNoise
)


class BioInspiredCTM(ContinuousThoughtMachine):
    """
    CTM with bio-inspired dynamics that operate during the internal tick loop.
    
    These mechanisms augment (not replace) backprop training:
    - Short-term plasticity: modulates synapse strengths within a forward pass
    - Homeostatic regulation: prevents neuron saturation/silence
    - Lateral inhibition: maintains neural diversity
    - Refractory dynamics: encourages sequential activation patterns
    """
    
    def __init__(
        self,
        # Bio-inspired flags
        use_short_term_plasticity=True,
        use_homeostasis=True,
        use_lateral_inhibition=True,
        use_refractory=True,
        use_synaptic_noise=True,
        # STP parameters
        tau_facilitate=5.0,
        tau_depress=20.0,
        # Homeostasis parameters
        target_firing_rate=0.5,
        homeostasis_adaptation_rate=0.01,
        # Lateral inhibition parameters
        inhibition_strength=0.1,
        inhibition_neighborhood=8,
        # Refractory parameters
        refractory_strength=0.3,
        refractory_decay=0.8,
        # Noise parameters
        noise_scale=0.01,
        # Standard CTM parameters
        **ctm_kwargs
    ):
        super().__init__(**ctm_kwargs)
        
        # Store flags
        self.use_short_term_plasticity = use_short_term_plasticity
        self.use_homeostasis = use_homeostasis
        self.use_lateral_inhibition = use_lateral_inhibition
        self.use_refractory = use_refractory
        self.use_synaptic_noise = use_synaptic_noise
        
        d_model = ctm_kwargs['d_model']
        
        # Initialize bio-inspired modules
        if use_short_term_plasticity:
            self.stp = ShortTermPlasticity(d_model, tau_facilitate, tau_depress)
            
        if use_homeostasis:
            self.homeostasis = HomeostaticRegulation(
                d_model, target_firing_rate, homeostasis_adaptation_rate
            )
            
        if use_lateral_inhibition:
            self.lateral_inhibition = LateralInhibition(
                d_model, inhibition_strength, inhibition_neighborhood
            )
            
        if use_refractory:
            self.refractory = RefractoryDynamics(
                d_model, refractory_strength, refractory_decay
            )
            
        if use_synaptic_noise:
            self.synaptic_noise = SynapticNoise(d_model, noise_scale)

    def _init_bio_states(self, batch_size, device):
        """Initialize bio-inspired dynamic states."""
        states = {}
        
        if self.use_short_term_plasticity:
            states['facilitation'] = torch.ones(batch_size, self.d_model, device=device)
            states['depression'] = torch.ones(batch_size, self.d_model, device=device)
            
        if self.use_homeostasis:
            states['running_rates'] = torch.ones(
                batch_size, self.d_model, device=device
            ) * 0.5
            
        if self.use_refractory:
            states['refractory'] = torch.zeros(batch_size, self.d_model, device=device)
            
        return states

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
            'refractory': []
        }

        # --- Featurise Input Data ---
        kv = self.compute_features(x)

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        # --- Initialize Bio States ---
        bio_states = self._init_bio_states(B, device)

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

            # --- Apply Synapses ---
            state = self.synapses(pre_synapse_input)
            
            # --- Bio: Short-Term Plasticity (modulates synapse output) ---
            if self.use_short_term_plasticity:
                modulation, bio_states['facilitation'], bio_states['depression'] = \
                    self.stp(
                        pre_synapse_input[:, :self.d_model],  # Use activated_state portion
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

        # --- Return Values ---
        if track:
            return (
                predictions, 
                certainties, 
                (np.array(synch_out_tracking), np.array(synch_action_tracking)), 
                np.array(pre_activations_tracking), 
                np.array(post_activations_tracking), 
                np.array(attention_tracking),
                bio_tracking  # Additional bio-inspired tracking
            )
        return predictions, certainties, synchronisation_out