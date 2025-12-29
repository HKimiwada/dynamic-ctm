# models/ctm_hebbian.py
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
from models.hebbian_modules import HebbianPlasticity, HebbianSynapseModel


class HebbianCTM(ContinuousThoughtMachine):
    """
    CTM with Hebbian meta-learning for synapse weights.
    
    Key innovation: Synapse weights are not fixed after training.
    Instead, we learn the RULES for how weights should update,
    and weights self-organize during the internal tick loop.
    
    This combines:
    - Your working bio mechanisms (refractory, lateral inhibition)
    - Hebbian weight plasticity from Najarro & Risi
    """
    
    def __init__(
        self,
        # Hebbian plasticity settings
        use_hebbian_synapses: bool = True,
        hebbian_init_scale: float = 0.01,
        hebbian_weight_decay: float = 0.001,
        n_plastic_layers: int = 2,
        
        # Bio-inspired flags (your existing mechanisms)
        use_lateral_inhibition: bool = True,
        use_refractory: bool = True,
        use_synaptic_noise: bool = False,
        
        # Bio parameters
        inhibition_strength: float = 0.1,
        inhibition_neighborhood: int = 8,
        refractory_strength: float = 0.3,
        refractory_decay: float = 0.8,
        noise_scale: float = 0.01,
        
        # Standard CTM parameters
        **ctm_kwargs
    ):
        # Don't call super().__init__() yet - we need to override synapse creation
        nn.Module.__init__(self)
        
        # Store all parameters
        self.use_hebbian_synapses = use_hebbian_synapses
        self.use_lateral_inhibition = use_lateral_inhibition
        self.use_refractory = use_refractory
        self.use_synaptic_noise = use_synaptic_noise
        
        # Store CTM kwargs for later
        self.iterations = ctm_kwargs['iterations']
        self.d_model = ctm_kwargs['d_model']
        self.d_input = ctm_kwargs['d_input']
        self.memory_length = ctm_kwargs['memory_length']
        self.prediction_reshaper = ctm_kwargs.get('prediction_reshaper', [-1])
        self.n_synch_out = ctm_kwargs['n_synch_out']
        self.n_synch_action = ctm_kwargs['n_synch_action']
        self.backbone_type = ctm_kwargs['backbone_type']
        self.out_dims = ctm_kwargs['out_dims']
        self.positional_embedding_type = ctm_kwargs['positional_embedding_type']
        self.neuron_select_type = ctm_kwargs.get('neuron_select_type', 'random-pairing')
        
        # Verify args
        self._verify_args(ctm_kwargs)
        
        # Setup input processing (same as base CTM)
        self._setup_input_processing(ctm_kwargs)
        
        # Setup EITHER Hebbian OR static synapses
        if use_hebbian_synapses:
            self.synapses = HebbianSynapseModel(
                d_model=self.d_model,
                d_input=self.d_input,
                n_plastic_layers=n_plastic_layers,
                init_scale=hebbian_init_scale,
                weight_decay=hebbian_weight_decay,
            )
        else:
            self.synapses = self._get_static_synapses(
                ctm_kwargs['synapse_depth'],
                self.d_model,
                ctm_kwargs.get('dropout', 0)
            )
        
        # Setup NLMs (same as base CTM)
        self._setup_nlms(ctm_kwargs)
        
        # Setup synchronization (same as base CTM)
        self._setup_synchronization(ctm_kwargs)
        
        # Setup bio mechanisms
        if use_lateral_inhibition:
            self.lateral_inhibition = LateralInhibition(
                self.d_model, inhibition_strength, inhibition_neighborhood
            )
        
        if use_refractory:
            self.refractory = RefractoryDynamics(
                self.d_model, refractory_strength, refractory_decay
            )
        
        if use_synaptic_noise:
            self.synaptic_noise = SynapticNoise(self.d_model, noise_scale)
        
        # Output projector
        self.output_projector = nn.Sequential(nn.LazyLinear(self.out_dims))

    def _init_hebbian_states(self, batch_size: int, device: torch.device):
        """Initialize Hebbian plastic weights and activation history."""
        states = {}
        
        if self.use_hebbian_synapses:
            states['plastic_weights'] = self.synapses.init_plastic_weights(batch_size, device)
            states['prev_activations'] = None  # Will be set after first tick
        
        if self.use_refractory:
            states['refractory'] = torch.zeros(batch_size, self.d_model, device=device)
        
        return states

    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        tracking = self._init_tracking() if track else None

        # --- Featurise Input Data ---
        kv = self.compute_features(x)

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        # --- Initialize Hebbian + Bio States ---
        hebbian_states = self._init_hebbian_states(B, device)

        # --- Prepare Storage for Outputs ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=torch.float32)

        # --- Initialise Synchronization ---
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
            
            # --- Calculate Synchronisation for Action ---
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
            pre_synapse_input = torch.cat((attn_out, activated_state), dim=-1)

            # --- Apply Synapses (Hebbian or Static) ---
            if self.use_hebbian_synapses:
                state, new_plastic_weights, activations = self.synapses(
                    pre_synapse_input,
                    hebbian_states['plastic_weights'],
                    hebbian_states['prev_activations']
                )
                hebbian_states['plastic_weights'] = new_plastic_weights
                hebbian_states['prev_activations'] = activations
            else:
                state = self.synapses(pre_synapse_input)
            
            # Update trace
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply Neuron-Level Models ---
            activated_state = self.trace_processor(state_trace)
            
            # --- Bio: Lateral Inhibition ---
            if self.use_lateral_inhibition:
                activated_state = self.lateral_inhibition(activated_state)
            
            # --- Bio: Refractory Dynamics ---
            if self.use_refractory:
                activated_state, hebbian_states['refractory'] = \
                    self.refractory(activated_state, hebbian_states['refractory'])
            
            # --- Bio: Synaptic Noise ---
            if self.use_synaptic_noise:
                activated_state = self.synaptic_noise(activated_state, self.training)

            # --- Calculate Output Synchronisation ---
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
                self._update_tracking(tracking, state_trace, activated_state, 
                                     attn_weights, synchronisation_out, 
                                     synchronisation_action, hebbian_states)

        if track:
            return self._format_tracking_output(predictions, certainties, tracking)
        return predictions, certainties, synchronisation_out