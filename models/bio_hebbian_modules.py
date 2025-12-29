# models/hebbian_modules.py
import torch
import torch.nn as nn
import math

class HebbianPlasticity(nn.Module):
    """
    Hebbian plasticity for CTM synapses.
    
    Instead of static weights, synapses update via learned local rules:
        Δw_ij = η * (A*o_i*o_j + B*o_i + C*o_j + D)
    
    The coefficients (η, A, B, C, D) are learned via backprop/evolution,
    but the weight updates themselves are purely local during inference.
    
    Key difference from your STP:
    - STP: multiplicative modulation, weights unchanged
    - Hebbian: actual weight updates, self-organization from random init
    """
    
    def __init__(
        self, 
        d_model: int,
        init_scale: float = 0.01,
        weight_decay: float = 0.001,
        clamp_weights: tuple = (-2.0, 2.0),
        per_synapse: bool = True,  # Per-synapse vs per-neuron coefficients
    ):
        super().__init__()
        self.d_model = d_model
        self.weight_decay = weight_decay
        self.clamp_weights = clamp_weights
        self.per_synapse = per_synapse
        
        # Hebbian coefficients: η, A, B, C, D
        # These are LEARNED (via backprop or evolution) but applied LOCALLY
        if per_synapse:
            # Full flexibility: each synapse has its own rule
            # Shape: (d_model, d_model) for each coefficient
            self.eta = nn.Parameter(torch.ones(d_model, d_model) * 0.01)
            self.A = nn.Parameter(torch.zeros(d_model, d_model))  # Hebbian term
            self.B = nn.Parameter(torch.zeros(d_model, d_model))  # Pre-synaptic
            self.C = nn.Parameter(torch.zeros(d_model, d_model))  # Post-synaptic  
            self.D = nn.Parameter(torch.zeros(d_model, d_model))  # Bias/decay
        else:
            # Shared coefficients across all synapses (more constrained)
            self.eta = nn.Parameter(torch.tensor(0.01))
            self.A = nn.Parameter(torch.tensor(1.0))   # Standard Hebbian
            self.B = nn.Parameter(torch.tensor(0.0))
            self.C = nn.Parameter(torch.tensor(0.0))
            self.D = nn.Parameter(torch.tensor(-0.01))  # Slight decay
        
        # Initial weight distribution parameters (learned)
        self.init_mean = nn.Parameter(torch.tensor(0.0))
        self.init_std = nn.Parameter(torch.tensor(init_scale))
        
    def init_weights(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize plastic weights from learned distribution.
        Called at the start of each forward pass (or episode in RL).
        
        Returns:
            weights: (B, d_model, d_model) - plastic weights per sample
        """
        # Sample from learned initial distribution
        weights = torch.randn(
            batch_size, self.d_model, self.d_model, 
            device=device
        ) * torch.abs(self.init_std) + self.init_mean
        
        return weights
    
    def update_weights(
        self, 
        weights: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Hebbian update rule.
        
        Args:
            weights: (B, d_model, d_model) current plastic weights
            pre: (B, d_model) pre-synaptic activations
            post: (B, d_model) post-synaptic activations
            
        Returns:
            updated_weights: (B, d_model, d_model)
        """
        B = pre.size(0)
        
        # Compute Hebbian terms
        # pre: (B, d_model) -> (B, d_model, 1)
        # post: (B, d_model) -> (B, 1, d_model)
        pre_expanded = pre.unsqueeze(-1)      # (B, d_model, 1)
        post_expanded = post.unsqueeze(-2)    # (B, 1, d_model)
        
        # Hebbian product: o_i * o_j
        hebbian_product = pre_expanded * post_expanded  # (B, d_model, d_model)
        
        # Full plasticity rule: Δw = η * (A*o_i*o_j + B*o_i + C*o_j + D)
        if self.per_synapse:
            delta_w = self.eta * (
                self.A * hebbian_product +
                self.B * pre_expanded +
                self.C * post_expanded +
                self.D
            )
        else:
            delta_w = self.eta * (
                self.A * hebbian_product +
                self.B * pre_expanded +
                self.C * post_expanded +
                self.D
            )
        
        # Apply update with optional weight decay
        new_weights = weights + delta_w
        if self.weight_decay > 0:
            new_weights = new_weights * (1 - self.weight_decay)
        
        # Clamp to prevent explosion
        new_weights = torch.clamp(
            new_weights, 
            self.clamp_weights[0], 
            self.clamp_weights[1]
        )
        
        return new_weights
    
    def apply_weights(
        self, 
        weights: torch.Tensor, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply plastic weights to input.
        
        Args:
            weights: (B, d_model, d_model)
            x: (B, d_model)
            
        Returns:
            output: (B, d_model)
        """
        # Batched matrix-vector multiplication
        # weights: (B, d_model, d_model), x: (B, d_model, 1) -> (B, d_model)
        return torch.bmm(weights, x.unsqueeze(-1)).squeeze(-1)


class HebbianSynapseModel(nn.Module):
    """
    Replaces SynapseUNET with Hebbian plastic connections.
    
    The static synapse model learns fixed weights.
    This model learns the RULES for weight updates, and weights
    self-organize during the internal tick loop.
    """
    
    def __init__(
        self,
        d_model: int,
        d_input: int,
        hidden_dim: int = None,
        n_plastic_layers: int = 2,
        init_scale: float = 0.01,
        weight_decay: float = 0.001,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_input = d_input
        hidden_dim = hidden_dim or d_model
        
        # Input projection (static, learned via backprop)
        self.input_proj = nn.Sequential(
            nn.Linear(d_model + d_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # Plastic layers with Hebbian learning
        self.plastic_layers = nn.ModuleList([
            HebbianPlasticity(
                hidden_dim if i == 0 else d_model,
                init_scale=init_scale,
                weight_decay=weight_decay,
            )
            for i in range(n_plastic_layers)
        ])
        
        # Output projection (static)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Non-linearity between plastic layers
        self.activation = nn.SiLU()
        
    def init_plastic_weights(self, batch_size: int, device: torch.device):
        """Initialize all plastic weights for a new forward pass."""
        return [
            layer.init_weights(batch_size, device)
            for layer in self.plastic_layers
        ]
    
    def forward(
        self,
        x: torch.Tensor,
        plastic_weights: list,
        prev_activations: list = None,
    ):
        """
        Forward pass with Hebbian weight updates.
        
        Args:
            x: (B, d_model + d_input) - concatenated attention output + activated state
            plastic_weights: list of (B, dim, dim) weight tensors
            prev_activations: list of previous layer activations for Hebbian update
            
        Returns:
            output: (B, d_model)
            new_plastic_weights: updated weight tensors
            activations: current layer activations (for next tick's Hebbian update)
        """
        # Static input projection
        h = self.input_proj(x)
        
        activations = [h]
        new_weights = []
        
        # Apply plastic layers with Hebbian updates
        for i, (layer, weights) in enumerate(zip(self.plastic_layers, plastic_weights)):
            # Apply current weights
            h_new = layer.apply_weights(weights, h)
            h_new = self.activation(h_new)
            
            # Update weights based on pre/post activations
            if prev_activations is not None:
                pre = prev_activations[i]
                post = h_new
                updated_weights = layer.update_weights(weights, pre.detach(), post.detach())
            else:
                # First tick: no update yet
                updated_weights = weights
            
            new_weights.append(updated_weights)
            activations.append(h_new)
            h = h_new
        
        # Static output projection
        output = self.output_proj(h)
        
        return output, new_weights, activations