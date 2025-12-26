# models/bio_modules.py
import torch
import torch.nn as nn
import math

class ShortTermPlasticity(nn.Module):
    """
    Short-term synaptic plasticity that modulates connection strengths
    during the internal tick loop (not during training).
    
    Implements:
    - Facilitation: repeated activation strengthens connection temporarily
    - Depression: high activity depletes synaptic resources
    """
    def __init__(self, d_model, tau_facilitate=5.0, tau_depress=20.0):
        super().__init__()
        self.tau_facilitate = tau_facilitate
        self.tau_depress = tau_depress
        
        # Learnable baseline and scaling parameters
        self.facilitate_scale = nn.Parameter(torch.ones(d_model) * 0.1)
        self.depress_scale = nn.Parameter(torch.ones(d_model) * 0.1)
        
    def forward(self, pre_activation, post_activation, facilitation, depression):
        """
        Args:
            pre_activation: incoming signal (B, d_model)
            post_activation: outgoing signal (B, d_model)  
            facilitation: current facilitation state (B, d_model)
            depression: current depression state (B, d_model)
        Returns:
            modulation: multiplicative factor for synaptic weights
            new_facilitation, new_depression: updated states
        """
        # Update facilitation based on pre-post correlation
        correlation = (pre_activation * post_activation).detach()
        delta_f = self.facilitate_scale * correlation / self.tau_facilitate
        
        # Update depression based on pre-synaptic activity magnitude
        activity = pre_activation.abs().detach()
        delta_d = self.depress_scale * activity / self.tau_depress
        
        # Decay toward baseline (1.0)
        new_facilitation = 0.95 * facilitation + 0.05 + delta_f
        new_depression = 0.95 * depression + 0.05 - delta_d
        
        # Clamp to reasonable ranges
        new_facilitation = torch.clamp(new_facilitation, 0.5, 2.0)
        new_depression = torch.clamp(new_depression, 0.5, 1.5)
        
        modulation = new_facilitation * new_depression
        return modulation, new_facilitation, new_depression


class HomeostaticRegulation(nn.Module):
    """
    Maintains stable firing rates through adaptive thresholds.
    Prevents neurons from saturating or going silent.
    """
    def __init__(self, d_model, target_rate=0.5, adaptation_rate=0.01):
        super().__init__()
        self.target_rate = target_rate
        self.adaptation_rate = adaptation_rate
        self.register_buffer('thresholds', torch.zeros(d_model))
        
    def forward(self, activations, running_rates):
        """
        Args:
            activations: neuron activations (B, d_model)
            running_rates: exponential moving average of firing rates (B, d_model)
        Returns:
            adjusted_activations, new_running_rates
        """
        # Apply adaptive threshold
        adjusted = activations - self.thresholds.unsqueeze(0)
        
        # Update running rate estimate (using sigmoid as "firing probability")
        current_rate = torch.sigmoid(activations).detach()
        new_running_rates = 0.99 * running_rates + 0.01 * current_rate
        
        # Adjust thresholds toward target rate (done in-place for efficiency)
        with torch.no_grad():
            error = new_running_rates.mean(0) - self.target_rate
            self.thresholds += self.adaptation_rate * error
        
        return adjusted, new_running_rates


class LateralInhibition(nn.Module):
    """
    Neurons inhibit their neighbors to maintain diversity.
    Prevents synchronization collapse.
    """
    def __init__(self, d_model, inhibition_strength=0.1, neighborhood_size=8):
        super().__init__()
        self.inhibition_strength = inhibition_strength
        self.neighborhood_size = neighborhood_size
        
        # Create inhibition kernel (local connectivity)
        # Each neuron inhibits its neighbors within neighborhood_size
        kernel = torch.zeros(d_model, d_model)
        for i in range(d_model):
            for j in range(max(0, i - neighborhood_size), min(d_model, i + neighborhood_size + 1)):
                if i != j:
                    distance = abs(i - j)
                    kernel[i, j] = -inhibition_strength / (1 + distance)
        
        self.register_buffer('inhibition_kernel', kernel)
        
    def forward(self, activations):
        """
        Args:
            activations: (B, d_model)
        Returns:
            inhibited_activations: (B, d_model)
        """
        # Compute lateral inhibition
        inhibition = torch.matmul(activations.detach(), self.inhibition_kernel.T)
        return activations + inhibition


class RefractoryDynamics(nn.Module):
    """
    Neurons that just fired become temporarily less responsive.
    Encourages temporal sparsity and sequential activation patterns.
    """
    def __init__(self, d_model, refractory_strength=0.3, refractory_decay=0.8):
        super().__init__()
        self.refractory_strength = refractory_strength
        self.refractory_decay = refractory_decay
        
    def forward(self, activations, refractory_state):
        """
        Args:
            activations: current activations (B, d_model)
            refractory_state: accumulated refractory level (B, d_model)
        Returns:
            adjusted_activations, new_refractory_state
        """
        # Apply refractory suppression
        suppression = torch.sigmoid(refractory_state) * self.refractory_strength
        adjusted = activations * (1 - suppression)
        
        # Update refractory state based on current activity
        activity_level = torch.abs(activations).detach()
        new_refractory = self.refractory_decay * refractory_state + activity_level
        
        return adjusted, new_refractory


class SynapticNoise(nn.Module):
    """
    Adds stochastic noise to maintain exploration in synchronization space.
    Biologically inspired by synaptic vesicle release probability.
    """
    def __init__(self, d_model, noise_scale=0.01):
        super().__init__()
        self.noise_scale = nn.Parameter(torch.tensor(noise_scale))
        
    def forward(self, activations, training=True):
        if training:
            noise = torch.randn_like(activations) * self.noise_scale
            return activations + noise
        return activations