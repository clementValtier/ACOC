"""
ACOC - Composants du Réseau (PyTorch)
=====================================
Router et Expert : les briques de base du modèle.
Avec analyse du gradient flow et saturation des activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from collections import deque

from .structures import SaturationMetrics


class GradientFlowMonitor:
    """
    Moniteur de gradient flow pour détecter les blocages.
    
    Analyse:
    - Magnitude des gradients par couche
    - Ratio de gradients "vivants"
    - Détection de vanishing/exploding gradients
    """
    
    def __init__(self, threshold: float = 1e-6, history_size: int = 100):
        self.threshold = threshold
        self.history_size = history_size
        self.gradient_history: Dict[str, deque] = {}
    
    def register_layer(self, name: str):
        """Enregistre une couche à monitorer."""
        if name not in self.gradient_history:
            self.gradient_history[name] = deque(maxlen=self.history_size)
    
    def record_gradients(self, name: str, gradients: torch.Tensor):
        """Enregistre les gradients d'une couche."""
        if name not in self.gradient_history:
            self.register_layer(name)
        
        with torch.no_grad():
            grad_abs = gradients.abs()
            stats = {
                'mean': grad_abs.mean().item(),
                'max': grad_abs.max().item(),
                'alive_ratio': (grad_abs > self.threshold).float().mean().item()
            }
            self.gradient_history[name].append(stats)
    
    def get_flow_ratio(self, name: str) -> float:
        """
        Retourne le ratio de gradients vivants (moyenne récente).
        1.0 = tous les gradients circulent bien
        0.0 = gradient flow bloqué
        """
        if name not in self.gradient_history or len(self.gradient_history[name]) == 0:
            return 1.0
        
        recent = list(self.gradient_history[name])[-20:]
        avg_alive = sum(s['alive_ratio'] for s in recent) / len(recent)
        return avg_alive
    
    def get_all_flow_ratios(self) -> Dict[str, float]:
        """Retourne les ratios pour toutes les couches."""
        return {name: self.get_flow_ratio(name) for name in self.gradient_history}


class ActivationMonitor:
    """
    Moniteur de saturation des activations.
    
    Détecte:
    - Neurones saturés (toujours au max)
    - Neurones morts (toujours à 0)
    - Variance des activations
    """
    
    def __init__(
        self, 
        saturation_threshold: float = 0.95,
        dead_threshold: float = 1e-6,
        history_size: int = 100
    ):
        self.saturation_threshold = saturation_threshold
        self.dead_threshold = dead_threshold
        self.history_size = history_size
        self.activation_history: Dict[str, deque] = {}
    
    def register_layer(self, name: str):
        """Enregistre une couche à monitorer."""
        if name not in self.activation_history:
            self.activation_history[name] = deque(maxlen=self.history_size)
    
    def record_activations(self, name: str, activations: torch.Tensor):
        """Enregistre les activations d'une couche."""
        if name not in self.activation_history:
            self.register_layer(name)
        
        with torch.no_grad():
            # Flatten si nécessaire
            act = activations.view(activations.size(0), -1)  # [batch, neurons]
            
            # Statistiques par neurone (moyennées sur le batch)
            neuron_means = act.mean(dim=0)  # [neurons]
            
            # Théorique max pour ReLU (estimation basée sur les données)
            estimated_max = act.max().item() if act.max().item() > 0 else 1.0
            
            stats = {
                'neuron_means': neuron_means.cpu(),
                'saturated_ratio': (neuron_means > self.saturation_threshold * estimated_max).float().mean().item(),
                'dead_ratio': (neuron_means < self.dead_threshold).float().mean().item(),
                'variance': act.var().item(),
                'mean': act.mean().item()
            }
            self.activation_history[name].append(stats)
    
    def get_saturation_metrics(self, name: str) -> Tuple[float, float, float]:
        """
        Retourne (saturation_ratio, dead_ratio, variance) pour une couche.
        """
        if name not in self.activation_history or len(self.activation_history[name]) == 0:
            return 0.0, 0.0, 1.0
        
        recent = list(self.activation_history[name])[-20:]
        
        avg_saturated = sum(s['saturated_ratio'] for s in recent) / len(recent)
        avg_dead = sum(s['dead_ratio'] for s in recent) / len(recent)
        avg_variance = sum(s['variance'] for s in recent) / len(recent)
        
        return avg_saturated, avg_dead, avg_variance


class Router(nn.Module):
    """
    Routeur central qui dirige les inputs vers les bons experts/blocs.
    
    Avec protection EWC contre le catastrophic forgetting.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_routes: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_routes = num_routes
        
        # Réseau de routage (petit MLP)
        self.routing_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_routes)
        )
        
        # EWC: Fisher information et anciens poids
        self.fisher_info: Optional[Dict[str, torch.Tensor]] = None
        self.old_params: Optional[Dict[str, torch.Tensor]] = None
        
        # Monitoring
        self.gradient_monitor = GradientFlowMonitor()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [batch_size, input_dim]
            
        Returns:
            (selected_indices, probabilities)
        """
        logits = self.routing_net(x)
        probabilities = F.softmax(logits, dim=-1)
        selected = probabilities.argmax(dim=-1)
        
        return selected, probabilities
    
    def forward_with_exploration(
        self, 
        x: torch.Tensor, 
        force_route: Optional[int] = None,
        exploration_prob: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward avec exploration forcée vers certaines routes.
        Utilisé pour le warmup après expansion.
        """
        logits = self.routing_net(x)
        probabilities = F.softmax(logits, dim=-1)
        
        if force_route is not None and exploration_prob > 0:
            # Avec probabilité exploration_prob, forcer vers force_route
            batch_size = x.size(0)
            mask = torch.rand(batch_size, device=x.device) < exploration_prob
            selected = probabilities.argmax(dim=-1)
            selected[mask] = force_route
        else:
            selected = probabilities.argmax(dim=-1)
        
        return selected, probabilities
    
    def add_route(self, device: torch.device = None):
        """Ajoute une nouvelle route (pour un nouveau bloc)."""
        if device is None:
            device = next(self.parameters()).device
            
        old_out = self.routing_net[-1]
        new_out = nn.Linear(old_out.in_features, self.num_routes + 1).to(device)
        
        # Copier les anciens poids
        with torch.no_grad():
            new_out.weight[:self.num_routes] = old_out.weight
            new_out.bias[:self.num_routes] = old_out.bias
            # Initialiser la nouvelle route avec de petits poids
            nn.init.xavier_uniform_(new_out.weight[self.num_routes:self.num_routes+1])
            new_out.bias[self.num_routes] = 0.0
        
        self.routing_net[-1] = new_out
        self.num_routes += 1
        
        # Invalider EWC (les anciens paramètres ne sont plus valides)
        self.fisher_info = None
        self.old_params = None
    
    def compute_fisher(self, data_loader, num_samples: int = 500):
        """
        Calcule la Fisher information matrix pour EWC.
        """
        device = next(self.parameters()).device
        self.fisher_info = {n: torch.zeros_like(p) for n, p in self.named_parameters()}
        self.old_params = {n: p.clone().detach() for n, p in self.named_parameters()}
        
        self.eval()
        count = 0
        
        for batch in data_loader:
            if count >= num_samples:
                break
            
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(device)
            self.zero_grad()
            
            _, probs = self.forward(x)
            # Log-likelihood de la prédiction
            log_probs = torch.log(probs + 1e-8)
            selected = probs.argmax(dim=-1)
            loss = -log_probs.gather(1, selected.unsqueeze(1)).mean()
            loss.backward()
            
            for n, p in self.named_parameters():
                if p.grad is not None:
                    self.fisher_info[n] += p.grad.pow(2)
            
            count += x.size(0)
        
        # Normaliser
        for n in self.fisher_info:
            self.fisher_info[n] /= max(count, 1)
        
        self.train()
    
    def ewc_loss(self, lambda_ewc: float = 100.0) -> torch.Tensor:
        """Calcule la pénalité EWC."""
        device = next(self.parameters()).device
        
        if self.fisher_info is None or self.old_params is None:
            return torch.tensor(0.0, device=device)
        
        loss = torch.tensor(0.0, device=device)
        for n, p in self.named_parameters():
            if n in self.fisher_info:
                loss += (self.fisher_info[n] * (p - self.old_params[n]).pow(2)).sum()
        
        return lambda_ewc * 0.5 * loss
    
    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Expert(nn.Module):
    """
    Un expert (MLP) avec monitoring des activations et gradients.
    Supporte l'expansion Net2Net.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        name: str = "expert"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.name = name
        
        # Couches
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Activation
        self.activation = nn.ReLU()
        
        # Monitoring
        self.activation_monitor = ActivationMonitor()
        self.gradient_monitor = GradientFlowMonitor()
        
        # Buffer pour les activations
        self._last_hidden: Optional[torch.Tensor] = None
        self._hooks = []
        
        # Hook pour le gradient
        self._register_hooks()
    
    def _register_hooks(self):
        """Enregistre les hooks pour le monitoring."""
        # Supprimer les anciens hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        
        def save_hidden_hook(module, input, output):
            self._last_hidden = output.detach()
            self.activation_monitor.record_activations(f"{self.name}_hidden", output.detach())
        
        def save_gradient_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradient_monitor.record_gradients(
                    f"{self.name}_fc1", 
                    grad_output[0].detach()
                )
        
        h1 = self.fc1.register_forward_hook(save_hidden_hook)
        h2 = self.fc1.register_full_backward_hook(save_gradient_hook)
        self._hooks = [h1, h2]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        hidden = self.activation(self.fc1(x))
        output = self.fc2(hidden)
        return output
    
    def get_saturation_metrics(self) -> SaturationMetrics:
        """
        Calcule les métriques de saturation complètes.
        """
        metrics = SaturationMetrics()
        
        # Gradient flow
        metrics.gradient_flow_ratio = self.gradient_monitor.get_flow_ratio(f"{self.name}_fc1")
        
        # Activation saturation
        sat, dead, var = self.activation_monitor.get_saturation_metrics(f"{self.name}_hidden")
        metrics.activation_saturation = sat
        metrics.dead_neuron_ratio = dead
        metrics.activation_variance = var
        
        # Score combiné
        metrics.compute_combined_score()
        
        return metrics
    
    def expand_width(self, additional_neurons: int):
        """
        Expansion Net2Net en largeur.
        Préserve la fonction du réseau.
        """
        if additional_neurons <= 0:
            return
        
        device = self.fc1.weight.device
        
        with torch.no_grad():
            # Sélectionner des neurones à dupliquer
            indices = torch.randint(0, self.hidden_dim, (additional_neurons,))
            
            # === Expansion fc1 (colonnes = neurones de sortie) ===
            new_fc1 = nn.Linear(self.input_dim, self.hidden_dim + additional_neurons).to(device)
            new_fc1.weight[:self.hidden_dim] = self.fc1.weight
            new_fc1.weight[self.hidden_dim:] = self.fc1.weight[indices]
            new_fc1.bias[:self.hidden_dim] = self.fc1.bias
            new_fc1.bias[self.hidden_dim:] = self.fc1.bias[indices]
            
            # Ajouter du bruit pour casser la symétrie
            noise_scale = 0.001 * self.fc1.weight.std().item()
            new_fc1.weight[self.hidden_dim:] += torch.randn_like(
                new_fc1.weight[self.hidden_dim:]
            ) * noise_scale
            
            # === Expansion fc2 (lignes = neurones d'entrée) ===
            new_fc2 = nn.Linear(self.hidden_dim + additional_neurons, self.output_dim).to(device)
            
            # D'abord copier fc2.weight avec division pour les indices dupliqués
            old_weight = self.fc2.weight.clone()
            old_weight[:, indices] /= 2  # Diviser les colonnes dupliquées
            
            new_fc2.weight[:, :self.hidden_dim] = old_weight
            new_fc2.weight[:, self.hidden_dim:] = self.fc2.weight[:, indices] / 2
            new_fc2.bias = nn.Parameter(self.fc2.bias.clone())
            
            # Remplacer les couches
            self.fc1 = new_fc1
            self.fc2 = new_fc2
            self.hidden_dim += additional_neurons
            
            # Ré-enregistrer les hooks
            self._register_hooks()
    
    def reset_monitors(self):
        """Réinitialise les moniteurs (après expansion)."""
        self.activation_monitor = ActivationMonitor()
        self.gradient_monitor = GradientFlowMonitor()
        self._register_hooks()
    
    def get_param_count(self) -> int:
        """Retourne le nombre de paramètres."""
        return sum(p.numel() for p in self.parameters())


class ExpertBlock(nn.Module):
    """
    Un bloc contenant potentiellement plusieurs experts en séquence.
    """
    
    def __init__(
        self, 
        experts: List[Expert],
        name: str = "block"
    ):
        super().__init__()
        self.name = name
        self.experts = nn.ModuleList(experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for expert in self.experts:
            x = expert(x)
        return x
    
    def get_combined_saturation(self) -> SaturationMetrics:
        """Retourne les métriques combinées de tous les experts."""
        if not self.experts:
            return SaturationMetrics()
        
        all_metrics = [e.get_saturation_metrics() for e in self.experts]
        
        combined = SaturationMetrics()
        n = len(all_metrics)
        
        combined.gradient_flow_ratio = sum(m.gradient_flow_ratio for m in all_metrics) / n
        combined.activation_saturation = sum(m.activation_saturation for m in all_metrics) / n
        combined.dead_neuron_ratio = sum(m.dead_neuron_ratio for m in all_metrics) / n
        combined.activation_variance = sum(m.activation_variance for m in all_metrics) / n
        combined.compute_combined_score()
        
        return combined
    
    def expand_all_experts(self, ratio: float = 0.1):
        """Expand tous les experts du bloc."""
        for expert in self.experts:
            additional = max(1, int(expert.hidden_dim * ratio))
            expert.expand_width(additional)
    
    def reset_all_monitors(self):
        """Reset tous les moniteurs du bloc."""
        for expert in self.experts:
            expert.reset_monitors()
    
    def get_param_count(self) -> int:
        return sum(e.get_param_count() for e in self.experts)
