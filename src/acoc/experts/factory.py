"""
ACOC - Expert Factory
=====================
Fabrique pour instancier les experts selon leur type.
"""

from .mlp import MLPExpert
from .cnn import CNNExpert
from ..config import SystemConfig

class ExpertFactory:
    """
    Factory statique pour créer des experts.
    """
    
    @staticmethod
    def create(
        expert_type: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        name: str,
        config: SystemConfig
    ):
        expert_type = expert_type.lower()
        
        if expert_type == "mlp":
            return MLPExpert(input_dim, hidden_dim, output_dim, name, config)
        elif expert_type == "cnn":
            return CNNExpert(input_dim, hidden_dim, output_dim, name, config)
        else:
            # Fallback ou erreur
            raise ValueError(f"Type d'expert inconnu : {expert_type}")