"""
ACOC - Expert Factory
=====================
Fabrique pour instancier les experts selon leur type.
"""

from .mlp import MLPExpert, AudioMLPExpert
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

        elif expert_type == "audio_mlp":
            return AudioMLPExpert(input_dim, hidden_dim, output_dim, name, config)

        elif expert_type == "cnn":
            try:
                return CNNExpert(input_dim, hidden_dim, output_dim, name, config)
            except Exception as e:
                # Si le CNN ne peut pas être créé (dimensions incorrectes, etc.)
                # Fallback vers MLP avec un warning
                import warnings
                warnings.warn(
                    f"⚠️  Impossible de créer un CNN pour {name} (erreur: {e}). "
                    f"Fallback vers MLP.",
                    RuntimeWarning
                )
                return MLPExpert(input_dim, hidden_dim, output_dim, name, config)

        else:
            # Type inconnu, fallback vers MLP
            import warnings
            warnings.warn(
                f"Type d'expert inconnu: {expert_type}. Fallback vers MLP.",
                RuntimeWarning
            )
            return MLPExpert(input_dim, hidden_dim, output_dim, name, config)