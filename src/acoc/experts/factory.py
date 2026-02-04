"""
ACOC - Expert Factory
=====================
Factory to instantiate experts based on their type.
"""

from .mlp import MLPExpert, AudioMLPExpert
from .cnn import CNNExpert
from ..config import SystemConfig

class ExpertFactory:
    """
    Static factory to create experts.
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
                # If CNN cannot be created (incorrect dimensions, etc.)
                # Fallback to MLP with warning
                import warnings
                warnings.warn(
                    f"Cannot create CNN for {name} (error: {e}). "
                    f"Falling back to MLP.",
                    RuntimeWarning
                )
                return MLPExpert(input_dim, hidden_dim, output_dim, name, config)

        else:
            # Unknown type, fallback to MLP
            import warnings
            warnings.warn(
                f"Unknown expert type: {expert_type}. Falling back to MLP.",
                RuntimeWarning
            )
            return MLPExpert(input_dim, hidden_dim, output_dim, name, config)