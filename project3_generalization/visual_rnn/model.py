from __future__ import annotations

from project3_generalization.models.hippocampal_module import HippocampalConfig, HippocampalPredictiveRNN


def build_visual_model_config(
    *,
    include_head_direction: bool = True,
    hidden_size: int = 384,
    encoder_type: str = "cnn",
    encoder_hidden_size: int = 256,
    encoder_output_size: int = 128,
    rollout_steps: int = 3,
    rollout_loss_weight: float = 0.5,
    latent_loss_weight: float = 0.1,
    pRNNtype: str = "vRNN_0win",
    learning_rate: float = 5e-4,
    use_amp: bool = True,
    amp_dtype: str = "float16",
    gradient_checkpointing: bool = True,
) -> HippocampalConfig:
    """Build a visual-input config for the predictive RNN."""

    head_direction_size = 2 if include_head_direction else 0
    return HippocampalConfig(
        obs_size=147 + head_direction_size,
        action_size=3,
        hidden_size=hidden_size,
        pRNNtype=pRNNtype,
        learning_rate=learning_rate,
        truncation=64,
        chunk_length=64,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        gradient_checkpointing=gradient_checkpointing,
        encoder_type=encoder_type,
        encoder_hidden_size=encoder_hidden_size,
        encoder_output_size=encoder_output_size,
        visual_patch_size=147,
        visual_patch_width=7,
        visual_channels=3,
        head_direction_size=head_direction_size,
        rollout_steps=rollout_steps,
        rollout_loss_weight=rollout_loss_weight,
        latent_loss_weight=latent_loss_weight,
        rollout_mode="autoregressive",
    )


__all__ = [
    "HippocampalConfig",
    "HippocampalPredictiveRNN",
    "build_visual_model_config",
]
