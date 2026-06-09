from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    name: str
    logger_name: str
    output_dim: int
    loss_kind: str
    supports_video: bool = False
    final_activation: str | None = None


GRAYSCALE_TASK = TaskConfig(
    name='grayscale',
    logger_name='polytope_nn',
    output_dim=1,
    loss_kind='mse',
    supports_video=True,
)

CLASSIFIER_TASK = TaskConfig(
    name='classifier',
    logger_name='polytope_classifier',
    output_dim=3,
    loss_kind='cross_entropy',
    final_activation='relu',
)
