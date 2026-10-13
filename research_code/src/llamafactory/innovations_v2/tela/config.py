from dataclasses import dataclass


@dataclass
class TELAConfig:
    algorithm: str = "reptile"
    tasks_per_meta_batch: int = 2
    support_size: int = 4
    query_size: int = 4
    inner_steps: int = 3
    inner_lr: float = 1e-4
    meta_step_size: float = 0.1
    adapt_parameters: str = "lora_only"

    def validate(self) -> None:
        if self.algorithm != "reptile":
            raise ValueError("TELA currently supports only Reptile-style first-order adaptation.")
        if min(self.tasks_per_meta_batch, self.support_size, self.query_size, self.inner_steps) < 1:
            raise ValueError("TELA task, support, query and inner-step counts must be positive.")
        if self.inner_lr <= 0 or not 0 < self.meta_step_size <= 1:
            raise ValueError("TELA learning rates must be positive and meta_step_size must be in (0, 1].")
        if self.adapt_parameters != "lora_only":
            raise ValueError("TELA v2 only permits LoRA parameter adaptation.")

