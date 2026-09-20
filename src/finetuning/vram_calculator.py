"""
Fine-tuning memory estimator.

Checkpoint 4 asks for a fine-tuning analysis. The decisive question for a
student project is not "would fine-tuning help?" in the abstract, but "does it
fit on hardware I can actually get?" This module answers that arithmetically
instead of by assertion, so the numbers in docs/06_Fine_Tuning_Analysis.md can
be regenerated rather than trusted.

Run:  python src/finetuning/vram_calculator.py

Method. Training memory is dominated by four terms:

    weights     params x bytes_per_param
    gradients   trainable_params x bytes_per_param
    optimizer   trainable_params x bytes_per_param x optimizer_multiplier
    activations grows with batch size, sequence length and depth

Adam keeps two moments per trainable parameter, so its states cost roughly 2x
the trainable weights (8 bytes/param in fp32, the usual mixed-precision setup).
That term is what makes full fine-tuning explode, and it is exactly the term
LoRA removes by freezing the base model.

These are first-order estimates. They ignore gradient checkpointing, ZeRO
sharding, fused kernels and framework overhead, so treat them as a floor with
a margin, not a precise budget.
"""

from dataclasses import dataclass
from typing import Dict, List

BYTES_PER_GB = 1024**3

# Representative open-weight models a student might realistically fine-tune.
MODELS: Dict[str, float] = {
    "Llama-3.2-1B": 1.24e9,
    "Llama-3.2-3B": 3.21e9,
    "Mistral-7B": 7.24e9,
    "Llama-3.1-8B": 8.03e9,
    "Llama-3.1-70B": 70.6e9,
}

# GPUs a student can plausibly reach, and their usable VRAM in GB.
GPUS: Dict[str, float] = {
    "Colab free (T4)": 15.0,
    "RTX 3060": 12.0,
    "RTX 4090": 24.0,
    "A100 40GB": 40.0,
    "A100 80GB": 80.0,
}

# Adam stores two moment tensors per trainable parameter.
ADAM_STATE_MULTIPLIER = 2.0


@dataclass
class FineTuneEstimate:
    """Estimated training memory for one model/strategy pair."""

    model: str
    strategy: str
    trainable_params: float
    weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float

    @property
    def total_gb(self) -> float:
        """Total estimated VRAM in GB."""
        return self.weights_gb + self.gradients_gb + self.optimizer_gb + self.activations_gb

    @property
    def trainable_percent(self) -> float:
        """Share of parameters actually being trained."""
        return 100.0 * self.trainable_params / MODELS[self.model]

    def fits_on(self) -> List[str]:
        """Return the GPUs whose VRAM covers this estimate."""
        return [name for name, vram in GPUS.items() if vram >= self.total_gb]

    def summary(self) -> dict:
        """Return a compact record for reporting."""
        return {
            "model": self.model,
            "strategy": self.strategy,
            "trainable_percent": round(self.trainable_percent, 3),
            "total_gb": round(self.total_gb, 1),
            "fits_on": self.fits_on(),
        }


def lora_trainable_params(
    total_params: float, rank: int = 16, target_fraction: float = 0.35
) -> float:
    """
    Approximate the parameter count of LoRA adapters.

    LoRA replaces a weight update dW (d x k) with B @ A, where A is r x k and
    B is d x r. For a square-ish projection of width d, that is about 2*r*d
    parameters instead of d^2 -- the saving that makes this viable.

    Args:
        total_params: Base model parameter count
        rank: LoRA rank; higher means more capacity and more memory
        target_fraction: Share of parameters living in the attention and MLP
            projections that adapters are attached to

    Returns:
        Estimated trainable parameter count
    """
    # Empirically, adapters at rank r over the usual target modules land near
    # (2 * r / 1000) of the targeted weights for models in this size range.
    adapted = total_params * target_fraction
    return adapted * (2.0 * rank / 1000.0)


def estimate(
    model: str,
    strategy: str,
    rank: int = 16,
    batch_size: int = 4,
    sequence_length: int = 512,
) -> FineTuneEstimate:
    """
    Estimate training VRAM for one model and strategy.

    Args:
        model: A key of MODELS
        strategy: 'full', 'lora' or 'qlora'
        rank: LoRA rank, ignored for full fine-tuning
        batch_size: Training batch size
        sequence_length: Tokens per sequence

    Returns:
        The memory estimate

    Raises:
        ValueError: If the model or strategy is unknown
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}. Expected one of {list(MODELS)}.")
    if strategy not in ("full", "lora", "qlora"):
        raise ValueError(f"Unknown strategy {strategy!r}.")

    total = MODELS[model]

    # QLoRA quantises the frozen base to 4 bits; the others keep it in fp16.
    weight_bytes = 0.5 if strategy == "qlora" else 2.0
    weights_gb = total * weight_bytes / BYTES_PER_GB

    if strategy == "full":
        trainable = total
    else:
        trainable = lora_trainable_params(total, rank=rank)

    # Gradients and Adam moments are kept in fp32 for stability, even under
    # mixed precision.
    gradients_gb = trainable * 4.0 / BYTES_PER_GB
    optimizer_gb = trainable * 4.0 * ADAM_STATE_MULTIPLIER / BYTES_PER_GB

    # Activation memory scales with batch, sequence and model width; this is a
    # coarse proxy that tracks the right order of magnitude.
    hidden_estimate = (total / 1e9) ** 0.5 * 2048
    activations_gb = (
        batch_size * sequence_length * hidden_estimate * 2.0 * 24 / BYTES_PER_GB
    )

    return FineTuneEstimate(
        model=model,
        strategy=strategy,
        trainable_params=trainable,
        weights_gb=weights_gb,
        gradients_gb=gradients_gb,
        optimizer_gb=optimizer_gb,
        activations_gb=activations_gb,
    )


def compare_strategies(model: str, rank: int = 16) -> List[FineTuneEstimate]:
    """
    Estimate all three strategies for one model.

    Args:
        model: A key of MODELS
        rank: LoRA rank

    Returns:
        Estimates for full, LoRA and QLoRA
    """
    return [estimate(model, s, rank=rank) for s in ("full", "lora", "qlora")]


def markdown_table(model: str, rank: int = 16) -> str:
    """Render the three-strategy comparison for one model as Markdown."""
    rows = [
        "| Strategy | Trainable | Weights | Grads | Optimizer | Activations | **Total** | Fits on |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in compare_strategies(model, rank=rank):
        fits = item.fits_on()
        rows.append(
            f"| {item.strategy.upper()} | {item.trainable_percent:.2f}% | "
            f"{item.weights_gb:.1f} GB | {item.gradients_gb:.1f} GB | "
            f"{item.optimizer_gb:.1f} GB | {item.activations_gb:.1f} GB | "
            f"**{item.total_gb:.1f} GB** | "
            f"{fits[0] if fits else 'nothing listed'} |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    print("=" * 78)
    print("FINE-TUNING MEMORY ESTIMATES (batch=4, seq=512, LoRA rank=16)")
    print("=" * 78)

    for model in MODELS:
        print(f"\n### {model} ({MODELS[model] / 1e9:.2f}B parameters)\n")
        print(markdown_table(model))

    print("\n" + "=" * 78)
    print("The pattern: full fine-tuning is dominated by optimizer state, which")
    print("LoRA removes by freezing the base. QLoRA then shrinks the frozen base")
    print("itself, which is what brings a 7B model onto a consumer GPU.")
    print("=" * 78)
