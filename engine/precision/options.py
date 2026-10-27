"""Immutable numerical experiments. Defaults select a single, unmodified product."""
from dataclasses import asdict, dataclass

from engine.precision.formats import format_of


@dataclass(frozen=True)
class ProductOptions:
    algorithm: str = "format"
    shared_terms: int = 1
    scaling: str = "none"
    shared_format: str = "same"
    bf16_padding: str = "none"
    subnormals: str = "native"
    implementation: str = "auto"

    def __post_init__(self):
        choices = {"algorithm": ("format", "fp8_full"), "scaling": ("none", "power_of_two"),
                   "shared_format": ("same", "e4m3"), "bf16_padding": ("none", "gemm8"),
                   "subnormals": ("native", "normalize_fp16"), "implementation": ("auto", "emulated_fp32")}
        for name, values in choices.items():
            if getattr(self, name) not in values:
                raise ValueError(f"ProductOptions.{name} must be one of {values}")
        if type(self.shared_terms) is not int or not 1 <= self.shared_terms <= 4:
            raise ValueError("ProductOptions.shared_terms must be an integer in [1, 4]")

    def validate(self, compute):
        fmt = format_of(compute)
        if not fmt.below_float32:
            if self != ProductOptions():
                raise ValueError("product options apply to compute formats below float32")
            return
        if fmt.product == "bfloat16_3x" and (self.shared_terms != 1 or self.scaling != "none"):
            raise ValueError("bfloat16_3x already specifies its decomposition; shared_terms=1 and scaling='none'")
        if compute.endswith("_1x") and self.shared_terms != 1:
            raise ValueError("a _1x compute format requires shared_terms=1")
        if self.scaling == "power_of_two" and fmt.base not in ("float16", "float8_e4m3fn", "float8_e5m2"):
            raise ValueError("power_of_two product scaling requires a range-limited FP16/FP8 format")
        if self.shared_format == "e4m3" and fmt.base != "float8_e5m2":
            raise ValueError("shared_format='e4m3' requires float8_e5m2 compute")
        if self.bf16_padding != "none" and fmt.product != "bfloat16_3x":
            raise ValueError("bf16_padding='gemm8' requires bfloat16_3x compute")
        if self.subnormals != "native" and fmt.base != "float16":
            raise ValueError("subnormals='normalize_fp16' requires float16 compute")
        if self.algorithm == "fp8_full" and not fmt.base.startswith("float8"):
            raise ValueError("algorithm='fp8_full' requires FP8 compute")
        if self.implementation == "emulated_fp32" and (self.algorithm != "format" or self.subnormals != "native"):
            raise ValueError("emulated_fp32 cannot request a native FP8 algorithm or native FP16 normalization")

    @classmethod
    def compensated(cls, compute):
        """Explicitly select the engine's previous scaled/residual/padded profile."""
        fmt = format_of(compute)
        return cls(shared_terms=fmt.shared_terms,
                   scaling="power_of_two" if fmt.base in ("float16", "float8_e4m3fn", "float8_e5m2") else "none",
                   shared_format="e4m3" if fmt.shared_dtype else "same",
                   bf16_padding="gemm8" if fmt.product == "bfloat16_3x" else "none")

    def description(self, compute):
        self.validate(compute)
        return {**asdict(self), "qualification": "not_established_by_configuration",
                "max_subnormal_product_count": 4 if self.subnormals == "normalize_fp16" else 1,
                "subnormal_fast_path": self.subnormals == "normalize_fp16"}
