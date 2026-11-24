"""Misaligned Roles, Misplaced Images: structural input perturbations for MLLM alignment.

Library code for the ICLR 2026 paper. The main entry points are:

- :mod:`blindspots.models`  -- model adapters (LLaVA / Qwen2-VL / Phi-3.5-V) that build
  perturbed prompts (role swap + image-token placement), generate, and extract activations.
- :mod:`blindspots.metrics` -- prefix-based refusal / jailbreak scoring.
- :mod:`blindspots.data`    -- dataset loaders (AdvBench, Alpaca, HarmBench, XSTest, SFT data).
- :mod:`blindspots.attacks` -- text jailbreak templates used as comparison attacks.
- :mod:`blindspots.vectors` -- direction vectors and layer-wise similarity utilities.
- :mod:`blindspots.sft`     -- the adversarial SFT mitigation (data building, collators, configs).
- :mod:`blindspots.plotting` -- figures (layer-wise PCA, cosine-similarity curves).

Runnable experiments live in ``scripts/``.
"""

__version__ = "0.1.0"
