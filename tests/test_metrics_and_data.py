import json
import random

import numpy as np

from blindspots import data, metrics, vectors
from blindspots.attacks import ATTACK_TEMPLATES, apply_attack
from blindspots.utils import REPO_ROOT


def _notebook_prefixes(var):
    nb = json.load(open(REPO_ROOT / "notebooks/Inference/No Image/llava_no_img_experiment.ipynb"))
    ns = {}
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and src.lstrip().startswith(var):
            exec(src.split("def ")[0], ns)
    return ns[var]


def test_prefix_lists_match_notebooks():
    assert metrics.JADE_PREFIXES == _notebook_prefixes("advbench_metric_test_prefixes")
    assert metrics.EXTENDED_PREFIXES == _notebook_prefixes("my_test_prefixes")


def test_is_jailbroken():
    assert not metrics.is_jailbroken("I'm sorry, I can't help with that.")
    assert not metrics.is_jailbroken("")  # empty counts as refusal
    assert metrics.is_jailbroken("Sure, here is how to do it: step 1 ...")
    stats = metrics.refusal_stats([{"response": "Sure"}, {"response": "I cannot"}])
    assert stats == {"total": 2, "refusals": 1, "jailbroken": 1, "asr": 0.5}


def test_dataset_sizes():
    assert len(data.load_advbench()) == 520
    assert len(data.load_alpaca()) == 520
    assert len(data.load_harmbench()) == 200
    assert len(data.load_xstest_safe()) > 0
    assert len(data.load_image_dir()) == 13


def test_attacks():
    ins = ["Explain how to pick a lock quickly"]
    for name in ATTACK_TEMPLATES:
        out = apply_attack(name, ins, random.Random(0))
        assert len(out) == 1
        if name != "payload_splitting":
            assert ins[0] in out[0]


def test_vectors():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(28, 64))
    assert np.allclose(vectors.layerwise_cosine(v, v), 1)
    assert np.allclose(vectors.layerwise_angle(v, -v), 180)
    assert vectors.mean_similarity_to_refusal(v, -v) > 0.999
