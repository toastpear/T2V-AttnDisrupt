"""Validate the bundled five-image experiment without loading model weights."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main():
    samples = json.loads((ROOT / "data/samples.json").read_text())
    results = json.loads((ROOT / "examples/outputs/results.json").read_text())
    assert len(samples) == len(results) == 5
    for index, sample in enumerate(samples):
        directory = ROOT / "examples/outputs" / f"{sample['image_id']:012d}"
        result = json.loads((directory / "result.json").read_text())
        assert result == results[index]
        assert result["complete"] and result["image_id"] == sample["image_id"]
        assert result["iterations"] == 50 and result["seed"] == 42 + index
        assert result["layers"] == list(range(5, 19))
        assert result["weights"] == {"entropy": 1, "suppression": 5, "js": 1}
        assert result["clean_caption"].strip() and result["attacked_caption"].strip()
        original = ROOT / "data/images" / sample["file_name"]
        assert hashlib.sha256(original.read_bytes()).hexdigest() == result["source_sha256"]
        clean = np.asarray(Image.open(directory / "clean.png").convert("RGB"), dtype=np.int16)
        adversarial = np.asarray(Image.open(directory / "adv.png").convert("RGB"), dtype=np.int16)
        assert clean.shape == adversarial.shape == (336, 336, 3)
        maximum = int(np.abs(adversarial - clean).max())
        assert 0 < maximum <= 8
        history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
        assert [step["step"] for step in history] == list(range(1, 51))
        assert all(np.isfinite(step["gradient_mean_abs"]) and step["gradient_mean_abs"] > 0 for step in history)
        assert all(step["linf_rgb"] <= 8 / 255 + 2e-6 for step in history)
        print(f"COCO {sample['image_id']:012d}: 50 steps; matching source hash; PNG max channel change {maximum}/255")
    runtime = json.loads((ROOT / "validation/runtime_checks.json").read_text())
    assert runtime["all_parameters_frozen"] and runtime["model_eval"]
    for key in ["vision_gradient_bridge_forward_max_error", "clean.png_processor_roundtrip_max_error",
                "adv.png_processor_roundtrip_max_error", "gradient_max_abs_error"]:
        assert runtime[key] == 0
    print("Five pairs and the recorded runtime checks validated.")


if __name__ == "__main__":
    main()
