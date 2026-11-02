"""Check input gradients, vision bridge equivalence, and PNG input transport."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_attack import LlavaBackend, objective


def full_profile(backend, rgb):
    out = backend.model(input_ids=backend.ids, images=backend.pixel_values(rgb),
                        output_attentions=True, use_cache=False, return_dict=True)
    start, end, qstart, qend = backend.span
    scores = torch.stack([out.attentions[i][0, :, qstart:qend, start:end].float().mean((0, 1))
                          for i in backend.layers]).mean(0)
    return scores / scores.sum().clamp_min(1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backend = LlavaBackend(SimpleNamespace(model_path=args.model_path))
    rgb = backend.prepare(Image.open(args.image).convert("RGB"), "Please describe this image in detail.")
    checks = {"all_parameters_frozen": all(not p.requires_grad for p in backend.model.parameters()),
              "model_eval": not backend.model.training, "layers": backend.layers,
              "query_text": backend.query_text, "attention_span": list(backend.span)}
    tower = backend.model.get_vision_tower()
    with torch.no_grad():
        normalized = backend.pixel_values(rgb)
        original = type(tower).forward(tower, normalized)
        bridged = tower(normalized)
        error = (original - bridged).abs().max().item()
        checks["vision_gradient_bridge_forward_max_error"] = error
        torch.testing.assert_close(original, bridged, atol=0, rtol=0)
        del original, bridged
        reference = backend.profile(rgb).detach()
    saved = torch.load(args.results / "inputs.pt", map_location=backend.device)
    torch.testing.assert_close(saved["clean_rgb"], rgb, atol=0, rtol=0)
    for key, name in [("clean_rgb", "clean.png"), ("adv_rgb", "adv.png")]:
        transported = backend.prepare(Image.open(args.results / name).convert("RGB"),
                                      "Please describe this image in detail.")
        difference = (saved[key] - transported).abs().max().item()
        checks[name + "_processor_roundtrip_max_error"] = difference
        torch.testing.assert_close(saved[key], transported, atol=0, rtol=0)
    x = saved["adv_rgb"].detach()
    topk = reference.topk(50).indices
    gradients, losses = [], []
    for label, forward in [("stopped", backend.profile), ("full", lambda value: full_profile(backend, value))]:
        variable = x.clone().requires_grad_(True)
        # Offload saved tensors to CPU to fit the full differentiable decoder
        # forward on a 24 GB GPU. This changes storage only.
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            distribution = forward(variable)
            loss, terms = objective(distribution, reference, topk)
        gradient = torch.autograd.grad(loss * 1024, variable)[0] / 1024
        if not torch.isfinite(gradient).all():
            raise RuntimeError("Non-finite gradient in " + label)
        gradients.append(gradient.cpu())
        losses.append(loss.detach().item())
        del distribution, loss, terms, gradient, variable
        torch.cuda.empty_cache()
    absolute = (gradients[0] - gradients[1]).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(gradients[0].flatten(), gradients[1].flatten(), dim=0).item()
    sign_agreement = (gradients[0].sign() == gradients[1].sign()).float().mean().item()
    checks.update(objectives=losses, gradient_max_abs_error=absolute,
                  gradient_cosine=cosine, gradient_sign_agreement=sign_agreement)
    torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)
    args.output.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(checks, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
