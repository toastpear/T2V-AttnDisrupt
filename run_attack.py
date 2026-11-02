"""Five-image T2V-AttnDisrupt example: RGB PGD and clean/adversarial captions.

This is a new, paper-objective example, not a replay of historical run settings.
LLaVA uses the upstream LLaVA package; Qwen uses Transformers 4.51.3.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import types

import numpy as np
from PIL import Image
import torch


class AttentionReady(Exception):
    pass


def fraction(value):
    if "/" in value:
        a, b = value.split("/")
        return float(a) / float(b)
    return float(value)


def save_rgb(tensor, path):
    pixels = tensor.detach().float().clamp(0, 1)[0].permute(1, 2, 0)
    Image.fromarray(np.rint(pixels.cpu().numpy() * 255).astype(np.uint8)).save(path)


def entropy(p):
    p = p.float().clamp_min(1e-12)
    return -(p * p.log()).sum()


def objective(p, ref, topk):
    p, ref = p.float().clamp_min(1e-12), ref.float().clamp_min(1e-12)
    midpoint = (p + ref) / 2
    dh = entropy(p) - entropy(ref)
    suppression = 1 - p[topk].sum() / ref[topk].sum().clamp_min(1e-12)
    js = ((p * (p.log() - midpoint.log())).sum() +
          (ref * (ref.log() - midpoint.log())).sum()) / 2
    return dh + 5 * suppression + js, {"delta_entropy": dh, "suppression": suppression, "js": js}


def attention_profile(model, layers, span, forward, device):
    """Keep prompt-to-image profiles and stop after the last selected layer.

    Stopping the prefill after that layer preserves the objective and its input
    gradient, while avoiding unused decoder layers and vocabulary logits.
    """
    start, end, qstart, qend = span
    profiles, handles = [], []

    def capture(module, inputs, output):
        weights = output[1]
        if weights is None:
            raise RuntimeError("Attention weights missing: use eager attention")
        block = weights[0, :, qstart:qend, start:end]
        if not block.numel():
            raise RuntimeError("Empty prompt-to-image attention block")
        profiles.append(block.float().mean(dim=(0, 1)).to(device))

    def stop(module, inputs, output):
        raise AttentionReady()

    try:
        for index in layers:
            handles.append(model.model.layers[index].self_attn.register_forward_hook(capture))
        handles.append(model.model.layers[max(layers)].register_forward_hook(stop))
        try:
            forward()
        except AttentionReady:
            pass
        else:
            raise RuntimeError("Expected selected decoder layer was not reached")
    finally:
        for handle in handles:
            handle.remove()
    if len(profiles) != len(layers):
        raise RuntimeError(f"Captured {len(profiles)} of {len(layers)} requested layers")
    scores = torch.stack(profiles).mean(dim=0)
    return scores / scores.sum().clamp_min(1e-12)


class LlavaBackend:
    def __init__(self, args):
        from llava.model.builder import load_pretrained_model
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            args.model_path, None, "llava-v1.5-7b", device="cuda:0",
            attn_implementation="eager",
        )
        self.model.eval().requires_grad_(False)
        vision = self.model.get_vision_tower()

        def differentiable_forward(tower, images):
            result = tower.vision_tower(images.to(device=tower.device, dtype=tower.dtype),
                                        output_hidden_states=True)
            return tower.feature_select(result).to(images.dtype)

        # Upstream has @no_grad on this method; replace only this instance.
        vision.forward = types.MethodType(differentiable_forward, vision)
        self.device = torch.device("cuda:0")
        self.mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)
        self.layers = list(range(5, 19))

    def prepare(self, image, prompt):
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token, process_images
        conversation = conv_templates["llava_v1"].copy()
        conversation.append_message(conversation.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conversation.append_message(conversation.roles[1], None)
        full_prompt = conversation.get_prompt()
        ids = tokenizer_image_token(full_prompt, self.tokenizer,
                                    IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        normalized = process_images([image], self.processor, self.model.config).float().to(self.device)
        clean = ((normalized * self.std + self.mean).clamp(0, 1) * 255).round() / 255
        position = ids[0].tolist().index(IMAGE_TOKEN_INDEX)
        end = position + self.model.get_vision_tower().num_patches
        prefix = full_prompt[:full_prompt.rindex(prompt) + len(prompt)]
        prefix_ids = tokenizer_image_token(prefix, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        prefix_length = len(prefix_ids)
        if not torch.equal(ids[0, :prefix_length].cpu(), prefix_ids):
            raise RuntimeError("Prompt boundary tokenization changed")
        qend = prefix_length - 1 + self.model.get_vision_tower().num_patches
        self.ids = ids
        self.span = (position, end, end, qend)
        self.query_text = self.tokenizer.decode(ids[0, position + 1:prefix_length])
        return clean

    def pixel_values(self, rgb):
        return ((rgb - self.mean) / self.std).to(dtype=self.model.dtype)

    def profile(self, rgb):
        return attention_profile(self.model, self.layers, self.span,
            lambda: self.model(input_ids=self.ids, images=self.pixel_values(rgb),
                               output_attentions=True, use_cache=False, return_dict=True), self.device)

    @torch.inference_mode()
    def caption(self, rgb, args):
        tokens = self.model.generate(self.ids, images=self.pixel_values(rgb), do_sample=False,
                                     num_beams=args.num_beams, max_new_tokens=args.max_new_tokens,
                                     use_cache=True, output_attentions=False)
        # Upstream generate(inputs_embeds=...) may return continuation only.
        if tokens.shape[1] >= self.ids.shape[1] and torch.equal(tokens[:, :self.ids.shape[1]], self.ids):
            tokens = tokens[:, self.ids.shape[1]:]
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/samples.json"))
    parser.add_argument("--images", type=Path, default=Path("data/images"))
    parser.add_argument("--output", type=Path, default=Path("runs/llava5"))
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--eps", type=fraction, default="8/255")
    parser.add_argument("--alpha", type=fraction, default="2/255")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt", default="Please describe this image in detail.")
    args = parser.parse_args()
    if args.iters < 1 or args.limit < 1 or args.topk < 1:
        parser.error("--iters, --limit and --topk must be positive")
    if not 0 < args.alpha <= args.eps:
        parser.error("Require 0 < --alpha <= --eps")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new --output path")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(backend="llava", objective="delta_entropy + 5 * relative_topk_suppression + JS",
                  budget_space="RGB [0,1] after model-specific resizing/padding",
                  implementation="LLaVA-1.5-7B attention-disruption example")
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    print("CONFIG", config, flush=True)
    backend = LlavaBackend(args)
    rows = []
    for index, item in enumerate(json.loads(args.manifest.read_text())[:args.limit]):
        sample = args.output / f"{item['image_id']:012d}"
        sample.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed + index)
        np.random.seed(args.seed + index)
        torch.manual_seed(args.seed + index)
        torch.cuda.manual_seed_all(args.seed + index)
        image_path = args.images / item["file_name"]
        clean = backend.prepare(Image.open(image_path).convert("RGB"), args.prompt)
        save_rgb(clean, sample / "clean.png")
        print(f"SAMPLE {index + 1}/{args.limit} ID={item['image_id']} shape={list(clean.shape)}", flush=True)
        clean_caption = backend.caption(clean, args)
        print("CLEAN:", clean_caption, flush=True)
        (sample / "clean_caption.txt").write_text(clean_caption + "\n")
        with torch.no_grad():
            reference = backend.profile(clean).detach()
        indices = torch.topk(reference, min(args.topk, reference.numel())).indices
        adv = (clean + torch.empty_like(clean).uniform_(-args.eps, args.eps)).clamp(0, 1).detach()
        started = time.time()
        history = []
        with (sample / "history.jsonl").open("w") as stream:
            for step in range(args.iters):
                adv = adv.detach().requires_grad_(True)
                profile = backend.profile(adv)
                loss, terms = objective(profile, reference, indices)
                gradient = torch.autograd.grad(loss * 1024, adv)[0] / 1024
                if not torch.isfinite(gradient).all() or gradient.abs().max().item() == 0:
                    raise RuntimeError("Image gradient is non-finite or identically zero")
                with torch.no_grad():
                    adv = torch.maximum(torch.minimum(adv + args.alpha * gradient.sign(), clean + args.eps),
                                        clean - args.eps).clamp(0, 1)
                record = {"step": step + 1, "objective": loss.detach().item(),
                    "gradient_mean_abs": gradient.abs().mean().item(),
                    "linf_rgb": (adv - clean).abs().max().item(),
                    **{k: v.detach().item() for k, v in terms.items()}}
                history.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                if step % 5 == 0 or step == args.iters - 1:
                    print("STEP", index, record, flush=True)
                del gradient, loss, terms, profile
        adv = (adv.detach() * 255).round().clamp(0, 255) / 255
        linf = (adv - clean).abs().max().item()
        if linf > args.eps + 2e-6:
            raise RuntimeError(f"Saved image violates RGB budget: {linf}")
        save_rgb(adv, sample / "adv.png")
        save_rgb((adv - clean) / (2 * args.eps) + 0.5, sample / "delta.png")
        with torch.no_grad():
            final = backend.profile(adv).detach()
            _, final_terms = objective(final, reference, indices)
        torch.cuda.empty_cache()
        attacked_caption = backend.caption(adv, args)
        print("ATTACKED:", attacked_caption, flush=True)
        torch.save({"clean_rgb": clean.cpu(), "adv_rgb": adv.cpu()}, sample / "inputs.pt")
        result = {"backend": "llava", "image_id": item["image_id"], "file_name": item["file_name"],
            "references": item["references"], "source_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "prompt": args.prompt, "clean_caption": clean_caption, "attacked_caption": attacked_caption,
            "changed": clean_caption != attacked_caption, "rgb_linf": linf,
            "iterations": args.iters, "seed": args.seed + index, "layers": backend.layers,
            "weights": {"entropy": 1, "suppression": 5, "js": 1},
            "rgb_shape": list(clean.shape), "query_span": list(backend.span),
            "query_text": backend.query_text,
            "final_attention_terms": {k: v.item() for k, v in final_terms.items()},
            "attack_and_generation_seconds": time.time() - started,
            "complete": True}
        (sample / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        rows.append(result)
        (args.output / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        del clean, adv, reference, final
        torch.cuda.empty_cache()
    print("COMPLETE", "llava", len(rows), flush=True)


if __name__ == "__main__":
    main()
