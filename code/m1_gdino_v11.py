#!/usr/bin/env python3
"""
M1-GroundingDINO v11: Per-prompt inference (NO multi-prompt competition).

Protocol identical to v10 (OWLv2), with these model-specific adaptations:
  - Model: IDEA-Research/grounding-dino-tiny
  - Each prompt run as SEPARATE inference (single caption per forward pass)
  - BERT tokenizer: each prompt must end with "." and be lowercase
  - Bypass post_process_grounded_object_detection (it has same torch.max over
    token dim — safe for single-prompt, but we set our own threshold)
  - Score filter = 0.01 (memory only, does not bias AP)
  - No NMS (Grounding DINO's transformers impl has no NMS step)
  - max_dets = 100 per prompt per image (top-k by score)

Output:
  results/gdino_v11_raw_detections.json — per-image per-prompt raw detections
  results/gdino_v11_summary.json        — AP50 per class/prompt + Δ + bootstrap CI
"""
import os, sys, json, time
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ── Constants (identical to M1-v2 / v10) ─────────────────────────────────
COCO_IMG_DIR  = os.environ.get("COCO_IMG_DIR", "data/val2017")
COCO_ANN_FILE = os.environ.get("COCO_ANN_FILE", "data/annotations/instances_val2017.json")
MODEL_NAME    = "IDEA-Research/grounding-dino-tiny"

SCORE_FILTER = 0.01   # very low threshold for memory only; does not bias AP
MAX_DETS     = 100    # cap per prompt per image (same as GLIP / OWLv2 v10)

CLASS_INFO = {
    "person": 1,
    "car":    3,
    "dog":    18,
    "cat":    17,
    "chair":  62,
    "cup":    47,
    "bottle": 44,
    "book":   84,
}

PROMPT_BANK = {
    "person": [
        "person", "a person", "a photo of a person", "a person in the scene",
        "people", "human", "there is a person", "a human standing",
        "an individual person", "a pedestrian",
    ],
    "car": [
        "car", "a car", "a photo of a car", "a car in the scene",
        "vehicle", "automobile", "there is a car", "a car parked",
        "a sedan", "a motorcar",
    ],
    "dog": [
        "dog", "a dog", "a photo of a dog", "a dog in the scene",
        "puppy", "canine", "there is a dog", "a dog standing",
        "a pet dog", "a dog running",
    ],
    "cat": [
        "cat", "a cat", "a photo of a cat", "a cat in the scene",
        "kitten", "feline", "there is a cat", "a cat sitting",
        "a pet cat", "a cat lying down",
    ],
    "chair": [
        "chair", "a chair", "a photo of a chair", "a chair in the scene",
        "seat", "furniture", "there is a chair", "a wooden chair",
        "an empty chair", "a chair nearby",
    ],
    "cup": [
        "cup", "a cup", "a photo of a cup", "a cup in the scene",
        "mug", "drinking cup", "there is a cup", "a cup on the table",
        "a teacup", "a coffee cup",
    ],
    "bottle": [
        "bottle", "a bottle", "a photo of a bottle", "a bottle in the scene",
        "flask", "container", "there is a bottle", "a glass bottle",
        "a water bottle", "a plastic bottle",
    ],
    "book": [
        "book", "a book", "a photo of a book", "a book in the scene",
        "volume", "publication", "there is a book", "a book on the shelf",
        "a paperback", "a hardcover book",
    ],
}


# ── AP50 evaluation (identical to v10) ──────────────────────────────────
def compute_ap50(coco_gt, detections, selected_img_ids, cat_id):
    if not detections:
        return 0.0, 0
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = selected_img_ids
    coco_eval.params.catIds = [cat_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    precision = coco_eval.eval["precision"]
    prec_slice = precision[0, :, 0, 0, 2]  # IoU=0.5, area=all, maxDets=100
    valid_mask = prec_slice > -1
    if valid_mask.any():
        ap50 = float(prec_slice[valid_mask].mean())
    else:
        ap50 = 0.0
    num_gt = sum(1 for img_id in selected_img_ids
                 for ann in coco_gt.loadAnns(
                     coco_gt.getAnnIds(imgIds=[img_id], catIds=[cat_id]))
                 if ann.get("iscrowd", 0) == 0)
    return ap50, num_gt


# ── Bootstrap CI (identical to v10) ──────────────────────────────────────
def bootstrap_delta_ci(coco_gt, bare_dets, tmpl_dets, selected_img_ids,
                       cat_id, B=200, seed=42, alpha=0.05):
    rng = np.random.RandomState(seed)
    n = len(selected_img_ids)
    img_id_arr = np.array(selected_img_ids)

    ap_bare_full, n_gt = compute_ap50(coco_gt, bare_dets, selected_img_ids, cat_id)
    ap_tmpl_full, _    = compute_ap50(coco_gt, tmpl_dets, selected_img_ids, cat_id)
    delta_point = ap_bare_full - ap_tmpl_full

    def ap_on_subset(img_ids_sub, dets):
        if not dets:
            return 0.0
        img_set = set(img_ids_sub)
        sub_dets = [d for d in dets if d["image_id"] in img_set]
        if not sub_dets:
            return 0.0
        coco_dt = coco_gt.loadRes(sub_dets)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = list(img_ids_sub)
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        precision = coco_eval.eval["precision"]
        prec_slice = precision[0, :, 0, 0, 2]
        valid_mask = prec_slice > -1
        if valid_mask.any():
            return float(prec_slice[valid_mask].mean())
        return 0.0

    all_deltas = []
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        img_ids_sub = img_id_arr[idx].tolist()
        ap_b = ap_on_subset(img_ids_sub, bare_dets)
        ap_t = ap_on_subset(img_ids_sub, tmpl_dets)
        all_deltas.append(ap_b - ap_t)

    all_deltas = np.array(all_deltas)
    ci_low = float(np.percentile(all_deltas, 100 * alpha / 2))
    ci_high = float(np.percentile(all_deltas, 100 * (1 - alpha / 2)))

    return delta_point, ci_low, ci_high, all_deltas, n_gt


# ── Grounding DINO per-prompt inference ──────────────────────────────────
def center_to_corners(boxes):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - w/2, cx + w/2, cy + h/2], dim=-1)


def run_gdino_single_prompt(model, processor, img_pil, prompt_text, device="cuda"):
    """Run Grounding DINO for a single image with a SINGLE text prompt.

    BERT tokenizer: prompt must end with "." and be lowercase.
    We bypass post_process_grounded_object_detection to control thresholding.

    Returns: list of {bbox: [x, y, w, h], score: float}
    """
    # Grounding DINO requires text ending with period
    caption = prompt_text.strip().lower()
    if not caption.endswith("."):
        caption = caption + "."

    inputs = processor(text=caption, images=img_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # logits: [1, num_queries, seq_len], pred_boxes: [1, num_queries, 4] (cx,cy,w,h normalized)
    logits = outputs.logits[0]         # [num_queries, seq_len]
    pred_boxes = outputs.pred_boxes[0]  # [num_queries, 4] normalized [0,1]

    # Per-query score: max over token dimension (within single prompt — no competition)
    probs = torch.sigmoid(logits)      # [num_queries, seq_len]
    scores = probs.max(dim=-1).values  # [num_queries]

    # Convert boxes to corners [x1, y1, x2, y2]
    boxes_corners = center_to_corners(pred_boxes)  # [num_queries, 4] normalized

    # Scale to absolute image coordinates
    h, w = img_pil.height, img_pil.width
    scale_fct = torch.tensor([w, h, w, h], dtype=torch.float32, device=device)
    boxes_abs = boxes_corners * scale_fct  # [num_queries, 4] absolute

    scores_np = scores.cpu().numpy()
    boxes_np = boxes_abs.cpu().numpy()

    # Filter by very low threshold (memory only)
    mask = scores_np > SCORE_FILTER
    if not mask.any():
        return []

    filtered_boxes = boxes_np[mask]
    filtered_scores = scores_np[mask]

    # Cap at MAX_DETS by score
    if len(filtered_scores) > MAX_DETS:
        order = np.argsort(-filtered_scores)[:MAX_DETS]
        filtered_boxes = filtered_boxes[order]
        filtered_scores = filtered_scores[order]

    dets = []
    for i in range(len(filtered_scores)):
        x1, y1, x2, y2 = filtered_boxes[i]
        dets.append({
            "bbox": [float(x1), float(y1),
                     float(x2 - x1), float(y2 - y1)],  # COCO: x, y, w, h
            "score": float(filtered_scores[i]),
        })
    return dets


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference, load existing raw detections")
    args = parser.parse_args()

    if args.smoke:
        args.num_images = 10
        print("🔥 SMOKE TEST MODE: 10 images")

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    # ── Load M1-v2 raw detections for image_id list (same images as v10) ─
    with open(os.path.join(args.output_dir, "m1_v2_raw_detections.json")) as f:
        m1_raw = json.load(f)
    selected_img_ids = m1_raw["selected_img_ids"]
    if args.smoke:
        selected_img_ids = selected_img_ids[:10]
    print(f"Using {len(selected_img_ids)} images (seed={args.seed})")

    # ── Load COCO GT ──────────────────────────────────────────────────────
    print("Loading COCO val2017 annotations...")
    coco_gt = COCO(COCO_ANN_FILE)
    img_infos = coco_gt.loadImgs(selected_img_ids)

    raw_file = os.path.join(args.output_dir, "gdino_v11_raw_detections.json")

    # Build flat list of all (class_name, cat_id, prompt, prompt_idx)
    all_prompts = []
    for class_name, cat_id in CLASS_INFO.items():
        for pi, prompt in enumerate(PROMPT_BANK[class_name]):
            all_prompts.append((class_name, cat_id, prompt, pi))

    if not args.skip_inference:
        # ── Load Grounding DINO model ────────────────────────────────────
        print(f"Loading Grounding DINO model ({MODEL_NAME})...")
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        )
        model = model.to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded in {time.time()-t0:.1f}s | {n_params:,} params")
        print(f"Per-prompt mode: each prompt run as separate inference")
        print(f"Text format: BERT tokenizer, prompt + '.', lowercase\n")

        # ── Run inference ─────────────────────────────────────────────────
        # Structure: per_image_dets[img_id_str][f"{class_name}__{prompt_idx}"] = [dets]
        per_image_dets = {}
        t0 = time.time()
        total_inferences = len(img_infos) * len(all_prompts)
        done = 0

        for ii, img_info in enumerate(img_infos):
            img_id = img_info["id"]
            img_path = os.path.join(COCO_IMG_DIR, img_info["file_name"])
            img_pil = Image.open(img_path).convert("RGB")

            per_image_dets[str(img_id)] = {}

            for class_name, cat_id, prompt_text, prompt_idx in all_prompts:
                dets = run_gdino_single_prompt(
                    model, processor, img_pil, prompt_text, device
                )
                per_image_dets[str(img_id)][f"{class_name}__{prompt_idx}"] = dets
                done += 1

            if (ii + 1) % 10 == 0 or (ii + 1) == len(img_infos):
                elapsed = time.time() - t0
                eta = elapsed / (ii + 1) * len(img_infos)
                print(f"  {ii+1}/{len(img_infos)} images ({done}/{total_inferences} "
                      f"inferences) | {elapsed:.1f}s elapsed | ETA {eta:.1f}s",
                      flush=True)

        elapsed_total = time.time() - t0
        print(f"\nInference done in {elapsed_total:.1f}s "
              f"({elapsed_total/total_inferences:.3f}s/inference)")

        # ── Save raw detections ───────────────────────────────────────────
        with open(raw_file, "w") as f:
            json.dump({
                "selected_img_ids": selected_img_ids,
                "class_info": CLASS_INFO,
                "prompt_bank": PROMPT_BANK,
                "per_image_detections": per_image_dets,
                "config": {
                    "model": MODEL_NAME,
                    "score_filter": SCORE_FILTER,
                    "max_dets": MAX_DETS,
                    "method": "per_prompt_separate_inference",
                    "text_format": "BERT tokenizer: prompt + '.', lowercase",
                    "note": "Each prompt run as separate forward pass. "
                            "No multi-prompt competition. "
                            "Bypassed post_process_grounded_object_detection "
                            "to control thresholding (score_filter=0.01)."
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"Raw detections saved to {raw_file}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()
    else:
        print(f"Loading existing raw detections from {raw_file}...")
        with open(raw_file) as f:
            raw = json.load(f)
        per_image_dets = raw["per_image_detections"]

    # ── Compute AP50 per (class, prompt) ─────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPUTING AP50 + BOOTSTRAP CI")
    print(f"{'='*60}")

    # Build flat detection lists per (class, prompt)
    all_detections = {}
    for cn, ci, p, pi in all_prompts:
        all_detections[(cn, p)] = []

    for img_id_str, prompts_dict in per_image_dets.items():
        img_id = int(img_id_str)
        for key, dets in prompts_dict.items():
            cn, pi_str = key.split("__")
            pi = int(pi_str)
            prompt = PROMPT_BANK[cn][pi]
            for d in dets:
                d["image_id"] = img_id
                d["category_id"] = CLASS_INFO[cn]
            all_detections[(cn, prompt)].extend(dets)

    results = {}
    for class_name, cat_id in CLASS_INFO.items():
        results[class_name] = {}
        prompts = PROMPT_BANK[class_name]
        print(f"\n  Class: {class_name} (cat_id={cat_id})")

        for prompt in prompts:
            dets = all_detections[(class_name, prompt)]
            ap50, n_gt = compute_ap50(coco_gt, dets, selected_img_ids, cat_id)
            results[class_name][prompt] = {
                "ap50_all": ap50,
                "num_gt": n_gt,
                "num_detections": len(dets),
            }
            print(f"    {prompt:40s}: AP50={ap50:.4f}  n_det={len(dets)}")

    # ── Bootstrap CI for Δ (bare vs template) ────────────────────────────
    print(f"\n{'='*60}")
    print("BOOTSTRAP CI FOR Δ (bare − template), B=200")
    print(f"{'='*60}")

    delta_results = {}
    for class_name, cat_id in CLASS_INFO.items():
        prompts = PROMPT_BANK[class_name]
        bare_prompt = prompts[0]        # "person", "car", etc.
        tmpl_prompt = prompts[2]        # "a photo of a X"

        bare_dets = all_detections[(class_name, bare_prompt)]
        tmpl_dets = all_detections[(class_name, tmpl_prompt)]

        print(f"\n  {class_name}: bootstrap B=200...", end="", flush=True)
        delta_pt, ci_lo, ci_hi, all_d, n_gt = bootstrap_delta_ci(
            coco_gt, bare_dets, tmpl_dets, selected_img_ids, cat_id,
            B=200, seed=42
        )
        contains_pt = ci_lo <= delta_pt <= ci_hi
        contains_zero = ci_lo <= 0.0 <= ci_hi
        print(f" Δ={delta_pt:+.4f} CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] "
              f"n_gt={n_gt} pt_in_ci={contains_pt} 0_in_ci={contains_zero}")

        delta_results[class_name] = {
            "ap_bare": results[class_name][bare_prompt]["ap50_all"],
            "ap_template": results[class_name][tmpl_prompt]["ap50_all"],
            "delta_point": delta_pt,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "ci_contains_point": contains_pt,
            "ci_contains_zero": contains_zero,
            "n_gt": n_gt,
            "n_det_bare": len(bare_dets),
            "n_det_template": len(tmpl_dets),
            "all_bootstrap_deltas": all_d.tolist(),
        }

    # ── Save summary ──────────────────────────────────────────────────────
    mean_delta = np.mean([delta_results[c]["delta_point"] for c in CLASS_INFO])
    n_significant = sum(1 for c in CLASS_INFO if not delta_results[c]["ci_contains_zero"])
    n_positive = sum(1 for c in CLASS_INFO if delta_results[c]["delta_point"] > 0)

    mean_ap_bare = np.mean([delta_results[c]["ap_bare"] for c in CLASS_INFO])

    summary = {
        "experiment": "M1-GroundingDINO v11: per-prompt separate inference",
        "model": MODEL_NAME,
        "method": "per_prompt_separate_inference",
        "text_format": "BERT tokenizer: prompt + '.', lowercase. "
                       "All 10 prompts treated identically (same format).",
        "threshold_policy": {
            "grounding_dino": f"score > {SCORE_FILTER} (memory filter only, no AP bias), "
                              f"max_dets={MAX_DETS}, no NMS",
            "owlv2_v10": "score > 0.01 (memory filter only), max_dets=100, no NMS",
            "glip": "pre_nms_thresh=0.05 (on sigmoid scores), nms_thresh=0.6, "
                    "max_dets=100, no post-NMS score threshold",
            "comparability": "All three keep all post-processing detections for AP "
                              "evaluation. The low score filter does not bias AP "
                              "because COCOeval sweeps over score thresholds.",
        },
        "num_images": len(selected_img_ids),
        "seed": args.seed,
        "classes": list(CLASS_INFO.keys()),
        "per_prompt_ap50": results,
        "delta_bootstrap": delta_results,
        "overall": {
            "mean_delta": float(mean_delta),
            "mean_ap_bare": float(mean_ap_bare),
            "n_positive": n_positive,
            "n_total": len(CLASS_INFO),
            "n_significant": n_significant,
        },
    }

    out_file = os.path.join(args.output_dir, "gdino_v11_summary.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {out_file}")
    print(f"\n{'='*60}")
    print(f"Mean Δ = {mean_delta:+.4f} ({n_positive}/{len(CLASS_INFO)} positive)")
    print(f"Mean AP50(bare) = {mean_ap_bare:.4f}")
    print(f"Bootstrap significant (CI excludes 0): {n_significant}/{len(CLASS_INFO)}")
    for c in CLASS_INFO:
        d = delta_results[c]
        print(f"  {c:8s}: Δ={d['delta_point']:+.4f} "
              f"CI=[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] "
              f"pt_in_ci={d['ci_contains_point']} "
              f"0_in_ci={d['ci_contains_zero']} "
              f"n_gt={d['n_gt']}")


if __name__ == "__main__":
    main()
