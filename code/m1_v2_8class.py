#!/usr/bin/env python3
"""
M1-v2: 8-class × 10-prompt × 500-image AP-based prompt sensitivity.

Changes from v1:
  - 8 classes (added cup, bottle, book)
  - Dumps per-image raw detections to JSON for bootstrap analysis
  - Same model, preprocessing, and post-processing as v1 (no changes to inference)

Output:
  results/m1_v2_raw_detections.json  — per-image raw detections (for bootstrap)
  results/m1_v2_prompt_sensitivity.json — summary with AP50, PSI per class/prompt
"""
import os, sys, json, argparse, time, random
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import AutoTokenizer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ── Setup paths ──────────────────────────────────────────────────────────
STANDALONE_DIR = os.environ.get("GLIP_ROOT", "third_party/glip")
sys.path.insert(0, STANDALONE_DIR)
import torch.nn as nn
from modules import (
    Backbone, LanguageBackbone, VLDyHead, ConvMapModule,
    AnchorGenerator, ATSSPostProcessor, to_image_list
)

# ── Constants ─────────────────────────────────────────────────────────────
PIXEL_MEAN = torch.tensor([103.530, 116.280, 123.675]).view(3, 1, 1)
PIXEL_STD  = torch.tensor([57.375, 57.120, 58.395]).view(3, 1, 1)

COCO_IMG_DIR    = os.environ.get("COCO_IMG_DIR", "data/val2017")
COCO_ANN_FILE   = os.environ.get("COCO_ANN_FILE", "data/annotations/instances_val2017.json")
GLIP_WEIGHTS    = os.environ.get("GLIP_WEIGHTS", "weights/glip_tiny_model_o365_goldg_cc_sbu.pth")

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

# Prompt bank: 10 prompts per class, same structural template:
#   bare name, "a X", "a photo of a X", "a X in the scene",
#   synonym, generic term, "there is a X", posture/state,
#   "a pet X" or "a X nearby", action/state
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

# Flatten: list of (class_name, cat_id, prompt, prompt_idx)
ALL_PROMPTS = []
for class_name, cat_id in CLASS_INFO.items():
    for pi, prompt in enumerate(PROMPT_BANK[class_name]):
        ALL_PROMPTS.append((class_name, cat_id, prompt, pi))


# ── Model definition ─────────────────────────────────────────────────────
def _norm(t):
    return (t[[2, 1, 0]] * 255.0 - PIXEL_MEAN) / PIXEL_STD

def preprocess(path, min_size=800, max_size=1333):
    img = Image.open(path).convert("RGB")
    ow, oh = img.size  # PIL: (width, height)
    s = min_size / min(ow, oh)
    if max(ow, oh) * s > max_size:
        s = max_size / max(ow, oh)
    nw, nh = int(round(ow * s)), int(round(oh * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    return _norm(TF.to_tensor(img)), (ow, oh), (nh, nw)

class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = VLDyHead(num_convs=6, skip_stages={0})
        self.anchor_generator = AnchorGenerator()

class VisTex(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.language_backbone = LanguageBackbone()
        self.rpn = RPN()
        self.fcto768 = ConvMapModule()

def build_positive_map(tok):
    ids = tok["input_ids"][0]
    pmap, phrase, pos = {}, 0, []
    for idx in range(1, len(ids)):
        t = ids[idx].item()
        if t == 0:
            break
        if t in (1012, 102):  # period or [SEP]
            if pos:
                phrase += 1
                pmap[phrase] = pos
                pos = []
        else:
            pos.append(idx)
        if phrase >= 79:
            break
    return pmap


# ── GLIP-T inference (single image, single prompt) ───────────────────────
def run_glip_inference(model, tokenizer, img_t, orig_w, orig_h,
                       text_prompt, device="cuda"):
    batch, sizes = to_image_list([img_t], 32)
    batch = batch.to(device)
    resized_h, resized_w = sizes[0]

    caption = text_prompt.strip() + ". "
    tk = tokenizer(caption, max_length=256, padding="max_length",
                   return_tensors="pt", truncation=True)
    tk = {k: v.to(device) for k, v in tk.items()}
    pm = build_positive_map(tk)

    with torch.no_grad():
        vfeat = model.backbone(batch)
        lang = model.language_backbone(tk)
        logits, bbox_reg, ctr, dot = model.rpn.head(vfeat, lang)
        anchors = model.rpn.anchor_generator(sizes, vfeat)
        results = ATSSPostProcessor()(bbox_reg, ctr, anchors, logits, dot, pm)

    pred = results[0]
    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    sc = pred.get_field("scores").cpu().numpy()
    boxes = pred.bbox.cpu().numpy()
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y

    if len(sc) > 100:
        order = np.argsort(-sc)
        sc, boxes = sc[order[:100]], boxes[order[:100]]

    detections = []
    for i in range(len(sc)):
        detections.append({
            "bbox": [float(boxes[i][0]), float(boxes[i][1]),
                     float(boxes[i][2] - boxes[i][0]),
                     float(boxes[i][3] - boxes[i][1])],  # COCO: x,y,w,h
            "score": float(sc[i]),
        })
    return detections


# ── Image selection ──────────────────────────────────────────────────────
def select_images(coco, num_images, seed=42):
    """Select a fixed random subset of COCO val images that contain our classes."""
    random.seed(seed)
    cat_ids = list(CLASS_INFO.values())
    img_ids_with_our_classes = set()
    for cat_id in cat_ids:
        ann_ids = coco.getAnnIds(catIds=[cat_id])
        anns = coco.loadAnns(ann_ids)
        for a in anns:
            img_ids_with_our_classes.add(a["image_id"])

    eligible = sorted(img_ids_with_our_classes)
    random.shuffle(eligible)
    selected = eligible[:num_images]
    print(f"Selected {len(selected)} images from {len(eligible)} eligible "
          f"(seed={seed})")
    return selected


# ── AP50 from COCOeval ──────────────────────────────────────────────────
def compute_ap50(coco_gt, detections, selected_img_ids, cat_id):
    """Compute AP50 at IoU=0.5, area=all, maxDets=100 using pycocotools."""
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


def compute_ap50_area(coco_gt, detections, selected_img_ids, cat_id, area_key):
    """Compute AP50 at IoU=0.5 for a specific area range.

    COCOeval default areaRngLbl = ['all', 'small', 'medium', 'large']
    with areaRng = [[0,1e10], [0,1024], [1024,9216], [9216,1e10]]

    area_key options:
      'all'           → A index 0 (default)
      'small'         → A index 1
      'medium'        → A index 2
      'large'         → A index 3
      'medium_large'  → custom override: areaRng=[[1024, 1e10]], must
                        set BEFORE evaluate() so _prepare() filters correctly.
    """
    if not detections:
        return 0.0
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = selected_img_ids
    coco_eval.params.catIds = [cat_id]

    if area_key == "medium_large":
        # Override BEFORE evaluate() so _prepare() filters GT/dets by area
        coco_eval.params.areaRng = [[32**2, 1e10]]
        coco_eval.params.areaRngLbl = ["medium_large"]
        coco_eval.evaluate()
        coco_eval.accumulate()
        precision = coco_eval.eval["precision"]
        # After override, A dim should be 1
        assert precision.shape[3] == 1, \
            f"Expected A dim=1 for medium_large, got {precision.shape[3]}"
        prec_slice = precision[0, :, 0, 0, 2]
    else:
        coco_eval.evaluate()
        coco_eval.accumulate()
        precision = coco_eval.eval["precision"]
        assert precision.shape[3] == 4, \
            f"Expected A dim=4 for default, got {precision.shape[3]}"
        # COCOeval default order: [all, small, medium, large]
        area_idx_map = {"all": 0, "small": 1, "medium": 2, "large": 3}
        a_idx = area_idx_map[area_key]
        prec_slice = precision[0, :, 0, a_idx, 2]

    valid_mask = prec_slice > -1
    if valid_mask.any():
        return float(prec_slice[valid_mask].mean())
    return 0.0


# ── Main experiment ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="M1-v2: 8-class GLIP-T prompt sensitivity")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test with 10 images")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    if args.smoke:
        args.num_images = 10
        print("🔥 SMOKE TEST MODE: 10 images")

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    # ── Load COCO GT ──────────────────────────────────────────────────────
    print("Loading COCO val2017 annotations...")
    coco_gt = COCO(COCO_ANN_FILE)
    selected_img_ids = select_images(coco_gt, args.num_images, args.seed)
    img_infos = coco_gt.loadImgs(selected_img_ids)

    # ── Load GLIP-T model ─────────────────────────────────────────────────
    print("Loading GLIP-T model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )
    model = VisTex()
    ck = torch.load(GLIP_WEIGHTS, map_location="cpu", weights_only=False)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in ck["model"].items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    miss_real = [m for m in miss if "relative_position_index" not in m]
    print(f"Loaded GLIP-T: {len(sd)} keys | missing {len(miss_real)} unexpected {len(unexp)}")
    model = model.to(device).eval()
    print("Model loaded.\n")

    # ── Pre-compile prompt tokenization ───────────────────────────────────
    prompt_tokens = {}
    for class_name, cat_id, prompt, pi in ALL_PROMPTS:
        if prompt not in prompt_tokens:
            caption = prompt.strip() + ". "
            tk = tokenizer(caption, max_length=256, padding="max_length",
                           return_tensors="pt", truncation=True)
            prompt_tokens[prompt] = tk

    # ── Run inference: image-major loop ──────────────────────────────────
    # Per-image detections: {image_id: {(class_name, prompt_idx): [dets]}}
    per_image_dets = {}
    # Flat detections for COCOeval: {(class_name, prompt): [all dets]}
    all_detections = {}
    for cn, ci, p, pi in ALL_PROMPTS:
        all_detections[(cn, p)] = []

    total_inferences = 0
    t0 = time.time()

    for ii, img_info in enumerate(img_infos):
        img_id = img_info["id"]
        img_path = os.path.join(COCO_IMG_DIR, img_info["file_name"])

        img_t, (orig_w, orig_h), _ = preprocess(img_path)
        img_t = img_t.to(device)

        per_image_dets[img_id] = {}

        for class_name, cat_id, prompt, pi in ALL_PROMPTS:
            dets = run_glip_inference(model, tokenizer, img_t,
                                       orig_w, orig_h, prompt, device)
            for d in dets:
                d["image_id"] = img_id
                d["category_id"] = cat_id
            all_detections[(class_name, prompt)].extend(dets)
            # Store per-image: only bbox+score (image_id/cat_id redundant)
            per_image_dets[img_id][(class_name, pi)] = [
                {"bbox": d["bbox"], "score": d["score"]} for d in dets
            ]
            total_inferences += 1

        if (ii + 1) % 10 == 0 or (ii + 1) == len(img_infos):
            elapsed = time.time() - t0
            eta = elapsed / (ii + 1) * len(img_infos)
            print(f"  {ii+1}/{len(img_infos)} images | "
                  f"{total_inferences} inferences | "
                  f"{elapsed:.1f}s elapsed | ETA {eta:.1f}s")

    elapsed_total = time.time() - t0

    # ── Compute AP50 per (class, prompt) — all area ranges ───────────────
    print(f"\n{'='*60}")
    print("COMPUTING AP50 METRICS (all / medium / large)")
    print(f"{'='*60}")

    all_results = {}
    for class_name, cat_id in CLASS_INFO.items():
        all_results[class_name] = {}
        prompts = PROMPT_BANK[class_name]
        print(f"\n  Class: {class_name} (cat_id={cat_id})")

        for prompt in prompts:
            dets = all_detections[(class_name, prompt)]
            ap50_all, n_gt = compute_ap50(coco_gt, dets, selected_img_ids, cat_id)
            ap50_med = compute_ap50_area(coco_gt, dets, selected_img_ids, cat_id, "medium")
            ap50_lg  = compute_ap50_area(coco_gt, dets, selected_img_ids, cat_id, "large")
            ap50_ml  = compute_ap50_area(coco_gt, dets, selected_img_ids, cat_id, "medium_large")
            all_results[class_name][prompt] = {
                "ap50_all": ap50_all,
                "ap50_medium": ap50_med,
                "ap50_large": ap50_lg,
                "ap50_medium_large": ap50_ml,
                "num_gt": n_gt,
                "num_detections": len(dets),
            }
            print(f"    {prompt:30s}: AP50_all={ap50_all:.4f} "
                  f"AP50_med={ap50_med:.4f} AP50_lg={ap50_lg:.4f} "
                  f"AP50_ml={ap50_ml:.4f} "
                  f"n_gt={n_gt} n_det={len(dets)}")

    # ── PSI summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PROMPT SENSITIVITY ANALYSIS")
    print(f"{'='*60}")

    summary = {}
    for class_name in CLASS_INFO:
        prompts = PROMPT_BANK[class_name]
        summary[class_name] = {}
        for metric in ["ap50_all", "ap50_medium", "ap50_large", "ap50_medium_large"]:
            vals = [all_results[class_name][p][metric] for p in prompts]
            psi_key = f"psi_{metric.replace('ap50_', '')}"
            summary[class_name][psi_key] = float(max(vals) - min(vals))
            summary[class_name][f"mean_{metric}"] = float(np.mean(vals))
            summary[class_name][f"max_{metric}"] = float(max(vals))
            summary[class_name][f"min_{metric}"] = float(min(vals))
        # best/worst prompts (based on ap50_all)
        ap50s = [all_results[class_name][p]["ap50_all"] for p in prompts]
        summary[class_name]["best_prompt"] = prompts[int(np.argmax(ap50s))]
        summary[class_name]["worst_prompt"] = prompts[int(np.argmin(ap50s))]
        summary[class_name]["per_prompt_stats"] = all_results[class_name]

        print(f"\n{class_name}:")
        print(f"  PSI_all    = {summary[class_name]['psi_all']:.4f} "
              f"({summary[class_name]['psi_all']*100:.2f} pts)")
        print(f"  PSI_medium = {summary[class_name]['psi_medium']:.4f}")
        print(f"  PSI_large  = {summary[class_name]['psi_large']:.4f}")
        print(f"  PSI_medium_large = {summary[class_name]['psi_medium_large']:.4f}")
        print(f"  Best:  '{summary[class_name]['best_prompt']}'")
        print(f"  Worst: '{summary[class_name]['worst_prompt']}'")

    # ── Save raw detections (per-image) for bootstrap ────────────────────
    # Convert tuple keys to string keys for JSON serialization
    raw_dets_serializable = {}
    for img_id, prompts_dict in per_image_dets.items():
        raw_dets_serializable[str(img_id)] = {}
        for (cn, pi), dets in prompts_dict.items():
            raw_dets_serializable[str(img_id)][f"{cn}__{pi}"] = dets

    raw_file = os.path.join(args.output_dir, "m1_v2_raw_detections.json")
    print(f"\nSaving raw per-image detections to {raw_file}...")
    with open(raw_file, "w") as f:
        json.dump({
            "selected_img_ids": selected_img_ids,
            "class_info": CLASS_INFO,
            "prompt_bank": PROMPT_BANK,
            "per_image_detections": raw_dets_serializable,
        }, f, indent=2, ensure_ascii=False)

    # ── Save summary ─────────────────────────────────────────────────────
    output_data = {
        "experiment": "M1-v2: 8-class GLIP-T AP-based prompt sensitivity",
        "model": "GLIP-T (glip_tiny_model_o365_goldg_cc_sbu.pth)",
        "data": f"COCO val2017 subset ({args.num_images} images, seed={args.seed})",
        "num_images": args.num_images,
        "seed": args.seed,
        "classes": list(CLASS_INFO.keys()),
        "coco_category_ids": CLASS_INFO,
        "prompt_bank": PROMPT_BANK,
        "threshold_info": {
            "pre_nms_thresh": 0.05,
            "pre_nms_top_n": 1000,
            "nms_thresh": 0.6,
            "detections_per_img": 100,
            "post_nms_score_threshold": "None (all detections kept for AP evaluation)",
            "official_glip_zeroshot_threshold": 0.3,
            "note": "M1 and official both use ATSSPostProcessor defaults (pre_nms_thresh=0.05). "
                    "M1 does NOT apply post-NMS score threshold (correct for AP). "
                    "Official glip_zeroshot.py applies 0.3 for visualization only.",
        },
        "results": summary,
        "overall": {
            "mean_psi_all": float(np.mean([summary[c]["psi_all"] for c in CLASS_INFO])),
            "total_inferences": total_inferences,
            "total_time_seconds": elapsed_total,
            "total_time_hours": elapsed_total / 3600,
        },
    }

    out_file = os.path.join(args.output_dir, "m1_v2_prompt_sensitivity.json")
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to {out_file}")
    print(f"Raw detections saved to {raw_file}")
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/3600:.2f}h)")


if __name__ == "__main__":
    main()
