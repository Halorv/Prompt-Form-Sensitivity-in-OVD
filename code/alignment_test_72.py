#!/usr/bin/env python3
"""
Extended alignment verification: 3 models × 8 classes × 3 prompts = 72 groups.

Extends alignment_test.py by adding the "there is a X" prompt (idx=6),
covering the full Δ_len comparison. If this third prompt tier doesn't
align within 1e-6, Δ_len CIs are untrustworthy.

Pass criterion: max |diff| < 1e-6 across all 72 groups.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from weighted_ap import weighted_ap50, load_gt_for_class, extract_dets

COCO_ANN_FILE = os.environ.get("COCO_ANN_FILE", "data/annotations/instances_val2017.json")
coco_gt = COCO(COCO_ANN_FILE)

CLASS_INFO = {"person": 1, "car": 3, "dog": 18, "cat": 17,
              "chair": 62, "cup": 47, "bottle": 44, "book": 84}

BARE_IDX = 0
TEMPLATE_IDX = 2
THERE_IS_IDX = 6

MODELS = [
    ("results/gdino_v11_raw_detections.json", "GDino"),
    ("results/owlv2_v10_raw_detections.json", "OWLv2"),
    ("results/m1_v2_raw_detections.json", "GLIP"),
]

CLASSES = ["person", "car", "dog", "cat", "chair", "cup", "bottle", "book"]

PROMPTS = [
    (BARE_IDX, "bare"),
    (TEMPLATE_IDX, "template"),
    (THERE_IS_IDX, "there_is"),
]


def pycoco_ap50(raw, cls, prompt_idx, img_ids):
    """Compute AP50 using pycocotools, exactly as v14_bootstrap_ci.py does."""
    cat_id = CLASS_INFO[cls]
    pid = raw["per_image_detections"]
    dets = []
    key = f"{cls}__{prompt_idx}"
    for img_id in img_ids:
        s_id = str(img_id)
        if s_id in pid and key in pid[s_id]:
            for det in pid[s_id][key]:
                dets.append({
                    "image_id": int(img_id), "category_id": cat_id,
                    "bbox": det["bbox"], "score": det["score"],
                })
    if not dets:
        return 0.0
    coco_dt = coco_gt.loadRes(dets)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = list(img_ids)
    coco_eval.params.catIds = [cat_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    precision = coco_eval.eval["precision"]
    prec = precision[0, :, 0, 0, 2]
    valid = prec > -1
    return float(prec[valid].mean()) if valid.any() else 0.0


# ── Run extended alignment (72 groups) ──────────────────────────────

print("=" * 90)
print("EXTENDED ALIGNMENT: 3 models × 8 classes × 3 prompts = 72 groups")
print("  prompts: bare(idx=0), template(idx=2), there_is(idx=6)")
print("=" * 90)

diffs = []
count = 0
fail_count = 0

for fname, mname in MODELS:
    with open(fname) as f:
        raw = json.load(f)
    img_ids = raw["selected_img_ids"]
    weights = {img_id: 1 for img_id in img_ids}

    for cls in CLASSES:
        if cls not in raw.get("prompt_bank", {}):
            continue
        cat_id = CLASS_INFO[cls]
        gts = load_gt_for_class(coco_gt, cat_id, img_ids)

        for prompt_idx, prompt_name in PROMPTS:
            ap_pycoco = pycoco_ap50(raw, cls, prompt_idx, img_ids)
            dets = extract_dets(raw, cls, prompt_idx, img_ids)
            ap_weighted = weighted_ap50(dets, gts, weights, iou_thr=0.5)

            diff = abs(ap_pycoco - ap_weighted)
            diffs.append(diff)
            count += 1

            flag = "✓" if diff < 1e-6 else "✗ FAIL"
            if diff >= 1e-6:
                fail_count += 1
            print(f"{mname:<8} {cls:<10} {prompt_name:<12} pycoco={ap_pycoco:.10f}  weighted={ap_weighted:.10f}  diff={diff:.2e}  {flag}")

print("-" * 90)
max_diff = max(diffs)
mean_diff = sum(diffs) / len(diffs)
print(f"\nGroups tested: {count}")
print(f"Max |diff|:    {max_diff:.2e}")
print(f"Mean |diff|:   {mean_diff:.2e}")
print(f"Threshold:     1e-6")
if max_diff < 1e-6:
    print(f"\n✅ ALIGNMENT PASSED — all {count} groups within 1e-6 tolerance.")
    print("   Δ_len CIs are trustworthy (there_is tier aligned).")
else:
    print(f"\n❌ ALIGNMENT FAILED — max diff {max_diff:.2e} exceeds 1e-6.")
    print(f"   {fail_count} group(s) failed.")
    print("   ⚠️ Δ_len CIs are NOT trustworthy — there_is tier misaligned.")
