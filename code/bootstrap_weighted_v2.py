#!/usr/bin/env python3
"""
Step 3 v2: Bootstrap rerun with weighted AP50 — FIXED delta_len formula.

B=200, seed=42, image-level resampling with replacement.
Uses weighted_ap50 instead of pycocotools to correctly handle
duplicate image IDs in bootstrap resamples.

Computes:
  delta_bare = AP50(bare)      - AP50(template)        [bare - "a photo of a X"]
  delta_len  = AP50(there_is)  - AP50(template)        ["there is a X" - "a photo of a X"]

Formula change (v1 → v2):
  v1 (BUG):  delta_len = AP50(bare) - AP50("there is a X")   ← wrong
  v2 (FIX):  delta_len = AP50("there is a X") - AP50(template)  ← correct

Prompt resolution: string-based lookup, NO hardcoded indices.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pycocotools.coco import COCO
from weighted_ap import weighted_ap50, load_gt_for_class, extract_dets

COCO_ANN_FILE = os.environ.get("COCO_ANN_FILE", "data/annotations/instances_val2017.json")
coco_gt = COCO(COCO_ANN_FILE)

CLASS_INFO = {"person": 1, "car": 3, "dog": 18, "cat": 17,
              "chair": 62, "cup": 47, "bottle": 44, "book": 84}

MODELS = [
    ("results/gdino_v11_raw_detections.json", "GDino"),
    ("results/owlv2_v10_raw_detections.json", "OWLv2"),
    ("results/m1_v2_raw_detections.json", "GLIP"),
]

CLASSES = ["person", "car", "dog", "cat", "chair", "cup", "bottle", "book"]

B = 200
SEED = 42


def resolve_prompt_indices(raw, cls):
    """Resolve prompt indices by string matching.

    Returns (bare_idx, template_idx, there_is_idx) for the given class.

    Hard requirements:
      - bare     = cls itself (e.g. "person")
      - template = "a photo of a " + cls  (e.g. "a photo of a person")
      - there_is = "there is a " + cls     (e.g. "there is a person")

    Raises KeyError if any prompt is not found — no silent fallback.
    """
    prompts = raw["prompt_bank"][cls]

    bare_str = cls
    template_str = "a photo of a " + cls
    there_is_str = "there is a " + cls

    def find_idx(prompt_list, target):
        for i, p in enumerate(prompt_list):
            if p == target:
                return i
        raise KeyError(f"Prompt {target!r} not found in prompt_bank[{cls!r}]: {prompt_list}")

    bare_idx = find_idx(prompts, bare_str)
    template_idx = find_idx(prompts, template_str)
    there_is_idx = find_idx(prompts, there_is_str)

    return bare_idx, template_idx, there_is_idx, (bare_str, template_str, there_is_str)


def bootstrap_weighted(raw, cls, img_ids, gts, B=200, seed=42):
    """Bootstrap with weighted AP50.

    For each resample:
      - Sample n image IDs with replacement
      - weights = {img_id: count}
      - Compute weighted AP50 for bare, template, there_is prompts
      - delta_bare = AP_bare - AP_template
      - delta_len  = AP_there_is - AP_template   (FIXED v2)
    """
    bare_idx, template_idx, there_is_idx, prompt_strs = resolve_prompt_indices(raw, cls)

    rng = np.random.RandomState(seed)
    n = len(img_ids)
    arr = np.array(img_ids)

    # Point estimates (full set, weights=1)
    weights_full = {img_id: 1 for img_id in img_ids}
    dets_bare = extract_dets(raw, cls, bare_idx, img_ids)
    dets_tmpl = extract_dets(raw, cls, template_idx, img_ids)
    dets_this = extract_dets(raw, cls, there_is_idx, img_ids)

    ap_bare_pt = weighted_ap50(dets_bare, gts, weights_full)
    ap_tmpl_pt = weighted_ap50(dets_tmpl, gts, weights_full)
    ap_this_pt = weighted_ap50(dets_this, gts, weights_full)

    delta_bare_pt = ap_bare_pt - ap_tmpl_pt
    delta_len_pt = ap_this_pt - ap_tmpl_pt    # FIXED v2

    # Bootstrap
    deltas_bare = []
    deltas_len = []
    unique_counts = []

    for b in range(B):
        idx = rng.randint(0, n, size=n)
        ids = arr[idx]

        # Compute weights (count of each image)
        weights = {}
        for img_id in ids:
            img_id = int(img_id)
            weights[img_id] = weights.get(img_id, 0) + 1

        unique_counts.append(len(weights))

        ap_b = weighted_ap50(dets_bare, gts, weights)
        ap_t = weighted_ap50(dets_tmpl, gts, weights)
        ap_s = weighted_ap50(dets_this, gts, weights)

        if np.isnan(ap_b) or np.isnan(ap_t):
            deltas_bare.append(np.nan)
        else:
            deltas_bare.append(ap_b - ap_t)

        if np.isnan(ap_s) or np.isnan(ap_t):
            deltas_len.append(np.nan)
        else:
            deltas_len.append(ap_s - ap_t)    # FIXED v2

    deltas_bare = np.array(deltas_bare)
    deltas_len = np.array(deltas_len)

    # Filter NaN for percentile computation
    valid_bare = deltas_bare[~np.isnan(deltas_bare)]
    valid_len = deltas_len[~np.isnan(deltas_len)]

    result = {
        "delta_bare": float(delta_bare_pt),
        "ci_low": float(np.percentile(valid_bare, 2.5)) if len(valid_bare) > 0 else float('nan'),
        "ci_high": float(np.percentile(valid_bare, 97.5)) if len(valid_bare) > 0 else float('nan'),
        "ci_contains_zero": bool(np.percentile(valid_bare, 2.5) <= 0 <= np.percentile(valid_bare, 97.5)) if len(valid_bare) > 0 else False,
        "point_in_ci": bool(np.percentile(valid_bare, 2.5) <= delta_bare_pt <= np.percentile(valid_bare, 97.5)) if len(valid_bare) > 0 else False,
        "delta_len": float(delta_len_pt),
        "delta_len_ci_low": float(np.percentile(valid_len, 2.5)) if len(valid_len) > 0 else float('nan'),
        "delta_len_ci_high": float(np.percentile(valid_len, 97.5)) if len(valid_len) > 0 else float('nan'),
        "delta_len_ci_contains_zero": bool(np.percentile(valid_len, 2.5) <= 0 <= np.percentile(valid_len, 97.5)) if len(valid_len) > 0 else False,
        "delta_len_point_in_ci": bool(np.percentile(valid_len, 2.5) <= delta_len_pt <= np.percentile(valid_len, 97.5)) if len(valid_len) > 0 else False,
        "n_bootstrap_bare_valid": int(len(valid_bare)),
        "n_bootstrap_bare_nan": int(np.sum(np.isnan(deltas_bare))),
        "n_bootstrap_len_valid": int(len(valid_len)),
        "n_bootstrap_len_nan": int(np.sum(np.isnan(deltas_len))),
        "prompt_strings": {
            "bare": prompt_strs[0],
            "template": prompt_strs[1],
            "there_is": prompt_strs[2],
        },
    }

    return result, float(np.mean(unique_counts))


# ── Main ─────────────────────────────────────────────────────────────

results = {}
all_unique_counts = []

for fname, mname in MODELS:
    print(f"\nProcessing {mname}...", flush=True)
    t0 = time.time()
    with open(fname) as f:
        raw = json.load(f)
    img_ids = raw["selected_img_ids"]
    results[mname] = {}

    for cls in CLASSES:
        if cls not in raw.get("prompt_bank", {}):
            continue

        # Resolve and print prompt strings for verification
        bare_idx, tmpl_idx, this_idx, pstrs = resolve_prompt_indices(raw, cls)
        print(f"  {cls:8s}: bare=[{bare_idx}]={pstrs[0]!r}  "
              f"template=[{tmpl_idx}]={pstrs[1]!r}  "
              f"there_is=[{this_idx}]={pstrs[2]!r}", flush=True)

        cat_id = CLASS_INFO[cls]
        gts = load_gt_for_class(coco_gt, cat_id, img_ids)

        res, mean_unique = bootstrap_weighted(raw, cls, img_ids, gts, B=B, seed=SEED)
        results[mname][cls] = res
        all_unique_counts.append(mean_unique)

        db = res["delta_bare"]
        db_lo = res["ci_low"]
        db_hi = res["ci_high"]
        db_sig = "SIG" if not res["ci_contains_zero"] else "ns"
        dl = res["delta_len"]
        dl_lo = res["delta_len_ci_low"]
        dl_hi = res["delta_len_ci_high"]
        dl_sig = "SIG" if not res["delta_len_ci_contains_zero"] else "ns"
        nan_b = res["n_bootstrap_bare_nan"]
        nan_l = res["n_bootstrap_len_nan"]

        print(f"           Δ_bare={db:+.4f} CI=[{db_lo:+.4f},{db_hi:+.4f}] {db_sig}"
              f"  Δ_len={dl:+.4f} CI=[{dl_lo:+.4f},{dl_hi:+.4f}] {dl_sig}"
              f"  NaN(bare={nan_b},len={nan_l})", flush=True)

    elapsed = time.time() - t0
    print(f"  {mname} done in {elapsed:.1f}s", flush=True)

output = {
    "bootstrap_B": B,
    "seed": SEED,
    "method": "weighted_ap50_v2_fixed_delta_len",
    "delta_bare_formula": "AP50(bare) - AP50(template)",
    "delta_len_formula": "AP50(there_is_a_X) - AP50(template)",  # FIXED
    "v1_bug_description": "v1 computed delta_len = AP50(bare) - AP50(there_is_a_X), which is WRONG",
    "effective_unique_per_resample_mean": float(np.mean(all_unique_counts)),
    "results": results,
}

with open("results/v14_bootstrap_weighted_v2.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*80}")
print(f"Saved to results/v14_bootstrap_weighted_v2.json")
print(f"Mean unique images per resample: {np.mean(all_unique_counts):.1f} / {len(img_ids)}")
print(f"Method: weighted_ap50 v2 (delta_len FIXED: there_is - template)")
