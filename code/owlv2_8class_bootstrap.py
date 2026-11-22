#!/usr/bin/env python3
"""
Bootstrap CI for OWLv2 8-class single-query protocol.

Reads per-image raw detections from results/m6_40class/owlv2_raw/<class>.json
(produced by m6_40class.py --model owlv2 --save-raw).

Computes:
  delta_bare = AP50(bare) - AP50(template)
  delta_len  = AP50(there_is) - AP50(template)

B=200, seed=42, image-level resampling with replacement.
Uses weighted_ap50 for correct bootstrap with duplicate image IDs.

Output: results/m6_40class/owlv2_8class_bootstrap.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pycocotools.coco import COCO
from weighted_ap import weighted_ap50, load_gt_for_class

COCO_ANN_FILE = os.environ.get("COCO_ANN_FILE", "data/annotations/instances_val2017.json")

CLASSES = ["person", "car", "dog", "cat", "chair", "cup", "bottle", "book"]
CAT_IDS = {"person": 1, "car": 3, "dog": 18, "cat": 17,
           "chair": 62, "cup": 47, "bottle": 44, "book": 84}

B = 200
SEED = 42

RAW_DIR = "results/m6_40class/owlv2_raw"


def load_raw(cls):
    """Load raw detections for a class from the single-query run."""
    path = os.path.join(RAW_DIR, f"{cls}.json")
    with open(path) as f:
        raw = json.load(f)
    return raw


def extract_dets_from_raw(raw, prompt_name, img_ids):
    """Extract detections for a specific prompt from raw JSON.

    raw["per_image_detections"] = {str(img_id): {"bare": [...], "template": [...], "there_is": [...]}}
    """
    dets = {}
    pid = raw["per_image_detections"]
    for img_id in img_ids:
        s_id = str(img_id)
        if s_id in pid and prompt_name in pid[s_id]:
            img_dets = []
            for det in pid[s_id][prompt_name]:
                bbox = det["bbox"]
                w, h = bbox[2], bbox[3]
                img_dets.append({
                    'score': det['score'],
                    'bbox': bbox,
                    'area': float(w * h),
                })
            dets[img_id] = img_dets
        else:
            dets[img_id] = []
    return dets


def bootstrap_weighted(raw, cls, img_ids, gts, B=200, seed=42):
    """Bootstrap with weighted AP50.

    For each resample:
      - Sample n image IDs with replacement
      - weights = {img_id: count}
      - Compute weighted AP50 for bare, template, there_is prompts
      - delta_bare = AP_bare - AP_template
      - delta_len  = AP_there_is - AP_template
    """
    dets_bare = extract_dets_from_raw(raw, "bare", img_ids)
    dets_tmpl = extract_dets_from_raw(raw, "template", img_ids)
    dets_this = extract_dets_from_raw(raw, "there_is", img_ids)

    # Verify prompt strings
    pb = raw["prompt_bank"]
    assert pb["bare"] == cls, f"bare prompt mismatch: {pb['bare']} != {cls}"
    assert pb["template"] == "a photo of a " + cls
    assert pb["there_is"] == "there is a " + cls

    rng = np.random.RandomState(seed)
    n = len(img_ids)
    arr = np.array(img_ids)

    # Point estimates (full set, weights=1)
    weights_full = {img_id: 1 for img_id in img_ids}
    ap_bare_pt = weighted_ap50(dets_bare, gts, weights_full)
    ap_tmpl_pt = weighted_ap50(dets_tmpl, gts, weights_full)
    ap_this_pt = weighted_ap50(dets_this, gts, weights_full)

    delta_bare_pt = ap_bare_pt - ap_tmpl_pt
    delta_len_pt = ap_this_pt - ap_tmpl_pt

    # Bootstrap
    deltas_bare = []
    deltas_len = []
    unique_counts = []

    for b in range(B):
        idx = rng.randint(0, n, size=n)
        ids = arr[idx]

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
            deltas_len.append(ap_s - ap_t)

    deltas_bare = np.array(deltas_bare)
    deltas_len = np.array(deltas_len)

    valid_bare = deltas_bare[~np.isnan(deltas_bare)]
    valid_len = deltas_len[~np.isnan(deltas_len)]

    ci_low_bare = float(np.percentile(valid_bare, 2.5)) if len(valid_bare) > 0 else float('nan')
    ci_high_bare = float(np.percentile(valid_bare, 97.5)) if len(valid_bare) > 0 else float('nan')
    ci_low_len = float(np.percentile(valid_len, 2.5)) if len(valid_len) > 0 else float('nan')
    ci_high_len = float(np.percentile(valid_len, 97.5)) if len(valid_len) > 0 else float('nan')

    # Bonferroni-corrected CI: alpha/8 = 0.00625, so 99.375% CI
    # percentiles: 0.3125 and 99.6875
    bonf_low_bare = float(np.percentile(valid_bare, 0.3125)) if len(valid_bare) > 0 else float('nan')
    bonf_high_bare = float(np.percentile(valid_bare, 99.6875)) if len(valid_bare) > 0 else float('nan')
    bonf_low_len = float(np.percentile(valid_len, 0.3125)) if len(valid_len) > 0 else float('nan')
    bonf_high_len = float(np.percentile(valid_len, 99.6875)) if len(valid_len) > 0 else float('nan')

    result = {
        "delta_bare": float(delta_bare_pt),
        "ci_low": ci_low_bare,
        "ci_high": ci_high_bare,
        "ci_contains_zero": bool(ci_low_bare <= 0 <= ci_high_bare) if len(valid_bare) > 0 else False,
        "ci_bonf_low": bonf_low_bare,
        "ci_bonf_high": bonf_high_bare,
        "ci_bonf_contains_zero": bool(bonf_low_bare <= 0 <= bonf_high_bare) if len(valid_bare) > 0 else False,
        "point_in_ci": bool(ci_low_bare <= delta_bare_pt <= ci_high_bare) if len(valid_bare) > 0 else False,
        "delta_len": float(delta_len_pt),
        "delta_len_ci_low": ci_low_len,
        "delta_len_ci_high": ci_high_len,
        "delta_len_ci_contains_zero": bool(ci_low_len <= 0 <= ci_high_len) if len(valid_len) > 0 else False,
        "delta_len_ci_bonf_low": bonf_low_len,
        "delta_len_ci_bonf_high": bonf_high_len,
        "delta_len_ci_bonf_contains_zero": bool(bonf_low_len <= 0 <= bonf_high_len) if len(valid_len) > 0 else False,
        "delta_len_point_in_ci": bool(ci_low_len <= delta_len_pt <= ci_high_len) if len(valid_len) > 0 else False,
        "n_bootstrap_bare_valid": int(len(valid_bare)),
        "n_bootstrap_bare_nan": int(np.sum(np.isnan(deltas_bare))),
        "n_bootstrap_len_valid": int(len(valid_len)),
        "n_bootstrap_len_nan": int(np.sum(np.isnan(deltas_len))),
        "ap_bare_point": float(ap_bare_pt),
        "ap_template_point": float(ap_tmpl_pt),
        "ap_there_is_point": float(ap_this_pt),
        "prompt_strings": {
            "bare": pb["bare"],
            "template": pb["template"],
            "there_is": pb["there_is"],
        },
    }

    return result, float(np.mean(unique_counts))


# ── Main ─────────────────────────────────────────────────────────────

print("Loading COCO val2017 annotations...", flush=True)
coco_gt = COCO(COCO_ANN_FILE)

results = {}
all_unique_counts = []

for cls in CLASSES:
    print(f"\nProcessing OWLv2 {cls}...", flush=True)
    t0 = time.time()

    raw = load_raw(cls)
    img_ids = raw["selected_img_ids"]

    cat_id = CAT_IDS[cls]
    gts = load_gt_for_class(coco_gt, cat_id, img_ids)

    res, mean_unique = bootstrap_weighted(raw, cls, img_ids, gts, B=B, seed=SEED)
    results[cls] = res
    all_unique_counts.append(mean_unique)

    db = res["delta_bare"]
    db_lo = res["ci_low"]
    db_hi = res["ci_high"]
    db_sig = "SIG" if not res["ci_contains_zero"] else "ns"
    db_pic = "✓" if res["point_in_ci"] else "✗"
    dl = res["delta_len"]
    dl_lo = res["delta_len_ci_low"]
    dl_hi = res["delta_len_ci_high"]
    dl_sig = "SIG" if not res["delta_len_ci_contains_zero"] else "ns"
    dl_pic = "✓" if res["delta_len_point_in_ci"] else "✗"
    nan_b = res["n_bootstrap_bare_nan"]
    nan_l = res["n_bootstrap_len_nan"]

    db_bonf_sig = "SIG" if not res["ci_bonf_contains_zero"] else "ns"

    print(f"  Δ_bare={db:+.4f} CI95=[{db_lo:+.4f},{db_hi:+.4f}] {db_sig}  CI99.375=[{res['ci_bonf_low']:+.4f},{res['ci_bonf_high']:+.4f}] {db_bonf_sig} pic={db_pic}"
          f"  Δ_len={dl:+.4f} CI95=[{dl_lo:+.4f},{dl_hi:+.4f}] {dl_sig} pic={dl_pic}"
          f"  NaN(bare={nan_b},len={nan_l})", flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s", flush=True)

output = {
    "bootstrap_B": B,
    "seed": SEED,
    "method": "weighted_ap50_single_query_protocol",
    "delta_bare_formula": "AP50(bare) - AP50(template)",
    "delta_len_formula": "AP50(there_is_a_X) - AP50(template)",
    "protocol": "single_query",
    "source": "results/m6_40class/owlv2_raw/<class>.json",
    "effective_unique_per_resample_mean": float(np.mean(all_unique_counts)),
    "results": results,
}

out_path = "results/m6_40class/owlv2_8class_bootstrap.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*80}")
print(f"Saved to {out_path}")
print(f"Mean unique images per resample: {np.mean(all_unique_counts):.1f} / 500")
print(f"Method: weighted_ap50, single-query protocol, B={B}, seed={SEED}")
