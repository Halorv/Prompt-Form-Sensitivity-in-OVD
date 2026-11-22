#!/usr/bin/env python3
"""
M-3 remediation: Bootstrap CI with B=1000 (community standard).

Answers audit item M-3 ("Bootstrap B=200 is below community standard").
No model re-inference is needed: all per-image raw detections are cached, so
this is a pure re-aggregation run.

Data sources (identical to the B=200 canonical pipeline):
  GLIP : results/raw/m1_v2_raw_detections.json      (80-query, prompt-index layout)
  GDino: results/raw/gdino_v11_raw_detections.json  (80-query, prompt-index layout)
  OWLv2: results/m6_40class/owlv2_raw/<cls>.json    (single-query, prompt-name layout)

Protocol — deliberately unchanged from bootstrap_weighted_v2.py /
owlv2_8class_bootstrap.py, so the ONLY difference from the paper numbers is B:
  - image-level resampling with replacement
  - weighted_ap50 (replicates pycocotools; correct under duplicate image IDs)
  - seed=42, same RNG stream -> the first 200 resamples coincide with the
    B=200 run, so B=1000 is a strict refinement, not a different draw
  - 95% percentile CI + Bonferroni-corrected 99.375% CI (alpha/8)

Output: results/BOOTSTRAP_B1000_8CLASS.json
Run from the openworld project root:
    nohup python3 experiments/bootstrap_b1000.py > results/b1000_run.log 2>&1 &
"""
import sys, os, json, time, gc
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pycocotools.coco import COCO
from weighted_ap import weighted_ap50, load_gt_for_class, extract_dets

COCO_ANN_FILE = os.environ.get("COCO_ANN_FILE", "data/annotations/instances_val2017.json")

CLASSES = ["person", "car", "dog", "cat", "chair", "cup", "bottle", "book"]
CAT_IDS = {"person": 1, "car": 3, "dog": 18, "cat": 17,
           "chair": 62, "cup": 47, "bottle": 44, "book": 84}

B = 1000
SEED = 42

OUT_PATH = "results/BOOTSTRAP_B1000_8CLASS.json"
CANON_PATH = "results/CANONICAL_8CLASS.json"


# ── Prompt resolution (copied verbatim from bootstrap_weighted_v2.py) ──

def resolve_prompt_indices(raw, cls):
    """Resolve prompt indices by string matching, no hardcoded indices."""
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
    return bare_idx, template_idx, there_is_idx


# ── Core bootstrap loop (identical logic to the B=200 scripts) ──

def bootstrap_core(dets_bare, dets_tmpl, dets_this, img_ids, gts, B, seed, tag):
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

    deltas_bare = np.empty(B)
    deltas_len = np.empty(B)
    unique_counts = []

    t0 = time.time()
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

        deltas_bare[b] = np.nan if (np.isnan(ap_b) or np.isnan(ap_t)) else ap_b - ap_t
        deltas_len[b] = np.nan if (np.isnan(ap_s) or np.isnan(ap_t)) else ap_s - ap_t

        if (b + 1) % 200 == 0:
            el = time.time() - t0
            print(f"    [{tag}] {b+1}/{B}  ({el:.0f}s, {el/(b+1):.3f}s/resample)", flush=True)

    valid_bare = deltas_bare[~np.isnan(deltas_bare)]
    valid_len = deltas_len[~np.isnan(deltas_len)]

    def pct(arr_in, q):
        return float(np.percentile(arr_in, q)) if len(arr_in) > 0 else float("nan")

    rec = {
        "delta_bare": float(delta_bare_pt),
        "ci_low": pct(valid_bare, 2.5),
        "ci_high": pct(valid_bare, 97.5),
        "ci_contains_zero": bool(pct(valid_bare, 2.5) <= 0 <= pct(valid_bare, 97.5)) if len(valid_bare) > 0 else False,
        "ci_bonf_low": pct(valid_bare, 0.3125),
        "ci_bonf_high": pct(valid_bare, 99.6875),
        "ci_bonf_contains_zero": bool(pct(valid_bare, 0.3125) <= 0 <= pct(valid_bare, 99.6875)) if len(valid_bare) > 0 else False,
        "point_in_ci": bool(pct(valid_bare, 2.5) <= delta_bare_pt <= pct(valid_bare, 97.5)) if len(valid_bare) > 0 else False,
        "delta_len": float(delta_len_pt),
        "delta_len_ci_low": pct(valid_len, 2.5),
        "delta_len_ci_high": pct(valid_len, 97.5),
        "delta_len_ci_contains_zero": bool(pct(valid_len, 2.5) <= 0 <= pct(valid_len, 97.5)) if len(valid_len) > 0 else False,
        "delta_len_ci_bonf_low": pct(valid_len, 0.3125),
        "delta_len_ci_bonf_high": pct(valid_len, 99.6875),
        "delta_len_ci_bonf_contains_zero": bool(pct(valid_len, 0.3125) <= 0 <= pct(valid_len, 99.6875)) if len(valid_len) > 0 else False,
        "delta_len_point_in_ci": bool(pct(valid_len, 2.5) <= delta_len_pt <= pct(valid_len, 97.5)) if len(valid_len) > 0 else False,
        "n_bootstrap_bare_valid": int(len(valid_bare)),
        "n_bootstrap_bare_nan": int(np.sum(np.isnan(deltas_bare))),
        "n_bootstrap_len_valid": int(len(valid_len)),
        "n_bootstrap_len_nan": int(np.sum(np.isnan(deltas_len))),
        "ap_bare_point": float(ap_bare_pt),
        "ap_template_point": float(ap_tmpl_pt),
        "ap_there_is_point": float(ap_this_pt),
        "bootstrap_B": B,
        "seed": seed,
        "n_images": n,
        "mean_unique_images_per_resample": float(np.mean(unique_counts)),
    }
    return rec


# ── Model runners ──

def run_prompt_index_model(raw, model_name, coco_gt, B, seed):
    """GLIP / GDino: raw['per_image_detections'][img_id][f'{cls}__{prompt_idx}']."""
    img_ids = raw["selected_img_ids"]
    results = {}
    for cls in CLASSES:
        print(f"\nProcessing {model_name} {cls} (B={B})...", flush=True)
        t0 = time.time()
        bare_idx, template_idx, there_is_idx = resolve_prompt_indices(raw, cls)
        gts = load_gt_for_class(coco_gt, CAT_IDS[cls], img_ids)
        dets_bare = extract_dets(raw, cls, bare_idx, img_ids)
        dets_tmpl = extract_dets(raw, cls, template_idx, img_ids)
        dets_this = extract_dets(raw, cls, there_is_idx, img_ids)

        rec = bootstrap_core(dets_bare, dets_tmpl, dets_this, img_ids, gts,
                             B, seed, f"{model_name}/{cls}")
        rec["prompt_strings"] = {
            "bare": cls,
            "template": "a photo of a " + cls,
            "there_is": "there is a " + cls,
        }
        results[cls] = rec
        print(f"  {model_name} {cls}: delta_bare={rec['delta_bare']:.4f} "
              f"CI=[{rec['ci_low']:.4f},{rec['ci_high']:.4f}] "
              f"({'SIG' if not rec['ci_contains_zero'] else 'ns'})  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    return results


def run_owlv2_model(coco_gt, B, seed):
    """OWLv2: per-class raw files with named prompts (single-query protocol)."""
    RAW_DIR = "results/m6_40class/owlv2_raw"
    results = {}
    for cls in CLASSES:
        print(f"\nProcessing OWLv2 {cls} (B={B})...", flush=True)
        t0 = time.time()
        with open(os.path.join(RAW_DIR, f"{cls}.json")) as f:
            raw = json.load(f)

        img_ids = raw["selected_img_ids"]
        pb = raw["prompt_bank"]
        assert pb["bare"] == cls, f"bare prompt mismatch: {pb['bare']} != {cls}"
        assert pb["template"] == "a photo of a " + cls
        assert pb["there_is"] == "there is a " + cls

        def extract(prompt_name):
            dets = {}
            pid = raw["per_image_detections"]
            for img_id in img_ids:
                s_id = str(img_id)
                img_dets = []
                if s_id in pid and prompt_name in pid[s_id]:
                    for det in pid[s_id][prompt_name]:
                        bbox = det["bbox"]
                        w, h = bbox[2], bbox[3]
                        img_dets.append({"score": det["score"], "bbox": bbox,
                                         "area": float(w * h)})
                dets[img_id] = img_dets
            return dets

        gts = load_gt_for_class(coco_gt, CAT_IDS[cls], img_ids)
        dets_bare = extract("bare")
        dets_tmpl = extract("template")
        dets_this = extract("there_is")

        rec = bootstrap_core(dets_bare, dets_tmpl, dets_this, img_ids, gts,
                             B, seed, f"OWLv2/{cls}")
        rec["prompt_strings"] = dict(pb)
        results[cls] = rec
        print(f"  OWLv2 {cls}: delta_bare={rec['delta_bare']:.4f} "
              f"CI=[{rec['ci_low']:.4f},{rec['ci_high']:.4f}] "
              f"({'SIG' if not rec['ci_contains_zero'] else 'ns'})  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    return results


# ── Comparison vs B=200 canonical ──

def compare_with_b200(b1000):
    canon = json.load(open(CANON_PATH))["models"]
    rows = []
    for m in ("GLIP", "GDino", "OWLv2"):
        for c in CLASSES:
            old = canon[m][c]
            new = b1000[m][c]
            for metric, lo_k, hi_k, new_lo_k, new_hi_k, sig_k, new_sig_k in (
                ("delta_bare", "ci_low", "ci_high", "ci_low", "ci_high",
                 "ci_contains_zero", "ci_contains_zero"),
                ("delta_len", "delta_len_ci_low", "delta_len_ci_high",
                 "delta_len_ci_low", "delta_len_ci_high",
                 "delta_len_ci_contains_zero", "delta_len_ci_contains_zero"),
            ):
                flip = bool(old[sig_k]) != bool(new[new_sig_k])
                rows.append({
                    "model": m, "class": c, "metric": metric,
                    "b200_point": old[metric],
                    "b1000_point": new[metric],
                    "point_diff": abs(old[metric] - new[metric]),
                    "b200_ci": [old[lo_k], old[hi_k]],
                    "b1000_ci": [new[new_lo_k], new[new_hi_k]],
                    "b200_sig": not bool(old[sig_k]),
                    "b1000_sig": not bool(new[new_sig_k]),
                    "significance_flip": flip,
                })
    return rows


def main():
    print(f"M-3 remediation run: bootstrap B={B}, seed={SEED}", flush=True)
    print("Loading COCO val2017 annotations...", flush=True)
    coco_gt = COCO(COCO_ANN_FILE)

    out = {"bootstrap_B": B, "seed": SEED,
           "method": "weighted_ap50_single_query_and_80query_protocol_same_as_B200",
           "note": ("Same implementation and RNG stream as the B=200 runs "
                    "(bootstrap_weighted_v2.py for GLIP/GDino, "
                    "owlv2_8class_bootstrap.py for OWLv2); only B differs."),
           "models": {}}

    # GLIP then GDino share the prompt-index layout; load one file at a time
    # to bound memory (gdino file is 824 MB).
    for path, name in (("results/raw/m1_v2_raw_detections.json", "GLIP"),
                       ("results/raw/gdino_v11_raw_detections.json", "GDino")):
        print(f"\n=== Loading {name} raw detections: {path} ===", flush=True)
        t0 = time.time()
        with open(path) as f:
            raw = json.load(f)
        print(f"  loaded in {time.time()-t0:.0f}s", flush=True)
        out["models"][name] = run_prompt_index_model(raw, name, coco_gt, B, SEED)
        del raw
        gc.collect()

    print("\n=== OWLv2 (per-class single-query raw) ===", flush=True)
    out["models"]["OWLv2"] = run_owlv2_model(coco_gt, B, SEED)

    # Sanity check: point estimates must match the B=200 canonical file
    canon = json.load(open(CANON_PATH))["models"]
    max_diff = 0.0
    for m in out["models"]:
        for c in CLASSES:
            for k in ("delta_bare", "delta_len", "ap_bare_point", "ap_template_point", "ap_there_is_point"):
                ko = {"ap_bare_point": "ap50"}.get(k, k)
                if k in ("ap_bare_point", "ap_template_point", "ap_there_is_point"):
                    continue  # canonical stores per-prompt ap50 dict separately
                d = abs(out["models"][m][c][k] - canon[m][c][ko])
                max_diff = max(max_diff, d)
    out["point_estimate_max_diff_vs_canonical"] = max_diff
    print(f"\nPoint-estimate max |diff| vs canonical B=200: {max_diff:.2e}", flush=True)

    out["comparison_b200_vs_b1000"] = compare_with_b200(out["models"])

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}", flush=True)

    # Summary of flips
    flips = [r for r in out["comparison_b200_vs_b1000"] if r["significance_flip"]]
    print(f"\n===== SUMMARY =====", flush=True)
    print(f"Point-estimate max diff vs B=200: {max_diff:.2e}", flush=True)
    print(f"Significance flips (95% CI, any model/class/metric): {len(flips)}", flush=True)
    for r in flips:
        print(f"  {r['model']:6s} {r['class']:7s} {r['metric']:10s} "
              f"B200 CI=[{r['b200_ci'][0]:.4f},{r['b200_ci'][1]:.4f}] -> "
              f"B1000 CI=[{r['b1000_ci'][0]:.4f},{r['b1000_ci'][1]:.4f}]", flush=True)


if __name__ == "__main__":
    main()
