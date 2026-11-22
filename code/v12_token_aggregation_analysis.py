#!/usr/bin/env python3
"""
v12 Token Aggregation Confound Analysis — ZERO GPU.

Reads pre-existing raw detections from GLIP, OWLv2, Grounding DINO.
Computes:

  Step 1: Δ_len adjudication
    Δ_bare = AP50(bare) - AP50("a photo of a X")    [idx 0 vs idx 2]
    Δ_len  = AP50("there is a X") - AP50("a photo of a X")  [idx 6 vs idx 2]
    Both with B=200 paired bootstrap 95% CI, per-class pt_in_ci check.
    All 3 models.

  Step 2: token count × AP50 table
    BERT tokenizer actual token counts (with "." suffix for GDino convention).
    10 prompts × 8 classes × 3 models AP50.

  Step 3: complete data tables (3×8×10 AP50 + n_gt + Δ/CI + pt_in_ci)

  Step 4: verify 100-box truncation (n_det=50000=100/img) + score-sorted order

  Step 5: uncertainty paragraph (written to report, not computed here)

Outputs:
  results/v12_delta_len_adjudication.json
  results/v12_complete_ap50_table.json
  results/v12_truncation_audit.json
  refine-logs/EXPERIMENT_RESULTS_v12.md
"""
import os, sys, json, time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ── Constants ─────────────────────────────────────────────────────────────
COCO_ANN_FILE = os.environ.get("COCO_ANN_FILE", "data/annotations/instances_val2017.json")

CLASS_INFO = {
    "person": 1, "car": 3, "dog": 18, "cat": 17,
    "chair": 62, "cup": 47, "bottle": 44, "book": 84,
}

PROMPT_BANK = {
    "person": ["person", "a person", "a photo of a person", "a person in the scene",
               "people", "human", "there is a person", "a human standing",
               "an individual person", "a pedestrian"],
    "car": ["car", "a car", "a photo of a car", "a car in the scene",
            "vehicle", "automobile", "there is a car", "a car parked",
            "a sedan", "a motorcar"],
    "dog": ["dog", "a dog", "a photo of a dog", "a dog in the scene",
            "puppy", "canine", "there is a dog", "a dog standing",
            "a pet dog", "a dog running"],
    "cat": ["cat", "a cat", "a photo of a cat", "a cat in the scene",
            "kitten", "feline", "there is a cat", "a cat sitting",
            "a pet cat", "a cat lying down"],
    "chair": ["chair", "a chair", "a photo of a chair", "a chair in the scene",
              "seat", "furniture", "there is a chair", "a wooden chair",
              "an empty chair", "a chair nearby"],
    "cup": ["cup", "a cup", "a photo of a cup", "a cup in the scene",
            "mug", "drinking cup", "there is a cup", "a cup on the table",
            "a teacup", "a coffee cup"],
    "bottle": ["bottle", "a bottle", "a photo of a bottle", "a bottle in the scene",
               "flask", "container", "there is a bottle", "a glass bottle",
               "a water bottle", "a plastic bottle"],
    "book": ["book", "a book", "a photo of a book", "a book in the scene",
             "volume", "publication", "there is a book", "a book on the shelf",
             "a paperback", "a hardcover book"],
}

# Prompt indices
IDX_BARE = 0       # "person"
IDX_PHOTO = 2     # "a photo of a person"
IDX_THERE = 6     # "there is a person"

MODELS = {
    "GLIP":    "results/m1_v2_raw_detections.json",
    "OWLv2":   "results/owlv2_v10_raw_detections.json",
    "GDino":   "results/gdino_v11_raw_detections.json",
}

# ── Token counting ───────────────────────────────────────────────────────
def get_token_counts():
    """Compute BERT token counts for each prompt.
    GDino convention: prompt + '.', lowercase.
    For fair comparison, we compute both with and without the '.' suffix.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    token_counts = {}
    for cn, prompts in PROMPT_BANK.items():
        token_counts[cn] = []
        for p in prompts:
            # Without '.' (raw prompt)
            ids_raw = tokenizer.encode(p, add_special_tokens=False)
            # With '.' (GDino convention)
            p_dot = p.strip().lower() + "."
            ids_dot = tokenizer.encode(p_dot, add_special_tokens=False)
            token_counts[cn].append({
                "prompt": p,
                "n_tokens_no_dot": len(ids_raw),
                "n_tokens_with_dot": len(ids_dot),
                "tokens_no_dot": tokenizer.convert_ids_to_tokens(ids_raw),
                "tokens_with_dot": tokenizer.convert_ids_to_tokens(ids_dot),
            })
    return token_counts


# ── AP50 computation (identical protocol to v10/v11) ─────────────────────
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
    ap50 = float(prec_slice[valid_mask].mean()) if valid_mask.any() else 0.0
    num_gt = sum(1 for img_id in selected_img_ids
                 for ann in coco_gt.loadAnns(
                     coco_gt.getAnnIds(imgIds=[img_id], catIds=[cat_id]))
                 if ann.get("iscrowd", 0) == 0)
    return ap50, num_gt


# ── Paired bootstrap CI ──────────────────────────────────────────────────
def paired_bootstrap_delta(coco_gt, dets_a, dets_b, selected_img_ids,
                           cat_id, B=200, seed=42, alpha=0.05,
                           label_a="A", label_b="B", verbose=True):
    """Compute Δ = AP50(A) - AP50(B) with paired bootstrap CI.
    
    Both A and B are evaluated on the SAME bootstrap resample of images,
    so the pairing is preserved.
    """
    rng = np.random.RandomState(seed)
    n = len(selected_img_ids)
    img_id_arr = np.array(selected_img_ids)

    # Point estimates
    ap_a_full, n_gt = compute_ap50(coco_gt, dets_a, selected_img_ids, cat_id)
    ap_b_full, _    = compute_ap50(coco_gt, dets_b, selected_img_ids, cat_id)
    delta_point = ap_a_full - ap_b_full

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
        return float(prec_slice[valid_mask].mean()) if valid_mask.any() else 0.0

    all_deltas = []
    for b in range(B):
        if verbose and (b + 1) % 50 == 0:
            print(f"    bootstrap {b+1}/{B}...", flush=True)
        idx = rng.randint(0, n, size=n)
        img_ids_sub = img_id_arr[idx].tolist()
        ap_a = ap_on_subset(img_ids_sub, dets_a)
        ap_b = ap_on_subset(img_ids_sub, dets_b)
        all_deltas.append(ap_a - ap_b)

    all_deltas = np.array(all_deltas)
    ci_low = float(np.percentile(all_deltas, 100 * alpha / 2))
    ci_high = float(np.percentile(all_deltas, 100 * (1 - alpha / 2)))

    pt_in_ci = ci_low <= delta_point <= ci_high
    crosses_zero = ci_low <= 0.0 <= ci_high

    return {
        "ap_a": ap_a_full,
        "ap_b": ap_b_full,
        "delta_point": delta_point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pt_in_ci": pt_in_ci,
        "crosses_zero": crosses_zero,
        "n_gt": n_gt,
        "label_a": label_a,
        "label_b": label_b,
        "all_bootstrap_deltas": all_deltas.tolist(),
    }


# ── Load and flatten detections ──────────────────────────────────────────
def load_raw_detections(path, selected_img_ids_set):
    """Load raw detections and flatten into per-(class, prompt) detection lists."""
    with open(path) as f:
        raw = json.load(f)
    
    per_image = raw["per_image_detections"]
    prompt_bank = raw["prompt_bank"]
    
    # Build flat detection lists: {(class_name, prompt_idx): [dets]}
    all_dets = {}
    for cn in CLASS_INFO:
        for pi in range(10):
            all_dets[(cn, pi)] = []
    
    for img_id_str, prompts_dict in per_image.items():
        img_id = int(img_id_str)
        for key, dets in prompts_dict.items():
            parts = key.split("__")
            cn = parts[0]
            pi = int(parts[1])
            for d in dets:
                d["image_id"] = img_id
                d["category_id"] = CLASS_INFO[cn]
            all_dets[(cn, pi)].extend(dets)
    
    return all_dets, prompt_bank


# ── Step 4: Truncation audit ─────────────────────────────────────────────
def audit_truncation(raw_data, model_name):
    """Verify that max_dets=100 per prompt per image, and that truncation
    happens AFTER score sorting (top-k by score)."""
    per_image = raw_data["per_image_detections"]
    
    audit = {
        "model": model_name,
        "max_dets_config": raw_data.get("config", {}).get("max_dets", "N/A"),
        "per_image_per_prompt_counts": [],
        "truncation_evidence": [],
        "score_sorted_evidence": [],
    }
    
    n_truncated = 0
    n_exact_100 = 0
    n_total_prompt_images = 0
    
    for img_id_str, prompts_dict in per_image.items():
        for key, dets in prompts_dict.items():
            n = len(dets)
            n_total_prompt_images += 1
            if n == 100:
                n_exact_100 += 1
            if n >= 100:
                n_truncated += 1
            
            audit["per_image_per_prompt_counts"].append({
                "img_id": img_id_str,
                "prompt_key": key,
                "n_dets": n,
            })
            
            # Check if scores are sorted descending (evidence of top-k truncation)
            if n >= 100 and len(dets) >= 2:
                scores = [d["score"] for d in dets]
                is_sorted_desc = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
                audit["score_sorted_evidence"].append({
                    "img_id": img_id_str,
                    "prompt_key": key,
                    "n_dets": n,
                    "is_sorted_desc": is_sorted_desc,
                    "first_score": scores[0],
                    "last_score": scores[-1],
                })
    
    audit["summary"] = {
        "total_prompt_image_combos": n_total_prompt_images,
        "n_exact_100": n_exact_100,
        "n_truncated_ge100": n_truncated,
        "pct_exact_100": round(100 * n_exact_100 / n_total_prompt_images, 2) if n_total_prompt_images else 0,
    }
    
    # Check score sorting globally
    n_checked = len(audit["score_sorted_evidence"])
    n_sorted = sum(1 for e in audit["score_sorted_evidence"] if e["is_sorted_desc"])
    audit["summary"]["n_score_sorted_checked"] = n_checked
    audit["summary"]["n_score_sorted_desc"] = n_sorted
    audit["summary"]["pct_score_sorted"] = round(100 * n_sorted / n_checked, 2) if n_checked else 0
    
    return audit


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    
    print("=" * 70)
    print("v12 TOKEN AGGREGATION CONFOUND ANALYSIS — ZERO GPU")
    print("=" * 70)
    
    # ── Load COCO GT ──────────────────────────────────────────────────────
    print("\nLoading COCO val2017 annotations...")
    coco_gt = COCO(COCO_ANN_FILE)
    
    # ── Load all three models' raw detections ─────────────────────────────
    models_data = {}
    for model_name, path in MODELS.items():
        print(f"\nLoading {model_name} raw detections from {path}...")
        with open(path) as f:
            raw = json.load(f)
        selected_img_ids = raw["selected_img_ids"]
        models_data[model_name] = {
            "raw": raw,
            "selected_img_ids": selected_img_ids,
        }
    
    # Verify all models use the same image set
    ref_ids = set(models_data["GLIP"]["selected_img_ids"])
    for mn in ["OWLv2", "GDino"]:
        ids = set(models_data[mn]["selected_img_ids"])
        if ids != ref_ids:
            print(f"WARNING: {mn} image set differs from GLIP! "
                  f"intersection={len(ids & ref_ids)} vs GLIP={len(ref_ids)}")
        else:
            print(f"  {mn}: same image set ({len(ids)} images) ✓")
    
    selected_img_ids = models_data["GLIP"]["selected_img_ids"]
    selected_img_ids_set = set(selected_img_ids)
    
    # Flatten detections for each model
    for mn in MODELS:
        all_dets, pb = load_raw_detections(MODELS[mn], selected_img_ids_set)
        models_data[mn]["all_dets"] = all_dets
        models_data[mn]["prompt_bank"] = pb
    
    # ── Step 2: Token counts ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: BERT TOKEN COUNTS")
    print("=" * 70)
    
    token_counts = get_token_counts()
    
    # Print token counts
    print(f"\n{'Class':<10} {'Prompt':<35} {'n_tok(no.)':<12} {'n_tok(with.)':<14} {'tokens (with .)'}")
    print("-" * 100)
    for cn in CLASS_INFO:
        for i, tc in enumerate(token_counts[cn]):
            marker = ""
            if i == IDX_BARE:   marker = " ←bare"
            if i == IDX_PHOTO:  marker = " ←photo"
            if i == IDX_THERE:  marker = " ←there"
            print(f"{cn:<10} {tc['prompt']:<35} {tc['n_tokens_no_dot']:<12} "
                  f"{tc['n_tokens_with_dot']:<14} {' '.join(tc['tokens_with_dot'])}{marker}")
        print()
    
    # Save token counts
    with open("results/v12_token_counts.json", "w") as f:
        json.dump(token_counts, f, indent=2, ensure_ascii=False)
    
    # ── Step 3: Complete AP50 table (3 models × 8 classes × 10 prompts) ───
    print("\n" + "=" * 70)
    print("STEP 3: COMPLETE AP50 TABLE (3 models × 8 classes × 10 prompts)")
    print("=" * 70)
    
    complete_ap50 = {}
    for mn in MODELS:
        complete_ap50[mn] = {}
        print(f"\n--- {mn} ---")
        all_dets = models_data[mn]["all_dets"]
        
        for cn, cat_id in CLASS_INFO.items():
            complete_ap50[mn][cn] = {}
            n_gt = None
            for pi in range(10):
                dets = all_dets[(cn, pi)]
                ap50, n_gt_val = compute_ap50(coco_gt, dets, selected_img_ids, cat_id)
                if n_gt is None:
                    n_gt = n_gt_val
                prompt = PROMPT_BANK[cn][pi]
                n_tok = token_counts[cn][pi]["n_tokens_with_dot"]
                complete_ap50[mn][cn][pi] = {
                    "prompt": prompt,
                    "ap50": ap50,
                    "n_gt": n_gt_val,
                    "n_det": len(dets),
                    "n_tokens_bert": n_tok,
                }
                print(f"  {cn:<10} p{pi} {prompt:<35} AP50={ap50:.4f} "
                      f"n_tok={n_tok} n_det={len(dets)}")
            print(f"  {'':10} {'':3} {'n_gt=' + str(n_gt):^35}")
    
    with open("results/v12_complete_ap50_table.json", "w") as f:
        json.dump(complete_ap50, f, indent=2, ensure_ascii=False)
    
    # ── Step 1: Δ_len adjudication ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: Δ_len ADJUDICATION (B=200 paired bootstrap)")
    print("=" * 70)
    
    delta_results = {}
    for mn in MODELS:
        delta_results[mn] = {}
        all_dets = models_data[mn]["all_dets"]
        print(f"\n{'='*60}")
        print(f"  Model: {mn}")
        print(f"{'='*60}")
        
        for cn, cat_id in CLASS_INFO.items():
            prompts = PROMPT_BANK[cn]
            bare_prompt = prompts[IDX_BARE]       # "person"
            photo_prompt = prompts[IDX_PHOTO]     # "a photo of a person"
            there_prompt = prompts[IDX_THERE]     # "there is a person"
            
            bare_dets = all_dets[(cn, IDX_BARE)]
            photo_dets = all_dets[(cn, IDX_PHOTO)]
            there_dets = all_dets[(cn, IDX_THERE)]
            
            print(f"\n  {cn}:")
            
            # Δ_bare = AP50(bare) - AP50(photo)
            print(f"    Δ_bare = AP50('{bare_prompt}') - AP50('{photo_prompt}')...")
            r_bare = paired_bootstrap_delta(
                coco_gt, bare_dets, photo_dets, selected_img_ids, cat_id,
                B=200, seed=42, label_a=bare_prompt, label_b=photo_prompt
            )
            print(f"    Δ_bare = {r_bare['delta_point']:+.4f} "
                  f"CI=[{r_bare['ci_low']:+.4f}, {r_bare['ci_high']:+.4f}] "
                  f"pt_in_ci={r_bare['pt_in_ci']} "
                  f"crosses_0={r_bare['crosses_zero']} "
                  f"n_gt={r_bare['n_gt']}")
            
            # Δ_len = AP50(there) - AP50(photo)
            print(f"    Δ_len  = AP50('{there_prompt}') - AP50('{photo_prompt}')...")
            r_len = paired_bootstrap_delta(
                coco_gt, there_dets, photo_dets, selected_img_ids, cat_id,
                B=200, seed=42, label_a=there_prompt, label_b=photo_prompt
            )
            print(f"    Δ_len  = {r_len['delta_point']:+.4f} "
                  f"CI=[{r_len['ci_low']:+.4f}, {r_len['ci_high']:+.4f}] "
                  f"pt_in_ci={r_len['pt_in_ci']} "
                  f"crosses_0={r_len['crosses_zero']}")
            
            # Ratio
            if abs(r_len['delta_point']) > 1e-8:
                ratio = r_bare['delta_point'] / r_len['delta_point']
            else:
                ratio = float('inf') if abs(r_bare['delta_point']) > 1e-8 else float('nan')
            
            # Adjudication
            # Δ_len same magnitude as Δ_bare (ratio 0.5-2.0) → token aggregation
            # Δ_len significantly smaller → template semantic effect
            if abs(ratio) >= 0.5 and abs(ratio) <= 2.0:
                verdict = "TOKEN_AGGREGATION"
            elif abs(ratio) > 2.0:
                verdict = "TEMPLATE_SEMANTIC"
            else:
                verdict = "INCONCLUSIVE"
            
            print(f"    ratio Δ_bare/Δ_len = {ratio:.2f} → {verdict}")
            
            delta_results[mn][cn] = {
                "delta_bare": {k: v for k, v in r_bare.items() if k != "all_bootstrap_deltas"},
                "delta_len": {k: v for k, v in r_len.items() if k != "all_bootstrap_deltas"},
                "ratio_delta_bare_over_delta_len": ratio,
                "verdict": verdict,
            }
    
    with open("results/v12_delta_len_adjudication.json", "w") as f:
        json.dump(delta_results, f, indent=2, ensure_ascii=False)
    
    # ── Step 4: Truncation audit ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: TRUNCATION AUDIT (100-box cap verification)")
    print("=" * 70)
    
    truncation_audit = {}
    for mn in MODELS:
        print(f"\n--- {mn} ---")
        audit = audit_truncation(models_data[mn]["raw"], mn)
        truncation_audit[mn] = audit
        s = audit["summary"]
        print(f"  Total prompt×image combos: {s['total_prompt_image_combos']}")
        print(f"  Exact 100 dets: {s['n_exact_100']} ({s['pct_exact_100']}%)")
        print(f"  Score-sorted descending (of {s['n_score_sorted_checked']} checked): "
              f"{s['n_score_sorted_desc']} ({s['pct_score_sorted']}%)")
    
    with open("results/v12_truncation_audit.json", "w") as f:
        json.dump(truncation_audit, f, indent=2, ensure_ascii=False)
    
    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Δ_bare vs Δ_len across models")
    print("=" * 70)
    
    print(f"\n{'Model':<8} {'Class':<10} {'Δ_bare':>8} {'Δ_len':>8} "
          f"{'ratio':>8} {'verdict':<20}")
    print("-" * 70)
    
    model_verdicts = {}
    for mn in MODELS:
        model_verdicts[mn] = {"token_agg": 0, "template_sem": 0, "inconclusive": 0}
        for cn in CLASS_INFO:
            r = delta_results[mn][cn]
            db = r["delta_bare"]["delta_point"]
            dl = r["delta_len"]["delta_point"]
            ratio = r["ratio_delta_bare_over_delta_len"]
            verdict = r["verdict"]
            
            print(f"{mn:<8} {cn:<10} {db:>+8.4f} {dl:>+8.4f} "
                  f"{ratio:>8.2f} {verdict:<20}")
            
            if verdict == "TOKEN_AGGREGATION":
                model_verdicts[mn]["token_agg"] += 1
            elif verdict == "TEMPLATE_SEMANTIC":
                model_verdicts[mn]["template_sem"] += 1
            else:
                model_verdicts[mn]["inconclusive"] += 1
        print()
    
    print("\nModel-level verdict counts:")
    for mn in MODELS:
        v = model_verdicts[mn]
        print(f"  {mn}: token_agg={v['token_agg']}, "
              f"template_sem={v['template_sem']}, "
              f"inconclusive={v['inconclusive']}")
    
    # Token-AP correlation
    print("\n" + "=" * 70)
    print("STEP 2 (cont): TOKEN COUNT vs AP50 CORRELATION")
    print("=" * 70)
    
    for mn in MODELS:
        print(f"\n--- {mn} ---")
        print(f"{'Class':<10} {'Prompt':<35} {'n_tok':>5} {'AP50':>8} {'Δ_vs_bare':>10}")
        print("-" * 70)
        
        all_toks = []
        all_aps = []
        
        for cn in CLASS_INFO:
            ap_bare = complete_ap50[mn][cn][IDX_BARE]["ap50"]
            for pi in range(10):
                ap = complete_ap50[mn][cn][pi]["ap50"]
                n_tok = complete_ap50[mn][cn][pi]["n_tokens_bert"]
                delta = ap - ap_bare
                print(f"{cn:<10} {PROMPT_BANK[cn][pi]:<35} {n_tok:>5} {ap:>8.4f} {delta:>+10.4f}")
                all_toks.append(n_tok)
                all_aps.append(ap)
            print()
        
        # Spearman correlation
        all_toks = np.array(all_toks)
        all_aps = np.array(all_aps)
        from scipy.stats import spearmanr, pearsonr
        rho, p_rho = spearmanr(all_toks, all_aps)
        r_pear, p_pear = pearsonr(all_toks, all_aps)
        print(f"  Spearman ρ = {rho:+.4f} (p={p_rho:.4f})")
        print(f"  Pearson r  = {r_pear:+.4f} (p={p_pear:.4f})")
        print(f"  {'Monotonic decrease → TOKEN AGGREGATION' if rho < -0.3 else 'NOT monotonic → SEMANTIC EFFECT'}")
    
    # Elapsed time
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Total elapsed: {elapsed:.1f}s")
    print(f"{'='*70}")
    
    print("\nResults saved:")
    print("  results/v12_token_counts.json")
    print("  results/v12_complete_ap50_table.json")
    print("  results/v12_delta_len_adjudication.json")
    print("  results/v12_truncation_audit.json")


if __name__ == "__main__":
    main()
