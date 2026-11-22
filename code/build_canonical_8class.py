#!/usr/bin/env python3
"""Build results/CANONICAL_8CLASS.json — the SINGLE source of truth for Table 1.

Why this exists
---------------
`v14_bootstrap_weighted_v2.json` is correct for GLIP and Grounding DINO, but its
OWLv2 block is from the 80-query protocol, which the paper does NOT use. OWLv2
must come from `m6_40class/owlv2_8class_bootstrap.json` (single-query).

Any audit that reads `v14_bootstrap_weighted_v2.json` for OWLv2 WILL report
spurious Table 1 mismatches. Merging the two sources into one file removes the
possibility of picking the wrong block.

Run from the openworld project root:
    python3 experiments/build_canonical_8class.py
"""
import json
import os

CLASSES = ["person", "car", "dog", "cat", "chair", "cup", "bottle", "book"]
THRESHOLD = 0.01  # |delta_bare| below this -> ratio is numerically unstable

# Source files may live at the top level or already be archived.
SEARCH_DIRS = ["results", "results/archive_superseded", "results/archive_80query"]


def load(rel):
    for d in SEARCH_DIRS:
        p = os.path.join(d, rel)
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError(f"{rel} not found in {SEARCH_DIRS}")


def norm(src, key_map):
    """Normalise one class record into the canonical shape."""
    get = lambda k: src.get(key_map.get(k, k))
    rec = {
        "delta_bare": get("delta_bare"),
        "ci_low": get("ci_low"),
        "ci_high": get("ci_high"),
        "ci_contains_zero": get("ci_contains_zero"),
        "delta_len": get("delta_len"),
        "delta_len_ci_low": get("delta_len_ci_low"),
        "delta_len_ci_high": get("delta_len_ci_high"),
        "delta_len_ci_contains_zero": get("delta_len_ci_contains_zero"),
    }
    db, dl = rec["delta_bare"], rec["delta_len"]
    if db is None:
        rec["ratio"] = None
    elif abs(db) < THRESHOLD:
        rec["ratio"] = None
        rec["ratio_note"] = "N/A: |delta_bare| < 0.01, ratio numerically unstable"
    else:
        rec["ratio"] = dl / db
    return rec


def per_prompt_ap50():
    """AP50 of the three prompts the paper quotes, for every class."""
    out = {}

    glip = load("glip_8class_ap50_summary.json")["results"]
    out["GLIP"] = {
        c: {
            "bare": glip[c]["per_prompt"][c]["ap50"],
            "template": glip[c]["per_prompt"][f"a photo of a {c}"]["ap50"],
            "there_is": glip[c]["per_prompt"][f"there is a {c}"]["ap50"],
        }
        for c in CLASSES
    }

    gd = load("gdino_v11_summary.json")["per_prompt_ap50"]
    out["GDino"] = {
        c: {
            "bare": gd[c][c]["ap50_all"],
            "template": gd[c][f"a photo of a {c}"]["ap50_all"],
            "there_is": gd[c][f"there is a {c}"]["ap50_all"],
        }
        for c in CLASSES
    }

    owl = load("m6_40class/owlv2_8class_bootstrap.json")
    owl = owl.get("results", owl)
    out["OWLv2"] = {
        c: {
            "bare": owl[c]["ap_bare_point"],
            "template": owl[c]["ap_template_point"],
            "there_is": owl[c]["ap_there_is_point"],
        }
        for c in CLASSES
    }
    return out


def main():
    v14 = load("v14_bootstrap_weighted_v2.json")["results"]
    owl = load("m6_40class/owlv2_8class_bootstrap.json")
    owl = owl.get("results", owl)

    models = {}
    for m in ("GLIP", "GDino"):
        models[m] = {c: norm(v14[m][c], {}) for c in CLASSES}
    models["OWLv2"] = {c: norm(owl[c], {}) for c in CLASSES}
    pp = per_prompt_ap50()
    for m in models:
        for c in CLASSES:
            models[m][c]["ap50"] = pp[m][c]

    means = {}
    for m, per in models.items():
        n = len(per)
        mb = sum(v["delta_bare"] for v in per.values()) / n
        ml = sum(v["delta_len"] for v in per.values()) / n
        means[m] = {
            "mean_delta_bare": mb,
            "mean_delta_len": ml,
            "n_sig_delta_bare": sum(1 for v in per.values() if v["ci_contains_zero"] is False),
            "n_sig_delta_len": sum(
                1 for v in per.values() if v["delta_len_ci_contains_zero"] is False
            ),
            "ratio_of_means": (ml / mb) if abs(mb) >= THRESHOLD else None,
        }

    out = {
        "schema": "canonical_8class_v1",
        "description": (
            "SINGLE SOURCE OF TRUTH for Table 1 and all 8-class OWLv2/GLIP/GDino claims. "
            "Do not substitute any other results/*.json for these numbers."
        ),
        "protocol": {
            "GLIP": "default model pipeline",
            "GDino": "default model pipeline",
            "OWLv2": "SINGLE-QUERY: one text query per forward pass (NOT 80-query batching)",
        },
        "bootstrap": {
            "B": 200,
            "seed": 42,
            "ci": "95% percentile, paired image-level resampling, weighted AP50",
            "effective_unique_per_resample_mean": 316.625,
        },
        "sources": {
            "GLIP": "results/archive_superseded/v14_bootstrap_weighted_v2.json",
            "GDino": "results/archive_superseded/v14_bootstrap_weighted_v2.json",
            "OWLv2": "results/m6_40class/owlv2_8class_bootstrap.json",
        },
        "definitions": {
            "delta_bare": "AP50(bare class name) - AP50('a photo of a X')",
            "delta_len": "AP50('there is a X') - AP50('a photo of a X')",
            "ratio": "delta_len / delta_bare; N/A when |delta_bare| < 0.01",
        },
        "means": means,
        "models": models,
    }

    os.makedirs("results", exist_ok=True)
    path = "results/CANONICAL_8CLASS.json"
    for m, per in models.items():
        for c, v in per.items():
            assert v["delta_bare"] is not None, f"missing delta_bare {m}/{c}"
            print(f"{m:6s} {c:7s} db={v['delta_bare']:+.4f} "
                  f"[{v['ci_low']:+.4f},{v['ci_high']:+.4f}] "
                  f"dl={v['delta_len']:+.4f} r="
                  f"{('%.2f' % v['ratio']) if v['ratio'] is not None else 'N/A'}")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    print("means:", json.dumps(means, indent=2))


if __name__ == "__main__":
    main()
