#!/usr/bin/env python3
"""Emit results/BONFERRONI_8CLASS.json — evidence for the multiple-comparison claim.

paper/sections/03_methodology.tex:42-44 states three things:
  * alpha/8 = 0.00625 -> 99.375% CIs
  * GLIP-T and Grounding DINO verdicts unchanged (8/8 significant)
  * OWLv2 drops from 3/8 to 1/8 (chair and book lose significance)

No single result file supports all three today:
  * archive_80query/v14_bonferroni_recalc.json holds the Bonferroni CIs, but its
    OWLv2 block is 80-query and must not be used.
  * results/m6_40class/owlv2_8class_bootstrap.json holds the single-query OWLv2
    Bonferroni CIs (ci_bonf_*), but only for OWLv2.

GLIP/GDino are identical under both protocols, so their block from the archived
file is valid; OWLv2 is taken from the single-query bootstrap. This script
asserts that the archived 95% CIs reproduce the canonical ones before copying.
"""
import json
import os

CLASSES = ["person", "car", "dog", "cat", "chair", "cup", "bottle", "book"]


def find(rel):
    for d in ("results", "results/archive_superseded", "results/archive_80query"):
        p = os.path.join(d, rel)
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError(rel)


bonf = find("v14_bonferroni_recalc.json")
canon = json.load(open("results/CANONICAL_8CLASS.json"))["models"]
owl = find("m6_40class/owlv2_8class_bootstrap.json")
owl = owl.get("results", owl)

models = {}
for m in ("GLIP", "GDino"):
    models[m] = {}
    for c in CLASSES:
        b = bonf[m][c]
        ref = canon[m][c]
        # The archived file must describe the SAME bootstrap as CANONICAL,
        # otherwise its GLIP/GDino block is not reusable either.
        assert abs(b["ci95_lo"] - ref["ci_low"]) < 1e-9, f"{m}/{c} ci95_lo mismatch"
        assert abs(b["ci95_hi"] - ref["ci_high"]) < 1e-9, f"{m}/{c} ci95_hi mismatch"
        models[m][c] = {
            "ci95_low": b["ci95_lo"],
            "ci95_high": b["ci95_hi"],
            "sig95": b["sig95"],
            "ci_bonf_low": b["ci_bonf_lo"],
            "ci_bonf_high": b["ci_bonf_hi"],
            "sig_bonf": b["sig_bonf"],
        }

models["OWLv2"] = {}
for c in CLASSES:
    o = owl[c]
    models["OWLv2"][c] = {
        "ci95_low": o["ci_low"],
        "ci95_high": o["ci_high"],
        "sig95": not o["ci_contains_zero"],
        "ci_bonf_low": o["ci_bonf_low"],
        "ci_bonf_high": o["ci_bonf_high"],
        "sig_bonf": not o["ci_bonf_contains_zero"],
    }

summary = {
    m: {
        "n_sig95": sum(1 for v in models[m].values() if v["sig95"]),
        "n_sig_bonf": sum(1 for v in models[m].values() if v["sig_bonf"]),
        "lose_significance": [c for c, v in models[m].items()
                              if v["sig95"] and not v["sig_bonf"]],
    }
    for m in models
}

out = {
    "schema": "bonferroni_8class_v1",
    "description": (
        "Bonferroni-corrected (alpha/8 = 0.00625, 99.375%) confidence intervals "
        "for the 8-class comparison, used by the multiple-comparison claim in "
        "03_methodology.tex."
    ),
    "sources": {
        "GLIP": "results/archive_80query/v14_bonferroni_recalc.json "
                "(GLIP/GDino blocks only; asserted identical to CANONICAL)",
        "GDino": "results/archive_80query/v14_bonferroni_recalc.json "
                 "(GLIP/GDino blocks only; asserted identical to CANONICAL)",
        "OWLv2": "results/m6_40class/owlv2_8class_bootstrap.json (single-query)",
    },
    "warning": (
        "The OWLv2 block inside archive_80query/v14_bonferroni_recalc.json is the "
        "80-query run and is NOT valid. Only the OWLv2 values in THIS file "
        "(single-query) may be compared against the paper."
    ),
    "correction": {"family_size": 8, "alpha": 0.00625, "ci_level": 0.99375},
    "summary": summary,
    "models": models,
}

with open("results/BONFERRONI_8CLASS.json", "w") as f:
    json.dump(out, f, indent=2)

for m, s in summary.items():
    print(f"{m:6s} 95%: {s['n_sig95']}/8   Bonferroni: {s['n_sig_bonf']}/8   "
          f"lost: {s['lose_significance']}")
print("\nwrote results/BONFERRONI_8CLASS.json")
