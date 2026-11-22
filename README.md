# Prompt-Form Sensitivity in Open-Vocabulary Detection

Code and curated result artifacts for the manuscript:

> **Prompt-Form Sensitivity in Open-Vocabulary Detection: The CLIP Template Is a Poor Default**
> Fadong Li, Bing Liu, Mingming Liu, Yong Zhou, Peng Liu, Ning Qiao
> Submitted to *Computer Vision and Image Understanding* (Elsevier), 2026.

This repository studies how the textual prompt form affects three frozen open-vocabulary detectors:

- GLIP-T
- Grounding DINO-tiny
- OWLv2-base-patch16-ensemble

The main experiment uses eight COCO classes on a 500-image subset of COCO val2017; the extension uses 41 classes. The primary metric is AP50.

## Main findings

- The CLIP-style template `a photo of a X` is the worst prompt form in 7/8 GLIP-T classes and 5/8 Grounding DINO classes.
- Mean AP50 degradation relative to the bare class name is 0.113 for GLIP-T and 0.491 for Grounding DINO.
- OWLv2 is largely insensitive to the tested prompt forms under the single-query protocol.
- On the held-out GLIP-T evaluation split, a per-image oracle improves AP50 by only 0.006 absolute (0.60 percentage points) over a global bare class name.

Absolute AP50 values are not compared across detector families because their inference and post-processing pipelines differ. The analysis focuses on within-model differences between prompt forms.

## Repository structure

```text
.
├── README.md
├── LICENSE                      # MIT for the authors' original code
├── THIRD_PARTY_NOTICES.md       # Third-party components and their terms
├── CITATION.cff
├── requirements.txt
├── code/                        # Experiment, evaluation, and aggregation scripts
├── results/                     # Curated derived results (small JSON/CSV)
│   └── m6_40class/              # 41-class summaries and OWLv2 raw detections
└── figures/                     # Manuscript figures (PDF) and Figure 1 source
```

## What is and is not included

Included:

- the analysis, bootstrap, aggregation, and figure scripts used for the paper;
- the derived result files that support the reported numbers;
- the manuscript figures;
- OWLv2 per-class raw detections for the single-query protocol (`results/m6_40class/owlv2_raw/`, about 28 MB).

Not included:

- COCO images and annotations;
- pretrained detector weights;
- third-party model source trees (including the external GLIP source tree);
- the very large GLIP and Grounding DINO raw detection caches (hundreds of MB each), which exceed normal GitHub file limits.

Because the large raw detection caches are not stored here, this repository supports inspection of all derived numbers and re-aggregation of the OWLv2 results, but a complete end-to-end rerun of GLIP and Grounding DINO requires re-running the inference scripts with the official models and then re-running the bootstrap.

## Setup

```bash
python -m pip install -r requirements.txt
```

Download COCO val2017 and its annotations from <https://cocodataset.org>, and obtain the pretrained models from their official sources:

- GLIP: official GLIP implementation and `glip_tiny_model_o365_goldg_cc_sbu.pth`
- Grounding DINO: `IDEA-Research/grounding-dino-tiny`
- OWLv2: `google/owlv2-base-patch16-ensemble`

Set paths either through environment variables or by placing files at the defaults used in the scripts:

```bash
export COCO_ANN_FILE=/path/to/instances_val2017.json
export COCO_IMG_DIR=/path/to/val2017
export GLIP_ROOT=/path/to/glip/source
export GLIP_WEIGHTS=/path/to/glip_tiny_model_o365_goldg_cc_sbu.pth
```

Default paths, if the environment variables are unset:

```text
data/annotations/instances_val2017.json
data/val2017
third_party/glip
weights/glip_tiny_model_o365_goldg_cc_sbu.pth
```

Run all commands from the repository root so that relative `results/` paths resolve correctly.

## Reproducing the reported analysis

### B=1000 paired bootstrap

```bash
python code/bootstrap_b1000.py
```

This uses:

- image-level resampling with replacement;
- 1,000 resamples;
- seed 42;
- weighted AP50 to handle duplicate image IDs;
- 95% percentile confidence intervals;
- Bonferroni-corrected 99.375% intervals for the eight-class family.

Output: `results/BOOTSTRAP_B1000_8CLASS.json`.

This script expects the GLIP and Grounding DINO raw detection caches at `results/raw/`. Those caches are not included; regenerate them with `code/m1_v2_8class.py` and `code/m1_gdino_v11.py`. The OWLv2 branch reads the included `results/m6_40class/owlv2_raw/` files.

### OWLv2 eight-class bootstrap

```bash
python code/owlv2_8class_bootstrap.py
```

This can be run with the included OWLv2 raw detections.

### Oracle table

```bash
python code/build_oracle_table.py
```

This reads `results/m1_v2_v7_analysis.json` and writes `results/ORACLE_TABLE.json`.

### Canonical table and Bonferroni table

```bash
python code/build_canonical_8class.py
python code/build_bonferroni_table.py
```

These scripts expect the archived B=200 bootstrap files that are not included in this repository; they are provided for provenance of how the canonical aggregates were constructed.

### Figure 2

```bash
python code/regen_fig2.py
```

This regenerates `figures/fig2_delta_bare_ci.pdf` from `results/BOOTSTRAP_B1000_8CLASS.json`.

## Key result files

| File | Contents |
|---|---|
| `results/CANONICAL_8CLASS.json` | Single source of truth for the main eight-class table |
| `results/BOOTSTRAP_B1000_8CLASS.json` | B=1000 bootstrap results and B=200 comparison metadata |
| `results/BONFERRONI_8CLASS.json` | Multiple-comparison correction results |
| `results/ORACLE_TABLE.json` | Oracle upper-bound aggregates |
| `results/PER_PROMPT_AP50.json` | Per-prompt AP50 values |
| `results/RUNNER_UP_MARGINS.json` | Template-versus-runner-up margins |
| `results/THRESHOLD_SENSITIVITY.json` | Mechanism-threshold sensitivity |
| `results/TOKEN_METRIC_ANNOTATION.json` | Token-metric definitions and annotations |
| `results/v14_step2_token_evidence.json` | Token-level evidence for Grounding DINO |
| `results/m6_40class/` | 41-class summaries, per-class results, and OWLv2 raw detections |

## Experimental scope

The reported study uses:

- COCO val2017;
- 500 sampled images, seed 42;
- eight classes in the main experiment and 41 classes in the extension;
- frozen pretrained models without fine-tuning;
- AP50 computed with COCO-style evaluation;
- no additional external NMS beyond each model's standard pipeline;
- a 100-detection-per-image cap and a 0.01 score filter for Grounding DINO and OWLv2, as described in the manuscript.

The conclusions are scoped to the tested checkpoints, prompt forms, class sets, dataset, and inference protocols.

## License

The authors' original code in this repository is released under the MIT License; see `LICENSE`.

Third-party components, datasets, pretrained models, and model outputs remain subject to their own terms. See `THIRD_PARTY_NOTICES.md`. The Elsevier CAS LaTeX files, if added to this repository, retain their original LPPL/Elsevier license notices.

## Citation

If you use this code or these results, please cite the manuscript:

```bibtex
@article{li2026promptforms,
  title   = {Prompt-Form Sensitivity in Open-Vocabulary Detection: The CLIP Template Is a Poor Default},
  author  = {Li, Fadong and Liu, Bing and Liu, Mingming and Zhou, Yong and Liu, Peng and Qiao, Ning},
  journal = {Computer Vision and Image Understanding},
  year    = {2026},
  note    = {Submitted}
}
```

## Contact

For questions about the code or results, please open an issue or contact the corresponding author listed in the manuscript.
