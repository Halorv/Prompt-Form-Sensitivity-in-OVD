# Third-Party Notices

The MIT License in `LICENSE` applies only to the original code and documentation contributed by the authors of this repository.

The following components are not relicensed by this repository and remain subject to their own terms.

## COCO dataset

- COCO val2017 images and annotations are obtained from <https://cocodataset.org>.
- They are not redistributed in this repository.
- Users are responsible for complying with the COCO terms of use and the terms of the original images.

## Pretrained models

- GLIP: official implementation and checkpoint.
- Grounding DINO: `IDEA-Research/grounding-dino-tiny`.
- OWLv2: `google/owlv2-base-patch16-ensemble`.

Model weights are not redistributed in this repository. Users should obtain them from their official sources and comply with the corresponding model licenses.

## Model implementations and libraries

The scripts depend on third-party software including, but not limited to:

- PyTorch
- torchvision
- Hugging Face Transformers
- pycocotools
- NumPy
- Pillow
- Matplotlib

Each dependency remains subject to its own license.

## Elsevier CAS LaTeX files

If the manuscript source is added to this repository, the files `cas-sc.cls`, `cas-common.sty`, and `cas-model2-names.bst` retain their original LaTeX Project Public License (LPPL) and Elsevier notices. They are not relicensed under the MIT License.

## Model outputs and derived results

The JSON result files in `results/` were produced by running the third-party models listed above. The MIT License covers the authors' original selection, aggregation, and documentation of these results; it does not grant rights to the underlying models or datasets.
