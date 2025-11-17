## Installation

First create aconda environment with python 3.10
```shell
conda create -n astroglial-morphology python=3.10
```


Activate the environment:

```shell
conda activate astroglial-morphology
```


Then install this package from github repository

```shell
pip install git+https://github.com/yaksilab/astroglial-morphology.git#subdirectory=astroglial-morphology
```

## Usage
To use this package, you need to have your raw data in .tif format.
```shell
python -m astroglial_morphology <path to your raw data in .tif format>
```


The package will do motion correction using suite2p and outputs the following projection images from the motion corrected data:

- mean image
- max projection image
- std deviation image
- sum image

### Exporting Correspondence & Trace Data

To reproduce the correspondence matrix and trace exports from the original `astroglialAnalysis` workflow, run the CLI with `--export-correspondence`. Optional knobs let you choose the sub-segmentation length and the x-axis grouping distance used during alignment:

```shell
python -m astroglial_morphology <data-dir> --export-correspondence --segment-length 10 --correspondence-delta-x 20
```

This command creates `subsegmented_masks_seg.npy`, `correspondence_matrix.(npy|mat)`, and `trace_matrix.(npy|mat)` inside your data directory while also extracting Suite2p traces for the new mask set.

Use `--subsegmentation-mode compartments` to split each aligned process into four biologically-inspired regions (soma, middle-near-soma, middle-near-distal, distal). In this mode every subsegment receives a class label 1–4 in addition to the original upper/lower class, while the default `equal_length` mode keeps the previous fixed-pixel segmentation that is controlled by `--segment-length`.
