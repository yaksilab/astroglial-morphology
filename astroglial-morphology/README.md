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
The legacy single-model workflow remains the default. It accepts raw `.tif` or
`.lif` data and runs Suite2p registration before segmentation.
```shell
python -m astroglial_morphology <path to your raw data in .tif format>
```

### Three-model Cellpose ensemble

Use the opt-in ensemble when you want separate complete-cell, cell-body, and
process predictions merged into one canonical Cellpose output:

```shell
python -m astroglial_morphology <data-dir> --segmentation-mode ensemble
```

The first ensemble run downloads the versioned CP3 model assets into the local
model cache and verifies their SHA-256 checksums. Download them in advance or
choose a shared cache location with:

```shell
python -m astroglial_morphology --download-models --model-cache-dir <cache-dir>
```

The default profile is `cp3-three-part`: complete-cell and processes use an
effective diameter of `7.89 µm`; cell body uses `6.0 µm`. For each role,
Cellpose receives either `effective_diameter_um × pixels_per_micron`, or the
diameter derived from `projected_area_um2`:

```text
diameter_px = 2 × sqrt(projected_area_um2 / π) × pixels_per_micron
```

Pass a complete custom role profile with `--ensemble-config profile.json` to
change model sources, Cellpose parameters, physical diameter/area, or merge
thresholds. The resulting files are saved beside the projection as
`*_complete_cell_seg.npy`, `*_processes_seg.npy`, `*_cell_body_seg.npy`, and
the combined canonical `*_seg.npy`. `pipeline_metadata.json` records the
role-specific target/pixel diameters and min/max/median mask sizes.

### Existing Suite2p `plane0` input

You can give the pipeline a registered Suite2p `plane0` folder directly; it
must contain `ops.npy` and `data.bin` (plus `data_chan2.bin` for two channels).
The pipeline skips raw conversion and registration, creates projections from
the existing binaries, and writes outputs in that same folder:

```shell
python -m astroglial_morphology <path-to-suite2p/plane0> --segmentation-mode ensemble \
  --pixels-per-micron 3.168
```

For direct Suite2p input, calibration is resolved from `--pixels-per-micron`
first, then `pixels_per_micron` or legacy `pixel_resolution` in the nearest
`pipeline_metadata.json`. Ensemble mode requires this calibration; single mode
can fall back to Cellpose's learned diameter when it is absent.


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
