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

## GUI

A Streamlit GUI is available for interactive registration, segmentation, mask
correction, and results inspection:

```shell
python -m astroglial_morphology.gui
# or, if console scripts are installed:
astroglial-morphology-gui
```

The GUI wraps the same Hydra pipeline. Every "Run" button spawns
`python -m astroglial_morphology` in a child process and streams its log into
the app so the browser stays responsive. Pages:

- **Home** – validate a data folder and see what artifacts already exist.
- **Registration** – Suite2p parameter form with common and advanced knobs;
  QC charts (`xoff`/`yoff`/`corrXY`, badframes, mean/reference images) drawn
  from `ops.npy` after each run.
- **Segmentation** – Cellpose parameters, overlay preview of the resulting
  masks.
- **Mask correction** – in-browser canvas editor for the Cellpose `*_seg.npy`
  file. Tools: select, brush, erase, split, pan. Saves back to the same file
  and leaves a `*.orig` backup the first time.
- **Inspect** – projections, mask overlays, and QC for any completed run,
  without launching anything.
- **Metadata** – groups the values recorded in `pipeline_metadata.json` and
  highlights the parameters that differ from the package defaults.

Classification and correspondence export are not exposed in the GUI; use the
CLI when you need them.

## Usage
The single-model workflow is the default. It accepts raw `.tif` or `.lif` data
and runs Suite2p registration before segmentation. The command line is managed
by Hydra: set values with `key=value`, inspect all available settings with
`--help`, and print the resolved settings without running the pipeline with
`--cfg job`.

```shell
python -m astroglial_morphology input.data_path=/path/to/data
```

Each invocation saves its resolved YAML configuration and log under
`outputs/<date>/<time>/`; pipeline results still stay beside the supplied input
data. Hydra does not change the working directory. For example, enable the GPU
and save registered TIFFs with:

```shell
python -m astroglial_morphology input.data_path=/path/to/data runtime.use_gpu=true registration.reg_tif=true
```

Hydra's standard `--multirun` syntax is also available for sequential parameter
sweeps; use distinct input/output data locations for runs that would otherwise
write the same pipeline artifacts.

### Three-model Cellpose ensemble

Use the opt-in ensemble when you want separate complete-cell, cell-body, and
process predictions merged into one canonical Cellpose output:

```shell
python -m astroglial_morphology input.data_path=/path/to/data segmentation=ensemble
```

The first ensemble run downloads the versioned CP3 model assets into the local
model cache and verifies their SHA-256 checksums. Download them in advance or
choose a shared cache location with:

```shell
python -m astroglial_morphology action=prefetch_models segmentation=ensemble segmentation.model_cache_dir=/path/to/cache
```

The default profile is `cp3-three-part`: complete-cell and processes use an
effective diameter of `7.89 µm`; cell body uses `6.0 µm`. For each role,
Cellpose receives either `effective_diameter_um × pixels_per_micron`, or the
diameter derived from `projected_area_um2`:

```text
diameter_px = 2 × sqrt(projected_area_um2 / π) × pixels_per_micron
```

Pass a complete custom role profile with
`segmentation.ensemble_config=/path/to/profile.json` to change model sources,
Cellpose parameters, physical diameter/area, or merge thresholds. The resulting
files are saved beside the projection as
`*_complete_cell_seg.npy`, `*_processes_seg.npy`, `*_cell_body_seg.npy`, and
the combined canonical `*_seg.npy`. `pipeline_metadata.json` records the
role-specific target/pixel diameters and min/max/median mask sizes.

### Existing Suite2p `plane0` input

You can give the pipeline a registered Suite2p `plane0` folder directly; it
must contain `ops.npy` and `data.bin` (plus `data_chan2.bin` for two channels).
The pipeline skips raw conversion and registration, creates projections from
the existing binaries, and writes outputs in that same folder:

```shell
python -m astroglial_morphology input.data_path=/path/to/suite2p/plane0 \
  segmentation=ensemble input.pixels_per_micron=3.168
```

For direct Suite2p input, calibration is resolved from `input.pixels_per_micron`
first, then `pixels_per_micron` or legacy `pixel_resolution` in the nearest
`pipeline_metadata.json`. Ensemble mode requires this calibration; single mode
can fall back to Cellpose's learned diameter for segmentation when it is absent.
Morphology classification and correspondence export still require calibration,
because they convert physical distances to pixels.


The package will do motion correction using suite2p and outputs the following projection images from the motion corrected data:

- mean image
- max projection image
- std deviation image
- sum image

### Exporting Correspondence & Trace Data

Correspondence matrix and trace exports are produced by default. Use
`correspondence.enabled=false` when you want to skip them. Optional knobs let
you choose the sub-segmentation length and the x-axis grouping distance used
during alignment:

```shell
python -m astroglial_morphology input.data_path=/path/to/data \
  correspondence.segment_length=10 correspondence.delta_x=20
```

This command creates `subsegmented_masks_seg.npy`, `correspondence_matrix.(npy|mat)`, and `trace_matrix.(npy|mat)` inside your data directory while also extracting Suite2p traces for the new mask set. The `cellpose_suite2p_output` folder is directly loadable in the Suite2p GUI: it contains `stat.npy`, `ops.npy`, `iscell.npy`, `F.npy`, `Fneu.npy`, and `spks.npy`. For two-channel data, the first `correspondence.trace_channels` selection is the GUI primary trace; the other channel is retained as `F_chan2.npy` and `Fneu_chan2.npy`.

Use `correspondence.subsegmentation_mode=compartments` to split each aligned process into four biologically-inspired regions (soma, middle-near-soma, middle-near-distal, distal). In this mode every subsegment receives a class label 1–4 in addition to the original upper/lower class, while the default `equal_length` mode keeps the previous fixed-pixel segmentation controlled by `correspondence.segment_length`.
