# astroglial-morphology

## For installation and usage instructions, please refer to the [README.md](astroglial-morphology/README.md).


## TODOS


-[x] Registration of data & alignment
Motion correction

- Segmentation
Per-frame segmentation and projection segmentation (max of all frames) [partially done]

- Cell compartment and annotation
Divide each segmented cell into compartments: soma/proximal, middle, and distal.
    middle: again devide in three parts, near distal, near proximal, and middle.

- Morphology and branching quantification per frame and across all frames
Skeleton length (distance from the soma to the distal tip), branch count and density, volume/area fraction, connectivity

- Time series signal extraction and normalisation
Extract fluorescence traces for each ROI 
Compute  ΔF/F

- Event detection
Detect calcium transients based on a threshold (e.g., ΔF/F₀ ≥ 100%).
Compute onset time and rise time per ROI for each event.
Compare onset order between soma, intermediate, and distal compartments.

- Repeat all steps for both channels in dual channel data

