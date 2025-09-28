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
