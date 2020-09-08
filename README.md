# DataCentricCrystalClassifier (DC<sup>3</sup>)

## Setup

### Requirements

TODO

### Installation

TODO

## Usage

Download model from this [link](https://drive.google.com/file/d/1oV_Gg2b6iihfbLLOg-ShmwYsZue_A3xi/view?usp=sharing) and unzip

This repo directory (`DataCentricCrystalClassification/`) should now include following files and dirs:
`dc3/`, `default_pipeline.zip`, `README.md`, `config/`, `default_pipeline/`, `examples/`

### Ovito modifier

```
from dc3.model.modifier import DC3Modifier
# add modifier to pipeline
```

## Components

*(file structure liable to change)*

`dc3/`: crystal classifier package

- `util/`: self-explanatory
    - `features.py`, `constants.py`
- `data/`: handle synthetic data generation, file i/o, downloading training data
- `model/`: SVM model and data scaler handling
    - `modifier.py`: source code to implement ovito modifier 
- `train/`: scripts to help train for new classes (defaults are: BCC, FCC, HCP, HD, CD, SC)

`examples/`: will feature example usages of dc3

