# DataCentricCrystalClassifier (DC<sup>3</sup>)

## Setup

### Requirements

TODO

### Installation

TODO

## Usage

TODO

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

