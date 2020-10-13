# DataCentricCrystalClassifier (DC)

## Setup

### Requirements

Tested with python3.7, ovito scripting package (called ovitos) -- future tasks include extending to open source packages

### Installation

`pip install -U .` or `ovitos -m pip install -U .`

### Ovito modifier

```
from dc3.model.modifier import DC3Modifier
# add modifier to pipeline, eg...
ovitos_pipeline.modifiers.append(DC3Modifier)
```

## Components

*(file structure liable to change)*

`dc3/`: crystal classifier package

- `util/`: contains featurizer, constants, etc
- `data/`: handle synthetic data generation, file i/o
- `model/`: SVM model and data scaler handling
    - `modifier.py`: source code to implement ovito modifier 
- `eval/`: scripts to help evaluate different pipeline versions

`examples/`: features example usages of dc3

