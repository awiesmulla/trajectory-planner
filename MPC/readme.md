# Model Predictive Control (MPC)

## Installation

All the python requirements are present in `requirements.txt`
```
pip install -r requirements.txt
```

## Usage

### `main.py`
#### Running with given paths
---
First, change the parameters in the `config.toml` file according to the requirement. Then run `main.py`.
```
python main.py
```

#### Using a custom path
---
To do this, add a function to the dict `ref_courses`  that returns your custom path. This path can then be selected by specifying the corresponding number in the `config.toml` .
That would be the recommended way as the user will have an option to quickly switch between paths by just changing the config.  
Currently we are calculating the custom path using the A* algorithm from `path_helper.py` on predefined maps present in maps directory.

### `mpc.py`
This module has 2 classes.
State: holds information about the robot state
MPC: holds the MPC specific information, non-linear solver, cost functions for unicycle and bicycle model and some other helper functions

Note: This file cannot be run as is. It must be imported into another file that calls the solver. Check `main.py` file for reference usage
