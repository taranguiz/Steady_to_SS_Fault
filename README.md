# Strike-Slip Fault Landscape Evolution Model

This repository contains the code for simulating landscape evolution under strike-slip faulting conditions, as described in the manuscript submitted to Geophysical Research Letters (GRL).

Steady State Topography data is available in https://doi.org/10.5281/zenodo.15870957

## Overview

This model simulates the transition from steady-state topography to strike-slip faulting, incorporating:
- **Hillslope processes**: Depth-dependent Taylor diffusion and Exponential weathering.
- **Fluvial processes**: SPACE Large Scale Eroder for sediment transport. 
- **Tectonic deformation**: Right-lateral strike-slip faulting with slip events that can move a minimum of 1 pixel. 
- **Climate variability**: Episodic fluvial periods with varying frequency and duration

## Repository Contents

### Core Simulation Files

- `updated_steady.py` - Steady-state topography generation
- `geomorph_dynamics_loop_trying_something.py` - Main simulation loop for experimental analysis
- `ss_fault_function.py` - Strike-slip fault numerical implementation
- `model_config.py` - Configuration management class
- `util.py` - Utility functions for file handling and grid state management

### Configuration
- `parameters_trying_something.yaml` - Model parameters configuration file

  ## Installation and Setup

### Prerequisites
- Python 3.7+
- Landlab (geomorphology modeling framework)
- NumPy, Matplotlib, PyYAML
- Additional dependencies (see requirements.txt)

### Installation
1. Clone or download this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Landlab is properly installed (see [Landlab documentation](https://landlab.readthedocs.io/))

If you do not want to create a new synthetic topography you can use the following for replication: 
Aranguiz-Rago, T. (2025). Steady State Topography used in models for "Climate oscillation and fault slip rate control sediment aggradation and channel morphology along strike-slip faults" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15870957
