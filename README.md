# Multiple Hypothesis Tracker with Joint Probabilistic Data Association (JPDA)

This repository contains a lightweight implementation of a Multiple Hypothesis Tracker (MHT) using Joint Probabilistic Data Association (JPDA). This code demonstrates a basic implementation of MHT-JPDA for multi-target tracking. The repository is intended for fast prototyping and educational purposes, not for production use.

## Implementation Details

1. **Track Representation**: Tracks are represented by the `Track` class, containing state (position and velocity), covariance, and history.

2. **Prediction**: Tracks are predicted using a constant velocity model with the `predict_track` function.

3. **Data Association**: JPDA is implemented in the `calculate_association_probabilities` function, which calculates association probabilities between observations and tracks.

4. **Update**: Tracks are updated using the JPDA algorithm in the `update_tracks_jpda` function, considering all possible associations weighted by their probabilities.

5. **Simulation**: The `Simulation` class generates observations with a 90% detection rate and Poisson-distributed false alarms.

6. **Visualization**: Real-time visualization is provided using PyQtGraph in the `VisualizationWidget` class.

## Key Components

- Track and Observation classes
- Prediction and update functions
- JPDA implementation
- Simulation class for generating scenarios
- Visualization widget for real-time display

## Dependencies

- NumPy
- SciPy
- PyQtGraph

## Usage

Run the `mht_jpda.py` script to start the simulation and visualization:
