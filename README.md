# GSplusOpt

This repo contains the implementation of the algorithms shown in the paper: [Combining global search inversion and optimization to estimate 1D electrical conductivity using frequency domain electromagnetic induction measurements].

The repository includes:
- Python 3 codes for creating the synthetic data
- Python 3 codes for implementing the inversion algorithms
- Python 3 codes for visualizing the results

 ## Getting Started

### Requirements
- Python 3
- Other libraries such as `empymod` and `pygimli` listed in `requirements.txt`

### Installation
1. Clone the repository
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run the codes by numeric order:
   - 1.
```
## Notes
- Data is simulated for an EMI device with the following characteristics
  - Frequency: 9000 Hz
  - Geometries: Horizontal coplanar (H) with offsets [2 m, 4 m, 8 m],
    Vertical coplanar (V) with offsets [2 m, 4 m, 8 m],
    and Perpendicular (P) with offsets [2.1 m, 4.1 m, 8.1 m]

## Data
- Field data acquired with a DUALEM842s EMI instrument is stored in file `FieldCase/data/Field_data.npy`
```

### Citation
Forward modelling using `empymod` from: [Werthmüller (2017)](<https://doi.org/10.1190/geo2016-0626.1>).
Gradient based inversion using `pygimli` from: [Rücker et. al. (2017)](<http://dx.doi.org/10.1016/j.cageo.2017.07.011>)
