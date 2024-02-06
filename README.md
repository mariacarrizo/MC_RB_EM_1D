# GSplusOpt

This repo contains the implementation of the algorithms shown in: [Combining global search inversion and optimization to estimate 1D electrical conductivity using frequency domain electromagnetic induction measurements].

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
2. Install `pygimli` in a separate conda environment
Open a terminal (Linux & Mac) or the Anaconda Prompt (Windows) and type
```
conda create -n pg -c gimli -c conda-forge pygimli=1.4.6
```
3. Activate your environment
If you are using Windows or Mac, a new environment named “pg” should be visible in the Anaconda Navigator. If you want to use pyGIMLi from the command line, you have to activate the environment. 
```
conda activate pg
```
For more information about pygimli instalation go to <https://www.pygimli.org/installation.html>

4. Install dependencies
```
pip install -r requirements.txt
```

5. Run the codes by numeric order in each folder:
   Example: Folder Synth-2Layers:
    - 1_LUTable2Lay.py
    - 2_CreateSyntheticData_A1.py
    - 3_GlobalSearch_A1.py
    - ...
    
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

### References
Forward modelling using `empymod` from: [Werthmüller (2017)](<https://doi.org/10.1190/geo2016-0626.1>).

Gradient based inversion using `pygimli` from: [Rücker et. al. (2017)](<http://dx.doi.org/10.1016/j.cageo.2017.07.011>)
