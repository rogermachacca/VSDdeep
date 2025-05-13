# VSDdeep: A deep learning model for Volcano-Seismic event Detection


VSDdeep is a modified version of the EQTransformer architecture tailored for volcano-seismic event detection. The decoder components related to P- and S-phase picking in the original model have been removed. This revised model was retrained using the VSED dataset, randomly split into training (80%), validation (10%), and test (10%) datasets.

-----------------
## Installation

**VSDdeep** supports a variety of platforms, including macOS, Windows, and Linux operating systems. Note that you will need to have Python 3.x (3.6 or 3.7) installed. The **VSDdeep** Python package can be installed using the following options:

#### Via Anaconda (recommended):

1. **Edit the Environment File**

   Open `env_VSDdeep.yml` and update the `prefix` field to match the desired path for the conda environment on your local machine. Alternatively, you can remove the `prefix` line and use the default location.

2. **Create the Conda Environment**

   ```bash
   conda env create -f env_VSDdeep.yml
   
   conda activate VSDCdeep 
    
-------------

# VSDdeep

**VSDdeep** is a deep learning model for volcano-seismic event detection, based on a modified version of the [EQTransformer](https://github.com/smousavi05/EQTransformer) architecture. The decoder components related to P- and S-phase picking in the original model have been removed, tailoring the model specifically for detection tasks.

This revised model was retrained using the VSED dataset, which was randomly divided into training (80%), validation (10%), and testing (10%) sets.

---

## 📁 Project Structure

- `src/` – Source code for model training, evaluation, and inference  
- `notebooks/` – Example notebooks for data exploration and usage  
- `env_VSDdeep.yml` – Conda environment specification  
- `README.md` – Project documentation  
- `LICENSE` – License file  

---

## ⚙️ Environment Setup

## Tutorials

See either:
-------------------
## A Quick Example


    python Masive_detection_and_classification_PE.py


-------------
## Reference

Mousavi, S.M., Ellsworth, W.L., Zhu, W., Chuang, L, Y., and Beroza, G, C. Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nat Commun 11, 3952 (2020). https://doi.org/10.1038/s41467-020-17591-w

BibTeX:

    @article{mousavi2020earthquake,
        title={Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking},
        author={Mousavi, S Mostafa and Ellsworth, William L and Zhu, Weiqiang and Chuang, Lindsay Y and Beroza, Gregory C},
        journal={Nature Communications},
        volume={11},
        number={1},
        pages={1--12},
        year={2020},
        publisher={Nature Publishing Group}
    }
