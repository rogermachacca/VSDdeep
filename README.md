# VSDdeep: A deep learning model for Volcano-Seismic event Detection


VSDdeep is a modified version of the EQTransformer architecture tailored for volcano-seismic event detection. The decoder components related to P- and S-phase picking in the original model have been removed. This modified model was retrained using the VSED dataset, randomly split into training (80%), validation (10%), and test (10%) datasets.

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

## Pre-trained model
Located in directory: model/VSDdeep_PF.h5

-------------

## A Quick Example


    python Masive_detection_and_classification_PE.py


-------------
## Reference

Machacca, R., Lesage, P., & Tavera, H. (2024). Deep learning and machine learning applied to the detection and classification of volcano-seismic events at Piton de la Fournaise volcano. Manuscript under review.
