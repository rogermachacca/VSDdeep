# AI-based models for volcano-seismic event detection and classification (VSDdeep and VSCdeep)


### Tested in Python 3.7.10 
Install using Anaconda from:

-----------------
## Installation

**EQTransformer** supports a variety of platforms, including macOS, Windows, and Linux operating systems. Note that you will need to have Python 3.x (3.6 or 3.7) installed. The **EQTransformer** Python package can be installed using the following options:

#### Via Anaconda (recommended):

    conda env create -f env_VSDCdeep.yml

    conda activate VSDCdeep 
    
-------------
## Tutorials

See either:
-------------------
## A Quick Example

```python

    from EQTransformer.core.mseed_predictor import mseed_predictor
    
    mseed_predictor(input_dir='downloads_mseeds',   
                    input_model='ModelsAndSampleData/EqT_model.h5',
                    stations_json='station_list.json',
                    output_dir='detection_results',
                    detection_threshold=0.2,                
                    P_threshold=0.1,
                    S_threshold=0.1, 
                    number_of_plots=10,
                    plot_mode='time_frequency',
                    batch_size=500,
                    overlap=0.3)
```

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
