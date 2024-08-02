# AI-based models for volcano-seismic event detection and classification (VSDdeep and VSCdeep)


### Tested in Python 3.7.10 
Install using Anaconda from:



### You must install the following dependencies
pip install keras==2.3.1 <br>
pip install tensorflow==2.0.0 <br>
pip install h5py==2.10.0 <br>
pip install obspy==1.3.1 <br>
pip install pandas==1.0.1 <br>
pip install --upgrade protobuf==3.20.0 <br>

-----------------
## Installation

**EQTransformer** supports a variety of platforms, including macOS, Windows, and Linux operating systems. Note that you will need to have Python 3.x (3.6 or 3.7) installed. The **EQTransformer** Python package can be installed using the following options:

#### Via Anaconda (recommended):

    conda create -n eqt python=3.7

    conda activate eqt

    conda install -c smousavi05 eqtransformer 
    
##### Note: sometimes you need to keep repeating executing the last line multiple time to succeed.  

#### Via PyPI:

If you already have `Obspy` installed on your machine, you can get **EQTransformer** through PyPI:

    pip install EQTransformer


#### From source:

The sources for **EQTransformer** can be downloaded from the `GitHub repo`.

##### Note: the GitHub version has been modified for Tensorflow 2.5.0

You can either clone the public repository:

    git clone git://github.com/smousavi05/EQTransformer
    
or (if you are working on Colab)

    pip install git+https://github.com/smousavi05/EQTransformer

Once you have a copy of the source, you can cd to **EQTransformer** directory and install it with:

    python setup.py install


If you have installed **EQTransformer** Python package before and want to upgrade to the latest version, you can use the following command:

    pip install EQTransformer -U
    
##### To install EqT on M1 laptop with python>=3.9 from the source (GitHub) code by changing tensorflow to tensorflow-maco in the setup.py and follow these steps:

      conda create -n eqt python=3.10
      conda activate eqt
      conda install -c apple tensorflow-deps
      conda install obspy jupyter pandas
      pip install tensorflow-macos
      python3 setup.py install

-------------
## Tutorials

See either:
