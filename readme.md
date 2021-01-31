### DeepTrio: a ternary prediction system for protein–protein interaction using mask multiple parallel convolutional neural networks

## Motivation
Protein-protein interaction (PPI), as a relative property, depends on two binding proteins of it, which brings a great challenge to design an expert model with unbiased learning and superior generalization performance. Additionally, few efforts have been made to grant models discriminative insights on relative properties.

# Installation

It's recommended to install dependencies in conda virtual environment so that only few installation commands are required for running DeepTrio. 
You can prepare all the dependencies just by the following commands.

  - Install Miniconda

    Miniconda is a free minimal installer for conda. It is a small, bootstrap version of Anaconda that includes only conda, Python, the packages they depend on, and a small number of other useful packages, including pip, zlib and a few others

    1. Download Miniconda installer for linux : https://docs.conda.io/en/latest/miniconda.html#linux-installers
    2. Check the hashes for the Miniconda from : https://docs.conda.io/en/latest/miniconda_hashes.html#miniconda-hash-information
    3. Go to the installation directory and run command : `bash Miniconda3-latest-Linux-x86_64.sh`
conda install tensorflow-gpu==2.1
conda install seaborn
conda install -c conda-forge scikit-learn
conda install -c conda-forge gpyopt
conda install -c conda-forge dotmap
