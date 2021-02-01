## DeepTrio: a ternary prediction system for protein–protein interaction using mask multiple parallel convolutional neural networks

# Motivation
Protein-protein interaction (PPI), as a relative property, depends on two binding proteins of it, which brings a great challenge to design an expert model with unbiased learning and superior generalization performance. Additionally, few efforts have been made to grant models discriminative insights on relative properties.

# Run DeepTrio for training

1. To run DeepSol on your own training data you need to prepare the following two things:

    * Protein-protein Interaction File: A pure protein ID file, in which two protein IDs are separated by the **Tab** key, alonge with their label (1 for 'interacting', 0 for 'non-interacting' and 2 for 'single protein'). For example:

      ```txt
      line1:    protein_id_1  [Tab]  protein_id_2  [Tab]  label
      line2:    protein_id_3  [Tab]  protein_id_4  [Tab]  label
      ```

    * Protein Sequence Database File: A file containing protein IDs and their sequences in fasta format, which are separated by the **Tab** key. For example:
   
      ```txt
      line1:    protein_id_1  [Tab]  protein_1_sequence  
      line2:    protein_id_3  [Tab]  protein_2_sequence
      ```
2. Execute command arguments with in shell:

    ```shell
    python build_model.py -p [ppi.tsv] -d [database.tsv] -e [the number of epochs] -h
    ```
    **Arguments:**

    |Abbr.|Arg.|Required|Description|
    |  ----   | ----  | ----  |----  |
    | -p  | --ppi | Yes | PPI file with it path|
    | -d  | --database | Yes | Database file with it path|
    | -e  | --epoch | No | The maximum number of epochs|
    | -h  | --help | No | Help message|

3. Select the best model according to **GpyOpt** log file:

    ```txt
    python build_model.py -p [ppi.tsv] -d [database.tsv] -e [the number of epochs] -h
    ```



# Run DeepTrio for prediction

# Installation

It is recommended to install dependencies in **conda** virtual environment so that only few installation commands are required for running DeepTrio. 
You can prepare all the dependencies just by the following commands.

  1. Install Miniconda

    > Miniconda is a free minimal installer for conda. It is a small, bootstrap version of Anaconda that includes only conda, Python, the packages they depend on, and a small number of other useful packages, including pip, zlib and a few others

    1. Download Miniconda installer for linux : https://docs.conda.io/en/latest/miniconda.html#linux-installers
    2. Check the hashes for the Miniconda from : https://docs.conda.io/en/latest/miniconda_hashes.html
    3. Go to the installation directory and run command : `bash Miniconda3-latest-Linux-x86_64.sh`

  2. Creating the environment

    If there is no environment in your Miniconda environment, it is recommeneded to create a new environment to run DeepTrio.

    - Run `conda create -n [your env name] python=3.7`
    2. Run `conda activate [your env name]`
    3. Run `conda install tensorflow-gpu==2.1`
    4. Run `conda install seaborn`
    5. Run `conda install -c conda-forge scikit-learn`
    6. Run `conda install -c conda-forge gpyopt`
    7. Run `conda install -c conda-forge dotmap`