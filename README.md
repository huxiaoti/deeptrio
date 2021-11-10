## DeepTrio: a ternary prediction system for protein–protein interaction using mask multiple parallel convolutional neural networks

![](https://img.shields.io/badge/language-python3.7-brightgreen.svg) &#160;&#160;&#160;![](https://img.shields.io/badge/backend-tensorflow2.1-orange.svg)&#160;&#160;&#160; ![](https://img.shields.io/badge/environment-conda-blue.svg)&#160;&#160;&#160; ![](https://img.shields.io/badge/verison-1.1.0-ff69b4.svg)



<!-- ![alt text](http://bis.zju.edu.cn/deeptrio/structure.svg) -->

<!-- # Motivation
Protein-protein interaction (PPI), as a relative property, depends on two binding proteins of it, which brings a great challenge to design an expert model with unbiased learning and superior generalization performance. Additionally, few efforts have been made to grant models discriminative insights on relative properties. -->
### Installation

It is recommended to install dependencies in **conda** virtual environment so that only few installation commands are required for running DeepTrio. 
You can prepare all the dependencies just by the following commands.

- Install Miniconda

    > Miniconda is a free minimal installer for conda. It is a small, bootstrap version of Anaconda that includes only conda, Python, the packages they depend on, and a small number of other useful packages, including pip, zlib and a few others

    1. Download Miniconda installer for linux : https://docs.conda.io/en/latest/miniconda.html#linux-installers
    2. Check the hashes for the Miniconda from : https://docs.conda.io/en/latest/miniconda_hashes.html
    3. Go to the installation directory and run command : `bash Miniconda3-latest-Linux-x86_64.sh`

- Creating the environment

    If there is no environment in your Miniconda environment, it is recommeneded to create a new environment to run DeepTrio.

    1. Run `conda create -n [your env name] python=3.7`
    2. Run `conda activate [your env name]`
    3. Run `conda install tensorflow-gpu==2.1`
    4. Run `conda install seaborn`
    5. Run `conda install -c conda-forge scikit-learn`
    6. Run `conda install -c conda-forge gpyopt`
    7. Run `conda install -c conda-forge dotmap`
### Run DeepTrio for Training

1. To run DeepTrio on your own training data you need to prepare the following two things:

    * Protein-protein Interaction File: A pure protein ID file, in which two protein IDs are separated by the **Tab** key, alonge with their label (1 for 'interacting', 0 for 'non-interacting' and 2 for 'single protein'). This file must be named as [(your customized name).pair.tsv]. For example:

      ```txt
      line1:    protein_id_1  [Tab]  protein_id_2  [Tab]  label
      line2:    protein_id_3  [Tab]  protein_id_4  [Tab]  label
      ```

    * Protein Sequence Database File: A file containing protein IDs and their sequences in fasta format, which are separated by the **Tab** key. This file must be named as [(your customized name).seq.tsv]. For example:
     
      ```txt
      line1:    protein_id_1  [Tab]  protein_1_sequence  
      line2:    protein_id_3  [Tab]  protein_2_sequence
      ```
2. Execute command with arguments in shell:

    ```shell
    python model.py [-h] [--dataset DATASET] [--fold_index FOLD_INDEX]
                         [--epoch EPOCH] [--seed SEED] [--outer_product OUTER_PRODUCT]
    ```
    **Arguments:**

    |Argument|Required|Default|Description|
    | ----  | ----  |  ----  |----  |
    | --dataset | Yes || The customized name of your dataset|
    | --fold_index | Yes |0| The fold index in 5-fold cross-validation|
    | --seed | No |555| The seed of the model|
    | --outer_product | No |False| Whether apply max-pooling on outer-product of two proteins|
    | --epoch | No |100| The maximum number of epochs|
    | --help | No || Help message|


### Run DeepTrio for Prediction
1. To run DeepSol for prediction on your own query protein pairs you need to prepare the following three things:

    * The first protein File: It can contain multiple proteins in fasta format. For example:

      ```txt
      line1:    >protein_id_1
      line2:    protein_1_sequence
      line3:    >protein_id_2
      line4:    protein_2_sequence
      ```

    * The second protein File: It can contain multiple proteins in fasta format. For example:
     
      ```txt
      line1:    >protein_id_3
      line2:    protein_3_sequence
      ```
    
    * The model file name and its path. 

    * The inputs of DeepTrio will be:

      ```txt
      the first query protein pair:   protein_1 and protein_3
      the second query protein pair:  protein_2 and protein_3
      ```

2. Execute command with arguments in shell:

    ```shell
    python main.py [-h] -p1 PROTEIN1 -p2 PROTEIN2 -m MODEL [-o OUTPUT]
    ```
    **Arguments:**

    |Abbreviation|Argument|Required|Description|
    |  ----   | ----  | ----  |----  |
    | -p1  | --protein1 | Yes | The first protein group in fasta format with its path|
    | -p2  | --protein2 | Yes | The second protein group in fasta format with its path|
    | -m | --model | Yes | The DeepTrio model with its path|
    | -o | --output | No | The output file name|
    | -h  | --help | No | Help message|

### Run DeepTrio for Visualization
1. To run DeepSol for visualization on your own query protein pairs you need to prepare the following three things:

    * The first protein File: which must contain only one protein in fasta format. For example:

      ```txt
      line1:    >protein_id_1
      line2:    protein_1_sequence
      ```

    * The second protein File: which must contain only one protein, like `the first protein File`. 

    * The model file name and its path.

2. Execute command with arguments in shell:

    ```shell
    python visual_DeepTrio.py [-h] -p1 PROTEIN1 -p2 PROTEIN2 -m MODEL
    ```
    **Arguments:**

    |Abbreviation|Argument|Required|Description|
    |  ----   | ----  | ----  |----  |
    | -p1  | --protein1 | Yes | The first protein group in fasta format with its path|
    | -p2  | --protein2 | Yes | The second protein group in fasta format with its path|
    | -m | --model | Yes | The DeepTrio model with its path|
    | -h  | --help | No | Help message|

# FAQ

#### 1. Can I use pip to install the environment dependencies?

A) Yes, you need to install some addtional libraries, like GPU drivers, matplotlib, numpy, Gpy and so on, so we recommend to use conda to install dependencies.

#### 2. Can DeepTrio run on Windows?

A) Yes, you can configure conda virtual environment on your Windows PC.

#### 3. If I am not good at using Unix software, is there any conventient way to use DeepTrio?

A) Yes, you can visit our online website : http://bis.zju.edu.cn/deeptrio, where you can predicti PPIs and draw importance maps on the DeepTrio model without any configurations.

# Citation

If you find DeepTrio useful, please consider citing our publication:<br />
Hu, X., Feng, C., Zhou, Y., Harrison, A., & Chen, M. (2021). DeepTrio: a ternary prediction system for protein-protein interaction using mask multiple parallel convolutional neural networks. <i>Bioinformatics</i>.

# Updates

2021-11-09: Revision for article: the reference in the footnote of Table 4 should be Chen <i>et al</i>. (2019).<br />
2021-09-03: v1.0 - v1.1: adding an alternative function for applying max-pooling on the outer-product of two protein feature maps.
