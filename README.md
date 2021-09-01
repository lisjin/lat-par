# Latent Tree Decomposition Parsers for AMR-to-Text Generation
Implementation of the corresponding [paper](https://arxiv.org/abs/2108.12304).

**Credit:** This code is based on the [repo](https://github.com/jcyk/gtos) for the 2020 AAAI paper "Graph Transformer for Graph-to-Sequence Learning". We are grateful to the authors for open-sourcing their work.

## Environment Setup
The code is tested with Python 3.6. All dependencies are listed in [requirements.txt](requirements.txt).

## Data Preparation
The instructions to prepare AMR data are given in the [generator_data](./generator_data) folder.

## Model Training and Evaluation
The following steps should be done in the `generator` folder. The default settings in this repo should reproduce the results in our paper. Please check all scripts for correct arguments before use.

1. Preprocess data and train
    ```
    sh prepare.sh  # vocab and data preprocessing
    sh train.sh
    ```
2. Test and postprocess
    ```
    sh work.sh  # test
    sh test.sh  # postprocess (make sure --output is set)
    ```
3. Evaluate
    ```
    ./multi-bleu.perl  # BLEU eval
    python chrF++.py -H [hyp] -R [ref]  # chrF++ eval
    java -Xmx2G -jar meteor-1.5.jar [hyp] [ref] -l en  # Meteor eval
    ```
