# Model Editing Results
This repository holds the results from the paper ... together with some basic scripts for analysis. The paper and the results contained here were created with the [model editing benchmark](https://github.com/oneSebastian/model-editing/).

## Reproduction
The `experiments` directory holds the full evaluation results from the paper.

Run `paper_figures.py` to recreate most of the tables and figures. Note that the `model_editing` package from the [model editing benchmark](https://github.com/oneSebastian/model-editing/) needs to be installed in your environment.

## Manual Rating
In one experiment we had humand rate the answers given my edited language models. The code used to create the tables for rating as well as their analysis and the full results are contained under the `generate_length` directory.

Run `generate_length/n_grams.py` and `generate_length/analyse_results.py` to recreate the paper figures.