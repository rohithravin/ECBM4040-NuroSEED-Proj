# ECBM4040-NuroSEED-Proj


# Preliminaries:

## P3-scripts for dataset generation
```bash
p3-all-genomes --eq reference_genome,Representative | p3-get-genome-features --eq "product,Phenylalanyl-tRNA synthetase alpha chain" --attr "patric_id,aa_sequence,na_sequence" > phes.tbl
p3-all-genomes --eq reference_genome,Representative | p3-get-genome-features --eq "product,Small Subunit Ribosomal RNA" --attr "patric_id,aa_sequence,na_sequence" > 16s.tbl
```

## Setting up environment
You can create a new environment using conda:
```bash
conda create -n neuroseed python tensorflow-gpu keras jupyter -y
conda activate neuroseed
pip install -r requirements.txt
```

## Data
The data we are using is too large to be stored in GH. Instead we have stored our data files in a public Dropbox folder. There is not need to manually grab the data as we pull the data ourselves using the `wget` commannd in all our notebooks a scripts when we run our experiments.

You can view the the Dropbox folder [here](https://www.dropbox.com/sh/18imo1x0ojqukeh/AAADw9nKVc-NNpYzjJh9zqE_a?dl=0). 

## Notebooks
We have multiple notebooks, including previous versions of the notebooks. The `./archived_notebooks` contains the previous versions of our current notebooks. The only notebooks of concern that contain our most recent analysis and experiments are in the main directory `./`.

The `tutorial.ipynb` notebooks, is our tuturial notebooks and creates and trains on our basic model. This would be the notebook we encourage newcomers to start on! 