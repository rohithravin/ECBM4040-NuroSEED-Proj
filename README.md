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