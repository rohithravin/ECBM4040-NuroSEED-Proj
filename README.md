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


## How to run run code
To run our code, you just need to run any of the notebooks located in the root directory of the this repository. All the python scripts are used (i.e. imported) in these notebooks.

## Notebooks
We have multiple notebooks, including previous versions of the notebooks. The `./archived_notebooks` contains the previous versions of our current notebooks. The only notebooks of concern that contain our most recent analysis and experiments are in the main directory `./`.


### Main Notebook
The `tutorial.ipynb` notebooks, is our tuturial notebooks and creates and trains on our basic model. This would be the notebook we encourage newcomers to start on! There is no special instructions needed to run the main notebook (`tutorial.ipynb`). Simply ensure you have all the libraries installed in the `requirements.txt` file and you can run the notebook. 

 
## Key Functions in Python Scripts

### ./archived_notebooks
This folder contains previous versions of our notebooks.

### ./hyper_ param _ tunning
This directory contains the results and logs of the trials that were executed in our Hyperparameter Search experiment.

### ./model
#### generator.py
This script contains the code that creates our data generator. It is used in all of our notebooks.

#### layer.py
This script contains two custom keras layers we have created for this project. One is the DistanceLayer, that calculates the distance between two embeddings. The OneHotEncoding Layer simply one-hots the incoming data. 

### model _ csm.py
This is where we create our two custom models. One is an embedding model, where we create three different archectures of the embedding model, which are used in  our experiments. 

The other model is our general model - SiameseModel. This is a custom keras model, where we define unique `call` `loss` etc. methods. 

### train_model.py 
This script is where we combine our two custom models, compile and train the model. This function is used as an abstraction method, where we just have to call this method in our notebooks to build and train the model. 

## ./results
This directory is where all of our results (pickle files) are stored

## ./utils

The files in this directory relate to gather data from our data sources and preprocessing our data. 


## Special Notes
While working on this project, we created a public Github repository. This is where we did our main work and pushed everything. You can also see that the only collaborators in that public repository are the members of this project, which are the same members (usernames) in the classroom github repository. Once our codebase and analysis was complete, we moved everything to the classroom github repository. If you would like to view our commits and backtrack our progress, please do so on our [public repository](https://github.com/rohithravin/ECBM4040-NuroSEED-Proj). 

# Organization of this directory

```
./
├── HS_test_hyperparam_tuning_CNN.ipynb
├── HS_test_hyperparam_tuning_LINEAR.ipynb
├── HS_test_hyperparam_tuning_MLP.ipynb
├── README.md
├── Train-test\ split\ analysis.ipynb
├── __pycache__
│   └── train_model.cpython-38.pyc
├── archived_notebooks
│   ├── HS_test_hyperparam_tuning_CNN.ipynb.invalid
│   ├── hyper_param_tunning_DENSE.ipynb
│   ├── hyper_param_tunning_LINEAR.ipynb
│   ├── hyperparam_tuning__CNN_clipgrad.ipynb
│   ├── hyperparam_tuning__LINEAR_clipgrad.ipynb
│   └── hyperparam_tuning__MLP_clipgrad.ipynb
├── data
│   ├── __pycache__
│   │   └── generator.cpython-38.pyc
│   └── qiita
│       └── qiita_numpy.pkl
├── debug_IDE.py
├── hyper_param_tunning
│   ├── dense
│   │   ├── random_search_cv_euc_dense
│   │   │   ├── oracle.json
│   │   │   ├── trial_118c82f6ae4b9a9a6a7929ae615eea8f
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   ├── trial_1ea4a37db99e03d243ad3a825d73094e
│   │   │   │   └── trial.json
│   │   │   ├── trial_5406e392514b6f5d7da2515a14db5e20
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   ├── trial_7ed11e2d7de61f51a83b0d6263e80281
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   ├── trial_ae1c5e89509a7af067a305343fbcde57
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   ├── trial_d1bbebf0c002ce41f3be9f261a8824d4
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   ├── trial_dea10561cbe4c252552047dc10d5be51
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   └── tuner0.json
│   │   └── random_search_cv_hyp_dense
│   │       ├── oracle.json
│   │       ├── trial_285df21bf6d519449b036ab0cbbfa5e5
│   │       │   └── trial.json
│   │       ├── trial_2ee684bf6e07a8b4fc1a51bd263fc4b8
│   │       │   └── trial.json
│   │       ├── trial_3048217a84c9da1b18b48d8944bba32a
│   │       │   └── trial.json
│   │       ├── trial_474911924ef9e8669f7c74150125fab2
│   │       │   └── trial.json
│   │       ├── trial_6e102e0c0f4c5b0e2ba6e0cda0c27dfd
│   │       │   └── trial.json
│   │       ├── trial_8fd9f6e534e924328ab37790fb77afbb
│   │       │   └── trial.json
│   │       ├── trial_a94f288b2a2189b3e5e4d91aa8546a37
│   │       │   └── trial.json
│   │       ├── trial_b3e61eba771bd8f51a900a102881ef16
│   │       │   └── trial.json
│   │       ├── trial_e14d10b8d9ad8a30ab11548fd62a20cd
│   │       │   └── trial.json
│   │       ├── trial_e68c99293d3c22ffe579b7c09a314dbe
│   │       │   └── trial.json
│   │       └── tuner0.json
│   ├── linear
│   │   ├── random_search_cv_euc_dense
│   │   │   ├── oracle.json
│   │   │   ├── trial_787dac7a227e1bc7304f24ff068fb7e6
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   ├── trial_92f6243beca28b129e147b6ad90f2fc5
│   │   │   │   └── trial.json
│   │   │   ├── trial_971a81dee47584c53f6964283c7d079a
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   ├── trial_bd009870381db51812962f42d6c50d0a
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   └── tuner0.json
│   │   ├── random_search_cv_hyp_dense
│   │   │   ├── oracle.json
│   │   │   ├── trial_0cd2bc4dc9da1ea569958f91d35c19da
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   ├── trial_3b0d040154ed2eed69d12f6027d08145
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   ├── trial_589332c6a4c52353d14908f1ca555f0c
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   ├── trial_7eb6e152dd918cc09c666d522463ea10
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   ├── trial_95630e2fc0753c831074f96e06180793
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   ├── trial_a8f71d3ff7e3a640d31fa53b77c0f2a9
│   │   │   │   ├── checkpoints
│   │   │   │   │   └── epoch_0
│   │   │   │   │       ├── checkpoint
│   │   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │   │       └── checkpoint.index
│   │   │   │   └── trial.json
│   │   │   └── tuner0.json
│   │   └── random_search_cv_hyp_dense-test
│   │       ├── oracle.json
│   │       └── trial_ba97efeb310ccba4e46d73f200419ac0
│   │           └── trial.json
│   ├── random_search_cv_euc_dense
│   ├── random_search_cv_hyp_dense
│   ├── random_search_euc_cnn_clipgrad
│   │   ├── oracle.json
│   │   ├── trial_3df7d4c46d9183fd2abdaa1b5be563cf
│   │   │   └── trial.json
│   │   ├── trial_6215384f5f78e82b4907e8251757f4b7
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_880c133084a585e34692e3fa95fdd3b5
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_e1eafd8a0ef765a0ba8470d953c2ed29
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   └── tuner0.json
│   ├── random_search_euc_linear_clipgrad
│   │   ├── oracle.json
│   │   ├── trial_202b7dfff377260a6ed16da831b2351a
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_21e564e6d668b9ca2957d6c302e9fb00
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_97408108433d2ed6f6e7c19930a3f305
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_97755cf0664d7a85d468bcdc7f5c9d4b
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_e6ef99065b15b5badf0c96f14e37ee69
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   └── tuner0.json
│   ├── random_search_euc_mlp_clipgrad
│   │   ├── oracle.json
│   │   ├── trial_056c8eee2991318ef9d36f968fb28f75
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_2340eb0fcd278e52fa11510ac2bc6ae7
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_8214cde07a4d804373ac7d49cd6db079
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_a5c03e1193c540d1d97eadbdc25be95e
│   │   │   └── trial.json
│   │   ├── trial_aa3d667e8adadc707b42364b395ad153
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   └── tuner0.json
│   ├── random_search_hyp_cnn_clipgrad
│   │   ├── oracle.json
│   │   ├── trial_3cace12dc8395c1aa05cf6a842ade7e2
│   │   │   └── trial.json
│   │   ├── trial_4d523cd472a87161e3722cd3eae6daec
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_5675df4a0d1238b2ad3e9ea6cf73382a
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_62a245e17aecbee58c082a1c8e496f80
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_7cd2117a92ff36162658db2f48bbe0f7
│   │   │   └── trial.json
│   │   ├── trial_b877b0d0e764d940cbaf40809f5f4cbb
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   └── tuner0.json
│   ├── random_search_hyp_linear_clipgrad
│   │   ├── oracle.json
│   │   ├── trial_04b45c410889ad2a45c990a16e694136
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_23e47f9e7df96bb812fa10ff2a7f2820
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_3dda3585f147e48f45f6d500aeb561c4
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_5c10618116f787eff13390aec966fdba
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   ├── trial_aac458beda026c32d80bac70ca3b5816
│   │   │   ├── checkpoints
│   │   │   │   └── epoch_0
│   │   │   │       ├── checkpoint
│   │   │   │       ├── checkpoint.data-00000-of-00001
│   │   │   │       └── checkpoint.index
│   │   │   └── trial.json
│   │   └── tuner0.json
│   └── random_search_hyp_mlp_clipgrad
│       ├── oracle.json
│       ├── trial_09c00b624dc3d13908dc840815d4c29e
│       │   ├── checkpoints
│       │   │   └── epoch_0
│       │   │       ├── checkpoint
│       │   │       ├── checkpoint.data-00000-of-00001
│       │   │       └── checkpoint.index
│       │   └── trial.json
│       ├── trial_0da0d09920a9897ed25021678ccb4dfb
│       │   └── trial.json
│       ├── trial_3163ad2ff88e59d29c2fcebc4cfb8a77
│       │   ├── checkpoints
│       │   │   └── epoch_0
│       │   │       ├── checkpoint
│       │   │       ├── checkpoint.data-00000-of-00001
│       │   │       └── checkpoint.index
│       │   └── trial.json
│       ├── trial_a891225abe8fe0091cb4cb3f76ce1820
│       │   ├── checkpoints
│       │   │   └── epoch_0
│       │   │       ├── checkpoint
│       │   │       ├── checkpoint.data-00000-of-00001
│       │   │       └── checkpoint.index
│       │   └── trial.json
│       ├── trial_ead240ac7b206b66e880510dcd4c2b8f
│       │   ├── checkpoints
│       │   │   └── epoch_0
│       │   │       ├── checkpoint
│       │   │       ├── checkpoint.data-00000-of-00001
│       │   │       └── checkpoint.index
│       │   └── trial.json
│       └── tuner0.json
├── model
│   ├── __pycache__
│   │   ├── generator.cpython-38.pyc
│   │   ├── layer.cpython-38.pyc
│   │   ├── model.cpython-38.pyc
│   │   ├── models.cpython-38.pyc
│   │   ├── models_cstm.cpython-38.pyc
│   │   └── train_model.cpython-38.pyc
│   ├── generator.py
│   ├── layer.py
│   ├── models_cstm.py
│   └── train_model.py
├── requirements.txt
├── results
│   ├── cnn_clipgrad
│   │   ├── best_hyps_EUCLIDEAN.pkl
│   │   ├── best_hyps_HYPERBOLIC.pkl
│   │   ├── dist_func_tunning_COSINE.pkl
│   │   ├── dist_func_tunning_EUCLIDEAN.pkl
│   │   ├── dist_func_tunning_HYPERBOLIC.pkl
│   │   ├── dist_func_tunning_MANHATTAN.pkl
│   │   └── dist_func_tunning_SQUARE.pkl
│   ├── dense
│   │   ├── dist_func_tunning_COSINE.pkl
│   │   ├── dist_func_tunning_EUCLIDEAN.pkl
│   │   ├── dist_func_tunning_HYPERBOLIC.pkl
│   │   ├── dist_func_tunning_MANHATTAN.pkl
│   │   └── dist_func_tunning_SQUARE.pkl
│   ├── linear
│   │   ├── dist_func_tunning_COSINE.pkl
│   │   ├── dist_func_tunning_EUCLIDEAN.pkl
│   │   ├── dist_func_tunning_HYPERBOLIC.pkl
│   │   ├── dist_func_tunning_MANHATTAN.pkl
│   │   └── dist_func_tunning_SQUARE.pkl
│   ├── linear_clipgrad
│   │   ├── best_hyps_EUCLIDEAN.pkl
│   │   ├── best_hyps_HYPERBOLIC.pkl
│   │   ├── dist_func_tunning_COSINE.pkl
│   │   ├── dist_func_tunning_EUCLIDEAN.pkl
│   │   ├── dist_func_tunning_HYPERBOLIC.pkl
│   │   ├── dist_func_tunning_MANHATTAN.pkl
│   │   └── dist_func_tunning_SQUARE.pkl
│   ├── mlp_clipgrad
│   │   ├── best_hyps_EUCLIDEAN.pkl
│   │   ├── best_hyps_HYPERBOLIC.pkl
│   │   ├── dist_func_tunning_COSINE.pkl
│   │   ├── dist_func_tunning_EUCLIDEAN.pkl
│   │   ├── dist_func_tunning_HYPERBOLIC.pkl
│   │   ├── dist_func_tunning_MANHATTAN.pkl
│   │   └── dist_func_tunning_SQUARE.pkl
│   ├── temp.txt
│   └── train_test
│       └── phes
│           ├── history_0.pkl
│           ├── history_1.pkl
│           ├── history_2.pkl
│           ├── history_3.pkl
│           ├── history_4.pkl
│           ├── history_5.pkl
│           ├── history_6.pkl
│           ├── history_7.pkl
│           ├── history_8.pkl
│           ├── model_1.tf
│           │   ├── saved_model.pb
│           │   └── variables
│           │       ├── variables.data-00000-of-00001
│           │       └── variables.index
│           ├── model_2.tf
│           │   ├── saved_model.pb
│           │   └── variables
│           │       ├── variables.data-00000-of-00001
│           │       └── variables.index
│           ├── model_3.tf
│           │   ├── saved_model.pb
│           │   └── variables
│           │       ├── variables.data-00000-of-00001
│           │       └── variables.index
│           ├── model_4.tf
│           │   ├── saved_model.pb
│           │   └── variables
│           │       ├── variables.data-00000-of-00001
│           │       └── variables.index
│           ├── model_5.tf
│           │   ├── saved_model.pb
│           │   └── variables
│           │       ├── variables.data-00000-of-00001
│           │       └── variables.index
│           ├── model_6.tf
│           │   ├── saved_model.pb
│           │   └── variables
│           │       ├── variables.data-00000-of-00001
│           │       └── variables.index
│           ├── model_7.tf
│           │   ├── saved_model.pb
│           │   └── variables
│           │       ├── variables.data-00000-of-00001
│           │       └── variables.index
│           ├── model_8.tf
│           │   ├── saved_model.pb
│           │   └── variables
│           │       ├── variables.data-00000-of-00001
│           │       └── variables.index
│           └── scores.pkl
├── test_embedding_models.ipynb
├── tutorial.ipynb
└── utils
    ├── 16s_to_fasta.py
    ├── __pycache__
    │   ├── utils.cpython-38.pyc
    │   └── utils_func.cpython-38.pyc
    ├── compute_distances.py
    ├── phes_to_fasta.py
    ├── preprocessing.py
    ├── preprocessing_tests.py
    └── utils_func.py

187 directories, 301 files
```
