# Models of Sequential Data Project: COHORTNEY

This awesome repository contains implementation of methods considered in the project.

The project is about Deep Clustering of Event Sequences.

The main method to be considered: **COHORTNEY**

The main baseline is [A Dirichlet Mixture Model of Hawkes Processes for
Event Sequence Clustering](https://arxiv.org/pdf/1701.09177.pdf)

The final project report with overall methods' description can be found at [src/tex/COHORTNEY.pdf](src/tex/COHORTNEY.pdf)

Datasets:
- IPTV dataset
- Synthetic Hawkes processes realizations

# Table of Contents
0. [Results](#results)
1. [Repository structure](#repository-structure) 
2. [Installation](#installation)
3. [Setup](#setup)
4. [Usage](#usage)

## Results

| method   | DMHP    | Deep Cluster | CAE  | Optimal Transport |
|----------|---------|-----------------|----------------|------------------------------|
| K=5, C=5 | \.6108 | \.4384          | \.5306         | \.4537                       |
| K=4, C=5 | \.7505 | \.5613          | \.7113         | \.5372                       |
| K=3, C=5 | \.8723  | \.6945          | \.797          | \.6731                       |

## Repository structure

The overall list of directories is:

```bat
.
├── data
│   ├── IPTVdata
│   ├── IPTV_Data
│   └── simulated_Hawkes
│       ├── K10_C1
│       ├── K10_C10
│       ├── K2_C5
│       ├── K3_C1
│       ├── K3_C5
│       ├── K4_C5
│       └── K5_C5
├── models
│   ├── cnn
│   ├── cnn-for-sinkhorn-knopp
│   ├── hawkes
│   └── OTModel
├── src
│   ├── CAE
│   ├── Cohortney
│   ├── DeepCluster
│   ├── DMHP
│   ├── optimal_transport
│   └── Soft_DTW
└── tex
    ├── code
    └── figs
```
The inner directories  of `simulated_Hawkes` contain synthetic datasets for different number of clusters (K) and event types (C)

The structure of `src` dir contains differnt components of the project:

```bat
.
├── __init__.py
├── CAE
│   ├── __init__.py
│   ├── litautoencoder.py
│   └── test_CAE.ipynb
├── Cohortney
│   ├── __init__.py
│   ├── cohortney.py
│   ├── data_utils.py
│   ├── pure_cohortney.ipynb
│   └── utils.py
├── DeepCluster
│   ├── __init__.py
│   ├── clustering.py
│   ├── data_utils.py
│   ├── main.py
│   ├── README.md
│   └── test_deep_cluster.ipynb
├── DMHP
│   ├── __init__.py
│   ├── HP.py
│   ├── hp_test.ipynb
│   ├── metrics.py
│   └── README.md
├── optimal_transport
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── development.ipynb
│   ├── main.py
│   ├── sk.py
│   ├── test_ot.ipynb
│   └── trainer.py
└── Soft_DTW
    ├── README.ipynb
    ├── soft_dtw.py
    └── tests.ipynb
```

## Installation

```bat
git clone https://github.com/rodrigorivera/mds20_cohortney
```

## Setup


```bat
cd [VENV]

virtualenv cohortney

source cohortney/bin/activate
```

back in repository dir:

```bat
pip install -r requirements.txt
```

to install the dependencies run one of the following:

```bat
pip install -e .
```

```bat
python setup.py install
```
Correctly installed dependencies are needed for running Deep Clustering


## Usage

### Deep Clustering over Cohortney

run on synthetic dataset:

```bat
python src/DeepCluster/main.py \
    --data_dir data/simulated_Hawkes/K3_C1 \ 
    --verbose \
    --epochs 30 \ 
    --nruns 3 \
    --not_datetime \
    --ext csv \ 
    --batch 128 \ 
    --nmb_cluster 3 \
    --n 8
```

run on IPTV dataset:

```bat
python src/DeepCluster/main.py \
    --data_dir data/IPTV_Data \
    --verbose \ 
    --epochs 30 \ 
    --nruns 3 \
    --ext txt \ 
    --batch 128 \ 
    --nmb_cluster 10 \
    --n 8
 ```

more examples can be found in [src/DeepCluster/test_deep_cluster.ipynb](src/DeepCluster/test_deep_cluster.ipynb)

### Dirichlet Mixture Model of Hawkes Processes

examples of usage DMMHP can be found in [src/DMHP/hp_test.ipynb](src/DMHP/hp_test.ipynb)

### Convolutional Autoencoder Clustering over Cohortney

examples of usage can be found in [src/CAE/test_CAE.ipynb](src/CAE/test_CAE.ipynb)

### Optimal Transport for Clustering over Cohortney

example of usage can be found in [src/optimal_transport/test_ot.ipynb](src/optimal_transport/test_ot.ipynb)

### Original Cohortney

examples of usage can be found in [src/Cohortney/pure_cohortney.ipynb](src/Cohortney/pure_cohortney.ipynb)
