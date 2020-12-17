# Models of Sequential Data Project: COHORTNEY

The project is about implementation of methods for event sequence clustering.

The main method to be considered: COHORTNEY

The main baseline is [A Dirichlet Mixture Model of Hawkes Processes for
Event Sequence Clustering](https://arxiv.org/pdf/1701.09177.pdf)

Datasets:
- IPTV dataset
- Synthetic Hawkes processes
- other datasets from the PointProcesses.com

### Installation

```bat
git clone https://github.com/rodrigorivera/mds20_cohortney
```

### Setup


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


### Usage

#### Deep Clustering over Cohortney

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

more examples can be found in src/DeepCluster/test_deep_cluster.ipynb

#### Dirichlet Mixture Model of Hawkes Processes

examples of usage DMMHP can be found in src/DMHP/hp_test.ipynb

#### Convolutional Autoencoder Clustering over Cohortney

examples of usage can be found in src/Cohortney/test_CAE.ipynb

#### Optimal Transport for Clustering over Cohortney

example of usage can be found in src/optimal_transport/test_op.ipynb
