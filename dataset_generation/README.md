# Dataset Overview

We use four datasets to evaluate our model: **N-body**, **Protein**, **Water-3D**, and **Fluid113K**. This README provides instructions for obtaining each dataset.

## Dataset Statistics

|                                            | N-body System             | Protein Dynamics         | Water-3D                  | Fluid113K              |
|--------------------------------------------|----------------------------|---------------------------|---------------------------|-------------------------|
| **# Samples (Train / Valid / Test)**       | 5,000 / 2,000 / 2,000     | 2,481 / 827 / 863         | 15,000 / 1,500 / 1,500     | 1,600 / 320 / 320       |
| **# Nodes ($N$)**                          | 100                        | 855                       | 7,806 (avg)                | 113,140 (avg)           |
| **# Edges ($E$)**                      | 9,900                      | 55,107 (avg)              | 94,931 (avg)               | 1,706,973 (avg)         |
| **Default Radius ($r$)**                  | ∞                          | 10 Å                      | 0.035                      | 0.075                   |
| **Prediction Interval ($\Delta t$)**       | 10                         | 15                        | 15                         | 20                      |

---

## Dataset Details

### N-Body

A widely used benchmark for equivariant graph neural networks. We generate this dataset using the code provided by [GMN](https://github.com/hanjq17/GMN/tree/main/spatial_graph/n_body_system/dataset). You can use this [script](./nbody/run.sh) to generate N-Body dataset.

### Protein

This dataset is a long-range AdK equilibrium molecular dynamics (MD) trajectory from MDAnalysis.

You can download the dataset with the following code:

```python
from MDAnalysisData import datasets
import MDAnalysis

adk = datasets.fetch_adk_equilibrium(data_home='protein')
adk_data = MDAnalysis.Universe(adk.topology, adk.trajectory)
```

### Water-3D

The Water-3D dataset is sourced from [learning to simulate](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate). You can download it using the script [here](https://github.com/google-deepmind/deepmind-research/blob/master/learning_to_simulate/download_dataset.sh).

After downloading, using this [code](https://github.com/tumaer/lagrangebench/blob/main/data_gen/gns_data/tfrecord_to_h5.py) to convert tfrecord to h5 format, which can process by our code.

### Fluid113K

A large-scale dataset generated using [DeepLagrangianFluids](https://github.com/isl-org/DeepLagrangianFluids/tree/master/datasets). To increase graph size, we enlarge the simulation [box and fluid mesh](./Fluid113K/models/).

You can generate this dataset using the script provided [here](./Fluid113K/create_data.sh). Note that depending on your CPU performance, data generation may take up to a week.

⸻

## Direct Download

You can also download the datasets via Baidu Drive:

- File Name: DistEGNN_Dataset
- Link: https://pan.baidu.com/s/15v5t5Gdy2GbxcK0s-KLdaw
- Code: mfvv
