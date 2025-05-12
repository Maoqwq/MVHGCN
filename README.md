# MVHGCN

## Introduction

MVHGCN is a model designed for predicting associations between circRNAs and diseases. It has demonstrated outstanding performance across three datasets.

## Requirements

To run MVHGCN, ensure you have the following dependencies installed:

```plaintext
Numpy = 1.26.4
Random = 1.24
Pandas = 2.2.1
Torch = 2.1.1
Keras = 3.5.0
Sklearn = 1.5.0
h5py = 3.10.0
network = 3.2.1
```

## Datasets

The model leverages three datasets, containing circRNA, miRNA, lncRNA, disease entities, and their relationships (circRNA-disease, miRNA-disease, lncRNA-disease, circRNA-miRNA, miRNA-lncRNA). Below is a summary of the datasets:

| Dataset   | circRNA | Disease | miRNA | lncRNA | circRNA-disease | circRNA-miRNA | miRNA-disease | lncRNA-disease | lncRNA-miRNA |
|-----------|---------|---------|-------|--------|-----------------|---------------|---------------|----------------|--------------|
| Dataset1  | 2480    | 101     | 2746  | 3291   | 3602           | 79908         | 735           | 268            | 31732        |
| Dataset2  | 1080    | 172     | 2716  | 3279   | 1273           | 50081         | 780           | 429            | 32073        |
| Dataset3  | 2640    | 181     | 2743  | 3383   | 3368           | 80015         | 315           | 831            | 33071        |

Data download link: https://drive.google.com/file/d/1IX9Jve3fcCb_15JZIHKKswW1GpAVFbdZ/view?usp=sharing

## Usage

emd.py is used to calculate the circRNA similarity matrix and the disease similarity matrix, and to extract low-dimensional embedding features using a Deep Neural Network (DNN). The input is the circRNA-disease association matrix, and the output is the circRNA and disease embeddings.

view.py is used to calculate the relationship weights between nodes based on different meta-paths, ultimately generating multiple views and saving them as npy files. The input is the node association data, and the output is the npy file of the views.

dataload.py is used to load the data and generate positive and negative samples.

parsers.py is used to define and retrieve the hyperparameters required for program execution.

utils.py is used for task evaluation and training control.

model.py is the prediction model.

main.py is the training process script. The input is the circRNA embeddings, disease embeddings, and the npy files of the views, and the output is the prediction results.

After downloading the datasets, you can run the model with the following command:

```bash
python main.py


