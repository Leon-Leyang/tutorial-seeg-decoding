# Decoding Presence of Tony from sEEG Recordings

## Project Overview

In this tutorial, we delve into the principles of stereoencephalography (sEEG) data decoding with a practical, hands-on approach. Our challenge involves a binary classification task: using one-second sEEG recordings to determine whether Tony, a character from 'Greenbook', is present in the scene. This task is designed to provide a straightforward yet informative introduction to sEEG decoding.



## Methods

Our classification task is approached using two primary machine learning models:

1. **Fully Connected Neural Network (FCNN):** Achieved a testing accuracy of 63.34%, using a data split of 70% for training, 15% for validation, and 15% for testing.
2. **Support Vector Machines (SVMs):** With a polynomial kernel and C=0.001, this model reached our highest accuracy of 67.24%. The data was similarly split, with 70% used for training and 15% for testing.



## Dataset

### Data Access and Setup

The dataset for this tutorial has been pre-processed and is ready for use. Download the dataset using your Brown University email from the links provided in our Slack channel.

#### Steps to Set Up the Data:

1. **Create a Data Folder:** In the root directory of this project, create a folder named `/data`.
2. **Download and Organize the Data:** Use the links provided in our Slack channel to download the sEEG and label data. After downloading, place these files inside the `/data` folder.



## Code Organization

This repository is organized to facilitate easy navigation and understanding of our machine learning pipelines for the Fully Connected Neural Network (FCNN) using PyTorch, and Support Vector Machines (SVMs) using Scipy.

### PyTorch Pipeline for FCNN

- `main.py`: This script includes the core training and testing logic for the FCNN model.
- `/dataset`:
  - `/dataset.py`: Defines the custom dataset class for use in the PyTorch pipeline.
- `/eval`:
  - `/eval.py`: Contains the evaluation functions used during the validation and testing phases in the PyTorch pipeline.
- `/models`:
  - `fcnn.py`: Houses the architecture definition of the FCNN model.
- `/train`:
  - `train.py`: Implements the training procedures specific to the FCNN in the PyTorch pipeline.
- `/utils`: Contains utility scripts aiding in various tasks.
  - `data.py`: Manages preprocessing steps for the dataset.
  - `model.py`: Includes utility functions for model operations.

### Scipy Pipeline for SVMs

- `/svm`:
  - `svm_demo.ipynb`: A Jupyter Notebook demonstrating the implementation and usage of SVMs within the Scipy framework.

