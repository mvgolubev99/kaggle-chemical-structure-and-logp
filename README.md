# kaggle-chemical-structure-and-logp - README

## Project Overview

This project is educational in nature. Its goal is to improve practical skills and understanding across several areas of machine learning and chemoinformatics, including:

* Exploratory data analysis
* Data splitting strategies
* Converting molecular structures into numerical representations (fingerprints)
* Training scikit-learn–based models
* Hyperparameter tuning (including molecular representation as a hyperparameter) using Bayesian optimization
* Training PyTorch/Lightning models
* Evaluating model performance
* Defining a model’s applicability domain using Mahalanobis distance in the latent space of a multilayer perceptron
* Applying a simple generative approach for inverse molecular design using SELFIES mutations

The dataset used in this project comes from the Kaggle dataset [*chemical-structure-and-logp*](https://www.kaggle.com/datasets/matthewmasters/chemical-structure-and-logp).

## How to Run This Project

### Installation

#### Create and Activate the Environment

This project is confirmed to work with **Python 3.12.12**.

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

#### Install Dependencies

After activating the virtual environment, install the required packages:

```bash
python -m pip install -r requirements.txt
```

You will also need a way to run Jupyter notebooks.
If you're using VSCode, simply install the official [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
Alternatively, you can install and launch Jupyter Notebook via the command line:

```bash
python -m pip install notebook
python -m notebook
```

Make sure the correct kernel (the one from the virtual environment) is selected when running the project notebooks.

### Downloading the Dataset

To reproduce the results, download the [*chemical-structure-and-logp*](https://www.kaggle.com/datasets/matthewmasters/chemical-structure-and-logp) dataset.

Run the following commands from the project root:

```bash
mkdir -p ./data
```

```bash
curl -L -o ./data/archive_logp.zip \
https://www.kaggle.com/api/v1/datasets/download/matthewmasters/chemical-structure-and-logp
```

```bash
unzip ./data/archive_logp.zip -d ./data/
rm ./data/archive_logp.zip
```

After extraction, the expected file structure includes:

```
./data/logP_dataset.csv
```

### Project Structure

```
.
├── data/                           ← Dataset and data splits
├── notebooks/                      ← Notebooks
│   ├── 01_Data_manipulations.*
│   ├── 02_LightGBM.*
│   └── 03_MLP_and_GA.*
├── src/
│   ├── chemdata/                   ← Fingerprinting + data splitting utilities
│   ├── models/                     ← LightGBM search, MLP, metrics
│   ├── ga/                         ← Genetic algorithm engine
│   └── path_handling.py
├── tb_logs/                        ← TensorBoard logs for MLP training
├── tests/
│   └── test_mlp_forward.py
├── requirements.txt
└── README.md
```

## References

* Kaggle dataset: [https://www.kaggle.com/datasets/matthewmasters/chemical-structure-and-logp](https://www.kaggle.com/datasets/matthewmasters/chemical-structure-and-logp)