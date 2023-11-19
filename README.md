# ST-ABC: Spatio-Temporal Attention-Based Convolutional Network for Multi-Scale Lane-Level Traffic Prediction
This is the original pytorch implementation of Spatio-Temporal Attention-Based Convolutional Network(ST-ABC):


## Requirements
- python 3.8
- matplotlib
- numpy
- scipy
- pandas
- torch
- argparse


## Data Preparation

### In the Data_process directory, we provide training, validation, and test set files for the PeMS dataset and HuaNan dataset with input_window=12 and input_window=30, which you can use directly. 

### If you wish to generate your own specified input data, we also provide the original data for the PeMS dataset, and you can use the following way：

```python
python generate_training_data.py
```

## Train Commands

```
python train.py
```

