from dimension_reduction import *
from ml_algs import *
from dimension_reduction import *
import data
import pandas as pd
import numpy as np


dataset = data.obtine_date_procesate()
dataset.T.to_csv("processed_data.csv", index=False)
dataset = dataset.to_numpy().astype(float)
print(dataset)
print(dataset.shape)









