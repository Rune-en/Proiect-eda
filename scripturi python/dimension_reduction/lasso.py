from sklearn.linear_model import lasso
import numpy as np

lasso_regression_parameters = {
    'nr_of_features': [10, 20, 30, 40, 50],
    'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
}

