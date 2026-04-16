from sklearn.linear_model import Ridge
import numpy as np

ridge_regression_parameters = {
    'nr_of_features': [10, 20, 30, 40, 50],
    'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
}

def genereaza_date_reduse_lasso(predictors, target):
    for nr_of_features in ridge_regression_parameters['nr_of_features']:
        for alpha in ridge_regression_parameters['alpha']:
            model = Ridge(alpha=alpha)
            model.fit(predictors, target)

            importance = np.abs(model.coef_)
            indices = np.argsort(importance)[-nr_of_features:]

            reduced_predictors = predictors[:, indices]
            return reduced_predictors
