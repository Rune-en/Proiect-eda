poisson_linear_model_parameters = {
    'alpha': [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
}

import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

'''
data list sa contina obiecte de tipul:
{
    'name': -
    'Train_Predictors': -
    'Train_Target': -
    'Test_Predictors': -
    'Test_Target': -
}

'''

def grid_metrics_poisson_linear_model(data_list: dict):
    if isinstance(data_list, list):
        if len(data_list) != 1:
            raise ValueError("data_list list input must contain exactly one dataset dict")
        data_list = data_list[0]

    rows = []
    for alpha in poisson_linear_model_parameters['alpha']:
        model = make_pipeline(
            StandardScaler(),
            PoissonRegressor(alpha=alpha, max_iter=2000)
        )

        model.fit(data_list['Train_Predictors'], data_list['Train_Target'])

        predictions = model.predict(data_list['Test_Predictors'])

        y_true = data_list['Test_Target'].astype(int)
        pred_classes = np.rint(predictions).astype(int)
        pred_classes = np.clip(pred_classes, int(y_true.min()), int(y_true.max()))

        mse = mean_squared_error(data_list['Test_Target'], predictions)
        mae = mean_absolute_error(data_list['Test_Target'], predictions)
        f1 = f1_score(y_true, pred_classes, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_true, pred_classes)


        return_object = {
            'dimension_reduction_type': data_list['name'],
            'model': 'Poisson Regression',
            'alpha': alpha,
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'f1_score': f1,
            'accuracy_score': accuracy,
            'prediction_std': float(np.std(predictions))
        }

        #plot predictions vs true values
        
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, predictions, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Poisson Regression Predictions (alpha={alpha})')
        plt.grid()
        plt.savefig(f'poisson_predictions_alpha_{alpha}.png')
        plt.close()

        rows.append(return_object)


    return rows






