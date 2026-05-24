poisson_linear_model_parameters = {
    'alpha': [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
}

import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

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
    rows = []
    best_alpha = None
    best_mae = float('inf')
    for alpha in poisson_linear_model_parameters['alpha']:
        start_time = time.time()
        model = make_pipeline(
            StandardScaler(),
            PoissonRegressor(alpha=alpha, max_iter=2000)
        )

        model.fit(data_list['Train_Predictors'], data_list['Train_Target'])

        y_true = data_list['Test_Target'].astype(int)

        predictions = model.predict(data_list['Test_Predictors'])
        predictions = np.clip(predictions, int(y_true.min()), int(y_true.max()))

        pred_classes = np.rint(predictions).astype(int)
        pred_classes = np.clip(pred_classes, int(y_true.min()), int(y_true.max()))

        mse = mean_squared_error(data_list['Test_Target'], predictions)
        mae = mean_absolute_error(data_list['Test_Target'], predictions)
        f1 = f1_score(y_true, pred_classes, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_true, pred_classes)
        
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

        end_time = time.time()
        return_object = {
            'dimension_reduction_type': data_list['name'],
            'n_features': data_list['Train_Predictors'].shape[1],
            'model': 'Poisson Regression',
            'alpha': alpha,
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'f1_score': f1,
            'accuracy_score': accuracy,
            'prediction_std': float(np.std(predictions)),
            'execution_time_seconds': end_time - start_time
        }

        #plot predictions vs true values

    rows.append(return_object)

    model = make_pipeline(
        StandardScaler(),
        PoissonRegressor(alpha=best_alpha, max_iter=2000))
    model.fit(data_list['Train_Predictors'], data_list['Train_Target'])
    predictions = model.predict(data_list['Test_Predictors'])
    predictions = np.clip(predictions, int(y_true.min()), int(y_true.max()))

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, predictions, alpha=0.2, s=100)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Poisson Regression Predictions (alpha={best_alpha})')
    plt.grid()
    plt.savefig(f'poisson_predictions_{data_list["name"]}_alpha_{best_alpha}.png')
    plt.close()

    return rows






