random_forest_parameters = {
    'n_estimators': [10, 20, 50, 100],
    'max_depth': [10, 20, 30, None]
}

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error, accuracy_score
import numpy as np
import time
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

def grid_metrics_random_forest(data_list: list):
    rows = []
    best_params = None
    best_mae = float('inf')
    for n_estimators in random_forest_parameters['n_estimators']:
        for max_depth in random_forest_parameters['max_depth']:
            start_time = time.time()
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

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

            end_time = time.time()
            return_object = {
                'dimension_reduction_type': data_list['name'],
                'n_features': data_list['Train_Predictors'].shape[1],
                'model': 'Random Forest',
                'n_estimators': n_estimators,
                'max_depth': str(max_depth) if max_depth is not None else 'None',
                'mean_squared_error': mse,
                'mean_absolute_error': mae,
                'f1_score': f1,
                'accuracy_score': accuracy,
                'execution_time_seconds': end_time - start_time
            }

            rows.append(return_object)

            if mae < best_mae:
                best_mae = mae
                best_params = (n_estimators, max_depth)

    model = RandomForestRegressor(n_estimators=best_params[0], max_depth=best_params[1])
    model.fit(data_list['Train_Predictors'], data_list['Train_Target'])
    predictions = model.predict(data_list['Test_Predictors'])
    predictions = np.clip(predictions, int(y_true.min()), int(y_true.max()))

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, predictions, alpha=0.2, s=100)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Random Forest Predictions\n(n_estimators={best_params[0]}, max_depth={best_params[1]})')
    plt.grid()
    plt.savefig(f'random_forest_predictions_{data_list["name"]}_n_estimators_{best_params[0]}_max_depth_{best_params[1]}.png')
    plt.close()

    return rows











