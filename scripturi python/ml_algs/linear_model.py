from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
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

def grid_metrics_linear_model(data_list: list):
    start_time = time.time()
    rows = []

    model = LinearRegression()

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
        'model': 'Linear Regression',
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'f1_score': f1,
        'accuracy_score': accuracy,
        'execution_time_seconds': end_time - start_time
    }

    rows.append(return_object)

    plt.figure(figsize=(5, 5))
    plt.scatter(data_list['Test_Target'], predictions, alpha=0.2, s=100)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Linear Regression Predictions ({data_list["name"]})')
    plt.grid()
    plt.savefig(f'linear_regression_predictions_{data_list["name"]}.png')
    plt.close()

    return rows






