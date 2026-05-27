import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error, accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .plot_confusion_matrix import plot_CM, within_1_acc
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

poisson_linear_model_parameters = {
    'alpha': [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
}


def poisson_linear_model(data_list: dict, smote=False):
    rows = []
    best_alpha = None
    best_kappa = -1

    train_X, val_X, train_y, val_y = train_test_split(data_list['Train_Predictors'], data_list['Train_Target'], test_size=0.25, random_state=42)

    for alpha in poisson_linear_model_parameters['alpha']:
        
        model = make_pipeline(
            StandardScaler(),
            PoissonRegressor(alpha=alpha, max_iter=2000)
        )

        model.fit(train_X, train_y)

        y_true = val_y.astype(int)

        predictions = model.predict(val_X)
        predictions = np.nan_to_num(predictions, nan=0.0)
        pred_classes = np.rint(predictions).astype(int)
        pred_classes = np.clip(pred_classes, int(y_true.min()), int(y_true.max()))

        cohen_kappa = cohen_kappa_score(y_true, pred_classes, weights='quadratic')

        if cohen_kappa > best_kappa:
            best_kappa = cohen_kappa
            best_alpha = alpha

    start_time = time.time()
    
    model = make_pipeline(StandardScaler(),PoissonRegressor(alpha=best_alpha, max_iter=2000))
    model.fit(data_list['Train_Predictors'], data_list['Train_Target'])

    y_true = data_list['Test_Target'].astype(int)

    predictions = model.predict(data_list['Test_Predictors'])
    predictions = np.nan_to_num(predictions, nan=0.0)
    pred_classes = np.rint(predictions).astype(int)
    pred_classes = np.clip(pred_classes, int(y_true.min()), int(y_true.max()))
    
    mse = mean_squared_error(data_list['Test_Target'], pred_classes)
    mae = mean_absolute_error(data_list['Test_Target'], pred_classes)
    f1 = f1_score(y_true, pred_classes, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, pred_classes)
    w1acc = within_1_acc(y_true, pred_classes)
    cohen_kappa = cohen_kappa_score(y_true, pred_classes, weights='quadratic')

    end_time = time.time()
    return_object = {
        'dimension_reduction_type': data_list['name'],
        'n_features': data_list['Train_Predictors'].shape[1],
        'model': 'Poisson Regression',
        'parameters': f'alpha={best_alpha}',
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'f1_score': f1,
        'accuracy_score': accuracy,
        'cohen_kappa': cohen_kappa,
        'within_1_accuracy': w1acc,
        'execution_time_seconds': end_time - start_time
    }

    rows.append(return_object)

    plot_CM(forwhom=f'CM_Poisson_Regression_{data_list["name"]}', true=y_true, pred=pred_classes, smote=smote)

    return rows






