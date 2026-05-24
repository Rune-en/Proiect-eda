random_forest_parameters = {
    'n_estimators': [10, 20, 50, 100],
    'max_depth': [10, 20, 30, None]
}

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error, accuracy_score
import numpy as np
import time
from .plot_confusion_matrix import plot_CM, within_1_acc
from imblearn.over_sampling import SMOTE

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
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            smote = SMOTE(sampling_strategy='auto', random_state=42)

            X_train_resampled, y_train_resampled = smote.fit_resample(data_list['Train_Predictors'], data_list['Train_Target'])

            model.fit(X_train_resampled, y_train_resampled)
            y_true = data_list['Test_Target'].astype(int)

            predictions = model.predict(data_list['Test_Predictors'])
            predictions = np.clip(predictions, int(y_true.min()), int(y_true.max()))

            pred_classes = np.rint(predictions).astype(int)
            pred_classes = np.clip(pred_classes, int(y_true.min()), int(y_true.max()))

            mse = mean_squared_error(data_list['Test_Target'], predictions)
            mae = mean_absolute_error(data_list['Test_Target'], predictions)
            f1 = f1_score(y_true, pred_classes, average='weighted', zero_division=0)
            accuracy = accuracy_score(y_true, pred_classes)
            w1acc = within_1_acc(y_true, predictions)

            end_time = time.time()
            return_object = {
                'dimension_reduction_type': data_list['name'],
                'n_features': data_list['Train_Predictors'].shape[1],
                'model': 'Random Forest with SMOTE',
                'n_estimators': n_estimators,
                'max_depth': str(max_depth) if max_depth is not None else 'None',
                'mean_squared_error': mse,
                'mean_absolute_error': mae,
                'f1_score': f1,
                'accuracy_score': accuracy,
                'within_1_accuracy': w1acc,
                'execution_time_seconds': end_time - start_time
            }

            rows.append(return_object)

            if mae < best_mae:
                best_mae = mae
                best_params = (n_estimators, max_depth)

    model = RandomForestClassifier(n_estimators=best_params[0], max_depth=best_params[1])
    smote = SMOTE(sampling_strategy='auto', random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(data_list['Train_Predictors'], data_list['Train_Target'])

    model.fit(X_train_resampled, y_train_resampled)
    predictions = model.predict(data_list['Test_Predictors'])
    predictions = np.clip(predictions, int(y_true.min()), int(y_true.max()))

    plot_CM(forwhom=f'CM_Random_Forest_{data_list["name"]}', true=y_true, pred=np.rint(predictions).astype(int))
    return rows











