import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, cohen_kappa_score
from .plot_confusion_matrix import plot_CM, within_1_acc

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

def grid_metrics_naive_model(data_list: list, smote=False):
    start_time = time.time()

    rows = []
    group_frequencies = np.bincount(data_list['Train_Target'].astype(int))/len(data_list['Train_Target'])

    predictions = np.random.choice(len(group_frequencies), size=len(data_list['Test_Target']), p=group_frequencies)
    pred_classes = predictions.astype(int)

    y_true = data_list['Test_Target'].astype(int)

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
        'model': 'Naive Model',
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'f1_score': f1,
        'accuracy_score': accuracy,
        'cohen_kappa': cohen_kappa,
        'within_1_accuracy': w1acc,
        'execution_time_seconds': end_time - start_time
    }

    plot_CM(forwhom=f'CM_Naive_Model_{data_list["name"]}', true=y_true, pred=pred_classes, smote=smote)

    rows.append(return_object)
    return rows