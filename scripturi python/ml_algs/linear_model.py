from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score, mean_squared_error, f1_score, mean_absolute_error, accuracy_score
import numpy as np
import time
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

def linear_model(data_list: list, smote=False):
    start_time = time.time()

    rows = []
    model = LinearRegression()
    model.fit(data_list['Train_Predictors'], data_list['Train_Target'])

    y_true = data_list['Test_Target'].astype(int)

    predictions = model.predict(data_list['Test_Predictors'])
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
        'model': 'Linear Regression',
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'f1_score': f1,
        'accuracy_score': accuracy,
        'within_1_accuracy': w1acc,
        'cohen_kappa': cohen_kappa,
        'execution_time_seconds': end_time - start_time
    }

    rows.append(return_object)

    plot_CM(forwhom=f'CM_Linear_Regression_{data_list["name"]}', true=y_true, pred=pred_classes, smote=smote)

    return rows






