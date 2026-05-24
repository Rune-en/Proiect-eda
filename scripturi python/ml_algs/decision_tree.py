from time import time

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import cohen_kappa_score, mean_squared_error, f1_score, mean_absolute_error, accuracy_score
import numpy as np
import time
from .plot_confusion_matrix import plot_CM, within_1_acc
from sklearn.model_selection import train_test_split

decision_tree_parameters = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

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

def grid_metrics_decision_tree(data_list: list, smote=False):
    rows = []
    best_params = None
    best_kappa = -1
    train_X, val_X, train_y, val_y = train_test_split(data_list['Train_Predictors'], data_list['Train_Target'], test_size=0.25, random_state=42)
    for max_depth in decision_tree_parameters['max_depth']:
        for min_samples_split in decision_tree_parameters['min_samples_split']:
            model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)

            model.fit(train_X, train_y)

            y_true = val_y.astype(int)

            predictions = model.predict(val_X)
            pred_classes = np.rint(predictions).astype(int)
            pred_classes = np.clip(pred_classes, int(y_true.min()), int(y_true.max()))

            cohen_kappa = cohen_kappa_score(y_true, pred_classes, weights='quadratic')

            if cohen_kappa > best_kappa:
                best_kappa = cohen_kappa
                best_params = (max_depth, min_samples_split)

    start_time = time.time()

    model = DecisionTreeRegressor(max_depth=best_params[0], min_samples_split=best_params[1])
    model.fit(data_list['Train_Predictors'], data_list['Train_Target'])
    predictions = model.predict(data_list['Test_Predictors'])
    pred_classes = np.rint(predictions).astype(int)
    pred_classes = np.clip(pred_classes, int(y_true.min()), int(y_true.max()))

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
        'parameters': f'max_depth={best_params[0]}, min_samples_split={best_params[1]}',
        'model': 'Decision Tree',
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'f1_score': f1,
        'accuracy_score': accuracy,
        'cohen_kappa': cohen_kappa,
        'within_1_accuracy': w1acc,
        'execution_time_seconds': end_time - start_time
    }
    rows.append(return_object)

    plot_CM(forwhom=f'CM_Decision_Tree_{data_list["name"]}', true=y_true, pred=pred_classes, smote=smote) 

    return rows






