from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import cohen_kappa_score, mean_squared_error, f1_score, mean_absolute_error, accuracy_score
import numpy as np
import time
from .plot_confusion_matrix import plot_CM, within_1_acc
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

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

def decision_tree_StratKFold(data_list: list, smote=False):
    rows = []
    train_predictors = np.asarray(data_list['Train_Predictors'])
    train_target = np.asarray(data_list['Train_Target'])
    test_predictors = np.asarray(data_list['Test_Predictors'])
    test_target = np.asarray(data_list['Test_Target'])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = defaultdict(list)

    for train_index, val_index in skf.split(train_predictors, train_target):
        train_X, val_X = train_predictors[train_index], train_predictors[val_index]
        train_y, val_y = train_target[train_index], train_target[val_index]

        for max_depth in decision_tree_parameters['max_depth']:
            for min_samples_split in decision_tree_parameters['min_samples_split']:
                params = (max_depth, min_samples_split)
                model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
                model.fit(train_X, train_y)

                predictions = model.predict(val_X)
                pred_classes = np.rint(predictions).astype(int)
                pred_classes = np.clip(pred_classes, int(val_y.min()), int(val_y.max()))

                kappa = cohen_kappa_score(val_y.astype(int), pred_classes, weights='quadratic')
                cv_scores[params].append(kappa)

    best_params = max(cv_scores, key=lambda params: np.mean(cv_scores[params]))

    start_time = time.time()

    model = DecisionTreeRegressor(max_depth=best_params[0], min_samples_split=best_params[1], random_state=42)
    model.fit(train_predictors, train_target)
    predictions = model.predict(test_predictors)

    y_true = test_target.astype(int)
    
    pred_classes = np.rint(predictions).astype(int)
    pred_classes = np.clip(pred_classes, int(y_true.min()), int(y_true.max()))

    mse = mean_squared_error(test_target, pred_classes)
    mae = mean_absolute_error(test_target, pred_classes)
    f1 = f1_score(y_true, pred_classes, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, pred_classes)
    w1acc = within_1_acc(y_true, pred_classes)
    cohen_kappa = cohen_kappa_score(y_true, pred_classes, weights='quadratic')

    end_time = time.time()

    return_object = {
        'dimension_reduction_type': data_list['name'],
        'n_features': train_predictors.shape[1],
        'parameters': f'max_depth={best_params[0]}, min_samples_split={best_params[1]}',
        'model': 'Decision Tree StratKFold',
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'f1_score': f1,
        'accuracy_score': accuracy,
        'cohen_kappa': cohen_kappa,
        'within_1_accuracy': w1acc,
        'execution_time_seconds': end_time - start_time
    }
    rows.append(return_object)

    plot_CM(forwhom=f'CM_Decision_Tree_StratKFold_{data_list["name"]}', true=y_true, pred=pred_classes, smote=smote)

    return rows