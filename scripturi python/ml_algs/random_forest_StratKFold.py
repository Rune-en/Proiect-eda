random_forest_parameters = {
    'n_estimators': [50, 100, 300],
    'max_depth': [5, 10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error, accuracy_score,cohen_kappa_score
import numpy as np
import time
from .plot_confusion_matrix import plot_CM, within_1_acc
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

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

def random_forest_StratKFold(data_list: list, smote=False):
    rows = []
    best_params = None
    train_predictors = np.asarray(data_list['Train_Predictors'])
    train_target = np.asarray(data_list['Train_Target'])
    test_predictors = np.asarray(data_list['Test_Predictors'])
    test_target = np.asarray(data_list['Test_Target'])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = defaultdict(list)

    for train_index, val_index in skf.split(train_predictors, train_target):
        train_X, val_X = train_predictors[train_index], train_predictors[val_index]
        train_y, val_y = train_target[train_index], train_target[val_index]

        for n_estimators in random_forest_parameters['n_estimators']:
            for max_depth in random_forest_parameters['max_depth']:
                for min_samples_split in random_forest_parameters['min_samples_split']:
                    params = (n_estimators, max_depth, min_samples_split)
                    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,random_state=42)
                    model.fit(train_X, train_y)
                    predictions = model.predict(val_X).astype(int)
                    predictions = np.clip(predictions, int(val_y.min()), int(val_y.max()))
                    kappa = cohen_kappa_score(val_y.astype(int), predictions, weights='quadratic')
                    cv_scores[params].append(kappa)
    
    best_params = max(cv_scores, key=lambda p: np.mean(cv_scores[p]))

    start_time = time.time()

    model = RandomForestClassifier(n_estimators=best_params[0], max_depth=best_params[1], min_samples_split=best_params[2])
    model.fit(train_predictors, train_target)

    y_true = test_target.astype(int)

    predictions = model.predict(test_predictors)
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
        'model': 'Random Forest StratKFold',
        'parameters': f'n_estimators={best_params[0]}, max_depth={best_params[1]}, min_samples_split={best_params[2]}',
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'f1_score': f1,
        'accuracy_score': accuracy,
        'within_1_accuracy': w1acc,
        'cohen_kappa': cohen_kappa,
        'execution_time_seconds': end_time - start_time
    }
    rows.append(return_object)

    plot_CM(forwhom=f'CM_Random_Forest_StratKFold_{data_list["name"]}', true=y_true, pred=pred_classes, smote=smote)
    return rows











