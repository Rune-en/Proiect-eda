import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error, accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
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


def poisson_linear_model_StratKFold(data_list: dict, smote=False):
    rows = []
    train_predictors = np.asarray(data_list['Train_Predictors'])
    train_target = np.asarray(data_list['Train_Target'])
    test_predictors = np.asarray(data_list['Test_Predictors'])
    test_target = np.asarray(data_list['Test_Target'])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = {alpha: [] for alpha in poisson_linear_model_parameters['alpha']}

    for train_index, val_index in skf.split(train_predictors, train_target):
        train_X, val_X = train_predictors[train_index], train_predictors[val_index]
        train_y, val_y = train_target[train_index], train_target[val_index]

        for alpha in poisson_linear_model_parameters['alpha']:
            model = make_pipeline(
                StandardScaler(),
                PoissonRegressor(alpha=alpha, max_iter=2000)
            )
            model.fit(train_X, train_y)

            predictions = model.predict(val_X)
            pred_classes = np.rint(predictions).astype(int)
            pred_classes = np.clip(pred_classes, int(val_y.min()), int(val_y.max()))

            kappa = cohen_kappa_score(val_y.astype(int), pred_classes, weights='quadratic')
            cv_scores[alpha].append(kappa)

    best_alpha = max(cv_scores, key=lambda alpha: np.mean(cv_scores[alpha]))

    start_time = time.time()

    model = make_pipeline(
        StandardScaler(),
        PoissonRegressor(alpha=best_alpha, max_iter=2000))
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
        'model': 'Poisson Regression StratKFold',
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

    plot_CM(forwhom=f'CM_Poisson_Regression_StratKFold_{data_list["name"]}', true=y_true, pred=pred_classes, smote=smote)

    return rows