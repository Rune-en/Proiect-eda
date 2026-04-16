random_forest_parameters = {
    'n_estimators': [10, 20, 50, 100],
    'max_depth': [10, 20, 30, None]
}

from sklearn.linear_model import RandomForestRegressor
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error, accuracy_score

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

def grid_metrics_random_forest_model(data_list: list):
    rows = []
    for dimension_reduction in data_list:
            for n_estimators in random_forest_parameters['n_estimators']:
                for max_depth in random_forest_parameters['max_depth']:
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

                    model.fit(dimension_reduction['Train_Predictors'], dimension_reduction['Train_Target'])

                    predictions = model.predict(dimension_reduction['Test_Predictors'])

                    mse = mean_squared_error(dimension_reduction['Test_Target'], predictions)
                    mae = mean_absolute_error(dimension_reduction['Test_Target'], predictions)
                    f1 = f1_score(dimension_reduction['Test_Target'], predictions, average='weighted')
                    accuracy = accuracy_score(dimension_reduction['Test_Target'], predictions)


                    return_object = {
                        'dimension_reduction_type': dimension_reduction['name'],
                        'model': 'Random Forest',
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'scores': {
                            'mean_squared_error': mse,
                            'mean_absolute_error': mae,
                            'f1_score': f1,
                            'accuracy_score': accuracy
                        }
                    }

                    rows.append(return_object)


    return rows











