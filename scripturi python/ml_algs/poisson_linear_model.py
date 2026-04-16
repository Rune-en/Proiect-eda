poisson_linear_model_parameters = {
    'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
}

from sklearn.linear_model import PoissonRegressor
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

def grid_metrics_linear_model(data_list: list):
    rows = []
    for dimension_reduction in data_list:
        for alpha in poisson_linear_model_parameters['alpha']:
                model = PoissonRegressor(alpha=alpha)

                model.fit(dimension_reduction['Train_Predictors'], dimension_reduction['Train_Target'])

                predictions = model.predict(dimension_reduction['Test_Predictors'])

                mse = mean_squared_error(dimension_reduction['Test_Target'], predictions)
                mae = mean_absolute_error(dimension_reduction['Test_Target'], predictions)
                f1 = f1_score(dimension_reduction['Test_Target'], predictions, average='weighted')
                accuracy = accuracy_score(dimension_reduction['Test_Target'], predictions)


                return_object = {
                    'dimension_reduction_type': dimension_reduction['name'],
                    'model': 'Poisson Regression',
                    'alpha': alpha,
                    'scores': {
                        'mean_squared_error': mse,
                        'mean_absolute_error': mae,
                        'f1_score': f1,
                        'accuracy_score': accuracy
                    }
                }

                rows.append(return_object)


    return rows






