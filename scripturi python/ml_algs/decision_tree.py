from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error, accuracy_score

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

def grid_metrics_decision_tree(data_list: list):
    rows = []
    for dimension_reduction in data_list:
        for max_depth in decision_tree_parameters['max_depth']:
            for min_samples_split in decision_tree_parameters['min_samples_split']:
                model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)

                model.fit(dimension_reduction['Train_Predictors'], dimension_reduction['Train_Target'])

                predictions = model.predict(dimension_reduction['Test_Predictors'])

                mse = mean_squared_error(dimension_reduction['Test_Target'], predictions)
                mae = mean_absolute_error(dimension_reduction['Test_Target'], predictions)
                f1 = f1_score(dimension_reduction['Test_Target'], predictions, average='weighted')
                accuracy = accuracy_score(dimension_reduction['Test_Target'], predictions)


                return_object = {
                    'dimension_reduction_type': dimension_reduction['name'],
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'model': 'Decision Tree',
                    'scores': {
                        'mean_squared_error': mse,
                        'mean_absolute_error': mae,
                        'f1_score': f1,
                        'accuracy_score': accuracy
                    }
                }

                rows.append(return_object)


    return rows






