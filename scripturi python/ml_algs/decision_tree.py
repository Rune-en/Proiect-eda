from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error, accuracy_score

decision_tree_parameters = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}



'''
input data list sa contina obiecte de tipul:
{
    'name': -
    'data': -
}

'''

def grid_metrics_decision_tree(input_data_list: list, Y_train):
    rows = []
    for dimension_reduction in input_data_list:
        for max_depth in decision_tree_parameters['max_depth']:
            for min_samples_split in decision_tree_parameters['min_samples_split']:
                model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)

                return_object = {
                    'dimension_reduction_type': dimension_reduction['name'],
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'model': 'Decision Tree',
                    'scores': {
                        'mean_squared_error': ,
                        'mean_absolute_error': ,
                        ''
                    }
                }



