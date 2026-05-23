import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

from sklearn.decomposition import PCA

from dimension_reduction import *
<<<<<<< HEAD
from ml_algs import *
from dimension_reduction import *
import data
=======
from ml_algs.decision_tree import grid_metrics_decision_tree
from ml_algs.random_forest import grid_metrics_random_forest
from ml_algs.linear_model import grid_metrics_linear_model
from ml_algs.poisson_linear_model import grid_metrics_poisson_linear_model  

from dimension_reduction. lasso import reduce_dimensionality_lasso
from data import obtine_date_procesat
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

>>>>>>> bbbb1e8 (feat: add PCA/Lasso modeling pipeline and improve regression evaluation)
import pandas as pd
import numpy as np


<<<<<<< HEAD
dataset = data.obtine_date_procesate()
dataset.T.to_csv("processed_data.csv", index=False)
dataset = dataset.to_numpy().astype(float)
print(dataset)
print(dataset.shape)

=======
df = obtine_date_procesat()
print(df.head())

train, test = train_test_split(df, test_size=0.2, random_state=42)
X_train = train.drop(columns=["Gleason Group"])
y_train = train["Gleason Group"]
X_test = test.drop(columns=["Gleason Group"])
y_test = test["Gleason Group"]
>>>>>>> bbbb1e8 (feat: add PCA/Lasso modeling pipeline and improve regression evaluation)



pca = PCA(n_components=0.999)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("PCA - explained variance ratio:", pca.explained_variance_ratio_)

data = {
    'name': 'PCA',
    'Train_Predictors': X_train_pca,
    'Train_Target': y_train,
    'Test_Predictors': X_test_pca,
    'Test_Target': y_test
    }

poisson = grid_metrics_poisson_linear_model(data_list=data)
poisson_df = pd.DataFrame(poisson)
poisson_df.to_csv("poisson_linear_model_results.csv", index=False)

linear = grid_metrics_linear_model(data_list=data)
linear_df = pd.DataFrame(linear)
linear_df.to_csv("linear_model_results.csv", index=False)

lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)
lasso.fit(X_train, y_train)

selected_features = np.where(lasso.coef_ != 0)[0]

data = {
    'name': 'Lasso',
    'Train_Predictors': X_train.values[:, selected_features],
    'Train_Target': y_train,
    'Test_Predictors': X_test.values[:, selected_features],
    'Test_Target': y_test
    }

lasso_results = grid_metrics_poisson_linear_model(data_list=data)
lasso_df = pd.DataFrame(lasso_results)
lasso_df.to_csv("lasso_results.csv", index=False)

linear_results = grid_metrics_linear_model(data_list=data)
linear_df = pd.DataFrame(linear_results)
linear_df.to_csv("linear_results.csv", index=False)
