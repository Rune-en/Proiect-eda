import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_algs.naive_model import grid_metrics_naive_model
from ml_algs.decision_tree import grid_metrics_decision_tree
from ml_algs.random_forest import grid_metrics_random_forest
from ml_algs.linear_model import grid_metrics_linear_model
from ml_algs.poisson_linear_model import grid_metrics_poisson_linear_model  
from data import obtine_date_procesat
from imblearn.over_sampling import SMOTE

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import os

# resampling with SMOTE
smote = False

if not os.path.exists('SMOTE'):
    os.makedirs('SMOTE')
if not os.path.exists('NoSMOTE'):
    os.makedirs('NoSMOTE')

df = obtine_date_procesat()
print(df.head())

train, test = train_test_split(df, test_size=0.2, random_state=42)

X_train = train.drop(columns=["Gleason Group"])
y_train = train["Gleason Group"]
X_test = test.drop(columns=["Gleason Group"])
y_test = test["Gleason Group"]

if smote:
    print("before SMOTE, class distribution in training set:")
    print(pd.Series(y_train).value_counts())
    SMOTE = SMOTE(random_state=42)
    X_train, y_train = SMOTE.fit_resample(X_train, y_train)
    print("After SMOTE, class distribution in training set:")
    print(pd.Series(y_train).value_counts())


all_features = {
    'name': 'All Features',
    'Train_Predictors': X_train.values,
    'Train_Target': y_train,
    'Test_Predictors': X_test.values,
    'Test_Target': y_test
    }
#############################   PCA dimensionality reduction   #############################

pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


"""X_train_pca = pca.fit_transform(StandardScaler().fit_transform(X_train))
X_test_pca = pca.transform(StandardScaler().fit_transform(X_test))"""

plt.figure(figsize=(5, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance Ratio')
plt.grid()
plt.savefig('pca_explained_variance_ratio.png')
plt.close()


data_pca1 = {
    'name': 'PCA50',
    'Train_Predictors': X_train_pca,
    'Train_Target': y_train,
    'Test_Predictors': X_test_pca,
    'Test_Target': y_test
    }

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

data_pca2 = {
    'name': 'PCA100',
    'Train_Predictors': X_train_pca,
    'Train_Target': y_train,
    'Test_Predictors': X_test_pca,
    'Test_Target': y_test
    }

pca = PCA(n_components=300)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

data_pca3 = {
    'name': 'PCA300',
    'Train_Predictors': X_train_pca,
    'Train_Target': y_train,
    'Test_Predictors': X_test_pca,
    'Test_Target': y_test
    }

#############################   Lasso dimensionality reduction   #############################

lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)
lasso.fit(X_train, y_train)
#lasso.fit(StandardScaler().fit_transform(X_train), y_train)

selected_features = np.where(lasso.coef_ != 0)[0]

data_lasso1 = {
    'name': 'Lasso_0.01',
    'Train_Predictors': X_train.values[:, selected_features],
    'Train_Target': y_train,
    'Test_Predictors': X_test.values[:, selected_features],
    'Test_Target': y_test
    }

lasso = Lasso(alpha=0.1, max_iter=10000, random_state=42)
lasso.fit(X_train, y_train)
#lasso.fit(StandardScaler().fit_transform(X_train), y_train)

selected_features = np.where(lasso.coef_ != 0)[0]

data_lasso2 = {
    'name': 'Lasso_0.1',
    'Train_Predictors': X_train.values[:, selected_features],
    'Train_Target': y_train,
    'Test_Predictors': X_test.values[:, selected_features],
    'Test_Target': y_test
    }

lasso = Lasso(alpha=1, max_iter=10000, random_state=42)
lasso.fit(X_train, y_train)
#lasso.fit(StandardScaler().fit_transform(X_train), y_train)

selected_features = np.where(lasso.coef_ != 0)[0]

data_lasso3 = {
    'name': 'Lasso_1',
    'Train_Predictors': X_train.values[:, selected_features],
    'Train_Target': y_train,
    'Test_Predictors': X_test.values[:, selected_features],
    'Test_Target': y_test
    }

#############################   Ridge dimensionality reduction   #############################

#model = Ridge(alpha=1.0)
# here it's probably better to use a pipeline with standardization, since Ridge is sensitive to feature scales
model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

model.fit(X_train, y_train)
predictions = model.predict(X_train)

#coefs = ridge.coef_
coefs = model.named_steps['ridge'].coef_

top_idx = np.argsort(np.abs(coefs))[::-1][:50]
selected = top_idx

data_ridge1 = {
    'name': 'Ridge50',
    'Train_Predictors': X_train.values[:, selected],
    'Train_Target': y_train,
    'Test_Predictors': X_test.values[:, selected],
    'Test_Target': y_test
    }

top_idx = np.argsort(np.abs(coefs))[::-1][:100]
selected = top_idx

data_ridge2 = {
    'name': 'Ridge100',
    'Train_Predictors': X_train.values[:, selected],
    'Train_Target': y_train,
    'Test_Predictors': X_test.values[:, selected],
    'Test_Target': y_test
    }

top_idx = np.argsort(np.abs(coefs))[::-1][:300]
selected = top_idx

data_ridge3 = {
    'name': 'Ridge300',
    'Train_Predictors': X_train.values[:, selected],
    'Train_Target': y_train,
    'Test_Predictors': X_test.values[:, selected],
    'Test_Target': y_test
    }

###############################   Grid Search for each model   #############################
all_results = []
for data in [data_pca1, data_pca2, data_pca3, data_lasso1, data_lasso2, data_lasso3, data_ridge1, data_ridge2, data_ridge3, all_features]:
    for model in [grid_metrics_naive_model, grid_metrics_linear_model, grid_metrics_poisson_linear_model,
                  grid_metrics_decision_tree, grid_metrics_random_forest]:
        start_time = time.time()
        results = model(data_list=data, smote=smote)
        df_results = pd.DataFrame(results)
        #df_results.to_csv(f"{data['name']}_{model.__name__}_results.csv", index=False)
        all_results.extend(results)
        end_time = time.time()
        print(f"{model.__name__[13:]} on {data['name']}: {end_time - start_time:.2f} seconds")

df_all_results = pd.DataFrame(all_results)
if smote:
    df_all_results.to_csv("SMOTE/all_results_SMOTE.csv", index=False)
else:
    df_all_results.to_csv("NoSMOTE/all_results.csv", index=False)