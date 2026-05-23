import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


def reduce_dimensionality_lasso(X, y, alpha=0.01, max_iter=10000):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
    lasso.fit(X_scaled, y)
 
    selected_features = np.where(lasso.coef_ != 0)[0]

    
    return  selected_features
