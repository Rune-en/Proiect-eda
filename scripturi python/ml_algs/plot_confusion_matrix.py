import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, cohen_kappa_score
import numpy as np

def within_1_acc(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) <= 1)

def plot_CM(forwhom,true=0, pred=0, smote=False):

    cm = confusion_matrix(true, pred, labels=[0, 1, 2, 3, 4, 5])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4, 5])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False,im_kw={"vmin": 0, "vmax": 15})

    if smote:
        plt.title(f'SMOTE - {forwhom}')
    else:
        plt.title(f'{forwhom}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    acc = accuracy_score(true, pred)
    w1acc = within_1_acc(true, pred)

    mae = np.mean(np.abs(true - pred))
    mse = np.mean((true - pred) ** 2)

    plt.figtext(0.01, 0.05, f'Accuracy: {acc:.4f}', ha='left', va='bottom')
    plt.figtext(0.01, 0.01, f'Within 1 Accuracy: {w1acc:.4f}', ha='left', va='bottom')
    
    plt.figtext(0.99, 0.09, f'MAE: {mae:.4f}', ha='right', va='bottom')
    plt.figtext(0.99, 0.05, f'MSE: {mse:.4f}', ha='right', va='bottom')
    cohen_kappa = cohen_kappa_score(true, pred, weights='quadratic')
    plt.figtext(0.99, 0.01, f'Cohen Kappa: {cohen_kappa:.4f}', ha='right', va='bottom')

    cbar = plt.colorbar(disp.im_, ax=ax, shrink=0.7)
    plt.tight_layout()
    if smote:
        plt.savefig(f'SMOTE/{forwhom}_SMOTE.png')
    else:
        plt.savefig(f'NoSMOTE/{forwhom}.png')
    plt.close()

