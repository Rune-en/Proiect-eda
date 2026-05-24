import pandas as pd
import numpy as np
from pathlib import Path


def make_unique(columns):
    counts = {}
    new_columns = []
    for col in columns:
        if col in counts:
            counts[col] += 1
            new_columns.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            new_columns.append(col)

    return new_columns

def obtine_date_procesat():
    db_path = Path(__file__).resolve().parent / "DB.csv"
    df = pd.read_csv(db_path, low_memory=False)
    df.pop("gene_id")
    df.pop("gene_type")
    df = df.transpose()
    df.columns = df.iloc[0]
    df.columns = make_unique(df.columns)
    df = df.drop(df.index[0])

    df=df.drop(columns=df.columns[1:29])
    df.index.names = ['sample_id']
    gleason_group_map = {"Normal": 0, "Group 1": 1, "Group 2": 2, "Group 3": 3, "Group 4": 4, "Group 5": 5}


    a = df["Gleason Group"].map(gleason_group_map)

    print("Data shape before dropping low variability columns:", df.shape)
    #drop rows with little variablility
    df = df.loc[:, df.nunique() > 200]
    df = df.apply(pd.to_numeric, errors='coerce')
    df["Gleason Group"] = a
    print("Data shape after dropping low variability columns:", df.shape)
    df = df.dropna(axis=0, how="any")
    df = df.astype(float)
    return df
