import pandas as pd

df = pd.read_csv("DB.csv")
df.pop("gene_id")
df.pop("gene_type")
df = df.transpose()
df.columns = df.iloc[0]
df = df.drop(df.index[0])

print(df.head(5))

df=df.drop(columns=df.columns[1:29])
df.index.names = ['sample_id']
df["Gleason Group"] = df["Gleason Group"].replace({"Normal":0,"Group 1": 1, "Group 2": 2, "Group 3": 3, "Group 4": 4, "Group 5": 5})

print(df.head(5))