import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from scipy.stats import ttest_rel
from matplotlib.patches import Patch
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


df = pd.read_csv('all_all_results.csv')


#########################################################################################################################################
# Plot with SMOTE and StratKFold, with alpha based on whether StratKFold is used
#########################################################################################################################################

df["log_Time"] = np.log10(df["execution_time_seconds"])

# base model names
df["base_model"] = df["model"].str.replace(" StratKFold", "", regex=False)


cmap = plt.get_cmap('tab10')
base_models = df['base_model'].unique()

base_colors = {
    "Naive Model": "black",
    "Decision Tree": "maroon",
    "Random Forest": "goldenrod",
    "Linear Regression": "green",
    "Poisson Regression": "blue",

}
marker_map = {'No': 'o', 'Yes': 'X'}

fig, ax = plt.subplots(figsize=(9, 7))

for model in df['model'].unique():
    for smote in ['No', 'Yes']:

        subset = df[(df['model'] == model) & (df['SMOTE'] == smote)]

        base_model = model.replace(" StratKFold", "")
        base_color = base_colors[base_model]

        alpha = 0.3 if "StratKFold" in model else 0.9
        color = to_rgba(base_color, alpha=alpha)

        ax.scatter(subset['cohen_kappa'], subset['log_Time'], color=color, marker=marker_map[smote], edgecolor='k', s=100)

model_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=base_colors[m], markeredgecolor='k', markersize=10, label=m) 
                 for m in base_models]

leg = ax.legend(handles=model_handles, title='Model', loc='upper right', bbox_to_anchor=(1.35, 1))
ax.add_artist(leg)

smote_handles = [plt.Line2D([0], [0], marker=marker_map[s], color='gray', linestyle='', markersize=10, label=f'SMOTE {s}') 
                 for s in ['No', 'Yes']]
leg2 = ax.legend(handles=smote_handles, title='SMOTE', loc='upper right', bbox_to_anchor=(1.31, 0.75))
ax.add_artist(leg2)

StratK_handles = [
    plt.Line2D([0], [0], marker='o', color=to_rgba('darkgray', alpha=0.9), linestyle='', markersize=10, label='No'),
    plt.Line2D([0], [0], marker='o', color=to_rgba('darkgray', alpha=0.3), linestyle='', markersize=10, label='Yes')
]
leg3 = ax.legend(handles=StratK_handles, title='StratKFold', loc='upper right', bbox_to_anchor=(1.27, 0.6))
ax.add_artist(leg3)

ax.set_xlabel('Cohen Kappa Score')
ax.set_ylabel('Log10 Execution Time (seconds)')

plt.title('Cohen Kappa Score vs Log10 Execution Time')
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(right=0.75)
plt.savefig('cohen_kappa_vs_execution_time.png', dpi=300)
plt.close()

#########################################################################################################################################
# Plot with dimension reduction, with color representing n_features using viridis
#########################################################################################################################################

df["log_n_features"] = np.log10(df["n_features"])

# normalize n_features to viridis colormap range with log scale
from matplotlib.colors import LogNorm
cmap = plt.get_cmap('viridis')
norm = LogNorm(vmin=df["n_features"].min(), vmax=df["n_features"].max())

dim_reductions = df["Dimension Reduction"].unique()
marker_map_dim = ['o', '*', 's', 'v']
marker_map_dim = dict(zip(dim_reductions, marker_map_dim))
fig, ax = plt.subplots(figsize=(9, 7))

for dim in dim_reductions:
    subset = df[df["Dimension Reduction"] == dim]
    colors = cmap(norm(subset["n_features"]))
    ax.scatter(subset["cohen_kappa"], subset["log_Time"], color=colors, marker=marker_map_dim[dim], edgecolor='k', s=100, alpha=0.7)

dim_handles = [plt.Line2D([0], [0], marker=marker_map_dim[d], color='w', markerfacecolor='gray', markeredgecolor='k', linestyle='', markersize=10, label=d)
               for d in dim_reductions]

leg1 = ax.legend(handles=dim_handles, title='Dimension Reduction', loc='upper right', bbox_to_anchor=(1.30, 1))
ax.add_artist(leg1)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label('Number of Features')

ax.set_xlabel('Cohen Kappa Score')
ax.set_ylabel('Log10 Execution Time (seconds)')

plt.title('Cohen Kappa Score vs Log10 Execution Time')
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(right=0.9)
cbar.ax.set_position([0.85, 0.20, 0.025, 0.50])


plt.savefig('cohen_kappa_vs_execution_time_dimred_ml_features.png', dpi=300)


#####################################################################################################
# plot SMOTE Vs No SMOTE boxplots

df["Model"] = df["model"].str.replace(" StratKFold", "", regex=False)

pair_cols = ["Model", "Dimension Reduction", "n_features", "StratKFold"]

paired = (df.pivot_table(index=pair_cols,columns="SMOTE",values="cohen_kappa",aggfunc="mean")
          .reset_index()
          .dropna(subset=["Yes", "No"]))

results = []

for model in paired["Model"].unique():
    sub = paired[paired["Model"] == model]

    stat, p = ttest_rel(sub["Yes"], sub["No"])

    results.append({
        "Model": model,
        "Mean No SMOTE": sub["No"].mean(),
        "Mean SMOTE": sub["Yes"].mean(),
        "Mean Difference": (sub["Yes"] - sub["No"]).mean(),
        "T-Statistic": stat,
        "P-Value": p,
        "N pairs": len(sub)
    })

# all models together
stat, p = ttest_rel(paired["Yes"], paired["No"])

results.append({
    "Model": "All Models",
    "Mean No SMOTE": paired["No"].mean(),
    "Mean SMOTE": paired["Yes"].mean(),
    "Mean Difference": (paired["Yes"] - paired["No"]).mean(),
    "T-Statistic": stat,
    "P-Value": p,
    "N pairs": len(paired)
})

results_df = pd.DataFrame(results)
results_df.to_csv("smote_vs_no_smote_ttest_results.csv", index=False)

plot_df = results_df[results_df["Model"] != "All Models"].copy()

order = ["Naive Model", "Decision Tree", "Random Forest", "Linear Regression", "Poisson Regression"]
plot_df["Model"] = pd.Categorical(plot_df["Model"], categories=order, ordered=True)
plot_df = plot_df.sort_values("Model").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 5))

data = []
positions = []

for i, model in enumerate(order):
    sub = paired[paired["Model"] == model]
    data.extend([sub["No"], sub["Yes"]])
    positions.extend([i * 3 + 1, i * 3 + 2])

bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True)

for i, box in enumerate(bp["boxes"]):
    box.set_facecolor("paleturquoise" if i % 2 == 0 else "cornflowerblue")

ax.set_xticks([i * 3 + 1.5 for i in range(len(order))])
ax.set_xticklabels(order, rotation=45, ha="right")

ax.axhline(0, color="gray", linestyle="--")
ax.set_ylabel("Cohen Kappa")
ax.set_title("SMOTE vs No SMOTE by Model")

ax.legend(handles=[Patch(facecolor="paleturquoise", label="No SMOTE"), Patch(facecolor="cornflowerblue", label="SMOTE")])

for i, row in plot_df.iterrows():
    xpos = i * 3 + 1.5
    sub = paired[paired["Model"] == row["Model"]]
    y = max(sub["No"].max(), sub["Yes"].max())
    ax.text(xpos, y + 0.01, f"p={row['P-Value']:.3f}", ha="center", va="bottom", fontsize=9)

ax.grid(axis="y", alpha=0.7)
plt.ylim(top=0.9) 
plt.tight_layout()
plt.savefig("smote_vs_no_smote_grouped_boxplot.png", dpi=300)
plt.close()

###########################################################################################################################
# plot StratKFold Vs No StratKFold boxplots


df["Model"] = df["model"].str.replace(" StratKFold", "", regex=False)

pair_cols = ["Model", "Dimension Reduction", "n_features", "SMOTE"]

paired = (df.pivot_table(index=pair_cols, columns="StratKFold", values="cohen_kappa", aggfunc="mean")
          .reset_index()
          .dropna(subset=["Yes", "No"]))

results = []

for model in paired["Model"].unique():
    sub = paired[paired["Model"] == model]
    stat, p = ttest_rel(sub["Yes"], sub["No"])

    results.append({"Model": model, "Mean No StratKFold": sub["No"].mean(), "Mean StratKFold": sub["Yes"].mean(),
                    "Mean Difference": (sub["Yes"] - sub["No"]).mean(), "T-Statistic": stat, "P-Value": p, "N pairs": len(sub)})

stat, p = ttest_rel(paired["Yes"], paired["No"])

results.append({"Model": "All Models", "Mean No StratKFold": paired["No"].mean(), "Mean StratKFold": paired["Yes"].mean(),
                "Mean Difference": (paired["Yes"] - paired["No"]).mean(), "T-Statistic": stat, "P-Value": p, "N pairs": len(paired)})

results_df = pd.DataFrame(results)
results_df.to_csv("stratkfold_vs_no_stratkfold_ttest_results.csv", index=False)

plot_df = results_df[results_df["Model"] != "All Models"].copy()

order = ["Decision Tree", "Random Forest", "Poisson Regression"]
plot_df["Model"] = pd.Categorical(plot_df["Model"], categories=order, ordered=True)
plot_df = plot_df.sort_values("Model").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(5, 5))

data = []
positions = []

for i, model in enumerate(order):
    sub = paired[paired["Model"] == model]
    data.extend([sub["No"], sub["Yes"]])
    positions.extend([i * 3 + 1, i * 3 + 2])

bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True)

for i, box in enumerate(bp["boxes"]):
    box.set_facecolor("lightsteelblue" if i % 2 == 0 else "blueviolet")

ax.set_xticks([i * 3 + 1.5 for i in range(len(order))])
ax.set_xticklabels(order, rotation=45, ha="right")

ax.axhline(0, color="gray", linestyle="--")
ax.set_ylabel("Cohen Kappa")
ax.set_title("StratKFold vs No StratKFold by Model")

ax.legend(handles=[Patch(facecolor="lightsteelblue", label="No StratKFold"), Patch(facecolor="blueviolet", label="StratKFold")])

for i, row in plot_df.iterrows():
    xpos = i * 3 + 1.5
    sub = paired[paired["Model"] == row["Model"]]
    y = max(sub["No"].max(), sub["Yes"].max())
    ax.text(xpos, y + 0.01, f"p={row['P-Value']:.3f}", ha="center", va="bottom", fontsize=9)

ax.grid(axis="y", alpha=0.7)
plt.tight_layout()
plt.ylim(top=0.9) 
plt.savefig("stratkfold_vs_no_stratkfold_grouped_boxplot.png", dpi=300)
plt.close()

######################################################################################################################
#plot boxplots number of features vs cohen kappa


df_anova = df[df["Model"] != "Naive Model"].copy()

groups = [g["cohen_kappa"].values for _, g in df_anova.groupby("dimension_reduction_type")]

f_stat, p_value = f_oneway(*groups)

print(f"ANOVA F = {f_stat:.3f}")
print(f"ANOVA p = {p_value:.6f}")

tukey = pairwise_tukeyhsd(
    endog=df_anova["cohen_kappa"],
    groups=df_anova["dimension_reduction_type"],
    alpha=0.05
)

print(tukey)
results_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
results_df.to_csv("dimension_reduction_tukey_hsd_results.csv", index=False)

order = ["All Features", "PCA50", "PCA100", "PCA300", 
            "Lasso_0.01", "Lasso_0.1", "Lasso_0.8",
            "Ridge50", "Ridge100", "Ridge300"]
data = [df_anova.loc[df_anova["dimension_reduction_type"] == dr, "cohen_kappa"] for dr in order]

fig, ax = plt.subplots(figsize=(8, 5))

global_median = df_anova["cohen_kappa"].median()
global_25th = df_anova["cohen_kappa"].quantile(0.25)
global_75th = df_anova["cohen_kappa"].quantile(0.75)
ax.axhline(global_median, color="red", linestyle="-", label=f"Global Median: {global_median:.3f}", alpha=0.2)
ax.axhline(global_25th, color="red", linestyle="--", label=f"Global 25th Percentile: {global_25th:.3f}", alpha=0.2)
ax.axhline(global_75th, color="red", linestyle="--", label=f"Global 75th Percentile: {global_75th:.3f}", alpha=0.2)
#fill between 25th and 75th percentile
ax.fill_between(x=[0, len(order) + 1], y1=global_25th, y2=global_75th, color="red", alpha=0.05)
bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="red", linewidth=2), 
                boxprops=dict(facecolor="lightblue", color="black", linewidth=1.5),
                whiskerprops=dict(color="black", linewidth=1.5), capprops=dict(color="black", linewidth=1.5))

ax.set_xticklabels(order, rotation=45, ha="right")
ax.set_ylabel("Cohen Kappa")
ax.set_xlabel("Dimension Reduction")
ax.set_title(f"Dimension Reduction Methods\nANOVA p = {p_value:.3g}")

ax.grid(axis="y", alpha=0.7)
plt.ylim(bottom =0)

plt.tight_layout()
plt.savefig("dimension_reduction_anova.png", dpi=300)
plt.close()

######################################################################################################################
# Pivot table: Dimension Reduction x ML Models for Cohen Kappa, Within1 Acc, and MAE

df_pivot = df[df["model"] != "Naive Model"].copy()
df_pivot = df_pivot[df_pivot["SMOTE"] == "Yes"]
df_pivot = df_pivot[df_pivot["StratKFold"] == "Yes"].copy()
print(df_pivot["dimension_reduction_type"].unique())
df_pivot = pd.concat([df_pivot, df[df["model"] == "Linear Regression"]], ignore_index=True)

# Create pivot tables for each metric
pivot_cohen_kappa = df_pivot.pivot_table(
    index="dimension_reduction_type",
    columns="model",
    values="cohen_kappa",
    aggfunc="mean"
)

pivot_within1_acc = df_pivot.pivot_table(
    index="dimension_reduction_type",
    columns="model",
    values="within_1_accuracy",
    aggfunc="mean"
)

pivot_mae = df_pivot.pivot_table(
    index="dimension_reduction_type",
    columns="model",
    values="mean_absolute_error",
    aggfunc="mean"
)

# Define dimension reduction order
dim_order = ["All Features", "PCA50", "PCA100", "PCA300", 
             "Lasso_0.01", "Lasso_0.1", "Lasso_0.8",
             "Ridge50", "Ridge100", "Ridge300"]

# Reindex to specified order
pivot_cohen_kappa = pivot_cohen_kappa.reindex([d for d in dim_order if d in pivot_cohen_kappa.index])
pivot_within1_acc = pivot_within1_acc.reindex([d for d in dim_order if d in pivot_within1_acc.index])
pivot_mae = pivot_mae.reindex([d for d in dim_order if d in pivot_mae.index])

# Save to CSV
pivot_cohen_kappa.to_csv("pivot_cohen_kappa.csv")
pivot_within1_acc.to_csv("pivot_within1_acc.csv")
pivot_mae.to_csv("pivot_mae.csv")

#merge them in a single csv
pivot_cohen_kappa = pivot_cohen_kappa.add_suffix("_cohen_kappa")
pivot_within1_acc = pivot_within1_acc.add_suffix("_within1_acc")
pivot_mae = pivot_mae.add_suffix("_mae")

merged_pivot = pd.concat([pivot_cohen_kappa, pivot_within1_acc, pivot_mae], axis=1)
merged_pivot.sort_index(axis=1, inplace=True)
merged_pivot.to_csv("merged_pivot.csv")
