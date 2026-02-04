# parkinson_project_optimised
# Author: Amir mahdi Taghizadeh


# imports

import os
import pandas as pd
import re
import io
import scanpy as sc
import gseapy as gp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
import random

# step 0: directories

base_dir = r"C:\Users\Asus\Desktop\projects\parkinson_project_optimised"
raw_dir = os.path.join(base_dir, 'data', 'raw')
processed_dir = os.path.join(base_dir, 'data', 'processed')
results_dir = os.path.join(base_dir, 'results')
figures_dir = os.path.join(base_dir, 'figures')


# step 1: loading df

df_path = os.path.join(raw_dir, 'GSE20295_series_matrix.txt')
df = pd.read_csv(df_path, sep='\t', comment='!', index_col=0)
df = df.drop_duplicates()
df = df.dropna()

print('Done with loading df. ')


# step 2: metadata

meta_lines = []
with open(df_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith('!Sample_characteristics_ch1'):
            meta_lines.append(line.strip('\n'))

disease_lines = [line for line in meta_lines if 'disease state' in line.lower()][0]
statuses = re.findall('"(.*?)"', disease_lines)

disease_labels = [
    'PD' if "parkinsons disease" in s.lower() or "parkinson's disease" in s.lower()
    else 'Control'
    for s in statuses
]

print('Done with metadata. ')


# step 3: anndata object

adata = sc.AnnData(df.T)
adata.var_names_make_unique()
adata.obs['disease state'] = disease_labels
sc.pp.calculate_qc_metrics(adata, inplace=True)

print('Done with anndata. ')


# step 4: normalization

sc.pp.normalize_total(adata, target_sum=1e4)
adata.write(os.path.join(processed_dir, 'rna_seq_pd_initial_anndata.h5ad'))
adata.raw = adata.copy()

print('Done with normalization. ')


# step 5: DEA

sc.tl.rank_genes_groups(adata, groupby='disease state', method='wilcoxon')

deg = sc.get.rank_genes_groups_df(adata, group='PD')
deg_sig = deg[(abs(deg['logfoldchanges']) > 1) & (deg['pvals_adj'] < 0.05)]
deg_path = os.path.join(results_dir, 'deg_all.csv')
deg.to_csv(deg_path, index=False)
deg_sig_path = os.path.join(results_dir, 'deg_sig.csv')
deg_sig.to_csv(deg_path, index=False)

print('Done with DEG. ')


# step 6: Annotation

annot_file = os.path.join(raw_dir, 'GPL96.annot')

table_lines = []
with open(annot_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        split_lines = line.strip().split('\t')
        if len(split_lines) > 1:
            table_lines.append(split_lines)

header = table_lines[0]
data = table_lines[1:]
annot_file = pd.DataFrame(data, columns=header)

prob_col = [c for c in annot_file.columns if 'id' in c.lower()][0]
symbol_col = [c for c in annot_file.columns if 'gene symbol' in c.lower()][0]

annot_file = annot_file[[prob_col ,symbol_col]].rename(columns={prob_col: 'ID', symbol_col: 'Gene'})
annot_file = annot_file[annot_file['Gene'].notna() & (annot_file['Gene'] != '')]

pd_deg_annot = deg.merge(annot_file[['ID', 'Gene']], left_on='names', right_on='ID', how='left')
pd_deg_annot = pd_deg_annot.drop(columns=['ID', 'names'])
col = ['Gene'] + [c for c in pd_deg_annot.columns if c != 'Gene']
pd_deg_annot = pd_deg_annot[col]
pd_deg_annot_path = os.path.join(results_dir, 'pd_deg_annotation.csv')
pd_deg_annot.to_csv(pd_deg_annot_path, index=False)

sig_deg_annot = deg_sig.merge(annot_file[['ID', 'Gene']], left_on='names', right_on='ID', how='left')
sig_deg_annot = sig_deg_annot.drop(columns=['ID', 'names'])
col = ['Gene'] + [c for c in sig_deg_annot.columns if c != 'Gene']
sig_deg_annot = sig_deg_annot[col]
sig_deg_annot_path = os.path.join(results_dir, 'sig_deg_annotation.csv')
sig_deg_annot.to_csv(sig_deg_annot_path, index=False)

print('Done with annotation. ')


# step 7: annot for ml

expr_path = df_path
expr_df = pd.read_csv(expr_path, sep='\t', comment='!', header=0, dtype=str)
expr_df = expr_df.rename(columns={expr_df.columns[0]: 'ID'})
expr_df['ID'] = expr_df['ID'].str.strip()

annot_file = os.path.join(raw_dir, 'GPL96.annot')

with open((annot_file), 'r', encoding='latin1') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if line.startswith('!platform_table_begin'):
        table_start = i + 1
        break
table_str = ''.join(lines[table_start:])
annot = pd.read_csv(io.StringIO(table_str), sep='\t', dtype=str)

annot['ID'] = annot['ID'].str.strip()
annot['Gene symbol'] = annot['Gene symbol'].str.strip()

annot = annot[annot['Gene symbol'].notna() & (annot['Gene symbol'] != '')]

merged_df = pd.merge(expr_df, annot[['ID','Gene symbol']], on='ID', how='inner')
sample_cols = merged_df.columns.difference(['ID', 'Gene symbol'])
merged_df[sample_cols] = merged_df[sample_cols].apply(pd.to_numeric, errors='coerce')
merged_agg = merged_df.groupby('Gene symbol')[sample_cols].mean()

print('Done with annotation for ML. ')


# step 8: creating new AnnData for ml

adata_parki = sc.AnnData(merged_agg.T)
adata_parki.obs['disease state'] = disease_labels
adata_parki.var_names_make_unique()

adata_pd = adata_parki[adata_parki.obs['disease state'].isin(['PD' , 'Control'])].copy()

sig_deg_annot['Gene'] = sig_deg_annot['Gene'].astype(str)
sig_deg_annot_exploded = sig_deg_annot.assign(Gene=sig_deg_annot['Gene'].str.split('///')).explode('Gene')
sig_deg_annot_exploded['Gene'] = sig_deg_annot_exploded['Gene'].str.strip()
sig_deg_annot_exploded = sig_deg_annot_exploded.reset_index(drop=True)

sig_deg_annot_unique = sig_deg_annot_exploded.loc[
    sig_deg_annot_exploded.groupby('Gene')['pvals_adj'].idxmin()
].reset_index(drop=True)

intersted_genes = sig_deg_annot_unique["Gene"].tolist()
genes_in_adata = [g for g in intersted_genes if g in adata_pd.var_names]
adata_selected = adata_pd[:, genes_in_adata].copy()
adata_selected.write(os.path.join(processed_dir, 'rna_seq_pd_selected_anndata.h5ad'))

print('Done with creating new AnnData for ML. ')



# step 9: ML


import numpy as np
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

adata_selected = adata_selected[:, sorted(adata_selected.var_names)]

X = adata_selected.X
y = adata_selected.obs['disease state'].map({'PD': 0, 'Control': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=SEED,
    stratify=y
)


rf = RandomForestClassifier(
    n_estimators=500,
    random_state=SEED,
    n_jobs=1,
    max_features="sqrt",
    bootstrap=True
)

rf.fit(X_train, y_train)

rf_y_pred = rf.predict(X_test)
rf_y_proba = rf.predict_proba(X_test)[:, 1]

rf_class_report = classification_report(y_test, rf_y_pred)
rf_roc_auc_score = roc_auc_score(y_test, rf_y_proba)
rf_confusion_matrix = confusion_matrix(y_test, rf_y_pred)


lr = LogisticRegression(
    penalty='l2',
    C=0.1,
    max_iter=5000,
    solver='lbfgs',
    random_state=SEED
)

lr.fit(X_train, y_train)

lr_y_pred = lr.predict(X_test)
lr_y_proba = lr.predict_proba(X_test)[:, 1]

lr_class_report = classification_report(y_test, lr_y_pred)
lr_roc_auc_score = roc_auc_score(y_test, lr_y_proba)
lr_confusion_matrix = confusion_matrix(y_test, lr_y_pred)


models_performance = os.path.join(results_dir, 'models_performance.txt')
with open(models_performance, 'w') as f:
    f.write("\n" + "="*60 + "\n")
    f.write("Random Forest\n")
    f.write("="*60 + "\n")
    f.write(f"Test ROC-AUC: {rf_roc_auc_score:.3f}\n\n")
    f.write(f'Classification Report:\n{rf_class_report}\n\n')
    f.write(f'Confusion Matrix:\n{rf_confusion_matrix}\n\n')

    f.write("\n" + "="*60 + "\n")
    f.write("Logistic Regression\n")
    f.write("="*60 + "\n")
    f.write(f"Test ROC-AUC: {lr_roc_auc_score:.3f}\n\n")
    f.write(f'Classification Report:\n{lr_class_report}\n\n')
    f.write(f'Confusion Matrix:\n{lr_confusion_matrix}\n\n')


feature_importance_rf = pd.DataFrame(
    {
        'Gene': adata_selected.var_names,
        'importance': rf.feature_importances_
    }
).sort_values('importance', ascending=False, kind='mergesort')

coefficients_lr = lr.coef_[0]
importance_lr = np.abs(coefficients_lr)

feature_importance_lr = pd.DataFrame(
    {
        'Gene': adata_selected.var_names,
        'importance': importance_lr
    }
).sort_values('importance', ascending=False, kind='mergesort')

feature_importance_rf.to_csv(
    os.path.join(results_dir, 'rf_feature_importance.csv'),
    index=False
)
feature_importance_lr.to_csv(
    os.path.join(results_dir, 'lr_feature_importance.csv'),
    index=False
)


rf_top20 = feature_importance_rf.head(20)
lr_top20 = feature_importance_lr.head(20)

overlap = sorted(
    set(rf_top20['Gene']).intersection(set(lr_top20['Gene']))
)

overlap_df = pd.DataFrame(overlap, columns=['Gene'])
overlap_df.to_csv(
    os.path.join(results_dir, 'model_genes_overlap.csv'),
    index=False
)

deg_overlap_genes = ['TAC1', 'CALM1', 'DCLK1', 'PRKACB', 'HMGN2', 'FGF13', 'SV2C']
deg_check = sig_deg_annot_unique[sig_deg_annot_unique['Gene'].isin(deg_overlap_genes)]
deg_check.to_csv(os.path.join(results_dir, 'models_overlapping_genes_expression_info.csv'), index=False)

print('Done with ML. ')


# step 10: pahtaway enrichment of final genes

final_significant_genes = deg_check['Gene'].tolist()

path_directory = os.path.join(results_dir, 'final_overlap_pathaway_enrichment')
os.makedirs(path_directory, exist_ok=True)

enrichr_libs = [
    "GO_Biological_Process_2021",
    "KEGG_2021_Human"
]
    
for l in enrichr_libs:
    gp.enrichr(
        gene_list=final_significant_genes,
        gene_sets=l,
        outdir=path_directory,
        no_plot=True,
        cutoff=0.05
    )

print('Done with pathaway enrichment. ')


# step 11: final ML(XGB)

X_final = adata_selected[:, deg_overlap_genes].X
y_final = adata_selected.obs['disease state'].map({'PD': 0, 'Control': 1})

final_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

final_model.fit(X_final, y_final)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_cv = cross_val_score(final_model, X_final, y_final, cv=5, scoring='roc_auc')
cross_validation_final = os.path.join(results_dir, 'cv_final_model.txt')

with open(cross_validation_final, 'w') as f:
    f.write('\nFinal Model ROC-AUC: \n')
    f.write(f'\n{final_cv.mean(): .3f}, ± {final_cv.std(): .3f}\n')

print('Done with final ML. ')


# step 12: external validation

## loading df
df2_path = os.path.join(raw_dir, 'GSE7621_series_matrix.txt')
df2 = pd.read_csv(df2_path, sep='\t', comment='!', index_col=0)
df2 = df2.drop_duplicates()
df2 = df2.dropna()

## metadata
meta_lines2 = []
with open(df2_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith('!Sample_characteristics_ch1'):
            meta_lines2.append(line.strip('\n'))

disease_lines2 = meta_lines2[0]
statuses2 = re.findall('"(.*?)"', disease_lines2)

disease_labels2 = [
    'PD' if "parkinson's disease" in s.lower()
    else 'Control'
    for s in statuses2
]

## annotation
expr2_path = df2_path
expr2_df = pd.read_csv(expr2_path, sep='\t', comment='!', header=0, dtype=str)
expr2_df = expr2_df.rename(columns={expr2_df.columns[0]: 'ID'})
expr2_df['ID'] = expr2_df['ID'].str.strip()

annot_file2 = os.path.join(raw_dir, 'GPL570.annot')

with open((annot_file2), 'r', encoding='latin1') as f:
    lines2 = f.readlines()
for i, line in enumerate(lines2):
    if line.startswith('!platform_table_begin'):
        table_start = i + 1
        break
table_str2 = ''.join(lines2[table_start:])
annot2 = pd.read_csv(io.StringIO(table_str2), sep='\t', dtype=str)

annot2['ID'] = annot2['ID'].str.strip()
annot2['Gene symbol'] = annot2['Gene symbol'].str.strip()

annot2 = annot2[annot2['Gene symbol'].notna() & (annot2['Gene symbol'] != '')]

merged_df2 = pd.merge(expr2_df, annot2[['ID','Gene symbol']], on='ID', how='inner')
sample_cols2 = merged_df2.columns.difference(['ID', 'Gene symbol'])
merged_df2[sample_cols2] = merged_df2[sample_cols2].apply(pd.to_numeric, errors='coerce')
merged_agg2 = merged_df2.groupby('Gene symbol')[sample_cols2].mean()

## creating AnnData
adata_valid = sc.AnnData(merged_agg2.T)
adata_valid.obs['disease state'] = disease_labels2
adata_valid.var_names_make_unique()

adata_test = adata_valid[adata_valid.obs['disease state'].isin(['PD' , 'Control'])].copy()
adata_test.obs['disease state'].value_counts()
adata_test.write(os.path.join(processed_dir, 'rna_seq_pd_test_external_anndata.h5ad'))

## final ML validation
X_train_final = adata_selected[:, deg_overlap_genes].X
y_train_final = adata_selected.obs['disease state'].map({'PD':0, 'Control':1}).values

X_test_final = adata_test[:, deg_overlap_genes].X
y_test_final = adata_test.obs['disease state'].map({'PD':0, 'Control':1}).values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

rf_clf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
rf_clf.fit(X_train_scaled, y_train_final)

y_prob_test = rf_clf.predict_proba(X_test_scaled)[:, 1]

auc_test = roc_auc_score(y_test_final, y_prob_test)

rng = np.random.RandomState(42)
boot_aucs = []

for i in range(1000):
    idx = rng.choice(len(y_test_final), len(y_test_final), replace=True)
    if len(np.unique(y_test_final[idx])) < 2:
        continue
    boot_aucs.append(
        roc_auc_score(y_test_final[idx], y_prob_test[idx])
    )

ci_low = np.percentile(boot_aucs, 2.5)
ci_high = np.percentile(boot_aucs, 97.5)
external_data_model_performance = os.path.join(results_dir, 'external_data_model_performance.txt')
with open(external_data_model_performance, 'w') as f:
    f.write(f"External ROC-AUC (RF): {auc_test:.3f}\n\n")
    f.write(f"External ROC-AUC with CI: {auc_test:.3f} (95% CI {ci_low:.3f}–{ci_high:.3f})")
   
## deg validation
genes_of_interest = deg_overlap_genes
adata_test_subset = adata_test[:, genes_of_interest].copy()

import pandas as pd
from scipy.stats import mannwhitneyu

results = []

for gene in genes_of_interest:
    expr_pd = adata_test_subset[adata_test_subset.obs['disease state']=='PD', gene].X.flatten()
    expr_ctrl = adata_test_subset[adata_test_subset.obs['disease state']=='Control', gene].X.flatten()
    
    stat, pval = mannwhitneyu(expr_pd, expr_ctrl, alternative='two-sided')
    results.append([gene, expr_pd.mean(), expr_ctrl.mean(), expr_pd.mean()-expr_ctrl.mean(), pval])
    
deg_validation = pd.DataFrame(results, columns=['Gene','PD_mean','Control_mean','LogFC','pval'])
deg_validation.to_csv(os.path.join(results_dir, 'final_deg_validation.csv'), index=False)

print('Done with external validation. ')


# step 13: visualization

## PCA & UMAP

adata.X = adata.raw.X.copy()
adata.obs['disease state'] = adata.obs['disease state'].astype('category')
adata.uns['disease state_colors'] = ['blue', 'red'] 
sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=30, svd_solver='arpack')
plt.figure(figsize=(6,5))
sc.pl.pca(adata, color='total_counts', show=False)
pca_path = os.path.join(figures_dir, 'pca_rna_seq.png')
plt.savefig(pca_path, dpi=300, bbox_inches='tight')
plt.close()

sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
sc.tl.umap(adata)
plt.figure(figsize=(6,5))
sc.pl.umap(adata, color='disease state', show=False)
umap_path = os.path.join(figures_dir, 'umap_rna_seq.png')
plt.savefig(umap_path, dpi=300, bbox_inches='tight')
plt.close()

## volcano plot

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=deg,
    x='logfoldchanges',
    y=-np.log10(deg['pvals']),
    hue=deg['pvals_adj'] < 0.05,
    palette={True: 'red', False: 'gray'},
    alpha=0.7
    )

plt.axvline(0, color="black", lw=1)
plt.xlabel("log2 Fold Change (PD vs Control)")
plt.ylabel("-log10(p-value)")
plt.title("Volcano Plot: PD vs Control")
volcano_path = os.path.join(figures_dir, 'Volcano_PD_deg.png')
plt.savefig(volcano_path, dpi=300, bbox_inches='tight')
plt.close()

## Confusion heatmap

sns.heatmap(rf_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['PD','Control'], yticklabels=['PD','Control'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('RF Confusion heatmap of PD vs Control')
confision_heatmap_rf_path = os.path.join(figures_dir, 'confusion_heatmap_rf.png')
plt.savefig(confision_heatmap_rf_path, dpi=300, bbox_inches='tight')
plt.close()

sns.heatmap(lr_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['PD','Control'], yticklabels=['PD','Control'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('LR Confusion heatmap of PD vs Control')
confision_heatmap_lr_path = os.path.join(figures_dir, 'confusion_heatmap_lr.png')
plt.savefig(confision_heatmap_lr_path, dpi=300, bbox_inches='tight')
plt.close()

## feature importance

rf_top20 = feature_importance_rf.head(20).iloc[::-1]

plt.figure(figsize=(6, 7))
plt.barh(
    rf_top20['Gene'],
    rf_top20['importance']
)
plt.xlabel("Feature Importance")
plt.title("Top 20 Features – Random Forest")
plt.tight_layout()
feature_importance_rf_path = os.path.join(figures_dir, 'feature_importance_rf.png')
plt.savefig(feature_importance_rf_path, dpi=300, bbox_inches='tight')

feature_importance_lr_plot = pd.DataFrame({
    'Gene': adata_selected.var_names,
    'Coefficient': coefficients_lr,
    'Abs_Coeff': np.abs(coefficients_lr)
}).sort_values('Abs_Coeff', ascending=False)

lr_top20 = feature_importance_lr_plot.head(20).iloc[::-1]
plt.figure(figsize=(6, 7))
plt.barh(
    lr_top20['Gene'],
    lr_top20['Coefficient']
)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel("Logistic Regression Coefficient")
plt.title("Top 20 Features – Logistic Regression")
plt.tight_layout()
feature_importance_lr_path = os.path.join(figures_dir, 'feature_importance_lr.png')
plt.savefig(feature_importance_lr_path, dpi=300, bbox_inches='tight')

## overlap venn

rf_top20_genes = set(rf_top20['Gene'])
lr_top20_genes = set(lr_top20['Gene'])
plt.figure(figsize=(8,6))
venn2([rf_top20_genes,
       lr_top20_genes], set_labels=('RF Top 20', 'LR Top 20'))
plt.figtext(0.5, -0.1, 'Overlaping features: TAC1, CALM1, DCLK1, PRKACB, HMGN2, FGF13, SV2C'
            ,wrap=True, ha='center',va='center', fontsize=14)
plt.title("Top Feature Overlap")
overlap_venn_path = os.path.join(figures_dir, 'models_overlap_venn.png')
plt.savefig(overlap_venn_path, dpi=300, bbox_inches='tight')

## roc-auc

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for train_idx, test_idx in cv.split(X_final, y_final):
    X_train, X_test = X_final[train_idx], X_final[test_idx]
    y_train, y_test = y_final.iloc[train_idx], y_final.iloc[test_idx]

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    aucs.append(roc_auc)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.figure(figsize=(6, 5))
plt.plot(
    mean_fpr,
    mean_tpr,
    color='blue',
    label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
    lw=2
)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Cross-validated ROC Curve (XGBoost)")
plt.legend(loc="lower right")
plt.tight_layout()
final_model_roc_auc = os.path.join(figures_dir, 'final_model_roc_auc.png')
plt.savefig(final_model_roc_auc, dpi=300, bbox_inches='tight')

fpr, tpr, _ = roc_curve(y_test_final, y_prob_test)

plt.figure(figsize=(6, 5))
plt.plot(
    fpr,
    tpr,
    lw=2,
    label=f"RF ROC (AUC = {auc_test:.3f})"
)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("External Validation ROC Curve (Random Forest)")
plt.legend(loc="lower right")
plt.tight_layout()
external_validation_roc_auc = os.path.join(figures_dir, 'external_validation_roc_auc.png')
plt.savefig(external_validation_roc_auc, dpi=300, bbox_inches='tight')

mean_fpr = np.linspace(0, 1, 100)
tprs = []

for i in range(1000):
    idx = rng.choice(len(y_test_final), len(y_test_final), replace=True)

    if len(np.unique(y_test_final[idx])) < 2:
        continue

    fpr_i, tpr_i, _ = roc_curve(
        y_test_final[idx],
        y_prob_test[idx]
    )

    tpr_interp = np.interp(mean_fpr, fpr_i, tpr_i)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)

tprs = np.array(tprs)
mean_tpr = tprs.mean(axis=0)
std_tpr = tprs.std(axis=0)

mean_tpr[-1] = 1.0

plt.figure(figsize=(6, 5))

plt.plot(
    mean_fpr,
    mean_tpr,
    color='blue',
    lw=2,
    label=f"RF ROC (AUC = {auc_test:.3f})"
)

plt.fill_between(
    mean_fpr,
    np.maximum(mean_tpr - 1.96 * std_tpr, 0),
    np.minimum(mean_tpr + 1.96 * std_tpr, 1),
    color='blue',
    alpha=0.2,
    label="95% CI"
)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("External Validation ROC Curve (Random Forest)")
plt.legend(loc="lower right")
plt.tight_layout()
external_roc_auc_ci = os.path.join(figures_dir, 'external_roc_auc_ci.png')
plt.savefig(external_roc_auc_ci, dpi=300, bbox_inches='tight')

## pathaway enrich

for l in enrichr_libs:
    file_name = f'{l}.human.enrichr.reports.txt'
    file_path = os.path.join(path_directory, file_name)

    path_df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    keep_cols = [c for c in ['Term', 'Adjusted P-value', 'Overlap', 'P-value', 'Combined Score'] if c in path_df.columns]
    path_df = path_df[keep_cols].sort_values(by=keep_cols[1]).head(50)

    out_file = os.path.join(path_directory, f"top50_{l}_enrichment.csv")
    path_df.to_csv(out_file, index=False)
    

libraries = [
    "GO_Biological_Process_2021",
    "KEGG_2021_Human"
]
for li in libraries:
    top_enr_path = os.path.join(path_directory, f"top50_{li}_enrichment.csv")
    enrich = pd.read_csv(top_enr_path)
    score_col = 'Adjusted P-value' if 'Adjusted P-value' in enrich.columns else 'P-value'
    enrich = enrich.sort_values(by=score_col).head(15)
    enrich['Term'] = (
    enrich['Term']
    .astype(str)
    .str.replace(r'\s*\(.*?\)', '', regex=True)
    .str.strip()
)
    
    plt.figure(figsize=(8, 6))
    plt.barh(enrich['Term'], -enrich[score_col].apply(lambda x: np.log10(x)))
    plt.xlim(0, enrich[score_col].apply(lambda x: -np.log10(x)).max() * 1.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('-log10(Adjusted P-value)', fontsize=20)
    plt.ylabel('Enriched Term', fontsize=20)
    plt.title(f'Top Enriched Terms: {li}')
    plt.title(f'Top Enriched Terms: {li}')
    plt.gca().invert_yaxis()
    
    
    plot_file = os.path.join(figures_dir, f"{li}_Final_barplot.png")
    
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')

print('Done with visualization. ')
