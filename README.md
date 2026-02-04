### **Parkinson’s Disease Transcriptomic Machine Learning Pipeline**





##### **Project Description**

##### 

This repository contains a computational analysis pipeline for transcriptomic data processing and machine learning–based classification of Parkinson’s disease (PD) versus control samples. The workflow integrates gene expression preprocessing, differential expression analysis, feature selection, supervised learning, pathway enrichment, and independent dataset validation.



The repository is structured to support methodological transparency and reproducibility while complying with data sensitivity and journal submission policies.

##### 

##### 

##### **Analysis Workflow**

##### 

##### The pipeline follows these major steps:

##### 

###### **Data Ingestion**

##### 

* Parsing GEO series matrix files



* Extraction of sample-level metadata



* Construction of AnnData objects



###### **Preprocessing**

##### 

* Quality control metric calculation



* Library size normalization



* Probe-to-gene annotation and aggregation

##### 

###### **Differential Expression Analysis**

###### 

* Non-parametric statistical testing



* Multiple testing correction



* Gene-level annotation

##### 

###### **Feature Selection for Machine Learning**

##### 

* DEG-based filtering



* Model-driven feature ranking



* Intersection of features across models

##### 

###### **Machine Learning Modeling**

##### 

* Random Forest



* Logistic Regression



* Gradient-boosted decision trees (XGBoost)



* Stratified cross-validation

##### 

###### **External Validation**

##### 

* Independent dataset preprocessing



* Feature-aligned model evaluation



* Statistical validation of selected genes

##### 

###### **Functional Enrichment**

##### 

* Gene Ontology (Biological Process)



* KEGG pathway analysis

##### 

###### **Visualization**

##### 

* Dimensionality reduction



* Differential expression plots



* Model diagnostics and evaluation figures



* Pathway enrichment plots





##### 

##### **Repository Structure (Public)**





parkinson\_project\_optimised/

│

├── scripts/        # Data analyses (parkinson_pipeline.py)

├── results/        # Tabular outputs and intermediate analysis artifacts

├── figures/        # Generated plots and visualizations

├── README.md       # Project documentation







##### **Data and Code Availability**



**Note on Data Sensitivity and Submission Policy**



The following components are excluded from this repository prior to submission due to data sensitivity, file size, and journal review requirements:



* data/raw/



* data/processed/




These directories contain raw expression matrices and processed intermediate objects. Their exclusion is intentional and does not affect the conceptual reproducibility of the workflow.



All analyses are based on publicly available GEO datasets. Complete data and code can be provided upon reasonable request or after manuscript acceptance, in accordance with journal and data-sharing policies.



##### 

##### **Software Environment**



###### Python (≥ 3.9)



**Core libraries:**



* scanpy



* scikit-learn



* xgboost



* gseapy



* pandas, numpy



* matplotlib, seaborn



##### 

##### **Author**



Amir Mahdi Taghizadeh



