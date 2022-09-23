import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#from google.colab import files
#import model
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import math
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta, Adam, Nadam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
import json
#
from sklearn.model_selection import train_test_split 
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
#%matplotlib inline
# Evaluations
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
# Random Forest
from sklearn.ensemble import RandomForestClassifier
from IPython.display import clear_output

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.metrics import plot_confusion_matrix

## Chargement dataset
def load_data_set(org, dataset):
    path = './'
    pathResult="Results/"+org
    #ajout du nouveau fichier du dataset
    metaDataset = pd.read_csv(dataset+".csv",sep=";")
    print(metaDataset)
    row,col = metaDataset.shape
    count=1
    return (row,col,count,metaDataset,path,pathResult)

def main_preprocessing(epath_file, balance):
    row, col, count, metaDataset, path, pathResult = load_data_set(organism, dataset)
    for index in range(row):
        print("############################"+metaDataset['featureGroup'][index]+"############################################")
        print(str(count)+" / "+str(row))
        META_DATA_FILE_NAME=epath_file
        FEAT=str(metaDataset['code'][index])
        FEAT_FILE=os.path.join(path, str(metaDataset['filename'][index]))
        META_FILE=os.path.join(path, str(META_DATA_FILE_NAME))
        RESULT_FILE=os.path.join(pathResult, str(FEAT)+"_Result.json")
        
        print(FEAT_FILE+"\n")
        print(META_FILE+"\n")
        
        print(RESULT_FILE)
        #open file 
		#FILE_SAVE=open(RESULT_FILE, 'a')
        DICT_XGB=dict()
        DICT_RF=dict()
        # feature de epath
        feature_data = pd.read_csv(FEAT_FILE)
        print(feature_data)
        genes = feature_data.index
        
        meta_df = pd.read_excel(META_FILE, sheet_name="Sheet1",engine="openpyxl")
        #nom du champ pour identifier les gene Gene_Locus
        meta_idx = meta_df['Gene_Locus']
        meta_idx = pd.Series([x.upper() for x in meta_idx.values])
        meta_df = meta_df.set_index(keys=meta_idx)
        ddf =meta_df
        print(meta_df['Gene_essentaility'])
        
        # get class labels for dataset
		#remplace class par Gene_essentaility
        df_full = feature_data.merge(meta_df[['Gene_essentaility']], how='inner', left_index=True, right_index=True)
        
        #df_full.to_csv('dffullctdcsimple.csv')
		#Nucleotide feature
        if 'X' in df_full.columns:
            df_full.drop('X', axis = 1, inplace=True)
        
        mappings = {'NE': 0, 'E': 1}
        classes = df_full.pop('Gene_essentaility')
        essential_labels = classes.map(mappings)
        df_full['essential'] = essential_labels
        
        # remove unknowns
        df_full = df_full[df_full['essential'] < 2]
        dataset_full=df_full.copy()
        df_essential = df_full[df_full['essential'] == 1]
        df_nonEssential = df_full[df_full['essential'] == 0]
        
        if (balance == 1):
            # rebalance the classes
            df_essential_oversample = pd.concat([df_essential, df_essential], ignore_index=True)
            df_essential_oversample = pd.concat([df_essential_oversample, df_essential], ignore_index=True)
            
            # sample non-essential genes
			#total_essential_samples = len(df_essential_oversample)
			#df_nonE_sample_RF = df_nonEssential.sample(3*total_essential_samples)
            df_nonE_sample_RF = pd.concat([df_essential, df_nonEssential], ignore_index=True)
            
            # combine essential and non-essential sets, drop gene name column
			#balance data
            df_full = pd.concat([df_essential_oversample.iloc[:,:],df_nonE_sample_RF.iloc[:,:]], ignore_index=True)
            #use unbalance data
            df_full = pd.concat([df_essential.iloc[:,:],df_nonEssential.iloc[:,:]], ignore_index=True)
            df_full = pd.concat([df_essential_oversample.iloc[:,1:],df_nonE_sample_RF.iloc[:,1:]], ignore_index=True)
            #df_equilib = df_full
    return (df_full,FEAT)
## compte les E et NE
def countE_NE(lst, x): 
    count = 0
    for ele in lst: 
        if (ele == x): 
            count = count + 1
    return count

file_dataset = open('subgrouplist.csv','r')
dataset_table = [line for line in file_dataset]
file_dataset.close()
dataset_table

organism = "top" # abreviation organism name 
epath_file ="top_Thermodesulfobacterium_geofontis.xlsx" # excel epath 
balance = 1 # 0 if classes are embalanced and 1 else
for i in range(1,len(dataset_table)):
  with open('dataset.csv','w') as file:
    file.write(dataset_table[0])
    file.write(dataset_table[i])
  dataset = "dataset" # dataset
  dataset1, FEAT = main_preprocessing(epath_file,balance)
  X=dataset1.copy()
  X.drop(columns=X.columns[-1], axis=1, inplace=True)
  Y=dataset1['essential'].copy()
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
  
  essential_qte = countE_NE(dataset1['essential'], 1)
  non_essential_qte = countE_NE(dataset1['essential'], 0)

  # fit model on training data
  model = XGBClassifier()
  model.fit(X_train, y_train)
  # make predictions for test data
  predit = model.predict(X_test)
  roc_auc = metrics.roc_auc_score(y_test,predit)
  results = classification_report(y_test,predit,output_dict=True)
  results['roc_auc'] = roc_auc
  results['essential_qte'] = essential_qte
  results['non_essential_qte'] = non_essential_qte
  
  with open("results/"+FEAT+"_results.json","w") as json_file:
    json.dump(results, json_file)

