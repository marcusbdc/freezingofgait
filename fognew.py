import os
import numpy as np 
import pandas as pd
import os
from tqdm import tqdm
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble, linear_model
from sklearn.metrics import average_precision_score

ROOT_DIR = "/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction"
FEATURES_LIST = ["AccV", "AccML", "AccAP"]
LABELS_LIST = ["StartHesitation", "Turn", "Walking"]

def read_data(dataset, datatype):
    metadata_df = pd.read_csv(os.path.join(ROOT_DIR,dataset + "_metadata.csv"))
    
    file_path = os.path.join(ROOT_DIR, datatype, dataset)
    
    df_res = pd.DataFrame()
    for _, _, files in os.walk(file_path):
        for name in files:
            f_path = os.path.join(file_path, name)
            csv_df = pd.read_csv(f_path)
            csv_df["file"] = name.replace(".csv", "")
            df_res = pd.concat([df_res,csv_df])
    
    df_res = metadata_df.merge(df_res, how = 'inner', left_on = 'Id', right_on = 'file')
    df_res = df_res.drop(["file"], axis = 1)
        
    return df_res
    
df_train_defog = read_data('defog','train')
df_train_tdcsfog = read_data('tdcsfog', 'train')
df_train = pd.concat([df_train_defog, df_train_tdcsfog])
X_train, X_valid, y_train, y_valid = train_test_split(df_train[FEATURES_LIST], df_train[LABELS_LIST], test_size=0.30, random_state=42)

import xgboost as xgb
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_valid, y_valid, enable_categorical=True)

params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
n_trees = 100
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n_trees,
)
print(average_precision_score(y_valid, model.predict(dtest_reg).clip(0.0,1.0)))

sample_submission_df = pd.read_csv(os.path.join(ROOT_DIR,'sample_submission.csv'))
test_files_list = glob(os.path.join(ROOT_DIR,'test/**/**'))

sample_submission_df['sample'] = 0
submission_df_list = []
for file_path in test_files_list:
    df = pd.read_csv(file_path)
    df['Id'] = file_path.split('/')[-1].split('.')[0]
    df = df.fillna(0).reset_index(drop=True)
    res_df = pd.DataFrame(np.round(model.predict(xgb.DMatrix(df[FEATURES_LIST])),3), columns=LABELS_LIST)
    df = pd.concat([df,res_df], axis=1)
    df['Id'] = df['Id'].astype(str) + '_' + df['Time'].astype(str)
    submission_df_list.append(df[['Id','StartHesitation', 'Turn' , 'Walking']])
submission_df = pd.concat(submission_df_list)
submission_df = pd.merge(sample_submission_df[['Id','sample']], submission_df, how='left', on='Id').fillna(0.0)
submission_df[['Id','StartHesitation', 'Turn' , 'Walking']].to_csv('submission.csv', index=False)
