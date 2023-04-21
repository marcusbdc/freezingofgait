import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble, linear_model
from sklearn.metrics import average_precision_score

import xgboost as xgb

root_dir = "/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction"
features = ["AccV", "AccML", "AccAP"]
labels = ["StartHesitation", "Turn", "Walking"]

def read_data(dataset, datatype):
    metadata = pd.read_csv(os.path.join(root_dir, f"{dataset}_metadata.csv"))
    
    file_path = os.path.join(root_dir, datatype, dataset)
    
    df_res = pd.concat([
        pd.read_csv(os.path.join(file_path, name)).assign(file=name.replace(".csv", ""))
        for _, _, files in os.walk(file_path)
        for name in files
    ])
    
    df_res = metadata.merge(df_res, how="inner", left_on="Id", right_on="file").drop("file", axis=1)
        
    return df_res
    
df_train_defog = read_data("defog", "train")
df_train_tdcsfog = read_data("tdcsfog", "train")
df_train = pd.concat([df_train_defog, df_train_tdcsfog])
X_train, X_valid, y_train, y_valid = train_test_split(df_train[features], df_train[labels], test_size=0.3, random_state=42)

dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_valid, y_valid, enable_categorical=True)

params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
n_rounds = 100
model = xgb.train(params=params, dtrain=dtrain_reg, num_boost_round=n_rounds)

y_pred = model.predict(dtest_reg).clip(0.0, 1.0)
print(average_precision_score(y_valid, y_pred))

sample_submission = pd.read_csv(os.path.join(root_dir, "sample_submission.csv"))
test_files = glob(os.path.join(root_dir, "test/**/**"))

df_list = []
for file in tqdm(test_files, desc="Processing test files"):
    df = pd.read_csv(file)
    df["Id"] = os.path.splitext(os.path.basename(file))[0]
    df = df.fillna(0).reset_index(drop=True)
    res = pd.DataFrame(np.round(model.predict(xgb.DMatrix(df[features])), 3), columns=labels)
    df = pd.concat([df, res], axis=1)
    df["Id"] = df["Id"].astype(str) + "_" + df["Time"].astype(str)
    df_list.append(df[["Id", "StartHesitation", "Turn", "Walking"]])

submission_df = pd.concat(df_list)
submission_df = pd.merge(sample_submission[["Id", "sample"]], submission_df, how="left", on="Id").fillna(0.0)
submission_df[["Id", "StartHesitation", "Turn", "Walking"]].to_csv("submission.csv", index=False)
