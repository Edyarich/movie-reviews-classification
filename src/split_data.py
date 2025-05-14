import pandas as pd
import sys
sys.path.append('./src/')
from data_processing import split_data

df = pd.read_csv("./data/IMDB Dataset.csv")
df_train, df_val, df_test = split_data(df)
df_train.to_csv("./data/train.csv", index=False)
df_val.to_csv("./data/val.csv", index=False)
df_test.to_csv("./data/test.csv", index=False)
