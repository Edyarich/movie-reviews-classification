import argparse
import sys
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append('./src/')

from data_processing import preprocess_train_data, preprocess_val_data
from model import instantiate_and_train_model, score_dl_model

os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"


def main():
    parser = argparse.ArgumentParser(description="Process paths for train and test dataframes.")

    # Adding arguments
    parser.add_argument('train_df_path', type=str, help='Path to the training dataframe')
    parser.add_argument('val_df_path', type=str, help='Path to the validation dataframe')
    parser.add_argument('--model_dir', type=str, default='sentiment/1', help='Path to save model')
    parser.add_argument('--metadata_filename', type=str, default='metadata.pkl', help='Path to save model')

    # Parsing arguments
    args = parser.parse_args()

    # Now you can use args.train_df and args.test_df as the paths to your dataframes
    df_train = pd.read_csv(args.train_df_path)
    df_val = pd.read_csv(args.val_df_path)

    os.makedirs(args.model_dir, exist_ok=True)
    train_dataset_tf, metadata = preprocess_train_data(df_train, tokenizer_dir=args.model_dir)
    val_dataset_tf = preprocess_val_data(df_val, metadata)
    model = instantiate_and_train_model(train_dataset_tf, val_dataset_tf)
    
    print("Prediction quality on a training dataset")
    score_dl_model(model, val_dataset_tf)
    
    model.export(args.model_dir)
    
    metadata_full_path = os.path.join(args.model_dir, args.metadata_filename)
    with open(metadata_full_path, "wb") as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    main()
