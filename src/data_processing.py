import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import os
import sentencepiece as spm

SEED = 42
VOCAB_SIZE = 20000
MAX_LENGTH = 500
OOV_TOKEN = "<OOV>"
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TOKENIZER_PREFIX = "tokenizer_model"


def preprocess_text(df: pd.DataFrame):
    """
    Preprocess text data: convert to lower case, remove non-alphanumeric characters, and remove HTML tags.

    Args:
        df (pd.DataFrame): dataframe with a column "review"
    """
    df["review"] = df["review"].str.lower()
    df["review"] = df["review"].str.replace("[^\w\s]", "")
    df["review"] = df["review"].str.replace("<br />", "")


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into train and test sets

    Args:
        df (pd.DataFrame): dataframe with columns "review" and "sentiment"

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train and test dataframes
    """
    texts = df["review"].values
    labels = (df["sentiment"] == "positive").astype(int).values

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=0.2,
        random_state=SEED,
        stratify=train_labels,
    )
    return (
        pd.DataFrame({"review": train_texts, "sentiment": train_labels}),
        pd.DataFrame({"review": val_texts, "sentiment": val_labels}),
        pd.DataFrame({"review": test_texts, "sentiment": test_labels}),
    )


def train_sentencepiece_tokenizer(df: pd.DataFrame, vocab_size: int, model_prefix: str):
    """
    Train and save a SentencePiece tokenizer model.

    Args:
        df (pd.DataFrame): Dataframe containing text data.
        vocab_size (int): Vocabulary size for SentencePiece.
        model_prefix (str): Prefix for the saved tokenizer model.
    """
    text_file = f"{model_prefix}.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        for review in df["review"]:
            f.write(review + "\n")

    spm.SentencePieceTrainer.train(
        input=text_file, model_prefix=model_prefix, vocab_size=vocab_size, 
        character_coverage=0.9995, model_type="unigram"
    )
    
    os.remove(text_file)


def preprocess_train_data(
    df: pd.DataFrame,
    tokenizer_dir: str,
    vocab_size: int = VOCAB_SIZE,
    max_length: int = MAX_LENGTH,
    batch_size: int = BATCH_SIZE,
    buffer_size: int = BUFFER_SIZE,
) -> Tuple[tf.data.Dataset, dict]:
    """
    Create tensorflow training dataloader from dataframe

    Args:
        df (pd.DataFrame): input dataframe with columns "review" and "sentiment"
        vocab_size (int, optional): vocabulary size. Defaults to VOCAB_SIZE.
        max_length (int, optional): max text length (in words). Defaults to MAX_LENGTH.
        batch_size (int, optional): batch size. Defaults to BATCH_SIZE.
        buffer_size (int, optional): buffer size. Defaults to BUFFER_SIZE.
        return_metadata (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[tf.data.Dataset, dict]: tensorflow dataloader and training metadata
    """
    preprocess_text(df)
    
    full_tokenizer_prefix = os.path.join(tokenizer_dir, TOKENIZER_PREFIX)
    train_sentencepiece_tokenizer(df, vocab_size, full_tokenizer_prefix)
    tokenizer = spm.SentencePieceProcessor(model_file=f"{full_tokenizer_prefix}.model")

    train_sequences = [tokenizer.encode_as_ids(text) for text in df["review"].values]
    train_padded = keras.preprocessing.sequence.pad_sequences(
        train_sequences, maxlen=max_length, padding="post", truncating="post"
    )
    train_labels_np = np.array(df["sentiment"].values)

    train_dataset_tf = tf.data.Dataset.from_tensor_slices(
        (train_padded, train_labels_np)
    )
    train_dataset_tf = (
        train_dataset_tf.shuffle(buffer_size=buffer_size, seed=SEED)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    metadata = {
        "tokenizer": tokenizer,
        "max_length": max_length,
        "batch_size": batch_size,
    }

    return train_dataset_tf, metadata


def preprocess_val_data(df: pd.DataFrame, metadata: dict) -> tf.data.Dataset:
    """
    Create tensorflow test dataloader from dataframe and training metadata

    Args:
        df (pd.DataFrame): input dataframe with columns "review" and "sentiment"
        metadata (dict): training metadata

    Returns:
        tf.data.Dataset: test dataloader
    """
    preprocess_text(df)
    tokenizer = metadata["tokenizer"]
    test_sequences = [tokenizer.encode_as_ids(text) for text in df["review"].values]
    test_padded = keras.preprocessing.sequence.pad_sequences(
        test_sequences, maxlen=metadata["max_length"], padding="post", truncating="post"
    )
    test_labels_np = np.array(df["sentiment"].values)

    test_dataset_tf = tf.data.Dataset.from_tensor_slices((test_padded, test_labels_np))
    test_dataset_tf = test_dataset_tf.batch(metadata["batch_size"]).prefetch(
        tf.data.AUTOTUNE
    )

    return test_dataset_tf
