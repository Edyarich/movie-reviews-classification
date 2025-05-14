import tensorflow as tf
from keras import layers, models, optimizers, losses, callbacks

import numpy as np
from sklearn.metrics import classification_report
from data_processing import VOCAB_SIZE, MAX_LENGTH


def score_dl_model(model, test_dataset: tf.data.Dataset, cutoff: float = 0.5):
    """
    Calculate prediction quality of DL model on a test dataset

    Args:
        model (_type_): DL model
        test_dataset (tf.data.Dataset): test dataset
        cutoff (float, optional): score cutoff. Defaults to 0.5.
    """
    y_pred_prob = model.predict(test_dataset)
    y_pred = (y_pred_prob >= cutoff).astype(int).reshape(-1)

    y_true = []
    for batch in test_dataset:
        _, labels = batch
        y_true.append(labels.numpy())
    y_true = np.concatenate(y_true, axis=0)

    print(classification_report(y_true, y_pred))


def create_cnn_model(
    vocab_size: int = VOCAB_SIZE,
    max_length: int = MAX_LENGTH,
    embedding_dim: int = 256,
) -> models.Model:
    """
    Instantiate CNN model

    Args:
        vocab_size (int, optional): vocabulary size. Defaults to VOCAB_SIZE.
        max_length (int, optional): max input length (in words). Defaults to MAX_LENGTH.
        embedding_dim (int, optional): embedding dim. Defaults to 256.

    Returns:
        models.Model: CNN model
    """
    inputs = layers.Input(shape=(max_length,), dtype="int32", name="input_layer")
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length,
        trainable=True,
        name="embedding_layer",
    )(inputs)
    x = layers.Dropout(0.3, name="dropout_1")(x)

    convs = []
    kernel_sizes = (2, 3, 4, 5)
    filter_sizes = (32, 128, 32, 128)
    for i, (kernel_size, filter_size) in enumerate(zip(kernel_sizes, filter_sizes)):
        conv = layers.Conv1D(
            filters=filter_size,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
            name=f"conv_{i}",
        )(x)
        pool = layers.GlobalMaxPooling1D(name=f"global_max_pool_{kernel_size}")(conv)
        convs.append(pool)

    concatenated = layers.concatenate(convs, axis=-1, name="concatenate_layer")
    concatenated = layers.Dropout(0.5, name="dropout_2")(concatenated)

    outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(concatenated)
    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_Model")
    return model


def fit_model(
    cnn_model: models.Model,
    train_dataset_tf: tf.data.Dataset,
    val_dataset_tf: tf.data.Dataset,
    epochs: int = 30,
):
    """
    Fit CNN model on training data

    Args:
        cnn_model (models.Model): CNN model to fit
        train_dataset_tf (tf.data.Dataset): train dataloader
        val_dataset_tf (tf.data.Dataset): validation dataloader
        epochs (int, optional): number of epochs. Defaults to 30.
    """
    early_stop_cnn = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    _ = cnn_model.fit(
        train_dataset_tf,
        validation_data=val_dataset_tf,
        epochs=epochs,
        callbacks=[early_stop_cnn],
    )


def instantiate_and_train_model(
    train_dataset_tf: tf.data.Dataset,
    val_dataset_tf: tf.data.Dataset,
    epochs: int = 30,
    vocab_size: int = VOCAB_SIZE,
    max_length: int = MAX_LENGTH,
    embedding_dim: int = 256,
) -> models.Model:
    """
    Instantiate and train CNN model

    Args:
        train_dataset_tf (tf.data.Dataset): train dataloader
        val_dataset_tf (tf.data.Dataset): validation dataloader
        epochs (int, optional): number of epochs. Defaults to 30.
        vocab_size (int, optional): vocabulary size. Defaults to VOCAB_SIZE.
        max_length (int, optional): max input length. Defaults to MAX_LENGTH.
        embedding_dim (int, optional): embedding dim. Defaults to 256.

    Returns:
        models.Model: trained CNN model
    """
    cnn_model = create_cnn_model(vocab_size, max_length, embedding_dim)
    cnn_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.00156, beta_1=0.94, beta_2=0.911),
        loss=losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    fit_model(cnn_model, train_dataset_tf, val_dataset_tf, epochs)
    return cnn_model
