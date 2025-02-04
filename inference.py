from flask import Flask, request, jsonify
import numpy as np
import os
import pickle
import pandas as pd
import argparse
from ai_edge_litert.interpreter import Interpreter

# Get rid of all possible problems with cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def preprocess_text(df: pd.DataFrame):
    """
    Preprocess text data: convert to lower case, remove non-alphanumeric characters, and remove HTML tags.

    Args:
        df (pd.DataFrame): dataframe with a column "review"
    """
    df["review"] = df["review"].str.lower()
    df["review"] = df["review"].str.replace("[^\w\s]", "")
    df["review"] = df["review"].str.replace("<br />", "")


def preprocess_test_data(df: pd.DataFrame, metadata: dict) -> np.ndarray:
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
    test_padded = np.array(
        [
            np.pad(
                seq, (0, max(0, metadata["max_length"] - len(seq))), mode="constant"
            )[: metadata["max_length"]]
            for seq in test_sequences
        ],
        dtype=np.int32,
    )

    return test_padded


# Initialize Flask app
app = Flask(__name__)

# Use argparse to specify the model path
parser = argparse.ArgumentParser(description="Run Flask app with specified model")
parser.add_argument(
    "model_path", type=str, help="Path to the model file (e.g., model.pkl)"
)
args = parser.parse_args()

# Load the model using the specified path
with open(args.model_path, "rb") as fd:
    model, metadata = pickle.load(fd)


interpreter = Interpreter(model_content=model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        input_df = pd.DataFrame(input_data["features"], index=[0], columns=["review"])
        test_padded = preprocess_test_data(input_df, metadata)

        interpreter.set_tensor(input_details[0]["index"], test_padded)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])

        prediction = (output_data >= 0.5).astype(int)

        return jsonify({"prediction": "positive" if prediction[0] == 1 else "negative"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
