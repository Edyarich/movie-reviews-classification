from flask import Flask, request, jsonify
import os
import argparse
import pickle
import requests
import numpy as np
import pandas as pd 
from proto import PredictRequest, PredictResponse, preprocess_test_data


# ── Load tokenizer metadata only ─────────────────────────
parser = argparse.ArgumentParser(description="Gateway for sentiment model via TF‑Serving")
parser.add_argument(
    "--metadata",
    default="metadata.pkl",
    help="Pickle with tokenizer & preprocessing info"
)
parser.add_argument(
    "--tfserving_url",
    default="http://sentiment-tf-serving:8501/v1/models/sentiment:predict",
    help="TensorFlow‑Serving REST endpoint"
)
args = parser.parse_args()

with open(args.metadata, "rb") as fd:
    metadata = pickle.load(fd)

TF_SERVING_URL = args.tfserving_url
TIMEOUT_SEC = 15.0
THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD", "0.5"))

# ── Flask app ─────────────────────────────────────────────
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        req = PredictRequest(review=payload["features"])

        df = pd.DataFrame([req.review], columns=["review"])
        instances = preprocess_test_data(df, metadata).tolist()

        resp = requests.post(
            TF_SERVING_URL,
            json={"instances": instances},
            timeout=TIMEOUT_SEC
        )
        resp.raise_for_status()
        prob = np.asarray(resp.json()["predictions"])[0][0]

        result = PredictResponse(
            score=round(prob, 3), 
            prediction="positive" if prob >= THRESHOLD else "negative"
        )
        return jsonify(result.__dict__)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
