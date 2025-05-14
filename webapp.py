import numpy as np
import pandas as pd
import requests
import os
import uuid
import tempfile

from flask import (
    Flask, render_template, request,
    jsonify, send_file, redirect, url_for, flash
)

GATEWAY_URL = os.getenv(
    "GATEWAY_URL",
    "http://sentiment-gateway.default.svc.cluster.local/predict"
)
THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD", "0.5"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "change-me")

def call_gateway(texts):
    """Send list[str] to sentiment gateway, return list[float] scores."""
    scores = []
    for t in texts:
        # gateway expects {"features": "<text>"}
        r = requests.post(GATEWAY_URL, json={"features": t}, timeout=5)
        r.raise_for_status()
        scores.append(r.json()["score"])
    return scores


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict-text", methods=["POST"])
def predict_text():
    text = request.form.get("free_text", "").strip()
    print(text[:200])
    
    if not text:
        flash("Please enter some text.", "warning")
        return redirect(url_for("index"))

    score = call_gateway([text])[0]
    label = "positive" if score >= THRESHOLD else "negative"
    return jsonify({"score": round(score, 4), "prediction": label})


@app.route("/predict-csv", methods=["POST"])
def predict_csv():
    if "csv_file" not in request.files or request.files["csv_file"].filename == "":
        flash("Please choose a CSV file.", "warning")
        return redirect(url_for("index"))

    f = request.files["csv_file"]
    try:
        df = pd.read_csv(f)
    except Exception as e:
        flash(f"Cannot parse CSV: {e}", "danger")
        return redirect(url_for("index"))

    if "text" not in df.columns:
        flash("CSV must contain a 'text' column.", "danger")
        return redirect(url_for("index"))

    scores = call_gateway(df["text"].tolist())
    df["score"] = np.round(scores, 4)
    df["prediction"] = (df["score"] >= THRESHOLD).map({True: "positive", False: "negative"})

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return send_file(tmp.name, as_attachment=True,
                     download_name=f"predictions_{uuid.uuid4().hex}.csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
