# IMDb Movie Reviews Classification

The repository contains an application of deep learning model for classifying movie reviews. The model is based on the [Keras](https://keras.io/) library and the [TensorFlow](https://www.tensorflow.org/) framework.

The goal is to deliver a **production‑ready sentiment‑analysis service** that scores IMDb movie reviews as *positive* or *negative*.  
The repository covers the full MLOps life‑cycle:

* **Training pipeline** – data cleaning, SentencePiece tokenizer, CNN model, evaluation.  
* **Model export** – SavedModel format + tokenizer artefacts.  
* **Serving stack** –  
  * **TensorFlow Serving** for high‑performance inference,  
  * a thin **Flask gateway** that handles all pre/post‑processing and exposes a simple REST API,  
  * a minimal **web UI** for manual testing and CSV batch uploads.  
* **Containerisation** – three independent images (model, gateway, web) for clear separation of concerns.  
* **Orchestration** – Docker Compose for local development and Kubernetes manifests for production, including Secrets for sensitive config.  
* **Upgrade/rollback workflow** – immutable version tags for every image and zero‑downtime roll‑outs via Kubernetes `kubectl set image`.


## Dataset

Link to the dataset: [kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Model quality ✨

On the validation set the model achieves **≈ 90 % accuracy** (macro F1 ≈ 0.90).

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
|     negative |    0.90   |  0.91  |   0.91   |   4000  |
|     positive |    0.91   |  0.90  |   0.91   |   4000  |
|     accuracy |           |        |   0.91   |   8000  |
|    macro avg |    0.91   |  0.91  |   0.91   |   8000  |
| weighted avg |    0.91   |  0.91  |   0.91   |   8000  |

## Prerequisites

| Tool            | Version (tested)              |
| --------------- | ----------------------------- |
| Python          |  3.10 +                       |
| Docker          |  24 +                         |
| Kubernetes CLI  |  1.30 +                       |
| kubectl context | points at your target cluster |

```bash
git clone git@github.com:Edyarich/movie-reviews-classification.git && cd movie-reviews-classification
# optional: create a virtual env for training
python3 -m venv env
source env/bin/activate
```

## Project structure
The application is split into **three cooperative services**, each packaged as an independent Docker image and deployed as its own Deployment ➜ Service pair in Kubernetes.

<br/>

```mermaid
flowchart LR
  %% ─────────── USER ───────────
  subgraph User
    BROWSER["Browser<br/>or curl"]:::client
  end

  %% ─────────── WEB UI ─────────
  subgraph Web["Web UI: 8080"]
    UI[/"index.html<br/>single-text form<br/>CSV upload"/]:::web
  end

  %% ────────── GATEWAY ─────────
  subgraph Gateway["Gateway Flask: 5000 CPU Pod"]
    PREP[/"Tokenise<br/>Pad<br/>JSON -> Tensor"/]:::proc
    THRESH[">= 0.5 ?"]:::proc
  end

  %% ──────── TF-SERVING ────────
  subgraph TF["TensorFlow Serving: 8501 GPU CPU Pod"]
    MODEL["SavedModel v1"]:::model
  end

  %% ────────── FLOW ────────────
  BROWSER -->|HTTP| UI
  UI      -->|REST JSON| PREP
  PREP    -->|gRPC REST instances list| MODEL
  MODEL   -->|raw score| THRESH
  THRESH  -->|JSON score prediction| UI
  UI      -->|HTTP| BROWSER

  %% ───────── STYLE ───────────
  classDef client fill:#fff,stroke:#777,stroke-width:1px
  classDef web    fill:#c6e0ff,stroke:#1a73e8,color:#000
  classDef proc   fill:#d0ffd8,stroke:#008a00,color:#000
  classDef model  fill:#ffe8c6,stroke:#ff9500,color:#000
```

<br/>

| Layer            | Goal / responsibility | Input | Output |
| ---------------- | --------------------- | ----- | ------ |
| **Web interface** (`webapp.py` + `index.html`) | Give users a friendly way to test the model.<br>• Single‑text box for ad‑hoc experiments.<br>• CSV upload (`text` column) for batch scoring. | Browser HTTP requests | • JSON response for single text.<br>• Downloadable CSV with `score` & `prediction` columns. |
| **Gateway** (`gateway.py`) | All **pre‑ and post‑processing in Python**.<br>• Lower‑case, clean HTML.<br>• Tokenise with SentencePiece.<br>• Pad/trim to max length.<br>• Call TF‑Serving.<br>• Apply 0.5 threshold.<br>• Hide TF‑Serving details from outside world. | JSON `{"features":"<raw text>"}` | JSON `{"score":0.93,"prediction":"positive"}` |
| **TF‑Serving** (custom image built from `docker/model.dockerfile`) | **High‑performance inference** on the exported SavedModel.<br>Handles model versioning, hot‑reload, batching, metrics. | gRPC / REST Predict request with an `instances` list of padded integer IDs | Raw tensor predictions (`[[0.93]]`) |


## Screenshots

### Web interface
[<img src="images/web-interface.png" width="800"/>](images/web-interface.png)

### Single text mode
Positive review:

[<img src="images/positive-review-example.png" width="800"/>](images/positive-review-example.png)

Negative review:

[<img src="images/negative-review-example.png" width="800"/>](images/negative-review-example.png)

### CSV upload mode
Upload `./data/test-web.csv`:

[<img src="images/csv-upload.png" width="800"/>](images/csv-upload.png)

Wait ~5 seconds to get results:

[<img src="images/csv-results.png" width="400"/>](images/csv-results.png)


## Local running
To begin with, install [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/). After that, run the commands:

```bash
python generate_env_secret.py
docker compose up  # spins tf‑serving, gateway, web on ports 8501,5000,8080
open http://localhost:8080   # interact via browser
```

## Kubernetes running
To begin with, install [kubectl](https://kubernetes.io/docs/tasks/tools/) and (optionally) [kind](https://kubernetes.io/docs/tasks/tools/). After that, run the commands:

```bash
kubectl create secret generic web-secret \
  --from-literal=FLASK_SECRET=$(openssl rand -hex 32)

kubectl apply -f kube-config/model-deployment.yaml
kubectl apply -f kube-config/model-service.yaml
kubectl apply -f kube-config/gateway-deployment.yaml
kubectl apply -f kube-config/gateway-service.yaml
kubectl apply -f kube-config/web-deployment.yaml
kubectl apply -f kube-config/web-service.yaml

# Web UI
kubectl port-forward svc/sentiment-web 8080:80 &
open http://localhost:8080
```

## Full pipeline

### 1. Training & Export

```bash
pip install -r requirements_train.txt

# split dataset (one‑off); creates data/train.csv, data/val.csv, data/test.csv
python src/split_data.py

# train & export SavedModel to ./sentiment/1/
python train.py data/train.csv data/val.csv --model_dir sentiment/1
```

### 2. Build container images

Don't forget to create an appropriate repository in Docker Hub.

At the end of the step, you can run `docker compose up --build` after you replace prepared images with your own.

#### 2.1 Custom TensorFlow Serving image (contains the model)

```bash
 docker build -f docker/model.dockerfile -t <REG>/sentiment-tf-serving:v1 .
 docker push <REG>/sentiment-tf-serving:v1
```

#### 2.2 Gateway image (talks to TF‑Serving)

```bash
docker build -f docker/gateway.dockerfile -t <REG>/sentiment-gateway:v1 .
docker push <REG>/sentiment-gateway:v1
```

#### 2.3 Web-interface image (talks to Gateway)

```bash
docker build -f docker/web.dockerfile -t <REG>/sentiment-web:v1 .
docker push <REG>/sentiment-web:v1
```


### 3. Kubernetes deployment

Before deployment, change the `containers.image` in `kube-config/model-deployment.yaml` to point to your registry.

```bash
kubectl apply -f kube-config/model-deployment.yaml      # TF‑Serving
kubectl apply -f kube-config/model-service.yaml

kubectl apply -f kube-config/gateway-deployment.yaml    # HTTP gateway
kubectl apply -f kube-config/gateway-service.yaml

kubectl apply -f kube-config/web-deployment.yaml        # Web server
kubectl apply -f kube-config/web-service.yaml
```

Wait for the pods:
```bash
kubectl rollout status deployment/sentiment-tf-serving
kubectl rollout status deployment/sentiment-gateway
kubectl rollout status deployment/sentiment-web
```

Smoke‑test:

```bash
# Gateway only
kubectl port-forward svc/sentiment-gateway 5000:80 &
curl -X POST http://localhost:5000/predict \
     -H 'Content-Type: application/json' \
     -d '{"features":"Absolute cinema!"}'
# Web UI
kubectl port-forward svc/sentiment-web 8080:80 &
open http://localhost:8080
```

The web Deployment references a Kubernetes Secret named **web-secret** that provides `FLASK_SECRET`. To rotate: 
```bash
kubectl delete secret web-secret
kubectl create secret generic web-secret \
  --from-literal=FLASK_SECRET=$(openssl rand -hex 32)
kubectl rollout restart deployment/sentiment-web
```

### 5. Updating the model

1. Re‑train, export to a *new* version folder e.g. `sentiment/2/`.
2. Copy `sentiment/2/` into `models/` and rebuild image **or** mount it via `subPath`.
3. Push as `sentiment-tf-serving:v2`, run:

   ```bash
   kubectl set image deployment/sentiment-tf-serving \
          tf-serving=<REG>/sentiment-tf-serving:v2
   ```

   Zero‑downtime roll‑out; gateway & web stay unchanged.

---

## Directory map

```
.
├── data/                       ← raw & split datasets + sample `test-web.csv`
│   └── test-web.csv            ← tiny CSV for UI smoke‑tests
├── docker/
│   ├── gateway.dockerfile      ← gateway container recipe
│   ├── model.dockerfile        ← TF‑Serving container recipe
│   └── web.dockerfile          ← web‑UI container recipe
├── docker-compose.yaml         ← local 3‑service stack (TF‑Serving, gateway, web)
├── download_data.sh            ← pulls IMDB dataset & triggers split
├── gateway.py                  ← HTTP façade that calls TF‑Serving
├── generate_env_secret.py      ← helper to create/update .env FLASK_SECRET
├── index.html                  ← Jinja template copied into /app/templates
├── kube-config/                ← 6 Kubernetes manifests (model / gateway / web)
│   ├── gateway-deployment.yaml
│   ├── gateway-service.yaml
│   ├── model-deployment.yaml
│   ├── model-service.yaml
│   ├── web-deployment.yaml
│   └── web-service.yaml
├── logs.txt                    ← handy CLI snippets & troubleshooting logs
├── notebooks/                  ← exploratory notebooks (Jupyter)
├── proto.py                    ← shared dataclasses + preprocessing utils
├── requirements.txt            ← runtime deps (Flask, pandas, requests …)
├── requirements_train.txt      ← heavier training deps (TensorFlow, etc.)
├── sentiment/1/                ← current SavedModel version + metadata
├── src/                        ← data pipeline & CNN model definition
│   ├── data_processing.py
│   ├── model.py
│   └── split_data.py
├── train.py                    ← train & export entry‑point
└── webapp.py                   ← Flask web interface code
```

