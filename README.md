# IMDb Movie Reviews Classification

The repository contains an application of deep learning model for classifying movie reviews. The model is based on the [Keras](https://keras.io/) library and the [TensorFlow](https://www.tensorflow.org/) framework.

## Dataset

Link to the dataset: [kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)


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
TODO

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
python train.py data/train.csv data/val.csv --output_dir sentiment/1
```

### 2. Build container images

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

