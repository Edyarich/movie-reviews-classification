# IMDb Movie Reviews Classification

The repository contains an application of deep learning model for classifying movie reviews. The model is based on the [Keras](https://keras.io/) library and the [TensorFlow](https://www.tensorflow.org/) framework.

## Dataset

Link to the dataset: [kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)


## 1  Prerequisites

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


## 2  Training & Export

```bash
pip install -r requirements_train.txt

# split dataset (one‑off); creates data/train.csv, data/val.csv, data/test.csv
python src/split_data.py

# train & export SavedModel to ./sentiment/1/
python train.py data/train.csv data/val.csv --output_dir sentiment/1
```

## 3  Build container images

TODO: add vizualization

### 3.1 Custom TensorFlow Serving image (contains the model)

```bash
# Don't forget to create an appropriate repository in Docker Hub
 docker build -f docker/model.dockerfile -t <REG>/sentiment-tf-serving:v1 .
 docker push <REG>/sentiment-tf-serving:v1
```

### 3.2 Gateway image (talks to TF‑Serving)

```bash
docker build -f docker/gateway.Dockerfile \
       -t <REG>/sentiment-gateway:v1 .
docker push <REG>/sentiment-gateway:v1
```


## 4  Local integration with Docker Compose (optional)

```bash
docker compose up --build   # spins tf‑serving, gateway, web on ports 8501,5000,8080
open http://localhost:8080   # interact via browser
```


## 5  Kubernetes deployment

Before deployment, change the `containers.image` in `kube-config/model-deployment.yaml` to point to your registry.

```bash
kubectl apply -f kube-config/model-deployment.yaml      # TF‑Serving
kubectl apply -f kube-config/model-service.yaml

kubectl apply -f kube-config/gateway-deployment.yaml    # HTTP gateway
kubectl apply -f kube-config/gateway-service.yaml
```

Smoke‑test:

```bash
# gateway only
kubectl port-forward svc/sentiment-gateway 5000:80 &
curl -X POST http://localhost:5000/predict \
     -H 'Content-Type: application/json' \
     -d '{"features":"Amazing movie!"}'
# web UI
kubectl port-forward svc/sentiment-web 8080:80 &
open http://localhost:8080
```


---

## 7  Updating the model

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
├── data/                       ← raw & split datasets
├── docker/
│   ├── gateway.dockerfile      ← gateway container recipe
│   └── model.dockerfile        ← TF‑Serving container recipe
├── docker-compose.yaml         ← local dev stack for localhost
├── download_data.sh            ← pulls IMDB dataset & triggers split
├── env/                        ← (optional) Python venv created locally
├── gateway.py                  ← HTTP facade that calls TF‑Serving
├── kube-config/                ← 4 Kubernetes manifests (model & gateway)
│   ├── gateway-deployment.yaml
│   ├── gateway-service.yaml
│   ├── model-deployment.yaml
│   └── model-service.yaml
├── notebooks/                  ← exploratory notebooks
├── proto.py                    ← shared dataclasses + preprocessing utils
├── requirements.txt            ← runtime deps (Flask, pandas, requests …)
├── requirements_train.txt      ← heavier training deps (TensorFlow, etc.)
├── sentiment/1/                ← current SavedModel version
├── src/                        ← data pipeline & CNN model definition
│   ├── data_processing.py
│   ├── model.py
│   └── split_data.py
├── train.py                    ← train & export entry‑point
```

