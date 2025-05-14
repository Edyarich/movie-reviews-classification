FROM tensorflow/serving:2.17.0

COPY sentiment /models/sentiment
ENV MODEL_NAME=sentiment
