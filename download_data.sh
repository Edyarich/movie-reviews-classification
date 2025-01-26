#!/bin/bash
curl -L -o ./data/imdb-dataset-of-50k-movie-reviews.zip\
  https://www.kaggle.com/api/v1/datasets/download/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
unzip ./data/imdb-dataset-of-50k-movie-reviews.zip -d ./data
rm ./data/imdb-dataset-of-50k-movie-reviews.zip

python split_data.py
