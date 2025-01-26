from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd
import sys
import argparse

sys.path.append('./src/')

from data_processing import preprocess_test_data

# Initialize Flask app
app = Flask(__name__)

# Use argparse to specify the model path
parser = argparse.ArgumentParser(description="Run Flask app with specified model")
parser.add_argument('--model-path', type=str, required=True, help='Path to the model file (e.g., model.pkl)')
args = parser.parse_args()

# Load the model using the specified path
with open(args.model_path, 'rb') as fd:
    model, metadata = pickle.load(fd)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        input_df = pd.DataFrame(input_data['features'], index=[0], columns=["review"])
        test_dataset_tf = preprocess_test_data(input_df, metadata)
        prediction = (model.predict(test_dataset_tf) >= 0.5).astype(int)

        return jsonify({
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
