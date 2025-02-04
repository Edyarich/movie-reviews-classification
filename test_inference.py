import requests
import pandas as pd

test_csv_path = "data/test.csv"
url = "http://127.0.0.1:5000/predict"

test_data = pd.read_csv(test_csv_path).head(5)
test_data.reset_index(inplace=True)

for index, row in test_data.iterrows():
    row_data = row.to_dict()
    payload = {"features": row_data['review']}
    print(payload)
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print(f"Row {index + 1} Prediction: {response.json()}")
    else:
        print(f"Row {index + 1} Error: {response.json()}")
