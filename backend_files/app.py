from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

superkart_api = Flask(__name__)

# Load the trained model
model = joblib.load('superkart_model.joblib')

@superkart_api.route('/')
def home():
    return jsonify({
        "message": "SuperKart Sales Prediction API",
        "endpoints": {
            "/v1/predict": "POST - Single prediction",
            "/v1/predictbatch": "POST - Batch predictions"
        }
    })

@superkart_api.route('/v1/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            "status": "success",
            "predicted_sales": round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@superkart_api.route('/v1/predictbatch', methods=['POST'])
def predict_batch():
    try:
        file = request.files['file']
        input_df = pd.read_csv(file)
        predictions = model.predict(input_df)
        results = {str(i): round(float(pred), 2) for i, pred in enumerate(predictions)}
        
        return jsonify(results)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    superkart_api.run(host='0.0.0.0', port=7860, debug=False)
