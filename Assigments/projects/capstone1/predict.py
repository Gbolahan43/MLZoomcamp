from flask import Flask, jsonify
import pickle
import numpy as np
from flask import request

classes = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
       'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
       'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
       'pigeonpeas', 'pomegranate', 'rice', 'watermelon']


with open('cr_rfmodel.bin', 'rb') as f_in:
    model = pickle.load(f_in)

with open('dv.bin', 'rb') as dict_vect:
    dv = pickle.load(dict_vect)

with open('scaler.bin', 'rb') as scaler_in:
    scaler = pickle.load(scaler_in)

app = Flask('crop_recommendation')

@app.route('/predict', methods = ['POST'])

def predict():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON format'}), 400
    
    X = dv.transform([data])
    X_scaled = scaler.transform(X)
    y_pred_prob = model.predict_proba(X_scaled)
    y_pred = np.argmax(y_pred_prob, axis=1)[0]

    result = {classes[y_pred]: int(y_pred)}
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port = 5000)