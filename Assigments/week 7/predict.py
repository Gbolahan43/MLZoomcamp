from flask import Flask, jsonify
import pickle
from flask import request


with open('rfcmodel.bin', 'rb') as f_in:
    model = pickle.load(f_in)

with open('dv.bin', 'rb') as dict_vect:
    dv = pickle.load(dict_vect)

app = Flask('rain_predict')

@app.route('/predict', methods = ['POST'])

def predict():
    data = request.get_json()
    
    X = dv.transform([data])
    y_pred = model.predict_proba(X)[0,1]
    rain = y_pred >= 0.5

    result = {
        'rain' : bool(rain),
        'Rain Probability' : float(y_pred)
    }
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port = 9696)