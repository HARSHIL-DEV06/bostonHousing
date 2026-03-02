import pickle
from flask import Flask,request,jsonify,app,render_template
import numpy as np
import pandas as pd


app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
scalar = pickle.load(open('scalar.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    payload = request.get_json(force=True)
    # support both: {"data": {...}} and direct dict payload
    if isinstance(payload, dict) and 'data' in payload:
        data = payload['data']
    else:
        data = payload

    try:
        # prefer a DataFrame to preserve column ordering by name
        new_df = pd.DataFrame([data])
        new_data = scaler.transform(new_df)
    except Exception:
        # fallback: transform values order (less reliable unless consistent)
        new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))

    output = regmodel.predict(new_data)
    return jsonify({'prediction': float(output[0])})


if __name__ =="__main__":
    app.run(debug=True)