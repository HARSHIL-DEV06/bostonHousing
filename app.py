import pickle
from flask import Flask,request,jsonify,app,render_template
import numpy as np
import pandas as pd


app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    
    print(data)

    values_array = np.array(list(data.values())).reshape(1, -1)
    print(values_array)

    new_data = scalar.transform(values_array)
    output = regmodel.predict(new_data)

    print(output[0])

    return jsonify(output[0])


if __name__ =="__main__":
    app.run(debug=True)