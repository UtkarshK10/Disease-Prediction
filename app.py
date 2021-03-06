import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True)
