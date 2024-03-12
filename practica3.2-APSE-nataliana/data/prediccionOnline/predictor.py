import pickle
from flask import Flask, jsonify, request
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)




# Load the trained scikit-learn models stored in pickle format
with open('C:/Users/natal/Downloads/APSELogisticSimulator-main (2)/APSELogisticSimulator-main/data/model_preparation/travelModel.pkl', 'rb') as f:
    modelo_tiempo_viaje = pickle.load(f)

with open('C:/Users/natal/Downloads/APSELogisticSimulator-main (2)/APSELogisticSimulator-main/data/model_preparation/deliveryModel.pkl', 'rb') as f:
    modelo_tiempo_entrega = pickle.load(f)

with open('C:/Users/natal/Downloads/APSELogisticSimulator-main (2)/APSELogisticSimulator-main/data/model_preparation/le.pkl', 'rb') as f:
    labelEncoder = pickle.load(f)


# Endpoint for route prediction model
# Input is a json object with attribute time
@app.route('/predict_eta', methods=['POST'])
def predict_eta():
    # Get the JSON data from the request body
    data = np.array(float(request.get_json()["time"]))


    # Realizar la predicción 
    prediction = modelo_tiempo_viaje.predict(data.reshape(-1,1))

    # Return the prediction as a JSON response
    return jsonify({"prediction": prediction[0].tolist()})

# Endpoint for load delivery endpoint.
# Input is a json object with attributes truckId and time
@app.route('/predict_delivery', methods=['POST'])
def predict_delivery():

    data = request.get_json()
    truck_id = np.array([data["truckId"]])  # Extraer el ID del camión
    time_features = np.array(data["time"]).reshape(1, -1)  # Extraer y dar forma a las características de tiempo
    # Transformar el ID del camión y combinarlo con las características de tiempo
    encoded_truck_id = labelEncoder.transform(truck_id).reshape(-1, 1)
    features = np.hstack((encoded_truck_id, time_features))
    prediction = modelo_tiempo_entrega.predict(features[:, 0].reshape(-1, 1)) # Realizar la predicción



    # Return the prediction as a JSON response
    return jsonify({"prediction": prediction[0].tolist()})


if __name__ == '__main__':
    app.run(debug=True, port=7777, host='0.0.0.0')

