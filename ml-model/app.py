from flask import Flask, request
import logging, json
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import joblib

app = Flask(__name__)
labels_save_path = 'ml/data/encoders/%s_classes.npy'

def get_labels():
    brands = np.load(labels_save_path % 'brand', allow_pickle=True).tolist()
    conditions = np.load(labels_save_path % 'condition', allow_pickle=True).tolist()
    sources = np.load(labels_save_path % 'source', allow_pickle=True).tolist()
    shippings = np.load(labels_save_path % 'shipping', allow_pickle=True).tolist()
    manufacturers = np.load(labels_save_path % 'manufacturer', allow_pickle=True).tolist()
    categories = np.load(labels_save_path % 'categories', allow_pickle=True).tolist()
    colors = np.load(labels_save_path % 'colors', allow_pickle=True).tolist()
    return {
        "success": True,
        "brands": brands,
        "conditions": conditions,
        "sources": sources,
        "shippings": shippings,
        "manufacturers": manufacturers,
        "categories": categories,
        "colors": colors
    }

def json_response(response_body, response_status=200):
    return app.response_class(
        response=json.dumps(response_body),
        status=response_status,
        mimetype='application/json'
    )

def transform_input_data(input_data):
    transformed_data = {}
    categorical_cols = ['brand', 'categories', 'colors',
                         'condition', 'source', 'shipping', 'manufacturer']
    boolean_cols = ['is_available', 'is_sale', 'is_offer', 'is_return']
    for col in categorical_cols:
        le = LabelEncoder()
        save_path = f'ml/data/encoders/{col}_classes.npy'
        le.classes_ = np.load(save_path, allow_pickle=True)
        transformed_data[col] = le.transform([input_data[col]])[0]
    for col in boolean_cols:
        if input_data[col] == 'Yes':
            transformed_data[col] = 1
        else:
            transformed_data[col] = 0
    return transformed_data

def get_prediction(data, model):
    return model.predict(data)
            
def transform_prediction(predict_value_norm):
    scaler_filename = 'ml/data/scalers/y_scaler.save'
    scaler = joblib.load(scaler_filename)
    return scaler.inverse_transform(np.array([predict_value_norm]).reshape(-1, 1))[0][0]

def predict_value(data):
    '''Предсказание значений на основе сохраненной модели'''
    data_keys = [
        'brand', 'categories', 'colors', 'is_available', 'condition', 'is_sale',
        'source', 'shipping', 'is_offer', 'is_return', 'manufacturer'
    ]
    ordered_data = np.array([data.get(key) for key in data_keys]).reshape(1, len(data))
    model_filename = 'ml/data/models/xgb.save'
    model = joblib.load(model_filename)

    return get_prediction(ordered_data, model)

# Define Flask routes
@app.route("/")
def index():
    return json_response({"success": True})

@app.route("/labels")
def labels():
    labels = get_labels()
    return json_response(labels)

@app.route("/predict")
def predict():
    user_input = {
        'brand': request.args.get('brand'),
        'is_available': request.args.get('available'),
        'condition': request.args.get('condition'),
        'is_sale': request.args.get('sale'),
        'source': request.args.get('source'),
        'shipping': request.args.get('shipping'),
        'is_offer': request.args.get('offer'),
        'is_return': request.args.get('return'),
        'manufacturer': request.args.get('manufacturer'),
        'categories': request.args.get('category'),
        'colors': request.args.get('color')
    }
    app.logger.debug('User input: %s', user_input)
    transformed_data = transform_input_data(user_input)
    app.logger.debug('Transformed data: %s', transformed_data)
    predict_value_norm = predict_value(transformed_data)
    predicted_value = transform_prediction(predict_value_norm)
    result = {"predicted_value": str(predicted_value.round(2))}
    return json_response(result)

if __name__ == '__main__':
    # logging.basicConfig(filename='log/development.log', level=logging.DEBUG)
    app.run(debug=True, port=5001)
