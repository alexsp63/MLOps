from flask import Flask, request, render_template
import random
import logging
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import joblib
import pickle

app = Flask(__name__)

def transform_input_data(input_data):
    transformed_data = {}
    categorical_cols = ['brand', 'categories', 'colors',
                         'condition', 'source', 'shipping', 'manufacturer']
    boolean_cols = ['is_available', 'is_sale', 'is_offer', 'is_return']
    for col in categorical_cols:
        le = LabelEncoder()
        save_path = f'D:/Maga/6th module/MLOps/ml/data/encoders/{col}_classes.npy'
        le.classes_ = np.load(save_path, allow_pickle=True)
        transformed_data[col] = le.transform([input_data[col]])[0]
    for col in boolean_cols:
        if input_data[col] == 'Yes':
            transformed_data[col] = 1
        else:
            transformed_data[col] = 0
    return transformed_data

def get_prediction(data, model_params):
    if 'prediction' in model_params:
        return model_params['prediction']
    else:
        field_name = model_params['field']
        conditions = model_params['conditions']
        bound_value = conditions['bound']
        if data[field_name] <= bound_value:
            return get_prediction(data, conditions['lower_or_equal'])
        else:
            return get_prediction(data, conditions['upper'])    

def predict_value(data):
    '''Предсказание значений на основе выведенного алгоритма'''
    data = {
        'source': data['source'],
        'manufacturer': data['manufacturer'],
        'is_sale': data['is_sale']
    }
    with open('D:/Maga/6th module/MLOps/ml/data/params/decision_tree.pkl', 'rb') as f:
        model_params = pickle.load(f)
        
    return get_prediction(data, model_params)
            
def transform_prediction(predict_value_norm):
    scaler_filename = 'D:/Maga/6th module/MLOps/ml/data/scalers/y_scaler.save'
    scaler = joblib.load(scaler_filename)
    return scaler.inverse_transform(np.array([predict_value_norm]).reshape(-1, 1))[0][0]

def create_main_page():
    save_path = 'D:/Maga/6th module/MLOps/ml/data/encoders/%s_classes.npy'
    brands = np.load(save_path % 'brand', allow_pickle=True)
    conditions = np.load(save_path % 'condition', allow_pickle=True)
    sources = np.load(save_path % 'source', allow_pickle=True)
    shippings = np.load(save_path % 'shipping', allow_pickle=True)
    manufacturers = np.load(save_path % 'manufacturer', allow_pickle=True)
    categories = np.load(save_path % 'categories', allow_pickle=True)
    colors = np.load(save_path % 'colors', allow_pickle=True)

    return render_template(
        'index.html', brands=brands, conditions=conditions,
        sources=sources, shippings=shippings, manufacturers=manufacturers,
        categories=categories, colors=colors
    )

# Define Flask routes
@app.route("/")
def index():
    return create_main_page()

@app.route('/submit', methods=['POST'])
def submit():
    # полученные с формы значения
    user_input = {
        'brand': request.form['brand'],
        'is_available': request.form['available'],
        'condition': request.form['condition'],
        'is_sale': request.form['sale'],
        'source': request.form['source'],
        'shipping': request.form['shipping'],
        'is_offer': request.form['offer'],
        'is_return': request.form['return'],
        'manufacturer': request.form['manufacturer'],
        'categories': request.form['category'],
        'colors': request.form['color']
    }
    app.logger.debug('User input: %s', user_input)
    transformed_data = transform_input_data(user_input)
    app.logger.debug('Transformed data: %s', transformed_data)

    predict_value_norm = predict_value(transformed_data)
    predicted_value = transform_prediction(predict_value_norm)
    formatted_price = "{:.2f}".format(predicted_value)
    app.logger.debug('Predicted value: %s', formatted_price)

    return render_template('prediction.html', formatted_price=formatted_price)

if __name__ == '__main__':
    logging.basicConfig(filename='log/development.log',level=logging.DEBUG)
    app.run(debug=True)
