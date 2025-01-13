from flask import Flask, request, render_template
import logging, os, requests, json
import urllib.parse

app = Flask(__name__)

try:
    ml_service_host = os.environ['ML_SERVICE_HOST']
except:
    ml_service_host = 'http://localhost'
ml_service_port = 5001
ml_service_url = f'{ml_service_host}:{ml_service_port}'

def ml_service_get_request(url_path, request_body={}):
    res = requests.get(
        f'{ml_service_url}{url_path}', 
        params=urllib.parse.urlencode(request_body)
    )
    return res.json()

def create_main_page():
    try:
        labels = ml_service_get_request('/labels')
        brands = labels.get('brands')
        conditions = labels.get('conditions')
        sources = labels.get('sources')
        shippings = labels.get('shippings')
        manufacturers = labels.get('manufacturers')
        categories = labels.get('categories')
        colors = labels.get('colors')

        return render_template(
            'index.html', brands=brands, conditions=conditions,
            sources=sources, shippings=shippings, manufacturers=manufacturers,
            categories=categories, colors=colors
        )
    except:
        return render_template('something-went-wrong.html')

# Define Flask routes
@app.route("/")
def index():
    return create_main_page()

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # полученные с формы значения
        user_input = {
            'brand': request.form['brand'],
            'available': request.form['available'].capitalize(),
            'condition': request.form['condition'],
            'sale': request.form['sale'].capitalize(),
            'source': request.form['source'],
            'shipping': request.form['shipping'],
            'offer': request.form['offer'].capitalize(),
            'return': request.form['return'].capitalize(),
            'manufacturer': request.form['manufacturer'],
            'category': request.form['category'],
            'color': request.form['color']
        }
        app.logger.debug('User input: %s', user_input)
        predicted_value_response = ml_service_get_request('/predict', user_input)
        return render_template('prediction.html', formatted_price=predicted_value_response.get('predicted_value'))
    except:
        return render_template('something-went-wrong.html')

if __name__ == '__main__':
    # logging.basicConfig(filename='web-service/log/development.log',level=logging.DEBUG)
    app.run(debug=True, port=5000)
