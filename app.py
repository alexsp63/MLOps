from flask import Flask, request, render_template
import random
import logging

app = Flask(__name__)

@app.before_request
def log_request_info():
    app.logger.debug('Body: %s', request.get_data())

def create_main_page():
    brands = ['Gucci', 'Adidas']
    conditions = ['New', 'Used']
    sources = ['Amazon', 'Shopping mall']
    shippings = ['Yes', 'No']
    manufacturers = ['Japan', 'Turkey']
    categories = ['Shoes', 'Glasses']
    colors = ['Black', 'White']

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
    # brand = request.form['brand']
    # available = request.form['available']
    # condition = request.form['condition']
    # sale = request.form['sale']
    # source = request.form['source']
    # shipping = request.form['shipping']
    # offer = request.form['offer']
    # is_return = request.form['return']
    # manufacturer = request.form['manufacturer']
    # category = request.form['category']
    # color = request.form['color']

    price = random.uniform(100.0, 1_000.0)
    formatted_price = "{:.2f}".format(price)
    app.logger.debug('Predicted value: %s', formatted_price)

    return render_template('prediction.html', formatted_price=formatted_price)

if __name__ == '__main__':
    logging.basicConfig(filename='log/development.log',level=logging.DEBUG)
    app.run(debug=True)
