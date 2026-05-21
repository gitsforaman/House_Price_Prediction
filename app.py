from flask import Flask, render_template, request
import joblib
import json

app = Flask(__name__)

model = joblib.load("model.pkl")

locations = ["Delhi","Mumbai","Lucknow","Noida","Bangalore","Pune"]

city_factor = {
    "Delhi": 90,
    "Mumbai": 140,
    "Lucknow": 50,
    "Noida": 70,
    "Bangalore": 110,
    "Pune": 85
}

cities = list(city_factor.keys())
prices = list(city_factor.values())

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/loading', methods=['POST'])
def loading():
    return render_template('loading.html', data=request.form)

@app.route('/result', methods=['POST'])
def result():

    sqft = float(request.form['sqft'])
    bath = float(request.form['bath'])
    bhk = float(request.form['bhk'])
    city = request.form['location']

    ml_price = float(model.predict([[sqft, bath, bhk]])[0])

    base = city_factor.get(city, 80)

    # SIMPLE + STABLE PRICE (NO RANDOMNESS)
    price = round((ml_price * 0.7) + (base * 0.3), 2)

    return render_template(
        "result.html",
        price=price,
        cities=json.dumps(cities),
        prices=json.dumps(prices),
        selected_city=city
    )

if __name__ == "__main__":
    app.run(debug=True)
