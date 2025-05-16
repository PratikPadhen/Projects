import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import json
import os
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

# User data file
USER_DATA_FILE = 'users.json'

# Load the rainfall prediction model
with open('model1.pkl', 'rb') as model_file:
    rainfall_model = pickle.load(model_file)

# Load the crop recommendation model and scalers
crop_model = pickle.load(open('model2.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Mock crop data for chatbot
crop_data = {
    "wheat": "Wheat is a staple crop grown in temperate regions. It requires well-drained soil and moderate rainfall.",
    "corn": "Corn (maize) is a widely grown cereal crop. It thrives in warm climates with fertile soil.",
    "rice": "Rice is typically grown during the Kharif season, sown in June-July and harvested in October-November.",
    "potato": "Potatoes grow best in cool climates with well-drained soil. They are rich in carbohydrates.",
    "tomato": "Tomatoes are warm-season crops. They need plenty of sunlight and well-drained soil.",
    "coffee": "Coffee plants grow in tropical regions. They require shade, consistent rainfall, and well-drained soil.",
    "sugarcane": "Sugarcane is a tropical crop. It requires high temperatures and abundant water.",
}

# Initialize user data file
def init_user_data():
    if not os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'w') as f:
            json.dump({}, f)

# Load user data
def load_users():
    init_user_data()
    with open(USER_DATA_FILE, 'r') as f:
        return json.load(f)

# Save user data
def save_users(users):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users, f)

# Mock function to simulate crop data
def get_crop_info(query):
    query = query.lower()
    if query in crop_data:
        return crop_data[query]
    else:
        return f"Sorry, I don't have information about {query}. Please ask about wheat, corn, rice, potato, tomato, coffee, or sugarcane etc."

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        users = load_users()
        
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            error = 'Invalid username or password'
            return render_template('login.html', error=error)
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            error = 'Passwords do not match'
            return render_template('register.html', error=error)
        
        users = load_users()
        
        if username in users:
            error = 'Username already exists'
            return render_template('register.html', error=error)
        
        users[username] = generate_password_hash(password)
        save_users(users)
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/crop-prediction', methods=['GET', 'POST'])
@login_required
def crop_prediction():
    if request.method == 'POST':
        try:
            N = float(request.form['Nitrogen'])
            P = float(request.form['Phosporus'])
            K = float(request.form['Potassium'])
            temp = float(request.form['Temperature'])
            humidity = float(request.form['Humidity'])
            ph = float(request.form['pH'])
            rainfall = float(request.form['Rainfall'])

            feature_list = [N, P, K, temp, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            mx_features = mx.transform(single_pred)
            sc_mx_features = sc.transform(mx_features)
            prediction = crop_model.predict(sc_mx_features)

            crop_dict = {
                1: {"name": "Rice", "image": "rice.png"},
                2: {"name": "Maize", "image": "maize.png"},
                3: {"name": "Jute", "image": "jute.png"},
                4: {"name": "Cotton", "image": "cotton.png"},
                5: {"name": "Coconut", "image": "coconut.png"},
                6: {"name": "Papaya", "image": "papaya.png"},
                7: {"name": "Orange", "image": "orange.png"},
                8: {"name": "Apple", "image": "apple.png"},
                9: {"name": "Muskmelon", "image": "muskmelon.png"},
                10: {"name": "Watermelon", "image": "watermelon.png"},
                11: {"name": "Grapes", "image": "grapes.png"},
                12: {"name": "Mango", "image": "mango.png"},
                13: {"name": "Banana", "image": "banana.png"},
                14: {"name": "Pomegranate", "image": "pomegranate.png"},
                15: {"name": "Lentil", "image": "lentil.png"},
                16: {"name": "Blackgram", "image": "blackgram.png"},
                17: {"name": "Mungbean", "image": "mungbean.png"},
                18: {"name": "Mothbeans", "image": "mothbeans.png"},
                19: {"name": "Pigeonpeas", "image": "pigeonpeas.png"},
                20: {"name": "Kidneybeans", "image": "kidneybeans.png"},
                21: {"name": "Chickpea", "image": "chickpea.png"},
                22: {"name": "Coffee", "image": "coffee.png"}
            }

            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                return render_template('crop_prediction.html', 
                                    crop_prediction_result=crop["name"],
                                    crop_image=crop["image"],
                                    input_values={
                                        'Nitrogen': N,
                                        'Phosporus': P,
                                        'Potassium': K,
                                        'Temperature': temp,
                                        'Humidity': humidity,
                                        'pH': ph,
                                        'Rainfall': rainfall
                                    })
            else:
                return render_template('crop_prediction.html', 
                                    crop_prediction_result="Unknown crop",
                                    crop_image="default_crop.png")
        
        except Exception as e:
            app.logger.error(f"Error in crop prediction: {str(e)}")
            return render_template('crop_prediction.html', 
                                crop_prediction_result=f"Error: {str(e)}",
                                crop_image="error.png")
    
    # GET request - just show the form
    return render_template('crop_prediction.html')
@app.route('/weather-intelligence', methods=['GET', 'POST'])
@login_required
def weather_intelligence():
    if request.method == 'POST':
        try:
            input_data = {
                'pressure': float(request.form['pressure']),
                'dewpoint': float(request.form['dewpoint']),
                'humidity': float(request.form['humidity']),
                'cloud': float(request.form['cloud']),
                'sunshine': float(request.form['sunshine']),
                'winddirection': float(request.form['winddirection']),
                'windspeed': float(request.form['windspeed'])
            }
            
            input_df = pd.DataFrame([input_data])
            prediction = rainfall_model.predict(input_df)
            
            if prediction[0] == 1:
                result = "Rainfall Expected"
                image_url = "static/rain1.gif"
            else:
                result = "No Rainfall Expected"
                image_url = "static/sun.png"
            
            return render_template('weather_intelligence.html', 
                                 rainfall_prediction_result=result, 
                                 rainfall_input_values=input_data, 
                                 rainfall_image_url=image_url)
        except Exception as e:
            return render_template('weather_intelligence.html', 
                                 rainfall_prediction_result=f"Error: {str(e)}")
    
    return render_template('weather_intelligence.html')

@app.route('/farming-assistant')
def farming_assistant():
    return render_template('farming_assistant.html')


@app.route('/chat', methods=['POST'])
@login_required
def chat():
    user_input = request.json.get('message')
    response = get_crop_info(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)