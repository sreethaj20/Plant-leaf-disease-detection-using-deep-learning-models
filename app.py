from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
from PIL import Image
import io
import base64
import os
import cv2 as cv
import numpy as np
import keras
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production
# Initialize Google Gemini API
GOOGLE_API_KEY = 'AIzaSyDWBFfx89qhJzE2HxJXs1ouvrsm_E5_Yug'

plant_disease_agent = Agent(
    model=Gemini(
        api_key=GOOGLE_API_KEY,
        id="gemini-2.0-flash-exp"
    ),
    tools=[DuckDuckGo()],
    markdown=True
)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Training', 'model', 'Leaf Deases(96,88).h5')
model = keras.models.load_model(MODEL_PATH)

# Label names for prediction
label_name = ['Apple scab','Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy', 
'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot', 
'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

# Database initialization
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Plant disease analysis prompt
PLANT_DISEASE_PROMPT = """
You are Pathologists AI, a highly skilled Plant Pathologist AI with extensive knowledge in plant diseases, agricultural science, and botanical diagnostics. 
Analyze the plant specimen information and structure your response as follows, also add relevant plant/nature emojis in your response:

### 1. Initial Assessment 🔍🌿
- Identify key visible symptoms and affected plant parts
- Note any relevant visual patterns in the leaf image
- Evaluate severity level of the condition

### 2. Key Findings 📋🍃
- List primary observations systematically
- Note any abnormalities with precise descriptions (spots, wilting, discoloration)
- Include affected area percentage where relevant
- Rate severity: Healthy/Minor/Moderate/Severe

### 3. Disease Identification 🦠🌱
- Provide primary disease possibilities with confidence levels
- List potential plant pathogens in order of likelihood (fungi, bacteria, virus, nutrient deficiency)
- Support each possibility with observed evidence
- Note any critical or urgent findings that require immediate action

### 4. Gardener-Friendly Explanation 👩‍🌾🌳
- Explain the findings in simple, clear language
- Avoid technical jargon or provide clear definitions
- Include basic information about the disease cycle
- Address common questions gardeners might have

### 5. Treatment Recommendations 💊🌿
- Suggest immediate actions for plant care
- Provide organic and chemical treatment options
- Recommend preventive measures for healthy plants
- List follow-up monitoring steps

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
Don't give very long responses, keep it short and simple, focusing on practical advice for plant care.

The model has identified the disease as: {{DISEASE_NAME}}. Please analyze this specific disease.
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        try:
            conn = sqlite3.connect('database.db')
            cur = conn.cursor()
            user = cur.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            
            if user and check_password_hash(user[2], password):
                session['username'] = username
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password!', 'error')
        except Exception as e:
            flash(f'Database error: {str(e)}', 'error')
        finally:
            conn.close()
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Basic validation
        if not username or not password:
            flash('Username and password are required!', 'error')
            return render_template('register.html')
            
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')
        
        try:
            conn = sqlite3.connect('database.db')
            cur = conn.cursor()
            
            # Check if user exists
            if cur.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone():
                flash('Username already exists!', 'error')
            else:
                hashed_password = generate_password_hash(password)
                cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
                conn.commit()
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
        except Exception as e:
            flash(f'Database error: {str(e)}', 'error')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Comment out the login check for now to make testing easier
    # if 'username' not in session:
    #     flash('Please login to access this feature.', 'error')
    #     return redirect(url_for('login'))
    
    prediction_result = None
    image_data_uri = None
    
    if request.method == 'POST':
        if 'leaf_image' in request.files:
            file = request.files['leaf_image']
            if file.filename != '':
                # Read and process the image
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv.imdecode(nparr, cv.IMREAD_COLOR)
                
                # Preprocess image for model
                normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)
                
                # Make prediction
                predictions = model.predict(normalized_image)
                confidence = predictions[0][np.argmax(predictions)] * 100
                
                if confidence >= 80:
                    disease_name = label_name[np.argmax(predictions)]
                    prediction_result = {
                        'disease': disease_name,
                        'confidence': confidence,
                        'success': True
                    }
                    
                    # Convert image to base64 for displaying and for potential AI analysis
                    buffered = io.BytesIO()
                    Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB)).save(buffered, format="JPEG")
                    image_data_uri = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
                    
                    # Store in session for AI analysis
                    session['last_prediction'] = {
                        'disease': disease_name,
                        'image_data': image_data_uri
                    }
                else:
                    prediction_result = {
                        'message': 'Confidence too low. Please try another image.',
                        'confidence': confidence,
                        'success': False
                    }
    
    return render_template('predict.html', 
                          prediction=prediction_result, 
                          image_data=image_data_uri)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get data directly from the request
        request_data = request.json
        
        if not request_data or 'disease' not in request_data or 'image_data' not in request_data:
            return jsonify({
                'success': False,
                'error': 'Missing required data (disease or image)'
            })
            
        disease_name = request_data['disease']
        image_data = request_data['image_data']
        
        # Save temporary image for Gemini
        image_data_binary = base64.b64decode(image_data.split(',')[1])
        temp_path = "temp_leaf_image.png"
        with open(temp_path, 'wb') as f:
            f.write(image_data_binary)
        
        try:
            # Create prompt with the disease name
            customized_prompt = PLANT_DISEASE_PROMPT.replace("{{DISEASE_NAME}}", disease_name)
            
            print(f"Sending analysis request for disease: {disease_name}")
            
            # Get analysis from Gemini
            response = plant_disease_agent.run(
                customized_prompt,
                images=[temp_path]
            )
            
            print("Analysis received from Gemini")
            
            return jsonify({
                'success': True,
                'analysis': response.content
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/about')
def about():
    return render_template('about.html')

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)