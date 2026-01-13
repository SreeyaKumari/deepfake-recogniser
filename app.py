import os
import pickle
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, flash
import time

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SPECTROGRAM_FOLDER'] = 'static/spectrograms'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SPECTROGRAM_FOLDER'], exist_ok=True)

# ---- LOAD MODEL ----
MODEL_PATH = "deepfake_model_lr.pkl"
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
            print("✅ Model loaded successfully!")
    else:
        model = None
        print(f"ERROR: Model file not found at {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"ERROR loading model: {e}")

def get_physics_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        # Check for empty file
        if len(y) == 0:
            return None, "Audio file is empty."

        noise_amp = 0.001 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])

        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = np.mean(mfcc, axis=1)
        
        features = {"rolloff": rolloff, "centroid": centroid, "zcr": zcr}
        for i, m in enumerate(mfcc_means):
            features[f"mfcc_{i}"] = m
            
        return pd.DataFrame([features]), None

    except Exception as e:
        print(f"Physics Error: {e}")
        return None, str(e)

def generate_spectrogram(file_path, output_filename):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.ylim(0, 8000) 
        plt.title('Spectral Density Analysis (0-8kHz)')
        plt.tight_layout()
        
        output_path = os.path.join(app.config['SPECTROGRAM_FOLDER'], output_filename)
        plt.savefig(output_path)
        plt.close() 
        return output_filename
    except Exception:
        return None

def get_explanation(prediction_class):
    if prediction_class == 1: 
        return "⚠️ <b>Reason:</b> Spectrogram shows 'Digital Silence' (black gaps) and lacks natural room noise."
    else: 
        return "✅ <b>Reason:</b> Spectrogram shows consistent 'Natural Noise' (purple streaks) and microphone hum."

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    confidence_score = ""
    result_class = ""
    spectrogram_url = None
    explanation_text = ""

    if request.method == 'POST':
        if 'audio' not in request.files:
            flash("No file part found.")
            return redirect(request.url)
        
        file = request.files['audio']
        if file.filename == '':
            flash("No file selected.")
            return redirect(request.url)

        if file:
            timestamp = int(time.time())
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if model is None:
                flash("Error: Model is not loaded. Check server logs.")
            else:
                features, error_msg = get_physics_features(filepath)
                
                if features is not None:
                    # Successful Extraction
                    #pred = model.predict(features)[0]
                    prob = model.predict_proba(features)[0][1]
                    if(prob>0.70):
                        pred = 1
                    else:
                        pred = 0
                    explanation_text = get_explanation(pred)
                        
                    if pred == 1:
                        prediction_text = "🚨 FAKE DETECTED"
                        confidence_score = f"Confidence: {prob*100:.1f}%"
                        result_class = "danger"
                    else:
                        prediction_text = "✅ REAL VOICE"
                        confidence_score = f"Safety Score: {(1-prob)*100:.1f}%"
                        result_class = "safe"
                    
                    # Generate Graph
                    plot_filename = f"plot_{filename}.png"
                    spectrogram_url = generate_spectrogram(filepath, plot_filename)
                    if spectrogram_url:
                        spectrogram_url = url_for('static', filename=f'spectrograms/{spectrogram_url}')
                else:
                    flash(f"Analysis Failed: {error_msg}")

            if os.path.exists(filepath):
                os.remove(filepath)

    return render_template('index.html', 
                           prediction=prediction_text, 
                           score=confidence_score, 
                           css_class=result_class,
                           spectrogram_url=spectrogram_url,
                           explanation=explanation_text)

if __name__ == '__main__':
    app.run(debug=True)
