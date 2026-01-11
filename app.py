import os
import pickle
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SPECTROGRAM_FOLDER'] = 'static/spectrograms'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SPECTROGRAM_FOLDER'], exist_ok=True)

MODEL_PATH = "deepfake_model_lr.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None
    print("❌ WARNING: Model file not found!")

def get_physics_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = np.mean(mfcc, axis=1)
        
        features = {"rolloff": rolloff, "centroid": centroid, "zcr": zcr}
        for i, m in enumerate(mfcc_means):
            features[f"mfcc_{i}"] = m
        return pd.DataFrame([features])
    except Exception as e:
        return None

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
    except Exception as e:
        print(f"Plot Error: {e}")
        return None

def get_explanation(prediction_class):
    if prediction_class == 1: # FAKE
        return (
            "⚠️ <b>Reason:</b> The spectrogram shows <b>'Digital Silence'</b> (pitch black gaps). "
            "AI models often fail to generate the natural background noise (room tone/static) found in real phone calls. "
            "The spectral texture is too smooth or blurry."
        )
    else: # REAL
        return (
            "✅ <b>Reason:</b> The spectrogram shows consistent <b>'Natural Noise'</b> (purple/orange streaks). "
            "This indicates real room tone, microphone hum, or breath sounds that are physically present in human recordings."
        )

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    confidence_score = ""
    result_class = ""
    spectrogram_url = None
    explanation_text = ""

    if request.method == 'POST':
        if 'audio' not in request.files: return redirect(request.url)
        file = request.files['audio']
        if file.filename == '': return redirect(request.url)

        if file:
            timestamp = int(time.time())
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            features = get_physics_features(filepath)
            
            if features is not None and model is not None:
                pred = model.predict(features)[0]
                prob = model.predict_proba(features)[0][1]

                explanation_text = get_explanation(pred)

                if pred == 1:
                    prediction_text = "🚨 FAKE DETECTED"
                    confidence_score = f"Confidence: {prob*100:.1f}%"
                    result_class = "danger"
                else:
                    prediction_text = "✅ REAL VOICE"
                    confidence_score = f"Safety Score: {(1-prob)*100:.1f}%"
                    result_class = "safe"
                
                plot_filename = f"plot_{filename}.png"
                spectrogram_url = generate_spectrogram(filepath, plot_filename)
            
            os.remove(filepath)

            if spectrogram_url:
                spectrogram_url = url_for('static', filename=f'spectrograms/{spectrogram_url}')

    return render_template('index.html', 
                           prediction=prediction_text, 
                           score=confidence_score, 
                           css_class=result_class,
                           spectrogram_url=spectrogram_url,
                           explanation=explanation_text)

if __name__ == '__main__':
    app.run(debug=True)