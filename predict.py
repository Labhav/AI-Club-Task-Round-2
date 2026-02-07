import numpy as np
import librosa
import tensorflow as tf
import sys
import os

# --- CONFIGURATION ---
MODEL_PATH = 'ravdess_emotion_model_GAPfinal.h5'

EMOTIONS = {
    0: 'Neutral',
    1: 'Calm',
    2: 'Happy',
    3: 'Sad',
    4: 'Angry',
    5: 'Fearful',
    6: 'Disgust',
    7: 'Surprised'
}

def preprocess_audio(file_path):
    """
    Loads and processes an audio file to match the training data format.
    """
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        
        if log_S.shape[1] < 130:
            pad_width = 130 - log_S.shape[1]
            log_S = np.pad(log_S, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_S = log_S[:, :130]
            
        norm_S = (log_S - log_S.min()) / (log_S.max() - log_S.min())
        
        return norm_S.reshape(1, 128, 130, 1)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def predict_emotion(file_path):
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
        print("Make sure the .h5 file is in the same folder as this script!")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"Processing audio: {file_path}...")
    processed_data = preprocess_audio(file_path)
    
    if processed_data is not None:
        predictions = model.predict(processed_data)
        class_id = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        emotion = EMOTIONS.get(class_id, "Unknown")
        
        print(f"\n🎧 Prediction Result:")
        print(f"--------------------------")
        print(f"🗣️  Emotion:    {emotion.upper()}")
        print(f"📊 Confidence: {confidence:.2f}%")
        print(f"--------------------------")
        return emotion
    else:
        print("Failed to process audio.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        predict_emotion(audio_path)
    else:
        print("Usage: python predict.py <path_to_wav_file>")
