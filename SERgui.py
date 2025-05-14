import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import librosa
import joblib
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ========== SER Setup ==========
ser_model_paths = {
    'KNN': 'SER_KNN.joblib',
    'SVM': 'SER_SVM.joblib',
    'NaiveBayes': 'SER_NaiveBayes.joblib',
    'RandomForest': 'SER_RandomForest.joblib',
    'MLP': 'SER_MLP.joblib',
    'DecisionTree': 'SER_DecisionTree.joblib'
}
ser_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
ser_models = {name: joblib.load(path) for name, path in ser_model_paths.items()}

def extract_ser_features(file_name, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    result = np.array([])
    if chroma:
        stft = np.abs(librosa.stft(X))
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_feat))
    if mel:
        mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel_feat))
    return result.reshape(1, -1)

# ========== FER Setup ==========
fer_model_path = "FER_CNN_model.h5"
fer_model = load_model(fer_model_path)
fer_emotions = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def prepare_fer_image(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# ========== GUI Setup ==========
root = tk.Tk()
root.title("Multimodal Emotion Recognition")

notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both')

# ---------- SER Tab ----------
ser_tab = ttk.Frame(notebook)
notebook.add(ser_tab, text='Speech Emotion Recognition')

selected_ser_model = tk.StringVar(value='KNN')
ser_file_path = tk.StringVar()

def browse_ser_file():
    path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    ser_file_path.set(path)

def predict_ser_emotion():
    if not ser_file_path.get():
        messagebox.showerror("Error", "Please select an audio file.")
        return
    features = extract_ser_features(ser_file_path.get())
    selected_model = ser_models[selected_ser_model.get()]
    selected_prediction = selected_model.predict(features)[0]
    comparison_results = {name: model.predict(features)[0] for name, model in ser_models.items()}
    result_text = f"Selected Model ({selected_ser_model.get()}): {selected_prediction}\n\n\n\n\n\n"
    result_text += "Here's What Other Models Predict:\n\n"
    for name, pred in comparison_results.items():
        if name != selected_ser_model.get():
            result_text += f"  {name}: {pred}\n"
    ser_output_label.config(text=result_text)

tk.Label(ser_tab, text="Select Model:").pack()
tk.OptionMenu(ser_tab, selected_ser_model, *ser_model_paths.keys()).pack()

tk.Label(ser_tab, text="Select Audio File (.wav):").pack()
tk.Entry(ser_tab, textvariable=ser_file_path, width=50).pack()
tk.Button(ser_tab, text="Browse", command=browse_ser_file).pack(pady=5)
tk.Button(ser_tab, text="Predict Emotion", command=predict_ser_emotion, bg="lightgreen").pack(pady=10)

ser_output_label = tk.Label(ser_tab, text="",bg="lightblue", justify="left", font=("Arial", 12), fg="blue")
ser_output_label.pack(padx=10, pady=10)

# ---------- FER Tab ----------
fer_file_path = tk.StringVar()

def browse_fer_file():
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
    fer_file_path.set(path)

def predict_fer_emotion():
    if not fer_file_path.get():
        messagebox.showerror("Error", "Please select an image file.")
        return
    img_array = prepare_fer_image(fer_file_path.get())
    predictions = fer_model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_emotion = fer_emotions[predicted_index]
    fer_output_label.config(text=f"Predicted Emotion: {predicted_emotion}")

# ---------- FER Tab ----------
fer_tab = ttk.Frame(notebook)
notebook.add(fer_tab, text='Facial Emotion Recognition')

fer_file_path = tk.StringVar()

# Emotion-to-image mapping (update paths as needed)
emotion_image_map = {
    "happy": "images/happy.png",
    "sad": "images/sad.png",
    "angry": "images/angry.png",
    "fearful": "images/fearful.png",
    "disgust": "images/disgusted.png",
    "neutral": "images/neutral.png",
    "surprised": "images/surprised.png"
}

# Image reference storage (to prevent garbage collection)
fer_loaded_images = {}

def browse_fer_file():
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
    fer_file_path.set(path)

def predict_fer_emotion():
    if not fer_file_path.get():
        messagebox.showerror("Error", "Please select an image file.")
        return
    try:
        img_array = prepare_fer_image(fer_file_path.get())
        predictions = fer_model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_emotion = fer_emotions[predicted_index].lower()

        # Show predicted emotion text
        fer_output_label.config(text=f"Predicted Emotion: {predicted_emotion.capitalize()}")

        # Display corresponding cartoon image
        image_path = emotion_image_map.get(predicted_emotion)
        if image_path and os.path.exists(image_path):
            pil_img = Image.open(image_path).resize((100, 100))
            tk_img = ImageTk.PhotoImage(pil_img)
            fer_loaded_images["current"] = tk_img  # Prevent garbage collection
            fer_image_display.config(image=tk_img)
        else:
            fer_image_display.config(image='', text='[Image not available]')

    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")

# Input Section (top)
tk.Label(fer_tab, text="Select Image File (.jpg, .png):").pack()
tk.Entry(fer_tab, textvariable=fer_file_path, width=50).pack()
tk.Button(fer_tab, text="Browse", command=browse_fer_file).pack(pady=5)
tk.Button(fer_tab, text="Predict Emotion", command=predict_fer_emotion, bg="lightblue").pack(pady=10)

# Output Text (middle)
fer_output_label = tk.Label(fer_tab, text="", font=("Arial", 12), fg="green")
fer_output_label.pack(pady=10)

# Image Display (bottom of the tab)
fer_image_display = tk.Label(fer_tab)
fer_image_display.pack(pady=20)

root.mainloop()
