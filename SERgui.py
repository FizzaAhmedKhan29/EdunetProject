import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import joblib
import os

# Define path mappings to your models
MODEL_PATHS = {
    "KNN": "SER_KNN.joblib",
    "MLP": "SER_MLP.joblib",
    "Naive Bayes": "SER_NaiveBayes.joblib",
    "Decision Tree": "SER_DecisionTree.joblib",
    "Random Forest": "SER_RandomForest.joblib",
    "SVM": "SER_SVM.joblib"
}

def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
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

    return result.reshape(1, -1)  # reshape for sklearn predict()

# GUI setup
class EmotionRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Emotion Recognition")

        self.selected_model = tk.StringVar()
        self.selected_model.set("KNN")  # Default

        # Dropdown to select model
        tk.Label(root, text="Select Model:").pack()
        tk.OptionMenu(root, self.selected_model, *MODEL_PATHS.keys()).pack()

        # Upload button
        tk.Button(root, text="Upload Audio File", command=self.upload_file).pack(pady=10)

        # Output
        self.output_label = tk.Label(root, text="", fg="blue", font=("Arial", 12))
        self.output_label.pack(pady=10)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not file_path:
            return

        try:
            features = extract_feature(file_path)
            model_file = MODEL_PATHS[self.selected_model.get()]
            if not os.path.exists(model_file):
                messagebox.showerror("Error", f"Model file {model_file} not found.")
                return

            model = joblib.load(model_file)
            prediction = model.predict(features)[0]
            self.output_label.config(text=f"Predicted Emotion: {prediction}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict emotion:\n{e}")

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognizerGUI(root)
    root.mainloop()
