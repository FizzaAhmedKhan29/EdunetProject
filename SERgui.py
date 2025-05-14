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

    return result.reshape(1, -1)

# GUI setup
class EmotionRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Emotion Recognition")

        self.selected_model = tk.StringVar()
        self.selected_model.set("KNN")  # Default

        # Dropdown to select model
        tk.Label(root, text="Select Model:").pack()
        tk.OptionMenu(root, self.selected_model, *MODEL_PATHS.keys()).pack(pady=5)

        # Upload button
        tk.Button(root, text="Upload Audio File", command=self.upload_file).pack(pady=10)
        self.file_name = tk.Label(root, text="Please select a file", font=("Arial", 8, "bold"), fg="black")
        self.file_name.pack(pady=2)

        # Output label for selected model
        self.primary_output = tk.Label(root, text="", font=("Arial", 12, "bold"), fg="blue")
        self.primary_output.pack(pady=10)

        # Output text widget for showing all model predictions
        self.all_output = tk.Text(root, height=10, width=60, font=("Arial", 10))
        self.all_output.pack(pady=5)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        self.file_name.config(text=file_path)
        if not file_path:
            return

        try:
            features = extract_feature(file_path)
            self.all_output.delete(1.0, tk.END)  # Clear previous results

            selected = self.selected_model.get()
            all_predictions = {}

            for model_name, model_file in MODEL_PATHS.items():
                if not os.path.exists(model_file):
                    all_predictions[model_name] = "Model file not found"
                    continue

                model = joblib.load(model_file)
                all_predictions[model_name] = model.predict(features)[0]

            # Show selected model prediction
            selected_emotion = all_predictions[selected]
            self.primary_output.config(text=f"{selected} Prediction: {selected_emotion}")

            self.all_output.insert(tk.END, "Here's what other models predicted:\n")
            # Show all predictions
            for model, emotion in all_predictions.items():
                self.all_output.insert(tk.END, f"{model}: {emotion}\n")

        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong:\n{str(e)}")


# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognizerGUI(root)
    root.mainloop()
