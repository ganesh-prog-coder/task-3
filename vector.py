import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import cv2

IMG_SIZE = (64, 64)

# Load trained model
df=pd.read_excel("C:/Users/SAI/Documents/dog vs cat.csv.xlsx")

# Colors
BG_COLOR = "#f0f4f8"
BTN_COLOR = "#4CAF50"
BTN_HOVER = "#45a049"
CAT_COLOR = "#ff8c94"
DOG_COLOR = "#8cb3ff"

def on_enter(e):
    e.widget.config(bg=BTN_HOVER)

def on_leave(e):
    e.widget.config(bg=BTN_COLOR)

def predict_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        return

    # Show image in GUI
    img = Image.open(file_path)
    img.thumbnail((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Preprocess image
    img_cv = cv2.imread(file_path)
    img_cv = cv2.resize(img_cv, IMG_SIZE)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).flatten().reshape(1, -1)

    # Prediction
    prediction = model.predict(img_gray)[0]
    if prediction == 0:
        label_text = "Cat 🐱"
        result_label.config(text=f"Prediction: {label_text}", bg=CAT_COLOR)
    else:
        label_text = "Dog 🐶"
        result_label.config(text=f"Prediction: {label_text}", bg=DOG_COLOR)

# Tkinter window
root = tk.Tk()
root.title("🐾 Cat vs Dog Classifier 🐾")
root.geometry("450x500")
root.configure(bg=BG_COLOR)

# Title label
title_label = tk.Label(root, text="Cat vs Dog Image Classifier", font=("Arial", 18, "bold"), bg=BG_COLOR, fg="#333")
title_label.pack(pady=15)

# Upload button
btn = tk.Button(root, text="📂 Upload Image", command=predict_image,
                font=("Arial", 14, "bold"), bg=BTN_COLOR, fg="white", activebackground=BTN_HOVER, relief="flat", padx=15, pady=5)
btn.pack(pady=10)
btn.bind("<Enter>", on_enter)
btn.bind("<Leave>", on_leave)

# Image preview area
image_label = tk.Label(root, bg=BG_COLOR)
image_label.pack(pady=20)

# Result label
result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), width=25, height=2)
result_label.pack(pady=10)

# Footer
footer_label = tk.Label(root, text="Powered by Machine Learning", font=("Arial", 10), bg=BG_COLOR, fg="#777")
footer_label.pack(side="bottom", pady=5)

root.mainloop()
