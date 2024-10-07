import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
import torch

# Load your trained model
model = YOLO('sighn-language_3.pt')  # Update path as needed


def preprocess_image(image_path):
    """
    Preprocess the image to match the input requirements of the model.
    Adjust the size and normalization as per your model's requirements.
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize((640, 640))  # Example size; adjust as needed
    return image

def draw_bounding_boxes(image, result):
    """
    Draw bounding boxes with labels and confidence scores on the image.
    """
    draw = ImageDraw.Draw(image)
    
    # Load a default font or you can specify a custom one
    font = ImageFont.truetype("arial.ttf", size=24)

    for box in result[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]

        # Draw the bounding box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

        # Add label and confidence score
        label = f"{class_name} {confidence:.2f}"

        # Calculate text size using textbbox instead of textsize
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw the label background rectangle
        draw.rectangle([(x1, y1 - text_height), (x1 + text_width, y1)], fill="red")

        # Add the text (label)
        draw.text((x1, y1 - text_height), label, fill="white", font=font)

    return image


def predict_sign(image_path):
    """
    Predict the sign from the uploaded image and visualize the bounding boxes.
    """
    try:
        processed_image = preprocess_image(image_path)
        result = model(processed_image)
        
        # Draw bounding boxes with labels on the image
        image_with_boxes = draw_bounding_boxes(processed_image.copy(), result)
        
        return image_with_boxes
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during prediction:\n{e}")
        return None

def upload_image():
    """
    Handle the image upload and prediction.
    """
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not file_path:
        return

    # Predict the sign and draw bounding boxes
    image_with_result = predict_sign(file_path)
    
    if image_with_result:
        # Display the processed image with bounding boxes
        img_tk = ImageTk.PhotoImage(image_with_result.resize((400, 400)))
        image_label.config(image=img_tk)
        image_label.image = img_tk
        result_label.config(text="Prediction Complete!")

# Initialize Tkinter window
root = tk.Tk()
root.title("Sign Language Recognition")

# Set window size and styling
root.geometry("500x600")
root.configure(bg="#f5f5f5")  # Background color
root.resizable(False, False)

# Title Label
title_label = tk.Label(root, text="Sign Language Recognition", font=("Helvetica", 20, "bold"), bg="#f5f5f5", fg="#333")
title_label.pack(pady=10)

# Subtitle Label
subtitle_label = tk.Label(root, text="Upload an image to recognize the sign", font=("Helvetica", 14), bg="#f5f5f5", fg="#666")
subtitle_label.pack(pady=5)

# Upload Button
upload_btn = tk.Button(root, text="Upload Image", command=upload_image, width=20, height=2, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
upload_btn.pack(pady=20)

# Image Display Label
image_label = tk.Label(root, bg="#f5f5f5")
image_label.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="Predicted Sign: None", font=("Helvetica", 16), bg="#f5f5f5", fg="#333")
result_label.pack(pady=20)

# Run the application
root.mainloop()
