import argparse
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from mtcnn.mtcnn import MTCNN

def main(source_path, destination_path):

    # Huggingface Model for Emotion Detection
    processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
    model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Emotion labels
    emotion_labels = {
        0: 'Angry', 
        1: 'Disgust', 
        2: 'Fear', 
        3: 'Happy',
        4: 'Neutral', 
        5: 'Sad', 
        6: 'Surprise'
    }
    
    # Move the model to the chosen device (CPU or GPU)
    model.to(device)
    
    # Create MTCNN detector for face detection
    mtcnn_detector = MTCNN()
    
    # Load the video capture object
    cap = cv2.VideoCapture(source_path)
    
    # Define video writer parameters (adjust as needed)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec for mp4 format
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(destination_path, fourcc, 20.0, (frame_width, frame_height))
    
    while True:
      # Capture frame-by-frame
      ret, frame = cap.read()
    
      # Exit if no frame is captured
      if not ret:
        break
    
      # Convert frame to RGB format (MTCNN expects RGB)
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
      # Detect faces using MTCNN
      faces = mtcnn_detector.detect_faces(rgb_frame)
    
      # Loop through detected faces
      for face in faces:
        # Extract bounding box coordinates
        x, y, w, h = face['box']
    
        # Extract the face region of interest (ROI)
        face_roi = frame[y:y+h, x:x+w]
    
        # Preprocess the face ROI for the ViT model
        preprocessed_face = processor(face_roi, return_tensors="pt")
    
        # Move the preprocessed data to the chosen device (CPU or GPU)
        preprocessed_face_gpu = preprocessed_face.to(device)
    
        # Use the ViT model to predict emotions on the preprocessed face (on GPU)
        with torch.no_grad():
          outputs = model(**preprocessed_face_gpu)
          predictions = torch.argmax(outputs.logits, dim=-1)
    
        # Extract the predicted emotion label
        predicted_emotion_label = predictions.item()
        predicted_emotion = emotion_labels[predicted_emotion_label]
    
        # Display the emotion on the frame
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
      # Write the processed frame to the output video
      out_video.write(frame)

if __name__=="__main__":
  parser = argparse.ArgumentParser(description="Process a video for emotion detection")
  parser.add_argument("--source", type=str, required=True, help="Path to the source video file")
  parser.add_argument("--destination", type=str, required=True, help="Path to the destination video file")
  args = parser.parse_args
