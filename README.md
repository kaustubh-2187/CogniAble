# CogniAble Assignment
### By Kaustubh Kamble

Following is my submission for the CogniAble Assignment, `Therapist and Child Engagement Analysis`

My approach uses **[Multi-Task Cascaded Convolutional Neural Network](https://github.com/ipazc/mtcnn)** for detecting and aligning faces in the video and  **[Vision Transformer (ViT) for Facial Expression Recognition](https://huggingface.co/trpakov/vit-face-expression)** for performing Emotion Detecion on the detected faces of the child and therapist.

## 1. Emotion Detection

Install the dependencies for running emotion-detection.py

    pip install argparse opencv-python numpy matplotlib transformers torch mtcnn tensorflow


