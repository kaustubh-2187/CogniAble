# CogniAble Assignment
### By Kaustubh Kamble

Following is my submission for the CogniAble Assignment, `Therapist and Child Engagement Analysis`

My approach uses **[Multi-Task Cascaded Convolutional Neural Network](https://github.com/ipazc/mtcnn)** for detecting and aligning faces in the video and  **[Vision Transformer (ViT) for Facial Expression Recognition](https://huggingface.co/trpakov/vit-face-expression)** for performing Emotion Detecion on the detected faces of the child and therapist.

After the Emotion Detection phase, the output video is passed through the **[Gaze Transformer](https://github.com/nizhf/hoi-prediction-gaze-transformer)** for predicting the gaze of the child and the therapist along with their engagement analysis with each other and objects present.

Open this *[Google Colab Notebook](https://colab.research.google.com/drive/15gPiSlRCTd2kzkJH9QktMAOmRMNZmACc#scrollTo=j8y39J0vdg6t)* for further details
