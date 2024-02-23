[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
 <img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" /> 
 
# Computer_vision_emotion_recognition
To recognize emotions in my image, I utilized the DeepFace library. If you're using Google Colab, you'll need to install DeepFace first. DeepFace is a Python library that provides tools for facial analysis, including emotion recognition, using deep learning models.

To install DeepFace in Google Colab, you can run the following command:

python
Copy code
!pip install deepface
Once DeepFace is installed, you can proceed with using it to recognize emotions in your images. This library simplifies the process of analyzing facial expressions, allowing you to extract emotions such as happiness, sadness, anger, and more from images.

My test image:
<div align="center">
<img src='https://github.com/Sarvandani/Computer_vision_emotion_recognition/blob/main/EMOTION.jpg' width='400' height='600'>
</div>

```python
import cv2
import gdown
from deepface import DeepFace

# Download the image from Google Drive
url = 'https://drive.google.com/uc?id=1xKtqLlRjvj8lAaxz9UE2KWPlIJjfwC6q'
output = 'image.jpg'
gdown.download(url, output, quiet=False)

# Load the image
image = cv2.imread(output)

# Perform emotion recognition
result = DeepFace.analyze(image, actions=['emotion'])

# Extract the dominant emotion
dominant_emotion = result[0]['emotion']

# Extract the dominant emotion label
emotion_label = max(dominant_emotion, key=dominant_emotion.get)

# Convert to title case for consistent labeling
emotion_label = emotion_label.title()

# Display the dominant emotion
print("Dominant emotion:", emotion_label)

```

    Downloading...
    From: https://drive.google.com/uc?id=1xKtqLlRjvj8lAaxz9UE2KWPlIJjfwC6q
    To: /content/image.jpg
    100%|██████████| 3.54M/3.54M [00:00<00:00, 43.5MB/s]


    Dominant emotion: Angry

