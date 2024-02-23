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

