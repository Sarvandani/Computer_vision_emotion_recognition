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



```python
!pip install deepface

```

    Collecting deepface
      Downloading deepface-0.0.84-py3-none-any.whl (87 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m87.7/87.7 kB[0m [31m2.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy>=1.14.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (1.25.2)
    Requirement already satisfied: pandas>=0.23.4 in /usr/local/lib/python3.10/dist-packages (from deepface) (1.5.3)
    Requirement already satisfied: gdown>=3.10.1 in /usr/local/lib/python3.10/dist-packages (from deepface) (4.7.3)
    Requirement already satisfied: tqdm>=4.30.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (4.66.2)
    Requirement already satisfied: Pillow>=5.2.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (9.4.0)
    Requirement already satisfied: opencv-python>=4.5.5.64 in /usr/local/lib/python3.10/dist-packages (from deepface) (4.8.0.76)
    Requirement already satisfied: tensorflow>=1.9.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (2.15.0)
    Requirement already satisfied: keras>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (2.15.0)
    Requirement already satisfied: Flask>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from deepface) (2.2.5)
    Collecting mtcnn>=0.1.0 (from deepface)
      Downloading mtcnn-0.1.1-py3-none-any.whl (2.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.3/2.3 MB[0m [31m29.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting retina-face>=0.0.1 (from deepface)
      Downloading retina_face-0.0.14-py3-none-any.whl (23 kB)
    Collecting fire>=0.4.0 (from deepface)
      Downloading fire-0.5.0.tar.gz (88 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m88.3/88.3 kB[0m [31m13.6 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
    Collecting gunicorn>=20.1.0 (from deepface)
      Downloading gunicorn-21.2.0-py3-none-any.whl (80 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m80.2/80.2 kB[0m [31m12.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from fire>=0.4.0->deepface) (1.16.0)
    Requirement already satisfied: termcolor in /usr/local/lib/python3.10/dist-packages (from fire>=0.4.0->deepface) (2.4.0)
    Requirement already satisfied: Werkzeug>=2.2.2 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.1.2->deepface) (3.0.1)
    Requirement already satisfied: Jinja2>=3.0 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.1.2->deepface) (3.1.3)
    Requirement already satisfied: itsdangerous>=2.0 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.1.2->deepface) (2.1.2)
    Requirement already satisfied: click>=8.0 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.1.2->deepface) (8.1.7)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from gdown>=3.10.1->deepface) (3.13.1)
    Requirement already satisfied: requests[socks] in /usr/local/lib/python3.10/dist-packages (from gdown>=3.10.1->deepface) (2.31.0)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from gdown>=3.10.1->deepface) (4.12.3)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from gunicorn>=20.1.0->deepface) (23.2)
    Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.23.4->deepface) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.23.4->deepface) (2023.4)
    Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (1.4.0)
    Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (1.6.3)
    Requirement already satisfied: flatbuffers>=23.5.26 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (23.5.26)
    Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (0.5.4)
    Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (0.2.0)
    Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (3.9.0)
    Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (16.0.6)
    Requirement already satisfied: ml-dtypes~=0.2.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (0.2.0)
    Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (3.3.0)
    Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (3.20.3)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (67.7.2)
    Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (4.9.0)
    Requirement already satisfied: wrapt<1.15,>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (1.14.1)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (0.36.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (1.60.1)
    Requirement already satisfied: tensorboard<2.16,>=2.15 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (2.15.2)
    Requirement already satisfied: tensorflow-estimator<2.16,>=2.15.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (2.15.0)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from astunparse>=1.6.0->tensorflow>=1.9.0->deepface) (0.42.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from Jinja2>=3.0->Flask>=1.1.2->deepface) (2.1.5)
    Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow>=1.9.0->deepface) (2.27.0)
    Requirement already satisfied: google-auth-oauthlib<2,>=0.5 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow>=1.9.0->deepface) (1.2.0)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow>=1.9.0->deepface) (3.5.2)
    Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow>=1.9.0->deepface) (0.7.2)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->gdown>=3.10.1->deepface) (2.5)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown>=3.10.1->deepface) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown>=3.10.1->deepface) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown>=3.10.1->deepface) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown>=3.10.1->deepface) (2024.2.2)
    Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown>=3.10.1->deepface) (1.7.1)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow>=1.9.0->deepface) (5.3.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow>=1.9.0->deepface) (0.3.0)
    Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow>=1.9.0->deepface) (4.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow>=1.9.0->deepface) (1.3.1)
    Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /usr/local/lib/python3.10/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow>=1.9.0->deepface) (0.5.1)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.10/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow>=1.9.0->deepface) (3.2.2)
    Building wheels for collected packages: fire
      Building wheel for fire (setup.py) ... [?25l[?25hdone
      Created wheel for fire: filename=fire-0.5.0-py2.py3-none-any.whl size=116934 sha256=fea8f2256448904100f1c80298e5a31261ccef0d2b29356b507cb99adeaeffa0
      Stored in directory: /root/.cache/pip/wheels/90/d4/f7/9404e5db0116bd4d43e5666eaa3e70ab53723e1e3ea40c9a95
    Successfully built fire
    Installing collected packages: gunicorn, fire, mtcnn, retina-face, deepface
    Successfully installed deepface-0.0.84 fire-0.5.0 gunicorn-21.2.0 mtcnn-0.1.1 retina-face-0.0.14

