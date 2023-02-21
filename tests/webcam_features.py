# feb 17 2023

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import cv2
from deepface import DeepFace
from deepface.commons import functions

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]

detector_backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]

img_path = "dataset/img1.jpg"

cam = cv2.VideoCapture(0)

captured, image = cam.read()
features_arr = DeepFace.analyze(img_path=image, enforce_detection=False, silent=True)
print(type(image))

while captured:
    captured, image = cam.read()

    image = cv2.resize(image, (640, 480))

    cv2.imshow("Webcam", image)

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == ord("s"):
        print("Scanning...")
        try:
            features_arr = DeepFace.analyze(img_path=image, enforce_detection=True)
            print(features_arr)
            for features in features_arr:
                print("Age: ", features.get("age"))
                print("Gender: ", features.get("dominant_gender"))
                print("Race: ", features.get("dominant_race"))
                print("Emotion: ", features.get("dominant_emotion"))
                region = features.get("region")
                print(region["x"])
                print("Finished scan")
        except:
            print("No face detected")
