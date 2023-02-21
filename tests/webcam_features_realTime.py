# feb 17 2023

import os
from numba import cuda
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import numpy as np
import cv2
from deepface import DeepFace
from deepface.commons import functions

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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

with tf.device('/CPU:0'):
    print("building age model...")
    DeepFace.build_model("Age")
    DeepFace.build_model("Emotion")
    DeepFace.build_model("Race")
    DeepFace.build_model("Gender")

    cam = cv2.VideoCapture(0)

    captured, image = cam.read()
    features_arr = DeepFace.analyze(img_path=image, enforce_detection=False, silent=True)
    print(type(image))

    while captured:
        captured, image = cam.read()

        image = cv2.resize(image, (320, 240))
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
                x_coord = region["x"]
                y_coord = region["y"]
                width = region["w"]
                height = region["h"]
                color = (0, 255, 0)
                image = cv2.rectangle(
                    image, (x_coord, y_coord), (x_coord + width, y_coord + height), color, 2
                )
                cv2.putText(
                    image,
                    f'Age: {features.get("age")}, {features.get("dominant_gender")}, {features.get("dominant_race")}, {features.get("dominant_emotion")}',
                    (x_coord, y_coord - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (36, 255, 12),
                    2,
                )
        except:
            print("No face detected")

        cv2.imshow("Webcam", image)

        key = cv2.waitKey(1)

        if key == 27:
            break
