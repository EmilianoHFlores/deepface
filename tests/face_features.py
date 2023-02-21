#feb 16 2023

import matplotlib.pyplot as plt
import numpy as np
import cv2
from deepface import DeepFace
from deepface.commons import functions

model_names = [
    "Facenet512",
    "DeepFace",
]

detector_backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]

img_path = "dataset/img1.jpg"
cv_img = cv2.imread(img_path)
cv_img = cv2.resize(cv_img, (100, 200))
cv2.imshow("Photo", cv_img)

features = DeepFace.analyze(img_path)



print(features)

cv2.waitKey(0)

cv2.destroyAllWindows()
