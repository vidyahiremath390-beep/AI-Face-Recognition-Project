import cv2
import os
import numpy as np

def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    ids = []

    for root, dirs, files in os.walk("dataset"):
        label = os.path.basename(root)
        if not label.isdigit():
            continue

        for file in files:
            if file.endswith("jpg"):
                path = os.path.join(root, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, (200, 200))
                faces.append(img)
                ids.append(int(label))

    if len(faces) == 0:
        print("No training images found. Model was not updated.")
        return

    recognizer.train(faces, np.array(ids))

    if not os.path.exists("trainer"):
        os.makedirs("trainer")

    recognizer.save("trainer/trainer.yml")
    print("Model trained successfully!")