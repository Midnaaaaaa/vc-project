import numpy as np
import cv2
import os

def haarCascade(video, scaleFactor, minNeighbors, minSize):
    base_path = video + '/img/'

    if video == "Bike":
        video = "MotorcycleChase"

    casc=cv2.CascadeClassifier(f'{video}.xml')
    # Check if the cascade classifier is loaded successfully
    if casc.empty():
        raise Exception("Failed to load cascade classifier.")

    Idir = os.listdir(base_path)
    paths = list(map(lambda img_name : base_path + img_name, Idir))
    for img_path in paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))
        for (x,y,w,h) in results:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow('img',img)
        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

haarCascade("MotorcycleChase", scaleFactor=1.1, minNeighbors=4, minSize=(100,100))
#haarCascade("PolarBear", scaleFactor=1.05, minNeighbors=4, minSize=(40,40))
#haarCascade("Elephants", scaleFactor=1.3, minNeighbors=8, minSize=(100,100))
#haarCascade("Bike", scaleFactor=1.1, minNeighbors=4, minSize=(100,100))
