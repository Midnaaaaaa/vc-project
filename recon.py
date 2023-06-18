import numpy as np
import cv2
import os

casc=cv2.CascadeClassifier('cascade.xml')

# Check if the cascade classifier is loaded successfully
if casc.empty():
    raise Exception("Failed to load cascade classifier.")

base_path = 'Bike/img/'
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
