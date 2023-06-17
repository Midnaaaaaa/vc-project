import numpy as np
import cv2

casc=cv2.CascadeClassifier('stage3.xml')

img = cv2.imread('Alladin/img/00001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
results = casc.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in results:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
