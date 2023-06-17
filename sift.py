import cv2
import numpy as np
import os

def sift_tracking():
    BB = np.genfromtxt('MotorcycleChase/groundtruth_rect.txt', delimiter=',', dtype=int)

    base_path = 'MotorcycleChase/img/'
    scale = 1
    Idir = os.listdir('MotorcycleChase/img/')  # Directorio que contiene las imágenes
    nf = len(Idir)  # Número total de imágenes
    i = 0
    filename = Idir[i]  # Nombre del archivo de imagen
    I = cv2.resize(cv2.imread(base_path + filename), None, fx=scale, fy=scale)  # Leer y redimensionar la imagen

    # Coordenadas del rectángulo de interés
    rect = (BB[0, 1] * scale, BB[0, 2] * scale, BB[0, 3] * scale, BB[0, 4] * scale)
    IC = I[int(rect[1]):int(rect[1] + rect[3]), int(rect[0]):int(rect[0] + rect[2])]  # Recortar la imagen

    IQ = cv2.rectangle(I, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 255, 0), 2)  # Dibujar el rectángulo de interés en la imagen
    cv2.imshow('Image', IC)
    cv2.waitKey(0)

    for i in range(1, 50):
        im_obj = cv2.cvtColor(IC, cv2.COLOR_BGR2GRAY)  # Convertir la imagen a escala de grises
        filename = Idir[i]
        I2 = cv2.resize(cv2.imread(base_path + filename), None, fx=scale, fy=scale)

        im_esc = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()  # Crear el objeto SIFT
        kp_obj, feat_obj = sift.detectAndCompute(im_obj, None)  # Detectar y describir características SIFT en la imagen de referencia
        kp_esc, feat_esc = sift.detectAndCompute(im_esc, None)  # Detectar y describir características SIFT en la imagen actual

        bf = cv2.BFMatcher()  # Crear el objeto correspondiente
        matches = bf.match(feat_obj, feat_esc)  # Encontrar los puntos coincidentes más cercanos
        matches = sorted(matches, key=lambda x: x.distance)

        # Filter good matches based on a distance threshold
        good_matches = [m for m in matches if m.distance < 150]

        # Find transformation between matched keypoints
        src_pts = np.float32([kp_obj[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_esc[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        # Transform the bounding box from the previous frame
        rect_pts = np.float32([[rect[0], rect[1]], [rect[0] + rect[2], rect[1]], [rect[0], rect[1] + rect[3]], [rect[0] + rect[2], rect[1] + rect[3]]]).reshape(-1, 1, 2)
        transformed_pts = cv2.transform(rect_pts, M)

        # Update the rect values based on the transformed bounding box
        rect = (min(transformed_pts[:, 0, 0]), min(transformed_pts[:, 0, 1]), max(transformed_pts[:, 0, 0]) - min(transformed_pts[:, 0, 0]), max(transformed_pts[:, 0, 1]) - min(transformed_pts[:, 0, 1]))

        # Draw the transformed bounding box on the current frame using cv2.rectangle
        I2 = cv2.rectangle(I2, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 255, 0), 2)

        #cv2.imshow('SIFT', im3)
        cv2.imshow('Bounding Box', I2)
        cv2.waitKey(0)

sift_tracking()
