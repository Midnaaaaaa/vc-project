from cv2 import cv2
import numpy as np


def sift_tracking():
    BB = np.genfromtxt('MotorcycleChase/groundtruth_rect.txt', delimiter=',', dtype=int)

    scale = 1
    Idir = ['MotorcycleChase/img/']  # Directorio que contiene las imágenes
    nf = len(Idir)  # Número total de imágenes
    i = 0
    filename = Idir[i]  # Nombre del archivo de imagen
    I = cv2.resize(cv2.imread(filename), None, fx=scale, fy=scale)  # Leer y redimensionar la imagen

    # Coordenadas del rectángulo de interés
    rect = (BB[0, 2] * scale, BB[0, 3] * scale, BB[0, 4] * scale, BB[0, 5] * scale)
    IC = I[int(rect[1]):int(rect[1] + rect[3]), int(rect[0]):int(rect[0] + rect[2])]  # Recortar la imagen

    IQ = cv2.rectangle(I, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 255, 0), 2)  # Dibujar el rectángulo de interés en la imagen
    cv2.imshow('Image', IQ)

    for i in range(1, 50):
        im_obj = cv2.cvtColor(IC, cv2.COLOR_BGR2GRAY)  # Convertir la imagen a escala de grises
        filename = Idir[i]
        I2 = cv2.resize(cv2.imread(filename), None, fx=scale, fy=scale)

        im_esc = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()  # Crear el objeto SIFT
        kp_obj, feat_obj = sift.detectAndCompute(im_obj, None)  # Detectar y describir características SIFT en la imagen de referencia
        kp_esc, feat_esc = sift.detectAndCompute(im_esc, None)  # Detectar y describir características SIFT en la imagen actual

        bf = cv2.BFMatcher()  # Crear el objeto correspondiente
        matches = bf.knnMatch(feat_obj, feat_esc, k=2)  # Encontrar los puntos coincidentes más cercanos

        # Aplicar el criterio de selección de relación de distancia más cercana
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        # Obtener los puntos coincidentes en ambas imágenes
        points_obj = np.float32([kp_obj[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        points_esc = np.float32([kp_esc[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Calcular la transformación afín
        T, _ = cv2.estimateAffine2D(points_obj, points_esc)

        # Obtener las dimensiones de la imagen de referencia
        f, c = im_obj.shape

        # Definir los puntos de la caja delimitadora en la imagen de referencia
        box = np.float32([[0, 0], [0, f], [c, f], [c, 0]])

        # Transformar los puntos de la caja delimitadora a la imagen actual
        nbox = cv2.transform(box.reshape(-1, 1, 2), T)
        bbox = cv2.boundingRect(nbox.reshape(-1, 2))

        IC = I2[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]  # Recortar la imagen actual

        # Calcular el overlap, intersection over union, de las ventanas
        overlapRatio = cv2.matchShapes(np.array(BB[i, 2:6]), np.array(bbox), cv2.CONTOURS_MATCH_I1, 0)

        # Mostrar los resultados en la imagen actual
        cv2.imshow('Image', I2)
        cv2.rectangle(I2, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)  # Dibujar el rectángulo predicho en la imagen actual
        cv2.rectangle(I2, (BB[i, 2], BB[i, 3]), (BB[i, 2] + BB[i, 4], BB[i, 3] + BB[i, 5]), (0, 255, 255), 2)  # Dibujar el rectángulo correcto en la imagen actual
        cv2.waitKey(0)

