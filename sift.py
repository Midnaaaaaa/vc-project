import cv2
import numpy as np
import os

def sift_tracking(video):
    BB = np.genfromtxt(video + '/groundtruth_rect.txt', delimiter=',', dtype=int)

    base_path = video+'/img/'
    scale = 1
    Idir = os.listdir(base_path)  # Directorio que contiene las imágenes
    nf = len(Idir)  # Número total de imágenes
    i = 0
    filename = Idir[i]  # Nombre del archivo de imagen
    I = cv2.resize(cv2.imread(base_path + filename), None, fx=scale, fy=scale)  # Leer y redimensionar la imagen

    # Coordenadas del rectángulo de interés
    rect = (BB[0, 1] * scale, BB[0, 2] * scale, BB[0, 3] * scale, BB[0, 4] * scale)
    IC = I[int(rect[1]):int(rect[1] + rect[3]), int(rect[0]):int(rect[0] + rect[2])]  # Recortar la imagen

    IQ = cv2.rectangle(I, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 255, 0), 2)  # Dibujar el rectángulo de interés en la imagen

    for i in range(1, len(Idir)):
        im_obj = cv2.cvtColor(IC, cv2.COLOR_BGR2GRAY)  # Convertir la imagen a escala de grises
        filename = Idir[i]
        I2 = cv2.resize(cv2.imread(base_path + filename), None, fx=scale, fy=scale)

        im_esc = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()  # Crear el objeto SIFT
        kp_obj, feat_obj = sift.detectAndCompute(im_obj, None)  # Detectar y describir características SIFT en la imagen de referencia
        kp_esc, feat_esc = sift.detectAndCompute(im_esc, None)  # Detectar y describir características SIFT en la imagen actual

        bf = cv2.BFMatcher()  # Crear el objeto correspondiente
        matches = bf.match(feat_obj, feat_esc)  # Encontrar los puntos coincidentes más cercanos
        matches = sorted(matches, key = lambda x:x.distance)

        # Filter good matches based on a distance threshold
        good_matches = [m for m in matches if m.distance < 150]

        #good_matches = matches[00:50]

        #print(good_matches)

        im3 = (cv2.drawMatches(im_obj, kp_obj, im_esc, kp_esc, good_matches, im_esc))
        cv2.imshow('SIFT', im3)
        cv2.waitKey(0)

        # Find transformation between matched keypoints
        src_points = np.float32([kp_obj[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp_esc[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        #M, _ = cv2.getAffineTransform(src_points, dst_points)
        #M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSA, 5.0)
        M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        h, w = im_obj.shape[:2]
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        #pts = np.float32([[0, 0], [rect[2], 0], [rect[2], rect[3]], [0, rect[3]]]).reshape(-1, 1, 2)
        #transformed_pts = cv2.warpAffine(pts, M, (I2.shape[0], I2.shape[1]))
        transformed_pts = cv2.perspectiveTransform(pts, M)
        x_min_new = int(np.min(transformed_pts[:, 0, 0]))
        y_min_new = int(np.min(transformed_pts[:, 0, 1]))
        x_max_new = int(np.max(transformed_pts[:, 0, 0]))
        y_max_new = int(np.max(transformed_pts[:, 0, 1]))

        #print(M)
        #print(pts)
        #print(transformed_pts)


        IC = I2[int(y_min_new):int(y_max_new), int(x_min_new):int(x_max_new)]  # Recortar la imagen

        IQ = cv2.rectangle(I2, (x_min_new, y_min_new), (x_max_new, y_max_new), (0, 255, 0), 2);

        #cv2.imshow('SIFT', im3)
        cv2.imshow('Bounding Box', IQ)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



sift_tracking('MotorcycleChase')
