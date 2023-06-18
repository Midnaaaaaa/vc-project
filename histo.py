import cv2
import os
import numpy as np

def get_histogram(image, roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist

def histogram_tracking(video):

    BB = np.genfromtxt(video + '/groundtruth_rect.txt', delimiter=',', dtype=int)

    path = video + '/img/'
    images = os.listdir(path)
    scale = 1
    # Inicialización
    frame = cv2.imread(path + images[0])
    x, y, w, h = BB[0, 1] * scale, BB[0, 2] * scale, BB[0, 3] * scale, BB[0, 4] * scale  # ROI inicial (modificar según la posición del objeto en la primera imagen)
    roi = frame[y:y+h, x:x+w]
    target_hist = get_histogram(frame, roi)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # Procesamiento de imágenes
    for image in images[1:]:
        frame = cv2.imread(path + image)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], target_hist, [0, 180], 1)

        # Seguimiento y actualización de la ROI
        _, track_window = cv2.meanShift(dst, (x, y, w, h), term_crit)
        x, y, w, h = track_window


        # Visualización
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Tracking', frame)

        key = cv2.waitKey(60) & 0xFF
        if key == 27:  # tecla ESC para salir
            break

    cv2.destroyAllWindows()

histogram_tracking('Elephants')
