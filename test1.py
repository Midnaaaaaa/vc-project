import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, color, transform
from skimage.feature import match_descriptors, plot_matches

BB = np.loadtxt('PolarBear1/groundtruth_rect.txt', delimiter=',')
base_path = "PolarBear1/img/"

scale = 1

nf = len(BB)  # nombre total de fitxers imatges
Idir = os.listdir(base_path)  # Directorio que contiene las imágenes
nf = len(Idir)  # Número total de imágenes
i = 0
filename = base_path + Idir[i]  # Nombre del archivo de imagen
I = cv2.resize(cv2.imread(filename), (0, 0), fx=scale, fy=scale)

rect = [int(val * scale) for val in BB[0, 1:]]
IC = I[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]

IQ = cv2.rectangle(I.copy(), (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
plt.imshow(IQ)
plt.show()

orb = cv2.ORB_create()

for i in range(1, 10):
    im_obj = IC.copy()
    filename = base_path + Idir[i]
    I2 = cv2.resize(cv2.imread(filename), (0, 0), fx=scale, fy=scale)
    im_esc = I2.copy()
    
    kp_obj, feat_obj = orb.detectAndCompute(im_obj, None)
    kp_esc, feat_esc = orb.detectAndCompute(im_esc, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    pairs = bf.match(feat_obj, feat_esc)
    
    # Ordenem les corresponencies per distància, les millors a l'inici
    pairs = sorted(pairs, key=lambda x: x.distance)
    
    m_kp_obj = [kp_obj[p.queryIdx].pt for p in pairs]
    m_kp_esc = [kp_esc[p.trainIdx].pt for p in pairs]
    
    plt.figure()
    out_img = cv2.drawMatches(im_obj, kp_obj, im_esc, kp_esc, pairs[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(out_img)
    plt.show()

    T_matrix = cv2.estimateAffinePartial2D(np.array([m_kp_obj]), 
                                           np.array([m_kp_esc]))
    if T_matrix[0] is True:
        T = T_matrix[1]
    else:
        continue
    
    f, c = im_obj.shape

    box = np.array([[1, 1], [1, f], [c, f], [c, 1], [1, 1]])
    nbox = cv2.transform(np.array([box]), T)[0]
    bbox = [nbox[0, :], nbox[2, :] - nbox[0, :]]
    IC = I2[int(bbox[0][1]):int(bbox[0][1] + bbox[1][1]), int(bbox[0][0]):int(bbox[0][0] + bbox[1][0])]

    def bbox_overlap_ratio(b1, b2):
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[0] + b1[2], b2[0] + b2[2])
        yB = min(b1[1] + b1[3], b2[1] + b2[3])
        
        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        box1_area = b1[2] * b1[3]
        box2_area = b2[2] * b2[3]
        
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    overlapRatio = bbox_overlap_ratio(bbox, [val * scale for val in BB[i, 1:]])
    print(overlapRatio)