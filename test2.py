import glob
import numpy as np
import cv2
from skimage import io, color, transform
from matplotlib import pyplot as plt

def load_annotations(filename):
    return np.loadtxt(filename, dtype=int, delimiter=',')

def open_image(path, scale=1.0):
    image = io.imread(path)
    return transform.rescale(image, scale)

def normalize_image(image):
    normalized_image = (image * 255 / np.max(image)).astype(np.uint8)
    return normalized_image


def plot_rect(image, rect):
    image_rgb = cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color=(0, 255, 0), thickness=1)
    plt.imshow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))

annotations_file = 'PolarBear1/groundtruth_rect.txt'
image_dir = 'PolarBear1/img/*.jpg'

bb = load_annotations(annotations_file)
idir = sorted(glob.glob(image_dir))
scale = 1

nf = len(idir)
i = 0
image = open_image(idir[i], scale)

rect = bb[i, 1:] * scale
cropped_image = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2], :]

plot_rect(normalize_image(image), rect)
plt.show()

for i in range(1, 50):
    im_obj = normalize_image(color.rgb2gray(cropped_image))
    image2 = open_image(idir[i], scale)
    im_esc = normalize_image(color.rgb2gray(image2))

    kp_obj, desc_obj = cv2.xfeatures2d.SIFT_create().detectAndCompute(im_obj, None)
    kp_esc, desc_esc = cv2.xfeatures2d.SIFT_create().detectAndCompute(im_esc, None)

    bf = cv2.BFMatcher()
    matches = bf.match(desc_obj, desc_esc)

    good_matches = []
    for m in matches:
        if m.distance < 10:
            good_matches.append(m)

    match_img = cv2.drawMatches(im_obj, kp_obj, im_esc, kp_esc, good_matches, None)
    plt.imshow(match_img)
    plt.show()

    src_pts = np.float32([kp_obj[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_esc[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    t, _ = cv2.estimateAffine2D(src_pts, dst_pts)

    f, c = im_obj.shape

    box = np.float32([[0, 0], [0, f - 1], [c - 1, f - 1], [c - 1, 0], [0, 0]])
    nbox = cv2.transform(np.float32([box]), t)
    bbox = np.int32([[nbox[0][0], nbox[0][1], nbox[0][2] - nbox[0][0], nbox[0][3] - nbox[0][1]]])

    cropped_image = image2[bbox[0][1]:(bbox[0][1] + bbox[0][3]), bbox[0][0]:(bbox[0][0] + bbox[0][2]), :]

    overlap_ratio = cv2.intersection_over_union(bbox[0], bb[i, 1:] * scale)[0]

print(overlap_ratio)