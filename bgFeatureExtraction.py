from sklearn.cluster import KMeans
from collections import Counter
import cv2 
import numpy as np


class bgFeatureExtractor():

    # step 1 read img
    # step 2 extract face and wipe out from img
    # step 3 get all remaining pixels to vector
    # step 4 find dominant color

    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        
    def run(self, image, filename):
        face_rect = self.get_face_rect(image=image)
        pixel_vec = self.add_mask_and_transform_to_pixel_vector(image=image, rect=face_rect)
        colors = self.get_main_colors(pixel_vec)
        return colors, filename

    def get_face_rect(self, image, scaleFactor = 2):
        image_copy = image.copy()
        gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        faces_rect = self.face_cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors = 5)
        
        if len(faces_rect) ==0:
            img_shape = np.shape(image)
            faces_rect = (img_shape[0] * 0.25, img_shape[0]* 0.25, img_shape[1] * 0.75, img_shape[1] * 0.75)
            return faces_rect
        return (faces_rect[0][0], faces_rect[0][1], faces_rect[0][0]+faces_rect[0][2], faces_rect[0][1]+faces_rect[0][3])
        

    def add_mask_and_transform_to_pixel_vector(self, image, rect):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.ones(np.shape(image)[:2], dtype=bool)
        mask[rect[0]: rect[2], rect[1] : rect[3]] = False
        result = image[mask]
        return result

    def get_main_colors(self, pixel_vec, k=4, colors_count = 7):
        if colors_count > k:
            k = colors_count+1
        clt = KMeans(n_clusters = k, n_init = 10)
        labels = clt.fit_predict(pixel_vec)
        label_count = Counter(labels)
        most_common_colors = label_count.most_common(colors_count)
        indexes = [col[0] for col in most_common_colors]
        main_colors = clt.cluster_centers_[indexes]
        test = np.ones((300,300,3), dtype=np.uint8)
        test[:100] = main_colors[0]
        test[100:200] = main_colors[1]
        test[200:] = main_colors[2]
        return list(main_colors)

bgExtractor = bgFeatureExtractor()

