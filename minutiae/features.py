import skimage
import numpy as np
import math
from skimage.morphology import convex_hull_image, erosion, square
from skimage.measure import label, regionprops
import cv2 as cv

class MinutiaeFeature:
    def __init__(self, X, Y, orientation, feature_type) -> None:
        self.X = X
        self.Y = Y
        self.orientation = orientation
        self.feature_type = feature_type

class FeatureExtractor:
    def __init__(self) -> None:
        self._mask = []
        self._skeleton = []
        self.minutiae_termination = []
        self.minutiae_bifurcation = []
        self._spurious_minutiae_thresh = 10

    def _skeletonise(self, img):
        img //= img.max()
        self._skeleton = skimage.morphology.skeletonize(img)
        self._skeleton = np.uint8(self._skeleton) * 255
        self._mask = img * 255

    def _compute_angle(self, block, minutiae_type):
        angle = []
        block_rows, block_cols = np.shape(block)
        center_x, center_y = (block_rows - 1) / 2, (block_cols - 1) / 2
        sum_val = 0
        if minutiae_type.lower() == 'termination':
            for i in range(block_rows):
                for j in range(block_cols):
                    if (i == 0 or i == block_rows - 1 or j == 0 or j == block_cols - 1) and block[i][j] != 0:
                        angle.append(-math.degrees(math.atan2(i - center_x, j - center_y)))
                        sum_val += 1
                        if sum_val > 1:
                            angle.append(math.nan)
            return angle
        elif minutiae_type.lower() == 'bifurcation':
            for i in range(block_rows):
                for j in range(block_cols):
                    if (i == 0 or i == block_rows - 1 or j == 0 or j == block_cols - 1) and block[i][j] != 0:
                        angle.append(-math.degrees(math.atan2(i - center_x, j - center_y)))
                        sum_val += 1
            if sum_val != 3:
                angle.append(math.nan)
            return angle

    def _get_terminations_bifurcations(self):
        self._skeleton = self._skeleton == 255
        (rows, cols) = self._skeleton.shape
        self.minutiae_termination = np.zeros(self._skeleton.shape)
        self.minutiae_bifurcation = np.zeros(self._skeleton.shape)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if self._skeleton[i][j] == 1:
                    block = self._skeleton[i - 1: i + 2, j - 1: j + 2]
                    block_val = np.sum(block)
                    if block_val == 2:
                        self.minutiae_termination[i, j] = 1
                    elif block_val == 4:
                        self.minutiae_bifurcation[i, j] = 1

        self._mask = convex_hull_image(self._mask > 0) 
        self._mask = erosion(self._mask, square(5))
        self.minutiae_termination = np.uint8(self._mask) * self.minutiae_termination

    def _remove_spurious(self, minutiae_list, img):
        img = img * 0
        spurious_min = []
        num_points = len(minutiae_list)
        D = np.zeros((num_points, num_points))
        for i in range(1, num_points):
            for j in range(0, i):
                (x1, y1) = minutiae_list[i]['centroid']
                (x2, y2) = minutiae_list[j]['centroid']

                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                D[i][j] = dist
                if dist < self._spurious_minutiae_thresh:
                    spurious_min.append(i)
                    spurious_min.append(j)
        spurious_min = np.unique(spurious_min)
        for i in range(0, num_points):
            if i not in spurious_min:
                x, y = np.int16(minutiae_list[i]['centroid'])
                img[x, y] = 1
        img = np.uint8(img)
        return img

    def _clean_minutiae(self, img):
        self.minutiae_termination = label(self.minutiae_termination, connectivity=2)
        rp = regionprops(self.minutiae_termination) 
        self.minutiae_termination = self._remove_spurious(rp, np.uint8(img))

    def _feature_extraction(self):
        FeatureTerm = []
        self.minutiae_termination = label(self.minutiae_termination, connectivity=2)
        rp = regionprops(np.uint8(self.minutiae_termination))
        window_size = 2
        for num, i in enumerate(rp):
            (row, col) = np.int16(np.round(i['Centroid']))
            block = self._skeleton[row - window_size:row + window_size + 1, col - window_size: col + window_size + 1]
            angle = self._compute_angle(block, "Termination")
            if len(angle) == 1:
                FeatureTerm.append(MinutiaeFeature(row, col, angle, 'Termination'))


        FeatureBif = []
        self.minutiae_bifurcation = label(self.minutiae_bifurcation, connectivity=2)
        rp = regionprops(np.uint8(self.minutiae_bifurcation))
        window_size = 1
        for i in rp:
            row, col = np.int16(np.round(i['Centroid']))
            block = self._skeleton[row-window_size:row+window_size+1, col-window_size:col+window_size+1]
            angle = self._compute_angle(block, "Bifurcation")
            if len(angle) == 3:
                FeatureBif.append(MinutiaeFeature(row, col, angle, 'Bifurcation'))

        return (FeatureTerm, FeatureBif)

    def extract_features(self, img):
        self._skeletonise(img)
        self._get_terminations_bifurcations()
        self._clean_minutiae(img)
        return self._feature_extraction()

    def show_results(self, img):
        FeatureTerm, FeatureBif = self.extract_features(img)
        rows, cols = self._skeleton.shape
        DispImg = np.zeros((rows, cols, 3), np.uint8)
        DispImg[:, :, 0] = 255*self._skeleton
        DispImg[:, :, 1] = 255*self._skeleton
        DispImg[:, :, 2] = 255*self._skeleton

        for idx, curr_minutiae in enumerate(FeatureTerm):
            row, col = curr_minutiae.X, curr_minutiae.Y
            (rr, cc) = skimage.draw.disk((row, col), 5)
            skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

        for idx, curr_minutiae in enumerate(FeatureBif):
            row, col = curr_minutiae.X, curr_minutiae.Y
            (rr, cc) = skimage.draw.disk((row, col), 5)
            skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))
        
        return DispImg

    def get_sift_descriptor(self, img):
        FeatureTerm, FeatureBif = self.extract_features(img)
        key_points = [cv.KeyPoint(fe.Y, fe.X, 3, fe.orientation[0]) for fe in FeatureTerm]
        key_points.extend([cv.KeyPoint(fe.Y, fe.X, 5, fe.orientation[0]) for fe in FeatureBif])
        sift = cv.SIFT_create()
        return sift.compute(img, key_points)
        


