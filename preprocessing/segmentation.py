import cv2 as cv
import numpy as np

def otsu_thresholding(input: np.ndarray, use_blur:bool = False, kernel_size = (3, 3)) -> np.ndarray:
    if use_blur:
        input = cv.GaussianBlur(input,kernel_size,0)
    _, bin_image = cv.threshold(input,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return bin_image
