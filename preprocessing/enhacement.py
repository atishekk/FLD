import numpy as np
import scipy
import cv2 as cv
import math

class ImageEnhancer:
    def __init__(self) -> None:
        """Initialise hyperparameters for the algorithm"""
        self.ridge_block_size = 16
        self.ridge_segment_thresh = 0.1
        self.gradient_sigma = 1
        self.block_sigma = 7
        self.orient_smooth_sigma = 7
        self.freq_block_size = 38
        self.freq_window_size = 5
        self.min_wavelength = 5
        self.max_wavelength = 15
        self.kx = 0.65
        self.ky = 0.65
        self.angle_inc = 3
        self.ridge_filter_thresh = -3

        self._enhanced = np.array([]) 

    def _normalise(self, img: np.ndarray) -> np.ndarray:
        if img.std() == 0:
            raise ValueError("Zero standard deviation for the image")
        normalised = (img - img.mean()) / img.std()
        return normalised

    def _ridge_segment(self, img: np.ndarray):
        """
        Generate the mask identifying the fingerprint region of the image
        also normalises the image to zero mean and unit standard deviation
        """
        r, c = img.shape
        normalised = self._normalise(img)

        rows = np.intc(self.ridge_block_size * np.ceil(np.double(r) / np.double(self.ridge_block_size)))
        cols = np.intc(self.ridge_block_size * np.ceil(np.double(c) / np.double(self.ridge_block_size)))

        padded_img = np.zeros((rows, cols))
        padded_img[0:r][:, 0:c] = normalised
        stddev_img = np.zeros((rows, cols))

        for i in range(0, rows, self.ridge_block_size):
            for j in range(0, cols, self.ridge_block_size):
                block = padded_img[i: i + self.ridge_block_size][:, j:j+self.ridge_block_size]
                stddev_img[i:i+self.ridge_block_size][:, j:j+self.ridge_block_size] = block.std() * np.ones(block.shape)

        stddev_img = stddev_img[0:r][:, 0:c]
        self._mask = stddev_img > self.ridge_segment_thresh
        mean = normalised[self._mask].mean()
        std = normalised[self._mask].std()
        self._norm_img = (normalised - mean) / std

    def _orientation(self):
        """
        Ridge Orientation estimation for the image
        """
        r, c = self._norm_img.shape
        # kernel size
        sze = np.fix(6*self.gradient_sigma)
        if np.remainder(sze,2) == 0:
            sze += 1

        # building the Gaussian Kernel
        gauss = cv.getGaussianKernel(np.intc(sze), self.gradient_sigma)
        f = gauss * gauss.T

        # derivative of Gaussian
        fy, fx = np.gradient(f)

        Gx = scipy.signal.convolve2d(self._norm_img, fx, mode='same')
        Gy = scipy.signal.convolve2d(self._norm_img, fy, mode='same')

        Gxx = np.power(Gx, 2)
        Gyy = np.power(Gy, 2)
        Gxy = Gx * Gy

        # block orientation estimation
        sze = np.fix(6*self.block_sigma)
        if np.remainder(sze, 2) == 0:
            sze += 1
        gauss = cv.getGaussianKernel(np.intc(sze), self.block_sigma)
        f = gauss * gauss.T

        Gxx = scipy.ndimage.convolve(Gxx, f)
        Gyy = scipy.ndimage.convolve(Gyy, f)
        Gxy = 2*scipy.ndimage.convolve(Gxy, f)

        denom = np.sqrt(np.power(Gxy, 2) + np.power((Gxx - Gyy) , 2)) + np.finfo(np.double).eps

        sin2theta = Gxy / denom
        cos2theta = (Gxx - Gyy)/denom

        # smoothing using continuous vector fields
        if self.orient_smooth_sigma:
            sze = np.fix(6*self.orient_smooth_sigma)
            if np.remainder(sze, 2) == 0:
                sze += 1
            gauss = cv.getGaussianKernel(np.intc(sze), self.orient_smooth_sigma)
            f = gauss * gauss.T
            cos2theta = scipy.ndimage.convolve(cos2theta, f)
            sin2theta = scipy.ndimage.convolve(sin2theta, f)

        self._orient_img = np.pi / 2 + np.arctan2(sin2theta, cos2theta)/2


    def _freq_block(self, block_norm, block_orient):
        """
        Ridge frequency estimation for a block
        """
        rows, cols = block_norm.shape

        # calculate the orientation angle for the block
        cos_orient = np.mean(np.cos(2 * block_orient))
        sin_orient = np.mean(np.sin(2 * block_orient))
        orient = math.atan2(sin_orient, cos_orient) / 2
        
        # rotating the image block so that ridge orientation is 
        # vertical
        rotated_img = scipy.ndimage.rotate(block_norm, orient / np.pi * 100 + 90, 
                axes=(1, 0), reshape=False, order=3, mode='nearest')

        # Crop out the invalid region, for better projection
        crop_size = int(np.fix(rows / np.sqrt(2)))
        offset = int(np.fix((rows - crop_size) / 2))
        rotated_img = rotated_img[offset:offset + crop_size][:, offset:offset+crop_size]

        proj = np.sum(rotated_img, axis=0)
        dilation = scipy.ndimage.grey_dilation(proj, self.freq_window_size, 
                structure=np.ones(self.freq_block_size))

        temp = np.abs(dilation - proj)

        peak_thresh = 2

        max_pts = (temp < peak_thresh) & (proj > np.mean(proj))
        max_ind = np.where(max_pts)

        rows_max_ind, cols_max_ind = np.shape(max_ind)

        if cols_max_ind < 2:
            return np.zeros(block_norm.shape)
        else:
            num_peaks = cols_max_ind
            wavelength = (max_ind[0][cols_max_ind - 1] - max_ind[0][0]) / (num_peaks - 1)
            if wavelength >= self.min_wavelength and wavelength < self.max_wavelength:
                return (1 / np.double(wavelength) * np.ones(block_norm.shape))
            else:
                return np.zeros(block_norm.shape)

    def _frequency(self):
        """
        Block-wise frequency estimation for the image
        """
        rows, cols = self._norm_img.shape
        freq = np.zeros((rows, cols))

        for r in range(0, rows - self.freq_block_size, self.ridge_block_size):
            for c in range(0, cols - self.freq_block_size, self.freq_block_size):
                block_norm = self._norm_img[r:r+self.freq_block_size][:, c:c+self.freq_block_size]
                block_orient = self._orient_img[r:r+self.freq_block_size][:, c:c+self.freq_block_size]
                freq[r:r+self.freq_block_size][:, c:c+self.freq_block_size] = self._freq_block(block_norm, block_orient)

        self._freq = freq * self._mask
        freq_id = np.reshape(self._freq, (1, rows * cols))
        ind = np.where(freq_id > 0)

        ind = np.array(ind)
        ind = ind[1, :]

        non_zero_elems_in_freq = freq_id[0][ind]

        self._mean_freq = np.mean(non_zero_elems_in_freq)
        self._median_freq = np.median(non_zero_elems_in_freq)

        self._freq = self._mean_freq * self._mask

    def _filter(self):
        """
        Image filtering using oriented gabor filters
        """
        img = np.double(self._norm_img)
        rows, cols = img.shape
        new_img = np.zeros((rows, cols))

        freq_1d = np.reshape(self._freq, (1, rows * cols))
        ind = np.where(freq_1d > 0)

        ind = np.array(ind)
        ind = ind[1, :]

        non_zero_elems_in_freq = freq_1d[0][ind]
        non_zero_elems_in_freq = np.array(np.round(non_zero_elems_in_freq * 100) / 100).astype(np.double)

        unfreq = np.unique(non_zero_elems_in_freq)

        sigmax = 1 / unfreq[0] * self.kx
        sigmay = 1 / unfreq[0] * self.ky

        sze = np.intc(np.round(3 * np.max([sigmax, sigmay])))

        x, y = np.meshgrid(np.linspace(-sze, sze, (2 * sze + 1)), 
                np.linspace(-sze, sze, (2 * sze + 1)))

        reffilter = np.exp(-(((np.power(x, 2)) / (sigmax * sigmax) + (np.power(y, 2)) / (sigmay * sigmay)))) * np.cos(2 * np.pi * unfreq[0] * x)

        filt_rows, filt_cols = reffilter.shape

        angleRange = np.intc(180 / self.angle_inc)

        gabor_filter = np.array(np.zeros((angleRange, filt_rows, filt_cols)))

        for o in range(0, angleRange):
            rot_filt = scipy.ndimage.rotate(reffilter, -(o * self.angle_inc + 90), reshape=False)
            gabor_filter[o] = rot_filt

        max_size = int(sze)

        temp = self._freq > 0
        validr, validc = np.where(temp)

        temp1 = validr > max_size
        temp2 = validr < rows - max_size
        temp3 = validc > max_size
        temp4 = validc < cols - max_size


        final_temp = temp1 & temp2 & temp3 & temp4

        finalind = np.where(final_temp)

        maxorientindex = np.round(180 / self.angle_inc)
        orientindex = np.round(self._orient_img / np.pi * 180 / self.angle_inc)

        # do the filtering
        for i in range(0, rows):
            for j in range(0, cols):
                if (orientindex[i][j] < 1):
                    orientindex[i][j] = orientindex[i][j] + maxorientindex
                if (orientindex[i][j] > maxorientindex):
                    orientindex[i][j] = orientindex[i][j] - maxorientindex
        finalind_rows, finalind_cols = np.shape(finalind)
        sze = int(sze)
        for k in range(0, finalind_cols):
            r = validr[finalind[0][k]]
            c = validc[finalind[0][k]]

            img_block = img[r - sze:r + sze + 1][:, c - sze:c + sze + 1]

            new_img[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r][c]) - 1])

        self._enhanced = new_img < self.ridge_filter_thresh


    def enhance(self, img: np.ndarray, resize: tuple[int, int] | None = None) -> np.ndarray:
        """
        Enhance a fingerprint image
        """
        if resize:
            img = cv.resize(img, resize)

        self._ridge_segment(img)
        self._orientation()
        self._frequency()
        self._filter()
        return self._enhanced.astype(np.uint8)
