import cv2
import numpy as np
from scipy import ndimage

class FilterImage:
    def gaussian_filter(self, image):
        if image is not None:
            return cv2.GaussianBlur(image, (15, 15), 0)
        return image

    def high_pass_filter(self, image):
        if image is not None:
            kernel = np.array([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)
        return image
    
    def histogramme_filter(self, image):
        if image is not None:
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return cv2.equalizeHist(gray_image)
            elif len(image.shape) == 2:
                return cv2.equalizeHist(image)
        return image
    
    def detect_contours(self, image):
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
            return image
        return image

    def add_gaussian_noise(self, image, progress_callback=None):
        if image is not None:
            if len(image.shape) == 3:  # Color image
                row, col, ch = image.shape
                sigma = np.sqrt(0.01)
                gauss = np.random.normal(0, sigma, (row, col, ch))
                noisy = image + gauss.reshape(row, col, ch) * 255
            elif len(image.shape) == 2:  # Grayscale image
                row, col = image.shape
                sigma = np.sqrt(0.01)
                gauss = np.random.normal(0, sigma, (row, col))
                noisy = image + gauss * 255
            
            # Clip the values to be between 0 and 255
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            
            if progress_callback:
                progress_callback(row * col)
            
            return noisy
        return image

    def mean_filter(self, image, n=3, m=3):
        if image is not None:
            kernel = np.ones((n, m), np.float32) / (n * m)
            return cv2.filter2D(image, -1, kernel)
        return image

    def low_pass_filter(self, image):
        if image is not None:
            H = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
            return cv2.filter2D(image, -1, H)
        return image

    def min_max_smoothing(self, image, progress_callback=None):
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(3):
                result[:,:,i] = self._min_max_smoothing_channel(image[:,:,i], progress_callback)
            return result
        else:
            return self._min_max_smoothing_channel(image, progress_callback)

    def _min_max_smoothing_channel(self, channel, progress_callback=None):
        rows, cols = channel.shape
        result = np.zeros_like(channel)
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                neighborhood = channel[i-1:i+2, j-1:j+2]
                min_val = np.min(neighborhood)
                max_val = np.max(neighborhood)
                result[i, j] = (min_val + max_val) / 2
            
            if progress_callback:
                progress_callback(cols)
        
        return result

    def median_filter(self, image, size=3):
        if image is not None:
            return cv2.medianBlur(image, size)
        return image


    def hybrid_median_filter(self, image):
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(3):
                result[:,:,i] = self._hybrid_median_channel(image[:,:,i])
            return result
        else:
            return self._hybrid_median_channel(image)

    def _hybrid_median_channel(self, channel):
        m1 = ndimage.median_filter(channel, footprint=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        m2 = ndimage.median_filter(channel, footprint=np.array([[1,0,1],[0,1,0],[1,0,1]]))
        return np.median([m1, m2, channel], axis=0)

    def morph_operation(self, image, operation, kernel_size=3, iterations=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if operation == 'dilate':
            return cv2.dilate(image, kernel, iterations=iterations)
        elif operation == 'erode':
            return cv2.erode(image, kernel, iterations=iterations)
        elif operation == 'open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            return image
        