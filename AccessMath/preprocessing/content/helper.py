
import cv2
import numpy as np
from scipy.ndimage.measurements import label

#=====================================================
# Defines miscellaneous operations that are required
# by other content processors
#=====================================================

class Helper:
    @staticmethod
    def grayscale_variance_map(original_image, ksize):
        result = np.zeros(original_image.shape)

        for y in range(original_image.shape[0]):
            for x in range(original_image.shape[1]):
                min_y = max(0, y - ksize)
                max_y = min(original_image.shape[0], y + ksize)
                min_x = max(0, x - ksize)
                max_x = min(original_image.shape[1], x + ksize)

                result[y, x] = original_image[min_y:max_y, min_x:max_x].var()

        return result

    @staticmethod
    def decompress_binary_images(compressed_images):
        all_binary = []
        for raw_data in compressed_images:
            image = cv2.imdecode(raw_data, cv2.IMREAD_GRAYSCALE)
            all_binary.append(image)

        return all_binary


