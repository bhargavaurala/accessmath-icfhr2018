
import math
import numpy as np

class FrameDifferencing:

    @staticmethod
    def absolute_difference(frame1, frame2):
        tempo_1 = frame1.astype('int32')
        tempo_2 = frame2.astype('int32')

        diff = np.abs(tempo_1 - tempo_2).astype('uint8')

        return diff

    @staticmethod
    def binary_log_motion_density(binary_1, binary_2):
        diff_image = FrameDifferencing.absolute_difference(binary_1, binary_2)

        motion = diff_image.sum() / 255

        height, width = diff_image.shape

        return math.log(1.0 + (motion / float(height * width)), 2.0)