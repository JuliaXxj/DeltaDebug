import numpy as np
import torch
from torchvision.transforms import functional as F
from numbers import Number




class EraseHalf(object):
    def __init__(self, part='right', value=0.5):
        self.part = part
        self.value = value

    def __call__(self, img):
        c, h, w = img.shape
        if self.part == 'upper':
            return F.erase(img, 0, 0, h // 2, w, self.value)
        elif self.part == 'lower':
            return F.erase(img, h // 2, 0, h // 2, w, self.value)
        elif self.part == 'left':
            return F.erase(img, 0, 0, h, w // 2, self.value)
        elif self.part == 'right':
            return F.erase(img, 0, w // 2, h, w // 2, self.value)
        else:
            return F.erase(img, 0, 0, 0, 0, self.value)


class KeepCenter(object):

    def __init__(self, percent=0.5, value=0.5):
        self.percent = percent
        self.value = value

    def __call__(self, img):
        c, h, w = img.shape
        if self.percent < 0 or self.percent > 1:
            return F.erase(img, 0, 0, 0, 0, self.value)

        center_mask = np.zeros((c, h, w))
        surrounding_mask = np.ones((c, h, w))

        center_mask[:, int((h - h * self.percent) // 2): int((h - h * self.percent) // 2) + int(h * self.percent),
        int((w - w * self.percent) // 2): int((w - w * self.percent) // 2) + int(w * self.percent)] = 1
        surrounding_mask[:, int((h - h * self.percent) // 2): int((h - h * self.percent) // 2) + int(h * self.percent),
        int((w - w * self.percent) // 2): int((w - w * self.percent) // 2) + int(w * self.percent)] = 0

        surrounding_mask = surrounding_mask * self.value
        center_mask = center_mask * img.numpy()

        mask = torch.from_numpy(center_mask + surrounding_mask)

        return F.erase(img, 0, 0, h, w, mask)

        # return F.erase(img, int((h - h * self.percent) // 2), int((w - w * self.percent)//2), int(h * self.percent), int(w * self.percent),
        #                self.value)


class MaskEveryOtherPixel(object):
    def __init__(self, value=0.5):
        self.value = value

    def __call__(self, img):
        c, h, w = img.shape
        start_one = ([1, 0] * (w // 2 + 1))[:w]
        start_zero = ([0, 1] * (w // 2 + 1))[:w]
        keep_mask = ([start_one, start_zero] * (h // 2 + 1))[:h]
        keep_mask = np.array([keep_mask] * c)

        remove_mask = np.abs(keep_mask - 1)

        mask = torch.from_numpy(keep_mask * img.numpy() + remove_mask * self.value)

        return F.erase(img, 0, 0, h, w, mask)


class CustomizeMask(object):
    def __init__(self, value=0.5):
        self.value = value

    def __call__(self, img, unmask_pixels):
        c, h, w = img.shape
        if not isinstance(unmask_pixels, torch.Tensor):
            raise Exception('Mask type wrong.')
        if unmask_pixels.shape != img.shape:
            raise Exception('Mask shape ({}) should match image shape {}.'.format(unmask_pixels.shape, img.shape))

        unmask_pixel_values = unmask_pixels * img
        mask_pixel_values = torch.logical_not(unmask_pixels) * self.value

        return F.erase(img, 0, 0, h, w, unmask_pixel_values + mask_pixel_values)


# class Eraser(object):
#
#     def __init__(self, value=0.5):
#         self.value = value
#
#     def __call__(self, mask, i, j, h, w):
#         return F.erase(mask, i, j, h, w, self.value)
