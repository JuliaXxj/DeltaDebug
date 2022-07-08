# Different way to update a mask/ do delta debugging
import torch
import numpy as np


def rectangle_half(unmasked_pixels: torch.Tensor, i, j, h, w):
    channel, height, width = unmasked_pixels.shape

    # Check rectangle first
    check_zero = unmasked_pixels.clone()
    check_one = unmasked_pixels.clone()
    check_zero[..., i: i + h, j: j + w] = 0
    check_one[..., i: i + h, j: j + w] = 1
    if torch.sum(check_zero).item() != 0 or torch.sum(check_one).item() != torch.sum(unmasked_pixels).item():
        raise Exception('Unmasked the pixels do not form a rectangle or the coordination given is not correct.')

    # Cutting from the middle of the long side of the rectangle
    first_i, first_j, first_h, first_w = i, j, h, w
    second_i, second_j, second_h, second_w = i, j, h, w
    if h > w:
        first_h = h // 2
        second_h = h - first_h
        second_i = i + first_h
    else:
        first_w = w // 2
        second_w = w - first_w
        second_j = j + first_w

    first_half = torch.zeros((channel, height, width))
    first_half[..., first_i: first_i + first_h, first_j: first_j + first_w] = 1
    second_half = torch.zeros((channel, height, width))
    second_half[..., second_i: second_i + second_h, second_j: second_j + second_w] = 1

    # Need to return the two halves of mask

    return first_half, first_i, first_j, first_h, first_w, second_half, second_i, second_j, second_h, second_w


def random_portion(unmasked_pixels: torch.Tensor, portion):
    if portion < 0 or portion > 1:
        raise Exception('Invalid portion value')
    unmasked_num = torch.sum(unmasked_pixels).item()
    num_to_mask = int(unmasked_num * portion)
    return random_number(unmasked_pixels, num_to_mask)


def random_number(unmasked_pixels: torch.Tensor, num):
    c, h, w = unmasked_pixels.shape
    total_pixel = int(torch.sum(unmasked_pixels).item())

    first_new_mask = unmasked_pixels.clone()

    chosen_to_mask = np.zeros(total_pixel)
    chosen_to_mask[:num] = 1
    np.random.shuffle(chosen_to_mask)

    checked_existed_pixel_num = 0
    for i in range(h):
        for j in range(w):
            if unmasked_pixels[0][i][j] == 0:
                continue
            if chosen_to_mask[checked_existed_pixel_num] == 1:
                first_new_mask[0][i][j] = 0
            checked_existed_pixel_num += 1

    # Check

    if torch.sum(first_new_mask).item() != total_pixel - chosen_to_mask.sum():
        raise Exception("Wrong calculation!")

    second_new_mask = unmasked_pixels - first_new_mask

    return first_new_mask, second_new_mask


def split_in_smaller_groups(unmasked_pixels: torch.Tensor, num_groups):
    """
    Split the pixels into num_groups groups sequentially and return the corresponding generated masks.
    """
    c, h, w = unmasked_pixels.shape
    total_pixel = int(torch.sum(unmasked_pixels[0]).item())

    # Each group needs to have at least one pixel.
    if num_groups > total_pixel:
        raise Exception("Too less pixels existed! Group number is even greater than total number of pixel.")

    all_new_masks = []
    each_group_num = total_pixel // num_groups
    res = total_pixel % num_groups
    current_group = 0
    checked_existed_pixel_num = each_group_num + (res != 0)

    current_mask = unmasked_pixels.clone()
    for i in range(h):
        for j in range(w):
            if unmasked_pixels[0][i][j] == 0:
                continue
            for k in range(c):
                current_mask[k][i][j] = 0
            checked_existed_pixel_num -= 1

            if checked_existed_pixel_num == 0:
                all_new_masks.append(current_mask)

                current_mask = unmasked_pixels.clone()
                current_group += 1
                checked_existed_pixel_num = each_group_num + (current_group < res)

    # all_new_masks.append(current_mask)
    assert current_group == num_groups
    return all_new_masks

def split_in_smaller_groups_new(unmasked_pixels: torch.Tensor, num_groups):

    """
    Split the pixels into num_groups groups sequentially and return the corresponding generated masks.
    """
    print("new")
    c, h, w = unmasked_pixels.shape
    total_pixel = int(torch.sum(unmasked_pixels[0]).item())

    # Each group needs to have at least one pixel.
    if num_groups > total_pixel:
        raise Exception("Too less pixels existed! Group number is even greater than total number of pixel.")

    all_new_unmasks = []
    all_new_unmasks_complement = []
    each_group_num = total_pixel // num_groups
    res = total_pixel % num_groups
    current_group = 0
    checked_existed_pixel_num = each_group_num + (res != 0)

    current_unmask = unmasked_pixels.clone()
    for i in range(h):
        for j in range(w):
            if unmasked_pixels[0][i][j] == 0:
                continue
            for k in range(c):
                current_unmask[k][i][j] = 0
            checked_existed_pixel_num -= 1

            if checked_existed_pixel_num == 0:
                all_new_unmasks.append(current_unmask)
                all_new_unmasks_complement.append(unmasked_pixels - current_unmask)

                current_unmask = unmasked_pixels.clone()
                current_group += 1
                checked_existed_pixel_num = each_group_num + (current_group < res)

            # if checked_existed_pixel_num % each_group_num == 0 and current_group < num_groups - 1:
            #     all_new_unmasks.append(current_unmask)
            #     all_new_unmasks_complement.append(unmasked_pixels - current_unmask)
            #
            #     current_unmask = unmasked_pixels.clone()
            #     current_group += 1
    # all_new_unmasks.append(current_unmask)
    # all_new_unmasks_complement.append(unmasked_pixels - current_unmask)
    assert current_group == num_groups
    return all_new_unmasks, all_new_unmasks_complement