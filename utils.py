import shutil
import imageio.v2 as imageio
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision


def generate_gif(gif_filename, img_folder, image_names, delete_folder=False):
    """
    Generate the gif of the process of how the image gets transformed from all the images image_names saved in img_folder.
    Save the generated gif file with name gif_filename.
    Delete the img_folder if delete_folder is True.
    """
    with imageio.get_writer(gif_filename, mode='I') as writer:
        for img_name in image_names:
            image = imageio.imread(os.path.join(img_folder, img_name))
            writer.append_data(image)

    if delete_folder:
        shutil.rmtree(img_folder)


def cifar10_imshow(img, img_title=None, save_img=False, img_path=None):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    figure = plt.figure()
    plt.title("Pred prob: {}".format(img_title))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if save_img:
        plt.savefig(img_path)
    plt.show()


def cifar10_img(img):
    """
    Transform the image from cifar10 to a form that can be inputted for matplotlib.
    """
    img = torchvision.utils.make_grid(img)
    # unnormalize the image, see the transform applied to img in train_cifar10.py.
    img = img / 2 + 0.5
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))
