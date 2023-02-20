import skimage
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from PIL import Image

IMAGE_PATH = "./dataset/Dataset_pages-to-jpg-0003.jpg"

image = skimage.io.imread(
    IMAGE_PATH, as_gray=True)
image = skimage.transform.rescale(image, 0.2)


def bounding_box_demo():
    def get_bounding_box(contour):
        min_x, min_y = np.min(contour[:, 1]), np.min(contour[:, 0])
        max_x, max_y = np.max(contour[:, 1]), np.max(contour[:, 0])
        return min_x, min_y, max_x - min_x, max_y - min_y

    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    contours = skimage.measure.find_contours(image)
    for contour in contours:
        #     ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
        bb_x, bb_y, bb_w, bb_h = get_bounding_box(contour)
        ax.add_patch(matplotlib.patches.Rectangle((bb_x, bb_y), bb_w, bb_h))

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def hog_demo():
    fd, hog_image = skimage.feature.hog(image, orientations=8, pixels_per_cell=(
        16, 16), cells_per_block=(1, 1), visualize=True, channel_axis=-1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = skimage.exposure.rescale_intensity(
        hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')


# bounding_box_demo()


res = pytesseract.image_to_string(Image.open(IMAGE_PATH))
pass
