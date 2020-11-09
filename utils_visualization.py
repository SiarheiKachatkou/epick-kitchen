import matplotlib.pyplot as plt
import numpy as np
import math


def show_snippet(snippet):
    imgs = snippet[0]
    imgs = imgs.squeeze(axis=1)
    imgs = [np.transpose(img, (1, 2, 0)).astype(np.uint8) for img in imgs]

    plt.figure()

    rows = int(math.sqrt(len(imgs)))
    cols = len(imgs) // rows + 1

    for c, img in enumerate(imgs):
        plt.subplot(rows, cols, c + 1)
        plt.imshow(img)
        plt.axis('off')

    plt.show()

