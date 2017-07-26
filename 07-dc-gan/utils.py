#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   25.07.2017
#-------------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

#-------------------------------------------------------------------------------
def figure_to_numpy(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

#-------------------------------------------------------------------------------
def gen_sample_summary(samples):
    fig, axes = plt.subplots(figsize=(5,3), nrows=3, ncols=5,
                             sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0, hspace=0)

    for ax, img in zip(axes.flatten(), samples):
        ax.axis('off')
        img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
        ax.set_adjustable('box-forced')
        im = ax.imshow(img, aspect='equal')

    arr = figure_to_numpy(fig)
    del(fig)
    return arr
