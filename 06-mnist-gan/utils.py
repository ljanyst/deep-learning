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
    fig, axes = plt.subplots(figsize=(5,3), nrows=3, ncols=5, \
                             sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')

    return figure_to_numpy(fig)
