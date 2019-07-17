# Methods to visualize the images
from matplotlib.pyplot import plt
def visualize_plot(item , xlab, ylab):
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.plot(item, lw=2)
    return


