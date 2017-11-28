import imageio
import os

def img2gif(imagePaths, gifPath, duration=0.10):
    """Merge image files into a single gif file.
    The order of image files are preserved when merged.
    """
    images = [imageio.imread(imagePath) for imagePath in imagePaths] 
    imageio.mimsave(gifPath, images, duration=duration)

    for imagePath in imagePaths:
        os.remove(imagePath)