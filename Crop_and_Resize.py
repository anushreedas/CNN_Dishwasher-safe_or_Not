# Importing Image class from PIL module
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

SIZE = 256, 256
raw_data_dir = '/Users/anushree/Desktop/Dishwasher-safe Or Not/Data/new_raw_data'
processed_data_dir = '/Users/anushree/Desktop/Dishwasher-safe Or Not/Data/new_processed_data/'

def crop_resize_image(filepath):
    im = Image.open(filepath)
    # Opens a image in RGB mode
    # plt.imshow(np.asarray(im))
    # plt.show()

    width, height = im.size   # Get dimensions
    new_width = min(width, height)
    new_height = min(width, height)

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop image to square
    im_cropped = im.crop((left, top, right, bottom))
    print(im_cropped.size)
    # plt.imshow(np.asarray(im_cropped))
    # plt.show()

    # Resize image
    im_cropped.thumbnail(SIZE, Image.Resampling.LANCZOS)
    # plt.imshow(np.asarray(im_cropped))
    # plt.show()
    im_cropped.save(processed_data_dir+filepath.split('/')[-1])

ext = ['JPG', 'jpeg', 'png', 'jpg', 'gif', 'webp']
# iterate over files
for filename in os.listdir(raw_data_dir):
    f = os.path.join(raw_data_dir, filename)
    # checking if it is a file
    if os.path.isfile(f) and filename.split('.')[-1] in ext:
        print(f)
        crop_resize_image(f)


# crop_resize_image('/Users/anushree/Desktop/Dishwasher-safe Or Not/Data/raw_data/teaspoonraw29.JPG')