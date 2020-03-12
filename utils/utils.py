import numpy as np
from skimage import io
from efficientnet.keras import center_crop_and_resize, preprocess_input

def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True

def preprocess_image(image,image_size):
    image = io.imread(image)
    x = center_crop_and_resize(image, image_size=image_size)
    x = preprocess_input(x)
    x = np.expand_dims(x, 0)

    return x

def clean_data():
    return 0    

def get_image_vector_pair():
    return 0

