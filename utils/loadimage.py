import numpy as np
import base64
import io

from PIL import Image
from tensorflow.keras.utils import img_to_array


def load_image_pixels(contents):
    if contents:
        image = contents.split(",")[1]
        image = base64.b64decode(image)
        image = Image.open(io.BytesIO(image)).convert('RGB')
    else:
        # default image if the image has not yet been uploaded by the user
        image = Image.open(r"img\000000000057.jpg")
    return image


def resize_and_scale(image, shape):
    # resizing to network input size (416, 416)
    # and converting to ndarray
    image = image.resize(shape)
    image = img_to_array(image)
    print(image.shape, "numpy shape")

    # scaling
    image = image.astype('float32')
    image /= 255.0

    # add one dimension to make a size on one batch
    image = np.expand_dims(image, 0)

    return image
