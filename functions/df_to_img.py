import os
import numpy as np
from PIL import Image

def create_img_dir(index,usage_type, emotion_label, data):

    newpath = os.path.join('data', usage_type, emotion_label)

    if not os.path.exists(newpath):
        os.makedirs(newpath)
        
    im = Image.fromarray(data.astype(np.uint8))
    im.save(newpath+f"/img_{index}.jpeg")
