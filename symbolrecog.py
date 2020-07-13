import numpy as np
import os
from keras.preprocessing import image
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

matching_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '-': 10, '+': 11,
                 '(': 12, ')': 13, '!alpha': 14, '!beta': 15, '!theta': 16, 'a': 17, 'b': 18, 'A': 19, 'B': 20,
                 'x': 21, 'y': 22, 'z': 23}


model = load_model('models/vgg16_v21.h5')


def img_to_string(target_images):
    x = image.img_to_array(target_images)
    x = np.expand_dims(x, axis=3)
    
    rst = model.predict(x)

    rtn = []
    for sym in rst:
        tmp = [name for name, val in matching_dict.items() if val == np.argmax(sym)]
        rtn.append(tmp[0])
    return rtn
