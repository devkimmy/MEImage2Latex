import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

matching_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                 '-': 10, '(': 11, ')': 12, '+': 13, 'alpha': 14, 'beta': 15, 'theta': 16, 'a': 17,
                 'b': 18, 'i': 19, 'j': 20, 'k': 21, 'm': 22, 'n': 23, 't': 24, 'x': 25, 'y': 26, 'z': 27}


def img_to_string(target_images):
    # load model
    model = load_model('models/norm_ext_v2.h5')

    x = image.img_to_array(target_images)
    x = np.expand_dims(x, axis=3)

    rst = model.predict_classes(x)
    rtn = []
    for sym in rst:
        tmp = [name for name, val in matching_dict.items() if val == sym]
        rtn.append(tmp[0])
    return rtn
