import numpy as np
from skimage.util import invert
import tensorflow as tf
import math
import cv2
from scipy import ndimage
import preprocess as pp
import symbolrecog as sbr
import matplotlib.pyplot as plt


class CNN_LATEX(object):
    def __init__(self, mean_train=None, std_train=None):
        # tf.logging.set_verbosity(tf.logging.WARN)
        self.mean_train = np.load(mean_train)
        self.std_train = np.load(std_train)

    def normalize_single(self, symbol):
        symbol = np.copy(symbol).astype(np.float32)

        # range 0-1
        symbol /= np.max(symbol)

        rows, cols = symbol.shape
        # scale to 40x40
        inner_size = 40
        if rows > cols:
            factor = inner_size / rows
            rows = inner_size
            cols = int(round(cols * factor))
            cols = cols if cols > 2 else 2
            inner = cv2.resize(symbol, (cols, rows))
        else:
            factor = inner_size / cols
            cols = inner_size
            rows = int(round(rows * factor))
            rows = rows if rows > 2 else 2
            inner = cv2.resize(symbol, (cols, rows))

        # pad to 48x48
        outer_size = 48
        colsPadding = (int(math.ceil((outer_size - cols) / 2.0)), int(math.floor((outer_size - cols) / 2.0)))
        rowsPadding = (int(math.ceil((outer_size - rows) / 2.0)), int(math.floor((outer_size - rows) / 2.0)))
        outer = np.pad(inner, (rowsPadding, colsPadding), 'constant', constant_values=(1, 1))
        return outer

    def getBestShift(self, img):
        inv = invert(img)
        cy, cx = ndimage.measurements.center_of_mass(inv)

        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)

        return shiftx, shifty

    def shift(self, img, sx, sy):
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows), borderValue=(1, 1))
        return shifted

    def add_rectangles(self, img, bounding_boxes):
        img_color = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
        for bounding_box in bounding_boxes:
            xmin, xmax = bounding_box['xmin'], bounding_box['xmax']
            ymin, ymax = bounding_box['ymin'], bounding_box['ymax']
            img_color[ymin, xmin:xmax] = [255, 0, 0]
            img_color[ymax - 1, xmin:xmax] = [255, 0, 0]
            img_color[ymin:ymax, xmin] = [255, 0, 0]
            img_color[ymin:ymax, xmax - 1] = [255, 0, 0]
        return img_color

    def get_bounding_boxes(self):
        thresh, thresh_inv = pp.blur_image(self.formula)
        contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        id_c = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 10 or w > 10:
                bounding_boxes.append({
                    'id': id_c,
                    'xmin': x,
                    'xmax': x + w,
                    'ymin': y,
                    'ymax': y + h,
                    'combined': []
                })
                id_c += 1
        bounding_boxes = sorted(bounding_boxes, key=lambda k: (k['xmin'], k['ymin']))
        formula_rects = self.add_rectangles(self.formula, bounding_boxes)
        self.bounding_boxes = bounding_boxes

    def normalize(self):
        self.possible_symbol_img = []
        self.pred_pos = []
        for bounding_box in self.bounding_boxes:
            xmin, xmax = bounding_box['xmin'], bounding_box['xmax']
            ymin, ymax = bounding_box['ymin'], bounding_box['ymax']
            dy = ymax - ymin
            dx = xmax - xmin
            normalized = self.normalize_single(self.formula[ymin:ymax, xmin:xmax])
            normalized = normalized - self.mean_train
            normalized = normalized / self.std_train
            self.possible_symbol_img.append(normalized)
            self.pred_pos.append(bounding_box)

    def predict(self, formula):
        self.formula = formula
        self.get_bounding_boxes()
        self.normalize()
        good_bounding_boxes = []
        pred_pos = self.pred_pos
        possible_symbol_img = self.possible_symbol_img
        symbols = sbr.img_to_string(possible_symbol_img)
        for sym, pos in zip(symbols, pred_pos):
            xmin, xmax = pos['xmin'], pos['xmax']
            ymin, ymax = pos['ymin'], pos['ymax']
            good_bounding_boxes.append({
                'xmin': xmin,
                'xmax': xmax,
                'ymin': ymin,
                'ymax': ymax,
                'symbol': sym,
                'did': False
            })
        good_bounding_boxes = sorted(good_bounding_boxes, key=lambda k: (k['xmin'], k['ymin']))
        left = []
        right = []
        divided = False
        for i in range(0, len(good_bounding_boxes)):
            if i == 0:
                left.append(good_bounding_boxes[i])
                continue
            if divided:
                right.append(good_bounding_boxes[i])
                continue
            prev = good_bounding_boxes[i - 1]
            now = good_bounding_boxes[i]
            xmin_gap = abs(prev['xmin'] - now['xmin'])
            xmax_gap = abs(prev['xmax'] - now['xmax'])
            if xmin_gap < 10 and xmax_gap < 10 and now['symbol'] == '-' and prev['symbol'] == '-':
                left.pop(len(left) - 1)
                divided = True
            else:
                left.append(good_bounding_boxes[i])
        return left, right
