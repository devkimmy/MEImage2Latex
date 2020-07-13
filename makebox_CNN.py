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
        self.mean_train = np.load(mean_train)
        self.std_train = np.load(std_train)
        # np.savetxt("mean_vgg16_v8.csv", self.mean_train, delimiter=',')
        # np.savetxt("std_vgg16_v8.csv", self.std_train, delimiter=',')
        self.zzz = 0
        self.good_bounding_boxes = []
        self.possible_symbol_img = []

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
        # center the mass
        shiftx, shifty = self.getBestShift(outer)
        shifted = self.shift(outer, shiftx, shifty)
        return shifted

    # def normalize_single(self, symbol):
    #     symbol = np.copy(symbol).astype(np.float64)
    #     symbol /= 255.0
    #     rows, cols = symbol.shape
    #     # scale to 40x40
    #     inner_size = 40
    #     if rows > cols:
    #         factor = inner_size / rows
    #         rows = inner_size
    #         cols = int(round(cols * factor))
    #         cols = cols if cols > 2 else 2
    #         inner = cv2.resize(symbol, (cols, rows), interpolation=cv2.INTER_LINEAR)
    #     else:
    #         factor = inner_size / cols
    #         cols = inner_size
    #         rows = int(round(rows * factor))
    #         rows = rows if rows > 2 else 2
    #         inner = cv2.resize(symbol, (cols, rows), interpolation=cv2.INTER_LINEAR)
    #
    #     outer_size = 48
    #     colsPadding = (int(math.ceil((outer_size - cols) / 2.0)), int(math.floor((outer_size - cols) / 2.0)))
    #     rowsPadding = (int(math.ceil((outer_size - rows) / 2.0)), int(math.floor((outer_size - rows) / 2.0)))
    #     outer = np.pad(inner, (rowsPadding, colsPadding), 'constant', constant_values=(1, 1))
    #     shiftx, shifty = self.getBestShift(outer)
    #     shifted = self.shift(outer, shiftx, shifty)
    #     return shifted

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
        shifted = cv2.warpAffine(img, M, (cols, rows), borderValue=1)
        return shifted

    def add_rectangles(self, img, bounding_boxes):
        img_color = np.copy(img)
        for bounding_box in bounding_boxes:
            xmin, xmax = bounding_box['xmin'], bounding_box['xmax']
            ymin, ymax = bounding_box['ymin'], bounding_box['ymax']
            img_color[ymin, xmin:xmax] = [0, 0, 255]
            img_color[ymax - 1, xmin:xmax] = [0, 0, 255]
            img_color[ymin:ymax, xmin] = [0, 0, 255]
            img_color[ymin:ymax, xmax - 1] = [0, 0, 255]
        return img_color

    def add_rectangle_single(self, img, bb, chanel):
        xmin, xmax = bb['xmin'], bb['xmax']
        ymin, ymax = bb['ymin'], bb['ymax']
        if chanel == 2:
            tmp = [0, 0, 255]
            img[ymin, xmin:xmax] = tmp
            img[ymax - 1, xmin:xmax] = tmp
            img[ymin:ymax, xmin] = tmp
            img[ymin:ymax, xmax - 1] = tmp
        elif chanel == 1:
            print("!!!!!!")
            tmp = [0, 255, 0]
            img[ymin, xmin:xmax] = tmp
            img[ymax - 1, xmin:xmax] = tmp
            img[ymin:ymax, xmin] = tmp
            img[ymin:ymax, xmax - 1] = tmp
        return img

    def get_bounding_boxes(self):
        thresh, thresh_inv = pp.blur_image(self.formula)
        contours, hierarchy = cv2.findContours(thresh_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # if h > 10 or w > 10:
            bounding_boxes.append({
                'xmin': x,
                'xmax': x + w,
                'ymin': y,
                'ymax': y + h,
                'combined': []
            })
        hierarchy = hierarchy[0]
        # tmp = cv2.merge((thresh, thresh, thresh))
        # cv2.imshow("tmp",tmp)
        # cv2.waitKey(0)
        box_formula = self.add_rectangles(self.formula, bounding_boxes)
        # cv2.imshow("box_formula", box_formula)
        # cv2.waitKey(0)

        # hierarchy : [다음형제 컨투어 인덱스, 이전형제 컨투어 인덱스, 부모 컨투어 인덱스, 자식 컨투어 인덱스]
        for (bb, i, h) in zip(bounding_boxes, range(0, len(bounding_boxes)), hierarchy):
            if bb['ymax'] - bb['ymin'] < 10 and bb['xmax'] - bb['xmin'] < 10:
                continue
            blank_image = np.full((bb['ymax'] - bb['ymin'], bb['xmax'] - bb['xmin'], 3), 255, np.uint8)
            # 1) 부모와 자식 모두 없다
            if h[2] < 0 and h[3] < 0:
                cv2.imwrite("empty_image/empty_" + str(i) + ".png", blank_image)
                blank_image = cv2.drawContours(blank_image, contours, i, (0, 0, 0), cv2.FILLED,
                                               offset=(-bb['xmin'], -bb['ymin']))
                # tmp = self.add_rectangle_single(tmp, bb, 2)
                self.good_bounding_boxes.append({
                    'xmin': bb['xmin'],
                    'xmax': bb['xmax'],
                    'ymin': bb['ymin'],
                    'ymax': bb['ymax'],
                    'img': blank_image
                })
            # 2) 자식이 있고 자신은 첫번째 부모이다
            elif h[3] < 0 and h[2] >= 0:
                blank_image = cv2.drawContours(blank_image, contours, i, (0, 0, 0), cv2.FILLED,
                                               offset=(-bb['xmin'], -bb['ymin']))

                # tmp = self.add_rectangle_single(tmp, bb, 2)
                blank_image = cv2.drawContours(blank_image, contours, h[2], (255, 255, 255), cv2.FILLED,
                                               offset=(-bb['xmin'], -bb['ymin']))
                # tmp = self.add_rectangle_single(tmp, bounding_boxes[h[2]], 1)

                h_child = hierarchy[h[2]]
                while h_child[0] >= 0:
                    blank_image = cv2.drawContours(blank_image, contours, h_child[0], (255, 255, 255), cv2.FILLED,
                                                   offset=(-bb['xmin'], -bb['ymin']))
                    h_child = hierarchy[h_child[0]]

                # cv2.imwrite("crop_image/crop_" + str(i) + ".png", image)
                self.good_bounding_boxes.append({
                    'xmin': bb['xmin'],
                    'xmax': bb['xmax'],
                    'ymin': bb['ymin'],
                    'ymax': bb['ymax'],
                    'img': blank_image
                })
        # cv2.imshow("tmp",tmp)
        # cv2.waitKey(0)

    def normalize(self):
        for gbb, i in zip(self.good_bounding_boxes, range(0, len(self.good_bounding_boxes))):
            # xmin, xmax = gbb['xmin'], gbb['xmax']
            # ymin, ymax = gbb['ymin'], gbb['ymax']
            # dy = ymax - ymin
            # dx = xmax - xmin
            gbb['img'] = cv2.cvtColor(gbb['img'], cv2.COLOR_RGB2GRAY)
            normalized = self.normalize_single(gbb["img"])
            # cv2.imshow("no1", normalized)
            # cv2.waitKey(0)
            # normalized = 1 - normalized
            normalized = normalized - self.mean_train
            normalized = normalized / self.std_train
            # cv2.imshow("no2", normalized * 255)
            # cv2.waitKey(0)
            self.possible_symbol_img.append(normalized)

    def show_images(self, images, cols=1, titles=None):
        assert ((titles is None) or (len(images) == len(titles)))
        n_images = len(images)
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_xticks([]), a.set_yticks([])
            a.set_title(title)
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        plt.show()

    def predict(self, formula):
        self.formula = formula
        self.get_bounding_boxes()
        self.normalize()
        good_bounding_boxes = []
        possible_symbol_img = self.possible_symbol_img
        symbols = sbr.img_to_string(possible_symbol_img)
        # self.show_images(self.possible_symbol_img, 4, symbols)
        for sym, pos, img, i in zip(symbols, self.good_bounding_boxes, possible_symbol_img,
                                    range(0, len(possible_symbol_img))):
            # cv2.imwrite('aalpha/' + sym + '_' + str(i) + ".png", img*255)
            xmin, xmax = pos['xmin'], pos['xmax']
            ymin, ymax = pos['ymin'], pos['ymax']
            good_bounding_boxes.append({
                'xmin': xmin,
                'xmax': xmax,
                'ymin': ymin,
                'ymax': ymax,
                'symbol': sym,
                'id': -1,
                'did': False
            })
        good_bounding_boxes = sorted(good_bounding_boxes, key=lambda k: (k['xmin'], k['ymin']))

        left = []
        right = []
        divided = False
        i = 0
        while i < len(good_bounding_boxes):
            if i == 0:
                left.append(good_bounding_boxes[i])
                i += 1
                continue
            if divided:
                right.append(good_bounding_boxes[i])
                i += 1
                continue
            prev = good_bounding_boxes[i - 1]
            now = good_bounding_boxes[i]
            if now['symbol'] == '-' and prev['symbol'] == '-':
                x_boundary = max(prev['xmax'], now['xmax'])
                if i == len(good_bounding_boxes) - 1:
                    divided = True
                    left.pop(len(left) - 1)
                else:
                    next = good_bounding_boxes[i + 1]
                    if next['xmin'] < x_boundary:
                        left.append(good_bounding_boxes[i])
                    else:
                        divided = True
                        left.pop(len(left) - 1)
                i += 1
            else:
                left.append(good_bounding_boxes[i])
                i += 1
        return left, right
