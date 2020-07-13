import cv2  # 문자단위로 인식 하기 위해
import numpy as np  # 배열 계산 및 조작
import matplotlib.pyplot as plt  # 이미지 띄워주기 위함
from keras.models import load_model
from keras.preprocessing import image


def show_images(images, cols=1, titles=None):
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


def read_image_gray_scale(path):
    cv2.namedWindow('grayScale')
    formula = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('grayScale', formula)
    cv2.imwrite('image_result/grayScale.png', formula)
    cv2.waitKey(0)
    cv2.destroyWindow('grayScale')
    return formula


def blur_image(formula):
    dst = cv2.fastNlMeansDenoisingColored(formula, None, 10, 10, 7, 21)
    # cv2.imshow("denoising",dst)
    # cv2.waitKey(0)
    rgb_planes = cv2.split(dst)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((23, 23), np.uint8), iterations=1)
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    # cv2.imshow("removeShadow", result_norm)
    # cv2.waitKey(0)

    img_gray = cv2.cvtColor(result_norm, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # cv2.imshow("GaussianBlur", img_gray)
    # cv2.waitKey(0)

    ret, img_inv = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, img_normal = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("THRESH_OTSU", img_inv)
    # cv2.waitKey(0)
    # plt.imshow(img_inv,cmap='gray')
    # plt.show()
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    #
    # encoder = load_model('models/3ch_encoder_v1.h5')
    # org_w, org_h, _ = dst.shape
    # formula = cv2.resize(dst, (768, 256), cv2.INTER_LINEAR)
    # formula = img_to_array(formula)
    # formula = formula.astype('float32') / 255.
    # formula = np.expand_dims(formula, axis=0)
    # predicted_formula = np.squeeze(encoder.predict(formula))
    #
    # # rst = cv2.resize(predicted_formula, (org_h, org_w), cv2.INTER_LINEAR)
    # predicted_formula = (predicted_formula * 255).astype(np.uint8)
    # ret, img_inv = cv2.threshold(predicted_formula, 127, 255, cv2.THRESH_BINARY_INV)
    # ret, img_normal = cv2.threshold(predicted_formula, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('img_inv', img_inv)
    # cv2.waitKey(0)

    # img_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # ret, img_inv = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ret, img_normal = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plt.imshow(img_inv)
    # plt.show()
    # cv2.imshow('img_inv', img_inv)
    # cv2.waitKey(0)

    # # original
    # img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # ret, img_inv = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ret, img_normal = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # img_blur = cv2.GaussianBlur(rst, (5, 5), 0)
    # _, rst_inv = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # cv2.imshow('img_inv', img_inv)
    # cv2.waitKey(0)
    # adaptive
    # img_blur = cv2.GaussianBlur(formula, (5, 5),0)
    # img_inv = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    # img_normal = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)

    # just thresh
    # ret, img_inv = cv2.threshold(formula, 140, 255, cv2.THRESH_BINARY_INV)
    # ret, img_normal = cv2.threshold(formula, 140, 255, cv2.THRESH_BINARY)

    # cv2.imshow('ADAPTIVE_THRESH_MEAN_C', img_result2)
    # cv2.imshow('ADAPTIVE_THRESH_GAUSSIAN_C', img_result3)
    # cv2.imshow('THRESH_OTSU', img_result4)
    return img_normal, img_inv


# def cleaned_image(formula):
#     encoder = load_model('models/3ch_encoder_v1.h5')
#     height, width, _ = formula.shape
#     ratio = width / height
#     width = 256 * ratio
#     formula = cv2.resize(formula, dsize=(int(width), 256))
#     cnt = width / 768
#     formula_array = []
#     for i in range(0, cnt + 1):
#         formula_array.append(formula[i * 768:(i + 1) * 768, 0:256])
#
#     # input array
#     formula = img_to_array(formula)
#     formula = formula.astype('float32') / 255.
#     formula = np.expand_dims(formula, axis=0)
#     # predicted_formula result array
#     predicted_formula = np.squeeze(encoder.predict(formula))
#
#     rst = cv2.resize(predicted_formula, (org_h, org_w), cv2.INTER_LINEAR)
#     rst = (rst * 255).astype(np.uint8)
#
#     img_blur = cv2.GaussianBlur(rst, (3, 3), 1)
#     _, rst_inv = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
#     # plt.imshow(rst, cmap='gray')
#     # plt.show()
#     return rst, rst_inv


# def cleaned_image_test(formula):
#     encoder = load_model('models/3ch_encoder_v1.h5')
#     height, width, _ = formula.shape
#     ratio = width / height
#     width = 256 * ratio
#     width = int(width)
#     cnt = width / 768
#     formula = cv2.resize(formula, dsize=(int(width), 256))
#     formula = cv2.fastNlMeansDenoisingColored(formula, None, 10, 10, 7, 21)
#     formula = cv2.GaussianBlur(formula, (5, 5), 0)
#     formula_array = []
#     # print("cnt: " , cnt)
#     for i in range(0, int(cnt) + 1):
#         # print("i : ", i)
#         if i == int(cnt):
#             black_img = np.full((256, (768 - (min(width, (i + 1) * 768) - (i * 768))), 3), 255, np.uint8)
#             tmp = formula[0:256, i * 768:max(width, (i + 1) * 768)]
#             black_img_height, black_img_width, _ = black_img.shape
#             tmp_height, tmp_width, _ = tmp.shape
#             # print("[black_img] height", black_img_height," width: ", black_img_width)
#             # print("[tmp] height", tmp_height," width: ", tmp_width)
#             tmp = cv2.hconcat([tmp, black_img])
#             formula_array.append(tmp)
#         else:
#             formula_array.append(formula[0:256, i * 768:(i + 1) * 768])
#
#     # for img in formula_array:
#     #     img_height, img_width, _ = img.shape
#     #     print("img : ", img_height, " width : ", img_width)
#     #     cv2.imshow("img", img)
#     #     cv2.waitKey(0)
#
#     # input array
#     formula_array = np.array(formula_array)
#     formula_array = formula_array.astype('float32') / 255.
#     # formula_array = np.expand_dims(formula_array, axis=0)
#     # predicted_formula result array
#     if int(cnt) == 0:
#         formula_array = encoder.predict(formula_array)
#     else:
#         formula_array = np.squeeze(encoder.predict(formula_array))
#
#     result = np.full((256, 0, 1), 1, np.float32)
#     for img in formula_array:
#         # cv2.imshow('predicted_formula_img', img)
#         result = cv2.hconcat([result, img])
#     result = result[0:256, 0:width]
#     plt.imshow(result)
#     plt.show()
#     # cv2.imshow('result', result)
#     # cv2.waitKey(0)
#
#     #
#     # # rst = cv2.resize(predicted_formula, (org_h, org_w), cv2.INTER_LINEAR)
#     # # rst = (rst * 255).astype(np.uint8)
#     #
#     # img_blur = cv2.GaussianBlur(rst, (3, 3), 1)
#     # _, rst_inv = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
#     # plt.imshow(rst, cmap='gray')
#     # plt.show()
#     result = (result * 255).astype(np.uint8)
#     img_gray = cv2.GaussianBlur(result, (5, 5), 0)
#     ret, img_inv = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     ret, img_normal = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return img_normal, img_inv
# formula = cv2.imread("test/images/fonts/!frac{ 33 }{ 71 }!sqrt { 2x ^{ 3 }+56A }=!frac { z }{ 3y!sqrt { frac { 1 }{ 2 } } } .png")
# cleaned_image_test(formula)

# def cleaned_image(formula):
#     encoder = load_model('models/3ch_encoder_v1.h5')
#     org_w, org_h, _ = formula.shape
#     plt.imshow(formula, cmap='gray')
#     plt.show()
#     formula = cv2.resize(formula, (768, 256), cv2.INTER_LINEAR)
#     formula = image.img_to_array(formula)
#     formula = formula.astype('float32') / 255.
#     formula = np.expand_dims(formula, axis=0)
#     predicted_formula = np.squeeze(encoder.predict(formula))
#
#     rst = cv2.resize(predicted_formula, (org_h, org_w), cv2.INTER_LINEAR)
#     rst = (rst * 255).astype(np.uint8)
#
#     img_blur = cv2.GaussianBlur(rst, (3, 3), 1)
#     _, rst_inv = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     plt.imshow(rst, cmap='gray')
#     plt.show()
#     return rst, rst_inv