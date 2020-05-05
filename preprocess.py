import cv2  # 문자단위로 인식 하기 위해
import pytesseract  # 문자단위로 잘려진 이미지 OCR(광학문자인식)
import numpy as np  # 배열 계산 및 조작
import matplotlib.pyplot as plt  # 이미지 띄워주기 위함

# path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = path


def show_images(images, cols=1, titles=None):
    """
    :param images: 배열을 한번에 한 plot에 띄어주기 위한 함수
    :param cols: 세로축 갯수
    :param titles: images와 같은 크기의 배열 각 이미지에 타이틀을 줌
    :return: none
    """
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
    """
    :param path: 상대경로로 프로젝트 폴더의 파일이름 쓰면됨 ex) test.png
    :return: grayScale된 이미지( 흑백 이미지로 변환, 이진화시키는것은 아님! 이진화는 다음 함수에서 진행 )
    """
    cv2.namedWindow('grayScale')
    formula = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('grayScale', formula)
    cv2.imwrite('image_result/grayScale.png', formula)
    cv2.waitKey(0)
    cv2.destroyWindow('grayScale')
    return formula


def blur_image(formula):
    """
    :param formula: 입력으로는 흑백사진이 드러옴
    :return: 입력으로 들어온 이미지를 blur 처리함 (밑에서 다시설명)
    # 부가설명
    ## cv2.GaussianBlur(formula, (3, 3), 0) : 3x3 크기를 기준으로 픽셀값들을 blur 블러처리함
    예를들어, 이미지속에서 {{1,1,1}{1,0,1}{1,1,1}} 가있으면 GaussainBlur 처리해주면 가운데 1,0,1이 1,1,1이 됨
    ##  cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshold :  흑백 이미지에서 이진화 파일로 만듬, 이미지속 픽셀하나하나를 0또는 1값만 가지게함
    0,255 : 0~255 의 값의 범위에 대해서 함수를 적용
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    - cv2.THRESH_BINARY_INV : 이진화 파일을 만드는데 거꾸로만듬 ( 즉 검은색은 흰색으로, 흰색은 검은색으로 ) INV가 INVERSE임
    - cv2.THRESH_OTSU : 이진화 파일을 만들때 어떤 값을 기준으로 0으로할꺼냐 1로할꺼냐를 정하는데, 이 옵션을 통해 전체 픽셀에서
    적당한값을 골라줌( 이게 만야 좋은 방법이라고 할 순 없음)
    """
    # cv2.namedWindow('binary')
    img_blur = cv2.GaussianBlur(formula, (3, 3), 0)
    ret, formula_inv = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, formula = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('binary', formula)
    # cv2.imwrite('image_result/binary.png', formula)
    # cv2.waitKey(0)
    # cv2.destroyWindow('binary')
    return formula, formula_inv

