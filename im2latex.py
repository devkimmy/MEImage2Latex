import cv2
from makebox_CNN import CNN_LATEX
import makeparsetree as mpt
import matplotlib.pyplot as plt


def translate(img):
    model = CNN_LATEX(mean_train="z_score_normalization/mean_vgg16_v20.npy",
                      std_train="z_score_normalization/std_vgg16_v20.npy")
    formula = cv2.imread(img, cv2.IMREAD_COLOR)
    # cv2.imshow("original", formula)
    # cv2.waitKey(0)
    height, width,_ = formula.shape
    ratio = width/height
    wdith = 512 * ratio
    formula = cv2.resize(formula,dsize=(int(wdith),512),interpolation=cv2.INTER_CUBIC)
    # formula = cv2.resize(formula,dsize=(1024,512),interpolation=cv2.INTER_CUBIC)
    # plt.imshow("resized", formula)
    # cv2.waitKey(0)
    # plt.imshow(formula)
    # plt.show()
    left_boxes, right_boxes = model.predict(formula)

    # 알고리즘 start
    # 분수 기호를 찾는다

    # print("# After set_fraction")
    # print("# converted_left\n" + mpt.print_compounds(left_compound))
    # print("# converted_right\n" + mpt.print_compounds(right_compound))

    # 근호를 찾고 파스 트리 확장
    left_compound = mpt.set_fraction(left_boxes)
    right_compound = mpt.set_fraction(right_boxes)

    left_compound = mpt.set_square_root(left_compound)
    right_compound = mpt.set_square_root(right_compound)
    # print("# After set_square_root")
    # print("# converted_left\n" + mpt.print_compounds(left_compound))
    # print("# converted_right\n" + mpt.print_compounds(right_compound))

    # 첨자를 찾고 파스 트리 확장
    left_compound = mpt.set_script(left_compound)
    right_compound = mpt.set_script(right_compound)

    # 생성된 파스트리를 DFS 순회하여 latex format 결과물 출력
    # print("# After set_super_script")
    # print("# converted_left\n" + mpt.print_compounds(left_compound))
    # print("# converted_right\n" + mpt.print_compounds(right_compound))
    left = mpt.print_compounds(left_compound)
    right = mpt.print_compounds(right_compound)
    if len(left) == 0:
        return right
    elif len(right) == 0:
        return left
    else:
        return (left + '=' + right).replace(' ', '')


# print(translate('test_formula/test_formula/!alpha!theta=36!beta.jpg'))
