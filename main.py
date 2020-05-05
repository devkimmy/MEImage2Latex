import cv2
from makebox_CNN import CNN_LATEX
import makeparsetree as mpt

model = CNN_LATEX(mean_train="z_score_normalization/mean_tr.npy", std_train="z_score_normalization/std_tr.npy")
formula = cv2.imread("test_image/test04.jpeg", cv2.IMREAD_GRAYSCALE)
left_boxes, right_boxes = model.predict(formula)

# 알고리즘 start
# 분수 기호를 찾는다
left_compound = mpt.set_fraction(left_boxes)
right_compound = mpt.set_fraction(right_boxes)
print("# After set_fraction")
print("# converted_left\n" + mpt.print_compounds(left_compound))
print("# converted_right\n" + mpt.print_compounds(right_compound))

# 근호를 찾고 파스 트리 확장
left_compound = mpt.set_square_root(left_compound)
right_compound = mpt.set_square_root(right_compound)
print("# After set_square_root")
print("# converted_left\n" + mpt.print_compounds(left_compound))
print("# converted_right\n" + mpt.print_compounds(right_compound))

# 첨자를 찾고 파스 트리 확장
left_compound = mpt.set_script(left_compound)
right_compound = mpt.set_script(right_compound)

# 생성된 파스트리를 DFS 순회하여 latex format 결과물 출력
print("# After set_super_script")
print("# converted_left\n" + mpt.print_compounds(left_compound))
print("# converted_right\n" + mpt.print_compounds(right_compound))
