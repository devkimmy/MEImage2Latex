'''
This module evaluates model with Test images
'''
import argparse
import os
import im2latex
from sklearn.metrics import *

parser = argparse.ArgumentParser(description='Evaluates model with test formula images.')
parser.add_argument('--path', required=False, default='test/images/handwriting/', help='image directory path')
args = parser.parse_args()

img_path = args.path  # Default: test/images/
file_list = os.listdir(img_path)
formulas = []
pred = []

print('Start Evaluation...\n')

for name in file_list:
    formulas.append(name[:-4].replace(' ', ''))

for (target,tmp) in zip(file_list,formulas):
    pred_formula = im2latex.translate(img_path + target)
    pred.append(pred_formula)
    if tmp != pred_formula:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("#origint:", target)
    print("#predict:", pred_formula + "\n")

# for tr, pr in zip(formulas, pred):
#     if tr not in pr:
#         print('True: %s\nPred: %s' % (tr, pr))

print("acc: %.2f%%" % (accuracy_score(formulas, pred) * 100))
