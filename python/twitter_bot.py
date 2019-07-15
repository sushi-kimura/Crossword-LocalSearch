"""
Crossword Local Search by command line

引数：
 1. 辞書ファイルのパス
 2. パズルの横幅
 3. パズルの縦幅
 4. シード値（-sまたは--seedオプションで指定. デフォルトは66666）
 5. エポック数（-eまたは--epochオプションで指定. デフォルトは10）
 6. パズルのタイトル（-tまたは--titleオプションで指定. デフォルトは{辞書名}_w{width}_h{height}_r{seed}_ep{epoch}）
 7. 重みを考慮するかどうか（-wまたは--weightオプションでフラグとして指定. デフォルトはFalse）
 8. 出力ファイル名（-oまたは--outputオプションで指定. デフォルトは{title}.png）

出力：
 問題画像のパス, 回答画像のパス

実行例：
python twitter_bot.py dict/pokemon.txt -w 15 -h 15 -s 1 -e 15
"""

import os
import argparse
import numpy as np
from matplotlib.font_manager import FontProperties

#os.chdir("/Users/taiga/Crossword-LocalSearch/Python")
from src import Puzzle, Dictionary, ObjectiveFunction, Optimizer
import puzzle_generator

# In[]
parser = argparse.ArgumentParser(description="make a puzzle with given parameters")
parser.add_argument("fpath", type=str,
                    help="file path of a dictionary")
parser.add_argument("width", type=int,
                    help="witdh of the puzzle")
parser.add_argument("height", type=int,
                    help="height of the puzzle")
parser.add_argument("-s", "--seed", type=int, default=66666,
                    help="random seed value, default=66666")
parser.add_argument("-e", "--epoch", type=int, default=10,
                    help="epoch number of local search, default=10")
parser.add_argument("-t", "--title", type=str,
                    help="title of the puzzle")
parser.add_argument("-w", "--weight", action="store_true",
                    help="flag of consider the weight, default=False")
parser.add_argument("-o", "--output", type=str,
                    help="name of the output image file")
args = parser.parse_args()

# settings
fpath = args.fpath # countries hokkaido animals kotowaza birds dinosaurs fishes sports pokemon typhoon cats s_and_p100
width = args.width
height = args.height
seed = args.seed
epoch = args.epoch
title = args.title
withWeight = args.weight
output = args.output

# solve
is_simple = False
loop_num = 0
while is_simple is False and loop_num != 50:
    pass_problem, pass_answer, is_simple = puzzle_generator.get(fpath, width, height, seed, epoch, title, withWeight, output)
    seed += 1
    loop_num += 1

if loop_num >= 50:
    print("False")
else:
    print(pass_problem, pass_answer)
