
# coding: utf-8
"""
Crossword Local Search
概要
 これはクロスワード(スケルトンパズル)自動生成用ソースをPythonにて再現できないかをテストしたものである。
 main.cはpythonで代用し、計算量の多い処理は既存のcommon.cをimportして使用する。
 結果やスコアの推移の視覚化も追加する。
"""
# ***
# ## Import
# 必要なライブラリをimportする：
import numpy as np
from numpy.random import *
import pandas as pd
import unicodedata
import itertools
import sys
import copy

from common import *
from IPython.display import display
from PIL import Image
from IPython.display import HTML

# countries hokkaido animals kotowaza birds dinosaurs fishes sports
fpath = "../dict/countries.txt"
width = 10
height = 10
randomSeed = 10
withweight = False
takemove = True
seed(seed = randomSeed)
print("==================================================================")
puzzle = Puzzle(width, height)
dic = Dictionary(fpath)
plc = Placeable(puzzle, dic)
objFunc = ObjectiveFunction(puzzle)
optimizer = Optimizer()
print("==================================================================")
puzzle.firstSolve(dic, plc)
objFunc.register(["solSize", "crossCount", "fillCount", "maxConnectedEmptys"])
optimizer.setMethod("localSearch")
puzzle.compile(dictionary=dic, placeable=plc, objFunc=objFunc, optimizer=optimizer)
print("==================================================================")
puzzle.solve(epock=10)
print("SimpleSolution: %s" % puzzle.isSimpleSol(plc))
