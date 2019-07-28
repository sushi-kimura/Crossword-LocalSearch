# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # FancyPuzzle
# ## 概要
# このノートブックでは、長方形に限らず、好きな形のパズルを生成する方法を考えます。

# +
import os
import sys
import copy
import datetime
import time
import math
import itertools
import unicodedata
import collections
import pickle
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from IPython.display import display, HTML
import matplotlib.pyplot as plt

sys.path.append('../python')
from sample_package import Puzzle, Dictionary, Placeable, ObjectiveFunction, Optimizer
from src import utils


# -

# ## FancyPuzzle

class FancyPuzzle(Puzzle):
    def __init__(self, width, height, mask=None, title="スケルトンパズル", msg=True):
        if mask is None:
            mask = np.ones(width*height, dtype="bool").reshape(height, width)
        self.mask = mask
        
        super().__init__(width, height, title, msg)

        self.enable = self.enable*mask

    def isEnabledAdd(self, div, i, j, word, wLen):
        """
        This method determines if a word can be placed
        """
        if div == 0:
            if np.any(self.mask[i:i+wLen, j] == False):
                return 7
        if div == 1:
            if np.any(self.mask[i, j:j+wLen] == False):
                return 7
    
        return super().isEnabledAdd(div, i, j, word, wLen)

    def saveImage(self, data, fpath, list_label="[Word List]", dpi=100):
        """
        This method generates and returns a puzzle image with a word list
        """
        # Generate puzzle image
        colors = np.where(self.cover<1, "#000000", "#FFFFFF")
        df = pd.DataFrame(data)

        fig=plt.figure(figsize=(16, 8), dpi=dpi)
        ax1=fig.add_subplot(121) # puzzle
        ax2=fig.add_subplot(122) # word list
        ax1.axis("off")
        ax2.axis("off")
        fig.set_facecolor('#EEEEEE')
        # Draw puzzle
        ax1_table = ax1.table(cellText=df.values, cellColours=colors, cellLoc="center", bbox=[0, 0, 1, 1])
        for _, cell in ax1_table.get_celld().items():
            cell.set_text_props(size=20)
        ax1.set_title(label="*** "+self.title+" ***", size=20)
        
        # delete unmasked cells
        mask = np.where(puzzle.mask== False)
        for i,j in list(zip(mask[0], mask[1])):
            del ax1_table._cells[i,j]

        # Draw word list
        words = [word for word in self.usedWords if word != ""]
        if words == []:
            words = [""]
        words.sort()
        words = sorted(words, key=len)

        rows = self.height
        cols = math.ceil(len(words)/rows)
        padnum = cols*rows - len(words)
        words += ['']*padnum
        words = np.array(words).reshape(cols, rows).T

        ax2_table = ax2.table(cellText=words, cellColours=None, cellLoc="left", edges="open", bbox=[0, 0, 1, 1])
        ax2.set_title(label=list_label, size=20)
        for _, cell in ax2_table.get_celld().items():
            cell.set_text_props(size=18)
        plt.tight_layout()
        plt.savefig(fpath, dpi=dpi)
        plt.close()


# ### フォント設定
# 本ライブラリにおける画像化には`matplotlib`が用いられますが、`matplotlib`はデフォルトで日本語に対応したフォントを使わないので、`rcParams`を用いてデフォルトのフォント設定を変更します。

# font setting
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP']

# ## 実行

# +
fpath = "../dict/pokemon.txt"  # countries hokkaido animals kotowaza birds dinosaurs fishes sports pokemon typhoon
width = 15
height = 15
seed = 2
withWeight = False

np.random.seed(seed=seed)
start = time.time()

# +
mask = np.array([
    [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
], dtype="bool")
# Make instances
puzzle = FancyPuzzle(width, height, mask, "ドーナツパズル")
dic = Dictionary(fpath)
if not withWeight:
    dic.calcWeight()
objFunc = ObjectiveFunction()
optimizer = Optimizer()

puzzle.importDict(dic)
# -

# Register and set method and compile
objFunc.register(["totalWeight", "solSize", "crossCount", "fillCount", "maxConnectedEmpties"])
optimizer.setMethod("localSearch")
puzzle.compile(objFunc=objFunc, optimizer=optimizer)

# Solve
puzzle.firstSolve()
puzzle.solve(epoch=1)
print(f"SimpleSolution: {puzzle.isSimpleSol()}")
puzzle.saveAnswerImage(f"fig/puzzle/{dic.name}_w{width}_h{height}_r{seed}.png", "【単語リスト】")

e_time = time.time() - start
print (f"e_time: {format(e_time)} s")
