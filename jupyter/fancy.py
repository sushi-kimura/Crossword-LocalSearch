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
        if type(width) is not int:
            raise TypeError(f"Type of 'width' must be integer. {type(width)} given")
        if type(height) is not int:
            raise TypeError(f"Type of 'height' must be integer. {type(height)} given")
        if mask is not None and type(mask) not in(list, np.ndarray):
            raise TypeError(f"Type of 'mask' must be list or np.ndarray. {type(mask)} given")
        self.width = width
        self.height = height
        self.totalWeight = 0
        self.title = title
        self.cell = np.full(width*height, "", dtype="unicode").reshape(height, width)
        self.cover = np.zeros(width*height, dtype="int").reshape(height, width)
        self.coverDFS = np.zeros(width*height, dtype="int").reshape(height, width)
        self.enable =  np.ones(width*height, dtype="bool").reshape(height, width)
        if mask is None:
             mask = np.ones(width*height, dtype="bool").reshape(height, width)
        self.mask = mask
        self.enable = self.enable*mask
        self.usedWords = np.full(width*height, "", dtype=f"U{max(width, height)}")
        self.usedPlcIdx = np.full(width*height, -1, dtype="int")
        self.solSize = 0
        self.history = []
        self.baseHistory = []
        self.log = None
        self.epoch = 0
        self.ccl = None
        self.initSol = False
        self.initSeed = None
        self.dic = Dictionary(msg=False)
        self.plc = Placeable(self.width, self.height, self.dic, msg=False)
        self.objFunc = None
        self.optimizer = None

        ## Message
        if msg is True:
            print("Puzzle object has made.")
            print(f" - title       : {self.title}")
            print(f" - width       : {self.width}")
            print(f" - height      : {self.height}")
            print(f" - cell' shape : (width, height) = ({self.cell.shape[0]},{self.cell.shape[1]})")

    def _drop(self, div, i, j, k, isKick=False):
        """
        This method removes the specified word from the puzzle.
        Note: This method pulls out the specified word without taking it into consideration, which may break the connectivity of the puzzle or cause LAOS / US / USA problems.
        """
        # Get p, pidx
        p = self.plc.invP[div, i, j, k]
        pidx = np.where(self.usedPlcIdx == p)[0][0]
        
        wLen = self.dic.wLen[k]
        weight = self.dic.weight[k]
        # Pull out a word
        if div == 0:
            self.cover[i:i+wLen,j] -= 1
            where = np.where(self.cover[i:i+wLen,j] == 0)[0]
            jall = np.full(where.size, j, dtype="int")
            self.cell[i+where,jall] = ""
        if div == 1:
            self.cover[i,j:j+wLen] -= 1
            where = np.where(self.cover[i,j:j+wLen] == 0)[0]
            iall = np.full(where.size, i, dtype="int")
            self.cell[iall,j+where] = ""
        # Update usedWords, usedPlcIdx, solSize, totalWeight
        self.usedWords = np.delete(self.usedWords, pidx)  # delete
        self.usedWords = np.append(self.usedWords, "")  # append
        self.usedPlcIdx = np.delete(self.usedPlcIdx, pidx)  # delete
        self.usedPlcIdx = np.append(self.usedPlcIdx, -1)  # append
        self.solSize -= 1
        self.totalWeight -= weight
        # Insert data to history
        code = 3 if isKick else 2
        self.history.append((code, k, div, i, j))
        # Release prohibited cells
        removeFlag = True
        if div == 0:
            if i > 0:
                if i > 2 and np.all(self.cell[[i-3,i-2],[j,j]] != ""):
                    removeFlag = False
                if j > 2 and np.all(self.cell[[i-1,i-1],[j-2,j-1]] != ""):
                    removeFlag = False
                if j < self.width-2 and np.all(self.cell[[i-1,i-1],[j+1,j+2]] != ""):
                    removeFlag = False
                if removeFlag == True:
                    self.enable[i-1,j] = True
            if i+wLen < self.height:
                if i+wLen < self.height-2 and np.all(self.cell[[i+wLen+1,i+wLen+2],[j,j]] != ""):
                    removeFlag = False
                if j > 2 and np.all(self.cell[[i+wLen,i+wLen],[j-2,j-1]] != ""):
                    removeFlag = False
                if j < self.width-2 and np.all(self.cell[[i+wLen,i+wLen],[j+1,j+2]] != ""):
                      removeFlag = False
                if removeFlag == True:
                    self.enable[i+wLen,j] = True
        if div == 1:
            if j > 0:
                if j > 2 and np.all(self.cell[[i,i],[j-3,j-2]] != ""):
                    removeFlag = False
                if i > 2 and np.all(self.cell[[i-2,i-1],[j-1,j-1]] != ""):
                    removeFlag = False
                if i < self.height-2 and np.all(self.cell[[i+1,i+2],[j-1,j-1]] != ""):
                    removeFlag = False
                if removeFlag == True:
                    self.enable[i,j-1] = True
            if j+wLen < self.width:
                if j+wLen < self.width-2 and np.all(self.cell[[i,i],[j+wLen+1,j+wLen+2]] != ""):
                    removeFlag = False
                if i > 2 and np.all(self.cell[[i-2,i-1],[j+wLen,j+wLen]] != ""):
                    removeFlag = False
                if i < self.height-2 and np.all(self.cell[[i+1,i+2],[j+wLen,j+wLen]] != ""):
                    removeFlag = False
                if removeFlag == True:
                    self.enable[i,j+wLen] = True
        self.enable *= self.mask

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
puzzle.solve(epoch=10)
print(f"SimpleSolution: {puzzle.isSimpleSol()}")
puzzle.saveAnswerImage(f"fig/puzzle/{dic.name}_w{width}_h{height}_r{seed}.png", "【単語リスト】")

e_time = time.time() - start
print (f"e_time: {format(e_time)} s")
