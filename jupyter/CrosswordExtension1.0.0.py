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

# # CrosswordExtension
# ## 概要
# このノートブックではクラスワード(スケルトンパズル)自動生成ツールの拡張機能について紹介します。

# ***
#
# ## Import
# 必要なライブラリをimportし, 日本語フォントの指定などを行う：

# +
import os
import sys
import glob
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
import cv2

sys.path.append('../python')
from sample_package import Puzzle, Dictionary, Placeable, ObjectiveFunction, Optimizer

start = time.time()


# -

# ## ユーザ用add
# [CrosswordBasic](CrosswordBasic.ipynb)で作成した`_add`メソッドを拡張し、ユーザにとって扱いやすい`add`メソッドを定義します。  
# `_add`メソッドと同じように辞書番号を指定することもできますが、追加する単語が指定されることを想定しています。辞書に存在しない単語が指定された場合は、辞書に単語を追加しPlaceableも再計算されます。

def add(self, div, i, j, word, weight=0):
    if type(word) is int:
        k = word
    elif type(word) is str:
        self.dic.add(word, weight)
        self.plc._compute([word], self.dic.size-1)
        k = self.dic.word.index(word)
    else:
        raise TypeError()
    self._add(div, i, j, k)
setattr(Puzzle, 'add', add)

# +
fpath = "../dict/pokemon.txt"  # countries hokkaido animals kotowaza birds dinosaurs fishes sports
width = 10
height = 10
seed = 1

puzzle = Puzzle(width, height, msg=False)
dic = Dictionary(fpath, msg=False)
puzzle.importDict(dic, msg=False)
puzzle.add(1, 6, 3, 'カラス')
puzzle.add(0, 4, 3, 'アシカ')
puzzle.add(1, 4, 2, 'コアラ')
puzzle.show()


# -

# ---
# ## 反復局所探索法
# [CrosswordBasic](CrosswordBasic.ipynb)では`局所探索法`を実装しました。局所探索法により局所最適解に到達した解に対して、`摂動`と呼ばれる操作を行うことで解を変更し、再び局所探索を行うという操作を繰り返す最適化手法を`反復局所探索法`と呼びます。以下ではこの手法を実装するための準備を行います。
#
# ## 摂動：パズルの平行移動
# 摂動として盤面のパズルを平行移動するという操作を実装します。まずは盤面からパズルの矩形を取り出す`getRect`メソッドを定義します。続いて、パズルを平行移動する`move`メソッドを実装します。

# +
def getRect(self):
    rows = np.any(self.cover, axis=1)
    cols = np.any(self.cover, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax
setattr(Puzzle, "getRect", getRect)

def move(self, direction, n=0, limit=False):
    rmin, rmax, cmin, cmax = self.getRect()
    str2int= {'U':1, 'D':2, 'R':3, 'L':4}
    if direction in ('U','D','R','L','u','d','r','l'):
        direction=str2int[direction.upper()]
    if direction not in (1,2,3,4):
        raise ValueError()
    if n < 0:
        reverse = {'1':2, '2':1, '3':4, '4':3}
        direction = reverse[str(direction)]
        n = -n
    if limit is True:
        n2limit = {1:rmin, 2:self.height-(rmax+1), 3:self.width-(cmax+1), 4:cmin}
        n = n2limit[direction] 
        
    if direction is 1:
        if rmin < n:
            raise RuntimeError()
        self.cell = np.roll(self.cell, -n, axis=0)
        self.cover = np.roll(self.cover, -n, axis=0)
        self.coverDFS = np.roll(self.coverDFS, -n, axis=0)
        self.enable = np.roll(self.enable, -n, axis=0)
        for i,p in enumerate(self.usedPlcIdx[:self.solSize]):
            self.usedPlcIdx[i] = self.plc.invP[self.plc.div[p], self.plc.i[p]-n, self.plc.j[p], self.plc.k[p]]
    if direction is 2:
        if self.height-(rmax+1) < n:
            raise RuntimeError()
        self.cell = np.roll(self.cell, n, axis=0)
        self.cover = np.roll(self.cover, n, axis=0)
        self.coverDFS = np.roll(self.coverDFS, n, axis=0)
        self.enable = np.roll(self.enable, n, axis=0)
        for i,p in enumerate(self.usedPlcIdx[:self.solSize]):
            self.usedPlcIdx[i] = self.plc.invP[self.plc.div[p], self.plc.i[p]+n, self.plc.j[p], self.plc.k[p]]
    if direction is 3:
        if self.width-(cmax+1) < n:
            raise RuntimeError()
        self.cell = np.roll(self.cell, n, axis=1)
        self.cover = np.roll(self.cover, n, axis=1)
        self.coverDFS = np.roll(self.coverDFS, n, axis=1)
        self.enable = np.roll(self.enable, n, axis=1)
        for i,p in enumerate(self.usedPlcIdx[:self.solSize]):
            self.usedPlcIdx[i] = self.plc.invP[self.plc.div[p], self.plc.i[p], self.plc.j[p]+n, self.plc.k[p]]
    if direction is 4:
        if cmin < n:
            raise RuntimeError()
        self.cell = np.roll(self.cell, -n, axis=1)
        self.cover = np.roll(self.cover, -n, axis=1)
        self.coverDFS = np.roll(self.coverDFS, -n, axis=1)
        self.enable = np.roll(self.enable, -n, axis=1)
        for i,p in enumerate(self.usedPlcIdx[:self.solSize]):
            self.usedPlcIdx[i] = self.plc.invP[self.plc.div[p], self.plc.i[p], self.plc.j[p]-n, self.plc.k[p]]
    
    self.history.append((4, direction, n))
setattr(Puzzle, "move", move)
# -

# `direction`引数は数値と文字列の両方で指定することができます。また、`limit`オプションをTrueにすることで、パズルを移動できるだけ平行移動することができます。
# 指定可能な値は以下の通り：
# * 上方向：`'U'`, `'u'`, `1`
# * 下方向：`'D'`, `'d'`, `2`
# * 左方向：`'L'`, `'l'`, `3`
# * 右方向：`'R'`, `'r'`, `4`

puzzle.move('U', 2)
puzzle.show()
puzzle.move(3, limit=True)
puzzle.show()

# ***
#
# ## Pickleオブジェクトのオープン
# ここでは、既に作成されたパズルデータを元に、様々な拡張機能について語ります。
# そこで、今回は`pickle/sample.pickle`というpickleファイルをロードします。

with open("pickle/sample.pickle", "rb") as f:
    sample_puzzle = pickle.load(f)
sample_puzzle.show()

# これで生成済みのパズルデータをオープンすることができました。

# ---
# ## （番外編）解の軌跡をアニメーション化
# 解の軌跡をアニメーション化してみましょう。
# パズルの巻き戻し・早送り機能を使って、作業履歴を最初から順番に画像化し、
# 外部ファイルを用いてそれを動画化します（このセルの実行には数分かかる場合があります）。

for p in glob.glob("fig/animation/*.png"):
     if os.path.isfile(p):
            os.remove(p)
# jump to top of the frame
tmpPuzzle = sample_puzzle.jump(0)
tmpPuzzle.saveAnswerImage(f"fig/animation/0000.png")
# save all history as image file
for histNum in range(len(tmpPuzzle.baseHistory)):
    tmpPuzzle = tmpPuzzle.getNext()
    tmpPuzzle.saveAnswerImage(f"fig/animation/{str(histNum+1).zfill(4)}.png")

# 動画化にはmovie_maker.pyを用います。コマンドライン引数で画像が入ったディレクトリとFPSを指定します。

# !python ../python/script/movie_maker.py "fig/animation/" -o "fig/animation/out.mp4" -f 10 -c mp4v

# これで、fig/animation内にout.mp4という動画ファイルが作成されました。
# 再生してみましょう。

from IPython.display import Video
Video("fig/animation/out.mp4", width=960, height=480)


