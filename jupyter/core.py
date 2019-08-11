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

# # 開発者向けJupyter Notebook
# ## 概要
# このノートブックは開発者が新規機能の実装や機能修正をする際に変更を共有するために使用します。

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

sys.path.append("../python")
from pyzzle import Puzzle, Dictionary, Placeable, ObjectiveFunction, Optimizer
# %load_ext autoreload
# %autoreload 2
# -

# ## フォント設定
# 本ライブラリにおける画像化には`matplotlib`が用いられますが、`matplotlib`はデフォルトで日本語に対応したフォントを使わないので、`rcParams`を用いてデフォルトのフォント設定を変更します。

# font setting
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP']

# ## 実行

# +
fpath = "../dict/typhoon.txt"  # countries hokkaido animals kotowaza birds dinosaurs fishes sports pokemon typhoon
width = 15
height = 15
seed = 1
withweight = False

np.random.seed(seed=seed)
start = time.time()

# +
# Make instances
puzzle = Puzzle(width, height)
dic = Dictionary(fpath)
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
puzzle.saveAnswerImage(f"fig/puzzle/{dic.name}_w{width}_h{height}_r{seed}.png")

e_time = time.time() - start
print (f"e_time: {format(e_time)} s")

# ## Package更新
# Jupytextによるpyファイル生成との合わせ技です。  
# まずは先にノートブックを保存してから、次のセルを実行してください。  
# `../python/pyzzle`ディレクトリのパッケージ情報が更新されます。

# +
# import os, shutil
# # !python ../python/script/ipynbpy2py.py core.py -n pyzzle
# if os.path.exists("../python/pyzzle") is True:
#     shutil.rmtree('../python/pyzzle')
# shutil.move("pyzzle", "../python")
# -

# ## Sphinxドキュメントを更新
# `conda install sphinx`と`pip install sphinx_rtd_theme`で必要ライブラリをインストールしてから次のセルを実行してください。

# +
###開発中### nbsphinxの利用を検討中
# os.chdir('../doc')
# # !sphinx-build -b html ./source ./source/_build
# # !sphinx-build -b html ./source ./source/_build #なぜか二度実行するとうまくいく
# os.chdir('../jupyter')

