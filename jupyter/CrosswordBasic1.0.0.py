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

# # CrosswordBasic
# ## 概要
# このノートブックではクロスワード(スケルトンパズル)自動生成ツールおよびそれらの可視化について紹介します。

# ***
#
# ## 入力データ・実行パラメータ設定
# 入力データを指定し、各種実行パラメータの設定を行います。
# 各パラメータは以下の通り：
#   * `fpath`      : 入力データ(単語リスト)のファイルパス
#   * `width`          : 盤面の大きさ(横)
#   * `height`          : 盤面の大きさ(縦)
#   * `seed`       : シード値
#   * `withWeight` : 辞書に重みを付すかどうか(bool)
#   * `title` : パズルのタイトル（デフォルトは「スケルトンパズル」）

fpath = f"../dict/typhoon.txt"  # countries hokkaido animals kotowaza birds dinosaurs fishes sports pokemon typhoon
width = 15
height = 15
seed = 6
withWeight = False
title = "台風パズル"  # default:スケルトンパズル

# ***
#
# ## Import
# 必要なライブラリをimportし, 乱数のシード値および, 日本語フォントの指定などを行う：

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
from src import utils

np.random.seed(seed = seed)
start = time.time()
# -

# ### フォント設定
# 本ライブラリにおける画像化には`matplotlib`が用いられますが、`matplotlib`はデフォルトで日本語に対応したフォントを使わないので、`rcParams`を用いてデフォルトのフォント設定を変更します。

# font setting
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP']


# ***
#
# ## クラス宣言
# 本プログラムで使用するクラスを定義する。  
# 見やすさのため、クラスメソッドやインスタンスメソッドは後から定義し、`setattr`関数で属性として追加します。
#
#
# ### Puzzle クラス
# 解となるスケルトンパズルそのものを表すクラス。
# メンバ変数は以下の通り：
#   * width : 盤面の大きさ(横)
#   * height : 盤面の大きさ(縦)
#   * totalWeight : 単語の重みの合計
#   * title：パズルのタイトル(str)
#   * enable : 配置禁止マスを保持した2次元(width*height)配列
#   * cell : パズルの解を保存する2次元(width*height)配列
#   * cover : セル上の文字数を保持する2次元(width*height)配列
#   * coverDFS : 連結成分を探すときに使われる2次元(width*height)配列
#   * usedWords : 解として使われた単語の一覧
#   * usedPlcIdx : 解として使われた(後に定義する)Placeable配列上の添え字一覧
#   * solSize : パズルに配置されている単語の数
#   * history : 単語増減の履歴
#   * baseHistory : 後に説明するjumpメソッドにより解を移動した際に保持されるhistory
#   * historyIdx : 現在参照している履歴番号
#   * log：目的関数値の履歴
#   * epoch : 初期解から局所探索した回数
#   * ccl : 連結成分標識
#   * initSol : 初期解が作られたかどうか(bool)
#   * initSeed：初期解作成開始時点のseed値
#   * dic：Dictionaryオブジェクト(後述)
#   * plc：Placeableオブジェクト(後述)
#   * objFunc：ObjectiveFunctionオブジェクト(後述)
#   * optimizer：Optimizerオブジェクト(後述)

class Puzzle:
    def __init__(self, width, height, title="スケルトンパズル", msg=True):
        self.width = width
        self.height = height
        self.totalWeight = 0
        self.title = title
        self.cell = np.full(width * height, "", dtype="unicode").reshape(height, width)
        self.cover = np.zeros(width * height, dtype="int").reshape(height, width)
        self.coverDFS = np.zeros(width * height, dtype="int").reshape(height, width)
        self.enable = np.ones(width * height, dtype="bool").reshape(height, width)
        self.usedWords = np.full(width * height, "", dtype=f"U{max(width, height)}")
        self.usedPlcIdx = np.full(width * height, -1, dtype="int")
        self.solSize = 0
        self.history = []
        self.baseHistory = []
        self.log = None
        self.epoch = 0
        self.ccl = None
        self.initSol = False
        self.initSeed = None
        self.dic = None
        self.plc = None
        self.objFunc = None
        self.optimizer = None
        ## Message
        if msg is True:
            print(f"{self.__class__.__name__} object has made.")
            print(f" - title       : {self.title}")
            print(f" - width       : {self.width}")
            print(f" - height      : {self.height}")
            print(f" - cell' shape : (width, height) = ({self.cell.shape[0]},{self.cell.shape[1]})")
    def __str__(self):
        return self.title
    def reinit(self, all=False):
        if all is True:
            self.dic = None
            self.plc = None
            self.objFunc = None
            self.optimizer = None
        self.totalWeight = 0
        self.enable = np.ones(self.width*self.height, dtype="bool").reshape(self.height, self.width)
        self.cell = np.full(self.width*self.height, "", dtype="unicode").reshape(self.height, self.width)
        self.cover = np.zeros(self.width*self.height, dtype="int").reshape(self.height, self.width)
        self.coverDFS = np.zeros(self.width*self.height, dtype="int").reshape(self.height, self.width)
        self.enable = np.ones(self.width*self.height, dtype="bool").reshape(self.height, self.width)
        self.usedWords = np.full(self.width*self.height, "", dtype=f"U{max(self.width, self.height)}")
        self.usedPlcIdx = np.full(self.width*self.height, -1, dtype="int")
        self.solSize = 0
        self.baseHistory = []
        self.history = []
        self.log = None
        self.epoch = 0
        self.initSol = False
        self.initSeed = None


sample_puzzle = Puzzle(width, height, title)


# ### Dictionary クラス
# 入力した単語リストを整理して保持するクラス。
# メンバ変数は以下の通り：
#   * fpath : 入力データのファイルパス
#   * size : 辞書の大きさ(単語数)
#   * dictType : 辞書のタイプ("English"/"Japanese")
#   * word : 単語配列
#   * weight : 重み配列
#   * wLen : 単語長配列
#   * removedWords : 削除された単語配列

class Dictionary:
    def __init__(self, fpath, msg=True):
        self.fpath = fpath
        self.name = os.path.basename(fpath)[:-4]
        # Read
        with open(self.fpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        self.word = [d[0] for d in data]
        self.weight = [d[1] for d in data]
        self.wLen = [len(w) for w in self.word]
        self.removedWords = []
        # Get a size of dictionary
        self.size = len(data)
        # Check dictionary type(English/Japanese/'Kanji')
        uniName = unicodedata.name(data[0][0])[0:10]
        if "HIRAGANA" in uniName or "KATAKANA" in uniName:
            self.dictType = "Japanese"
        elif "LATIN" in uniName:
            self.dictType = "English"
        elif "CJK" in uniName:
            self.dictType = "Kanji"

        # Remove "\n"
        def removeNewLineCode(word):
            line = word.rstrip("\n").split(" ")
            if len(line) == 1:
                line.append(0)
            line[1] = int(line[1])
            return line
        dic_list = list(map(removeNewLineCode, data))
        self.word = [d[0] for d in dic_list]
        self.weight = [d[1] for d in dic_list]
        self.wLen = [len(w) for w in self.word]

        # Message
        if msg is True:
            print("Dictionary object has made.")
            print(f" - file path         : {self.fpath}")
            print(f" - dictionary size   : {self.size}")
            print(f" - dictionary type   : {self.dictType}")
            print(f" - top of dictionary : {self[0]}")
    
    def include(self, word):
        return word in self.word

    def add(self, word=None, weight=None, fpath=None, msg=True):
        if (word,fpath) == (None,None):
            raise ValueError("'word' or 'fpath' must be specified")
        if word is not None and fpath is not None:
            raise ValueError("'word' or 'fpath' must be specified")
        if fpath is not None:
            self.read(fpath)
        if word is not None:
            if type(word) is str:
                    word = [word]
            if weight is None:
                weight = [0]*len(word)
            else:
                if type(weight) is int:
                    weight = [weight]
                if len(word) != len(weight):
                    raise ValueError(f"'word' and 'weight' must be same size")

            for wo, we in zip(word, weight):
                if self.include(wo) and msg is True:
                    print(f"The word '{wo}' already exists")
                self.word.append(wo)
                self.weight.append(we)
                self.wLen.append(len(wo))
                self.size += 1

    def __getitem__(self, key):
        return {'word': self.word[key], 'weight': self.weight[key], 'len': self.wLen[key]}
    
    def __str__(self):
        return self.name
    
    def __len__(self):
        return self.size
    
    def getK(self, word):
        return np.where(self.word == word)[0][0]


sample_dic = Dictionary(fpath)


# 無駄な計算を減らすため、他のどの単語とも接続(クロス)できない単語は辞書からあらかじめ削除しておきます。  
# `Dictionary`クラスに`deleteUnusableWords`メソッドを実装します：

def deleteUnusableWords(self, msg=True):
    """
    This method checks words in the dictionary and erases words that can not cross any other words.
    """
    mergedWords = "".join(self.word)
    counts = collections.Counter(mergedWords)
    for i, w in enumerate(self.word[:]):
        charValue = 0
        for char in set(w):
            charValue += counts[char]
        if charValue == len(w):
            self.removedWords.append(w)
            del self.word[i]
            del self.weight[i]
            del self.wLen[i]
            self.size -= 1
            if msg is True:
                print(f"'{w}' can not cross with any other words")
setattr(Dictionary, "deleteUnusableWords", deleteUnusableWords)

sample_dic.deleteUnusableWords()


# これで、他の単語とクロスしない単語を辞書から削除できました。  
# 削除された単語は`removedWords`プロパティから参照できます。
#
# 通常、辞書内の単語にはその重要性の指標として「重み」を付すことができますが、
# 重みなしの辞書を入力として与える場合は、簡単な計算を元に重みを自動計算させることができます。  
# 自動計算では、各単語に使用されるそれぞれの文字が、辞書内の全単語中で何回ずつ使用されたかをカウントし、  
# 単語内の各文字に対応するカウントの合計を重みとして設定します。  
# 例えば、辞書内の単語が
# 「アメリカ」「ロシア」「シリア」
# だった場合、各文字の出現回数は
# * ア：３回
# * シ：２回
# * リ：２回
# * メ：１回
# * カ：１回
# * ロ：１回  
#
# より、各単語は「アメリカ：７点」,「ロシア：６点」,「シリア：７点」となります。
# `Dictionary`クラスに`calcWeight`メソッドを実装します：

def calcWeight(self, msg=True):
    """
    Calculate word weights in the dictionary.
    """
    mergedWords = "".join(self.word)
    counts = collections.Counter(mergedWords)

    for i, w in enumerate(self.word):
        for char in w:
            self.weight[i] += counts[char]
            
    if msg is True:
        print("All weights are calculated.")
        print("TOP 5 characters:")
        print(counts.most_common()[:5])
        idx = sorted(range(self.size), key=lambda k: self.weight[k], reverse=True)[:5]
        print("TOP 5 words:")
        print(np.array(self.word)[idx])
setattr(Dictionary, "calcWeight", calcWeight)

if not withWeight:
    sample_dic.calcWeight()

# ### Placeable クラス
# 辞書内のすべての単語に対して、それぞれの単語が配置可能(placeable)な位置の一覧を作るクラス。  
# これは`Puzzle`クラス内で内部的に`Dictionary`クラスとパズルの盤面情報を用いて行われます。
#
# 配置可能な位置は、単語の先頭文字の座標で指定します。  
# ここでは、パズルの左上を(0,0)、右上を(n,0)、左下を(0,n)、右下を(n,n)とします。  
# 例えば、大きさが5×5のパズル上に`Dictionary`クラスの5番目に格納された長さ4文字の単語「HOGE」を配置しようとした場合、配置可能な位置は
#   * 横向きの場合：(0,0),(0,1),(0,2),(0,3)(0,4),(1,0),(1,1),(1,2),(1,3),(1,4)の10マス。
#   * 縦向きの場合：(0,0),(1,0),(2,0),(3,0)(4,0),(0,1),(1,1),(2,1),(3,1),(4,1)の10マス。  
# よって、配置する場合のパターンは全部で20通りになります。
# 詳しくは次の図をご参照ください。青が単語配置可能な位置、赤が配置不可能な位置を示します。

display(Image.open("fig/puzzles.png"))

# これらの情報は次のフォーマットで整理されます：
#   * k : 単語番号(辞書内の何番目の単語か)
#   * div : 単語を置く向き(0: 縦, 1: 横)
#   * j : 単語の先頭文字のx座標
#   * i : 単語の先頭文字のy座標

display(Image.open("fig/sample_placeable.png"))


# メンバ変数は以下の通り：
#   * size : Placeableオブジェクトの大きさ
#   * width : 引数のパズルの横幅
#   * height : 引数のパズルの縦幅
#   * div : Placeable成分の文字列の方向
#   * k : Placeable成分の単語番号
#   * i : Placeable成分のy方向の座標
#   * j : Placeable成分のx方向の座標
#   * invP : Placeableオブジェクトの逆写像

class Placeable:
    def __init__(self, width, height, dic, msg=True):
        self.size = 0
        self.width = width
        self.height = height
        self.div, self.i, self.j, self.k = [], [], [], []
        self.invP = np.full((2, self.height, self.width, dic.size), np.nan, dtype="int")
        
        self._compute(dic.word)

        if msg is True:
            print(f"Imported Dictionary name: `{dic.name}`, size: {dic.size}")
            print(f"Placeable size : {self.size}")

    def _compute(self, word, baseK=0):
        if baseK is not 0:
            ap = np.full((2, self.height, self.width, 1), np.nan, dtype="int")
            self.invP = np.append(self.invP, ap, axis=3)
        for div in (0,1):
            for k,w in enumerate(word):
                if div == 0:
                    iMax = self.height - len(w) + 1
                    jMax = self.width
                elif div == 1:
                    iMax = self.height
                    jMax = self.width - len(w) + 1
                for i in range(iMax):
                    for j in range(jMax):
                        self.invP[div,i,j,baseK+k] = len(self.div)
                        self.div.append(div)
                        self.i.append(i)
                        self.j.append(j)
                        self.k.append(baseK+k)
        self.size = len(self.k)

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        if type(key) in (int, np.int):
            return {"div": self.div[key], "i": self.i[key], "j": self.j[key], "k": self.k[key]}
        if type(key) is str:
            return eval(f"self.{key}")


# このクラスインスタンスをユーザが陽に作成することはほぼないでしょう。  
# ただし、内部では常に使われる重要なクラスです。  
# 実際には、`Dictionary`オブジェクトを`Puzzle`にインポートする際に内部で計算が行われます。

def importDict(self, dictionary, msg=True):
    self.dic = dictionary
    self.plc = Placeable(self.width, self.height, self.dic, msg=msg)
setattr(Puzzle, "importDict", importDict)

# それでは`Puzzle`に`Dictionary`をインポートしましょう。  
# このときに内部で`Placeable`の計算が行われます。

sample_puzzle.importDict(sample_dic)


# ### ObjectiveFunction クラス
# 生成したパズルは何らかの指標で定量的にその良し悪しを評価する必要があります。  
# そのパズルの良し悪しの指標として、「目的関数」を定義します。  
# 目的関数はパズルの初期解が得られてから、そのパズルを改善していくために使われます。  
# 目的関数には様々な指標が考えられるため、それらを管理する`ObjectiveFunction`クラスを定義します：

class ObjectiveFunction:
    def __init__(self, msg=True):
        self.flist = [
            "totalWeight",
            "solSize",
            "crossCount",
            "fillCount",
            "maxConnectedEmpties"
        ]
        self.registeredFuncs = []
        if msg is True:
            print("ObjectiveFunction object has made.")
    def __len__(self):
        return len(self.registeredFuncs)
    def getFuncs(self):
        return self.registeredFuncs


objFunc = ObjectiveFunction()


# `flist`は後ほど実装する目的関数名をリスト化したものです。`ObjectiveFunction`クラスについての詳しい説明と実装は後ほど行われます。

# ### Optimizer クラス
# 目的関数を指標にパズルを改善していく際、どのような手法で最適化していくのかも重要なカギになります。  
# この種の問題は「離散最適化問題(組み合わせ最適化)」と呼ばれ、巡回セールスマン問題などと近い分類に当たります。この手の問題で使われる最適化手法は「局所探索法」や「焼きなまし法」などが用いられます。  
# 最適化手法はアイデア次第で様々なものが考えられるため、これら管理する`Optimizer`クラスを定義しておきましょう：

class Optimizer:
    def __init__(self, msg=True):
        self.methodList = ["localSearch", "iteratedLocalSearch"]
        self.method = ""
        if msg is True:
            print("Optimizer object has made.")


optimizer = Optimizer()


# `Optimizer`クラスについての詳しい説明と実装は後ほど行われます。  
# また、`ObjectiveFunction`オブジェクトと`Optimizer`オブジェクトは同時に`Puzzle`オブジェクトにコンパイルすることで、`Puzzle`オブジェクトから呼び出しが可能になります。  
# 目的関数と最適化手法をパズルにコンパイルするための`compile`メソッドも後ほど実装します。

# ***
# ## パズル生成
# それでは、実際にパズルを生成していきます。  
# まずは、`placeable`オブジェクトから何らかの単語とその配置可能位置をランダムに取得し、パズルの盤面単語を一つ配置します。  
# さらにランダムサンプリングを行い、取得した単語とその配置可能位置が盤面上の単語とスケルトンパズルのルールに従って接続できる場合は、その単語を盤面に配置します。  
# この作業を反復し、どの単語をどの置き方にしても盤面上に乗らなくなった時、反復を終了します。
#
# そのために、単語を盤面上に乗せる`add`メソッドを`Puzzle`クラスに追加します。  
# ただしその前に、`add`メソッドで指定した単語と位置が、盤面上にスケルトンパズルのルールに従って配置可能かどうかをBoolianで返す`isEnabledAdd`メソッドを定義します：

# +
### isEnabledAdd
def isEnabledAdd(self, div, i, j, word, wLen):
    """
    This method determines if a word can be placed
    """
    if div == 0:
        empties = self.cell[i:i+wLen, j] == ""
    if div == 1:
        empties = self.cell[i, j:j+wLen] == ""
        
    # If 0 words used, return True
    if self.solSize is 0:
        return 0

    # If the preceding and succeeding cells are already filled
    if div == 0:
        if i > 0 and self.cell[i-1, j] != "":
            return 1
        if i+wLen < self.height and self.cell[i+wLen, j] != "":
            return 1
    if div == 1:
        if j > 0 and self.cell[i, j-1] != "":
            return 1
        if j+wLen < self.width and self.cell[i, j+wLen] != "":
            return 1
        
    # At least one place must cross other words
    if np.all(empties == True):
        return 2
        
    # Judge whether correct intersection
    where = np.where(empties == False)[0]
    if div == 0:
        jall = np.full(where.size, j, dtype="int")
        if np.any(self.cell[where+i, jall] != np.array(list(word))[where]):
            return 3
    if div == 1:
        iall = np.full(where.size, i, dtype="int")
        if np.any(self.cell[iall, where+j] != np.array(list(word))[where]):
            return 3
        
    # If the same word is in use, return False
    if word in self.usedWords:
        return 4

    # If neighbor cells are filled except at the intersection, return False
    where = np.where(empties == True)[0]
    if div == 0:
        jall = np.full(where.size, j, dtype="int")
        # Left side
        if j > 0 and np.any(self.cell[where+i, jall-1] != ""):
            return 5
        # Right side
        if j < self.width-1 and np.any(self.cell[where+i, jall+1] != ""):
            return 5
    if div == 1:
        iall = np.full(where.size, i, dtype="int")
        # Upper
        if i > 0 and np.any(self.cell[iall-1, where+j] != ""):
            return 5
        # Lower
        if i < self.height-1 and np.any(self.cell[iall+1, where+j] != ""):
            return 5
    
    # US/USA, DOMINICA/DOMINICAN problem
    if div == 0:
        if np.any(self.enable[i:i+wLen, j] == False) or np.all(empties == False):
            return 6
    if div == 1:
        if np.any(self.enable[i, j:j+wLen] == False) or np.all(empties == False):
            return 6

    # If Break through the all barrier, return True
    return 0

# Set attribute to Puzzle class
setattr(Puzzle, "isEnabledAdd", isEnabledAdd)


# -

# 準備ができたので、所望の単語をパズルに配置する`add`メソッドを定義します。
# `add`メソッドは次の機能を持ちます：
#   * `add`メソッドの引数は [単語を置く向き, 頭文字のy座標, 頭文字のx座標, 単語番号 ] で指定します。
#   * 指定した位置に単語が置ける場合は置き、置けない場合は何もしません。
#
# 実際に、`add`メソッドを定義しましょう：

### _add
def _add(self, div, i, j, k):
    """
    This method places a word at arbitrary positions. If it can not be arranged, nothing is done.
    """
    word = self.dic.word[k]
    weight = self.dic.weight[k]
    wLen = self.dic.wLen[k]

    # Judge whether adding is enabled
    code = self.isEnabledAdd(div, i, j, word, wLen)
    if code is not 0:
        return code
    
    # Put the word to puzzle
    if div == 0:
        self.cell[i:i+wLen, j] = list(word)[0:wLen]
    if div == 1:
        self.cell[i, j:j+wLen] = list(word)[0:wLen]

    # Set the prohibited cell before and after placed word
    if div == 0:
        if i > 0:
            self.enable[i-1, j] = False
        if i+wLen < self.height:
            self.enable[i+wLen, j] = False
    if div == 1:
        if j > 0:
            self.enable[i, j-1] = False
        if j+wLen < self.width:
            self.enable[i, j+wLen] = False
    
    # Update cover array
    if div == 0:
        self.cover[i:i+wLen, j] += 1
    if div == 1:
        self.cover[i, j:j+wLen] += 1
    
    # Update properties
    wordIdx = self.dic.word.index(word)
    self.usedPlcIdx[self.solSize] = self.plc.invP[div, i, j, wordIdx]
    self.usedWords[self.solSize] = self.dic.word[k]
    self.solSize += 1
    self.totalWeight += weight
    self.history.append((1, wordIdx, div, i, j))
    return 0
# Set attribute to Puzzle class  
setattr(Puzzle, "_add", _add)


# さあ、`add`メソッドが定義できました。  
# 早速パズルを生成して、初期解を作りましょう。  
# まずはランダムに選んだ単語を可能なだけパズルに置いていく`addToLimit`メソッドを用意し、`Puzzle`オブジェクトの初期解を得るための`firstSolve`メソッドの中でそれを呼び、初期解を得るメソッドとして実装します。
#   * `firstSolve`メソッドの引数は [ `Dictionary`オブジェクト, `Placeable`オブジェクト ] です。
#   * `firstSolve`メソッドにより、初期解が`Puzzle`オブジェクトの`cell`プロパティに保存されます。

# +
def addToLimit(self):
    """
    This method adds the words as much as possible 
    """
    # Make a random index of plc
    randomIndex = np.arange(self.plc.size)
    np.random.shuffle(randomIndex)
    
    # Add as much as possible
    solSizeTmp = None
    while self.solSize != solSizeTmp:
        solSizeTmp = self.solSize
        dropIdx = []
        for i, r in enumerate(randomIndex):
            code = self._add(self.plc.div[r], self.plc.i[r], self.plc.j[r], self.plc.k[r])
            if code is not 2:
                dropIdx.append(i)
        randomIndex = np.delete(randomIndex, dropIdx)
    return
setattr(Puzzle, "addToLimit", addToLimit)

def firstSolve(self):
    """
    This method creates an initial solution
    """
    # Check the initSol
    if self.initSol:
        raise RuntimeError("'firstSolve' method has already called")
        
    # Save initial seed number
    self.initSeed = np.random.get_state()[1][0]
    # Add as much as possible
    self.addToLimit()
    self.initSol = True
setattr(Puzzle, "firstSolve", firstSolve)
# -

sample_puzzle.firstSolve()


# どんなパズルができたのか、結果が気になりますよね。  
# 結果を確認するための`show`メソッドを定義します：

def show(self, ndarray=None):
    """
    This method displays a puzzle
    """
    if ndarray is None:
        ndarray = self.cell
    if utils.in_ipynb() is True:
        styles = [
            dict(selector="th", props=[("font-size", "90%"),
                                       ("text-align", "center"),
                                       ("color", "#ffffff"),
                                       ("background", "#777777"),
                                       ("border", "solid 1px white"),
                                       ("width", "30px"),
                                       ("height", "30px")]),
            dict(selector="td", props=[("font-size", "105%"),
                                       ("text-align", "center"),
                                       ("color", "#161616"),
                                       ("background", "#dddddd"),
                                       ("border", "solid 1px white"),
                                       ("width", "30px"),
                                       ("height", "30px")]),
            dict(selector="caption", props=[("caption-side", "bottom")])
        ]
        df = pd.DataFrame(ndarray)
        df = (df.style.set_table_styles(styles).set_caption(f"Puzzle({self.width},{self.height}), solSize:{self.solSize}, Dictionary:[{self.dic.fpath}]"))
        display(df) 
    else:
        ndarray = np.where(ndarray=="", "  ", ndarray)
        print(ndarray)
setattr(Puzzle, "show", show)

# `show`メソッドの引数として結果を与えることで、結果の確認ができます。  
# ここでは`puzzle.cell`を見てみます(`puzzle.cover`や`puzzle.enable`を指定することも可能)：

sample_puzzle.show(sample_puzzle.cell)


# 一つの島で繋がったパズルが出来上がっていますね。  
# まだまだ未熟なパズルですが、ちゃんとスケルトンパズルのルールに沿って辞書内の単語がパズル上に乗っているはずです。    

# ***
#
# ## 目的関数
# さて、初期解はまだ未熟であり、改善の余地がある場合が多いです。  
# そこで、解が未熟かどうかを定量的に良し悪しの指標として判断する関数を作ります。  
# 定量的な解の良し悪し指標は「目的関数」と呼ばれます。  
# 目的関数には様々なものが考えられます：  
#
# * 解に使われた単語数(solSize)
# * 単語のクロス数
# * 文字で埋まっているセルの個数
# * 文字なしマスの連結数の最大値(を最小化)  
#
# これ以外にも様々な目的関数が考えられるでしょう。そして、これらの目的関数は優先順位をつけて共存させることも可能です。後ほど実装する最適化手法は、これらの値を「スコア」として受け、それを最大化するように働きます。なので、4つ目の「文字なしマスの連結数の最大値」は、パズル全マスからその値を引いたものをスコアとして返すように設計します。
#
# 早速、目的関数を一つ作って、`ObjectiveFunction`クラスの属性として設定してみましょう。  
# まずは、最も単純な「解に使われた単語数」を返す目的関数を実装します：

def solSize(self, puzzle):
    """
    This method returns the number of words used in the solution
    """
    return puzzle.solSize
setattr(ObjectiveFunction, "solSize", solSize)


# 次に、単語のクロス数を判定して返す目的関数を実装します：

def crossCount(self, puzzle):
    """
    This method returns the number of crosses of a word
    """
    return np.sum(puzzle.cover == 2)
setattr(ObjectiveFunction, "crossCount", crossCount)


# 次に、文字で埋まっているセルの個数を返す目的関数を実装します：

def fillCount(self, puzzle):
    """
    This method returns the number of character cells in the puzzle
    """
    return np.sum(puzzle.cover >= 1)
setattr(ObjectiveFunction, "fillCount", fillCount)


# 次に、単語の重みの合計を返す目的関数を実装します：

def totalWeight(self, puzzle):
    """
    This method returns the sum of the word weights used for the solution
    """
    return puzzle.totalWeight
setattr(ObjectiveFunction, "totalWeight", totalWeight)

# それでは、それぞれのスコアを見てみましょう：

print(f"solSize: {objFunc.solSize(sample_puzzle)}")
print(f"crossCount: {objFunc.crossCount(sample_puzzle)}")
print(f"fillCount: {objFunc.fillCount(sample_puzzle)}")
print(f"totalWeight: {objFunc.totalWeight(sample_puzzle)}")


# 先ほどの初期解と見比べて目的関数の値が正しいかどうかを確認してください。
#
# 目的関数には「文字なしマスの連結数の最大値」なども考えられます。しかし、この「連結数」を数えるのは少し工夫が必要です。連結数のカウントには「深さ優先探索(Depth First Search:DFS)」を用います。
# * `DFS`メソッドはセルの値が1でそれ以外が0の2次元配列(coverDFS)を引数にとり、引数で与えられたセルと連結したセルには全て同じ番号(ccl)を振ります。
# * `DFS`メソッドは引数に[ coverDFS, 今見る行, 今見る列, 島番号]をとります。
#
# 後ほど、`DFS`メソッドは目的関数以外の場所でも使うため、`Puzzle`クラスのメソッドとして定義しておきます：

def DFS(self, i, j, ccl):
    """
    This method performs a Depth-First Search and labels each connected component
    """
    self.coverDFS[i,j] = ccl
    if i>0 and self.coverDFS[i-1, j] == 1:
        self.DFS(i-1, j, ccl)
    if i<self.height-1 and self.coverDFS[i+1, j] == 1:
        self.DFS(i+1, j, ccl)
    if j>0 and self.coverDFS[i, j-1] == 1:
        self.DFS(i, j-1, ccl)
    if j<self.width-1 and self.coverDFS[i, j+1] == 1:
        self.DFS(i, j+1, ccl)
setattr(Puzzle, "DFS", DFS)


# `DFS`メソッドは引数で与えられたセルに連結した島しか判定しません。
# 文字なしマスの最大連結数を見るためには、全ての島に対して`DFS`を使って番号を振る必要があります。
#
# さて、この`DFS`メソッドを使って文字なしマスの最大連結数を取り出す`maxConnectedEmpties`メソッドを実装しましょう(上でも述べたとおり、全マス数から最大連結数を引いたものをスコアとして返します)：

def maxConnectedEmpties(self, puzzle):
    """
    This method returns the maximum number of concatenations for unfilled squares
    """
    ccl = 2
    puzzle.coverDFS = np.where(puzzle.cover == 0, 1, 0)
    for i, j in itertools.product(range(puzzle.height), range(puzzle.width)):
        if puzzle.coverDFS[i, j] == 1:
            puzzle.DFS(i, j, ccl)
            ccl += 1
    score = puzzle.width*puzzle.height - np.max(np.bincount(puzzle.coverDFS.flatten())[1:])
    return score
setattr(ObjectiveFunction, "maxConnectedEmpties", maxConnectedEmpties)

# 早速結果を見てみましょう：

print(f"maxConnectedEmpties: {objFunc.maxConnectedEmpties(sample_puzzle)}")


# 次に、これらの目的関数をどの順番で見ていくかの優先順位をつけて、`ObjectiveFunction`オブジェクトに登録します。  
# そのために、`register`メソッドを実装します：

def register(self, funcNames, msg=True):
    """
    This method registers an objective function in an instance
    """
    for funcName in funcNames:
        if funcName not in self.flist:
            raise RuntimeError(f"ObjectiveFunction class does not have '{funcName}' function")
        if msg is True:
            print(f" - '{funcName}' function has registered.")
    self.registeredFuncs = funcNames
    return
setattr(ObjectiveFunction, "register", register)

objFunc.register(["totalWeight","solSize", "crossCount", "fillCount", "maxConnectedEmpties"])


# この場合、"totalWeight"から評価が始まり、最後に”maxConnectedEmpties”が評価されます。  
# 次に、こうして登録した目的関数値をスコアとして返す`getScore`メソッドを実装します：

def getScore(self, puzzle, i=0, func=None, all=False):
    """
    This method returns any objective function value
    """
    if all is True:
        scores=np.zeros(len(self.registeredFuncs), dtype="int")
        for n in range(scores.size):
            scores[n] = eval(f"self.{self.registeredFuncs[n]}(puzzle)")
        return scores
    if func is None:
        func = self.registeredFuncs[i]
    return eval(f"self.{func}(puzzle)")
setattr(ObjectiveFunction, "getScore", getScore)

# このメソッドは引数`i`を指定すれば`i`番目のスコアが得られ、`all=True`を与えればスコアを一覧で返します。  
# それでは、それぞれ試してみましょう：

print(f"score[0]: {objFunc.getScore(sample_puzzle, 0)}")
print(f"scores: {objFunc.getScore(sample_puzzle, all=True)}")


# ここで、解の改善過程を目的関数値の推移として記録するための`logging`メソッドを実装します。  

def logging(self):
    """
    This method logs the current objective function values
    """
    if self.objFunc is None:
        raise RuntimeError("Logging method must be executed after compilation method")
    if self.log is None:
        self.log = pd.DataFrame(columns=self.objFunc.getFuncs())
        self.log.index.name = "epoch"
    tmpSe = pd.Series(self.objFunc.getScore(self, all=True), index=self.objFunc.getFuncs())
    self.log = self.log.append(tmpSe, ignore_index=True)
setattr(Puzzle, "logging", logging)


# これで、`ObjectiveFunction`クラスを`Puzzle`クラスにコンパイルする準備はできました。  
# しかし、コンパイルは最適化手法(`Optimizer`クラス)と一緒にコンパイルする設計にするため、この後は`Optimizer`クラスの中身を作っていきましょう。

# ***
# ## 最適化手法
# 設定した目的関数値を最大化/最小化するための手法を実装しましょう。  
# まずは、数ある最適化手法の中から「局所探索法」を実装してみます。  
# これは、このノートのタイトルにもある「LocalSearch」と呼ばれる手法で、組み合わせ最適化問題を解く近似解法として代表的なものです。
#
# ここで、局所探索法に関して簡単に解説します。
# まず、今までの流れをおさらいします。
#   1. ランダムに単語を一つパズルに配置する。
#   2. その単語と必ずクロスするように、可能なだけランダムに単語を配置していく。
#   3. これ以上配置できない状態になったら、それを初期解とする。
#   4. 設定した目的関数で解の良さを判定する。
#   
# よろしいでしょうか。それではここから、初期解の「近傍領域」に存在する「近傍解」を探していきます。  
# 近傍解とは、初期解に似たような解を指します。よって、近傍解は初期解を元に得る必要があります。  
# しかし、初期解にはこれ以上単語を配置することはできません。  
# そのため、初期解に配置された単語をランダムに抜いていきます。  
# そして、単語の連結性が崩れ、1つだった単語の島が2つ以上の島に分離したら、その時点で単語を抜く処理をストップします。
#
# ここで一旦、盤面に置かれた単語を抜く処理を`drop`メソッドとして実装します：

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
setattr(Puzzle, "_drop", _drop)


# これを用いて単語を抜いた後、残った島の中で一番面積(+クロス数)の大きい島以外を全て消します(この判定にもDFSが使われています)。  
# これによって、初期解の情報をある程度保ちつつ、単語を減らしたことになります。  
# そこまでできたら、初期解を得た時と同じように、今の盤面に配置可能な単語をランダムに配置していき、
# これ以上配置できなくなった時点を「近傍解」とします。  
# それでは、近傍解を得るための`getNeighborSolution`メソッドを実装します。  
# この関数は近傍解のPuzzleオブジェクトを返します。その際、引数に与えたPuzzleオブジェクトには何もしません。  
# 実装においては、連結性が崩れるまで順番に単語を抜いていく`collapse`メソッドと、連結性が崩れた状態での使用を前提として一番大きな島以外をすべて消す`kick`メソッドを準備し、それらを`getNeighborSolution`メソッド内で呼ぶ形にします。

# +
def collapse(self):
    """
    This method collapses connectivity of the puzzle
    """
    # If solSize = 0, return
    if self.solSize == 0:
        return
    
    # Make a random index of solSize  
    randomIndex = np.arange(self.solSize)
    np.random.shuffle(randomIndex)
    
    # Drop words until connectivity collapses
    tmpUsedPlcIdx = copy.deepcopy(self.usedPlcIdx)
    for r, p in enumerate(tmpUsedPlcIdx[randomIndex]):
        # Get div, i, j, k, wLen
        div = self.plc.div[p]
        i = self.plc.i[p]
        j = self.plc.j[p]
        k = self.plc.k[p]
        wLen = self.dic.wLen[self.plc.k[p]]
        # If '2' is aligned in the cover array, the word can not be dropped
        if div == 0:
            if not np.any(np.diff(np.where(self.cover[i:i+wLen,j] == 2)[0]) == 1):
                self._drop(div, i, j, k)
        if div == 1:
            if not np.any(np.diff(np.where(self.cover[i,j:j+wLen] == 2)[0]) == 1):
                self._drop(div, i, j, k)
        
        # End with connectivity breakdown
        self.coverDFS = np.where(self.cover >= 1, 1, 0)
        self.ccl = 2
        for i, j in itertools.product(range(self.height), range(self.width)):
            if self.coverDFS[i,j] == 1:
                self.DFS(i, j, self.ccl)
                self.ccl += 1
        if self.ccl-2 >= 2:
            break
setattr(Puzzle, "collapse", collapse)

def kick(self):
    """
    This method kicks elements except largest CCL
    """
    # If solSize = 0, return
    if self.solSize == 0:
        return

    # Define 'largestCCL' witch has the largest score(fillCount+crossCount)
    cclScores = np.zeros(self.ccl-2, dtype="int")
    for c in range(self.ccl-2):
        cclScores[c] = np.sum(np.where(self.coverDFS == c+2, self.cover, 0))
    largestCCL = np.argmax(cclScores) + 2
    
    # Erase elements except CCL ('kick' in C-program)
    for idx, p in enumerate(self.usedPlcIdx[:self.solSize]):
        if p == -1:
            continue
        if self.coverDFS[self.plc.i[p], self.plc.j[p]] != largestCCL:
            self._drop(self.plc.div[p], self.plc.i[p], self.plc.j[p], self.plc.k[p], isKick=True)
setattr(Puzzle, "kick", kick)


# -

def getNeighborSolution(self, puzzle):   
    """
    This method gets the neighborhood solution
    """
    # Copy the puzzle
    _puzzle = copy.deepcopy(puzzle)
    # Drop words until connectivity collapse
    _puzzle.collapse()
    # Kick
    _puzzle.kick()
    # Add as much as possible 
    _puzzle.addToLimit()
    return _puzzle
setattr(Optimizer, "getNeighborSolution", getNeighborSolution)


# さあ、ようやく局所探索法を行う準備が整いました。  
# 局所探索法では近傍解を探し、そのスコアが暫定解よりも高ければ暫定解を更新し、低ければ近傍解を棄却します。  
# それを指定回数(epoch数)だけ繰り返し、指定回数分だけ終わった時点での暫定解を「局所最適解」として得ます。    
# 厳密にはそれが最適解かはわかりませんが、この解は近似的な最適解と言えるでしょう。  
# それでは、局所探索法を行う`localSearch`メソッドを実装します：

def localSearch(self, puzzle, epoch, show=True, move=False):
    """
    This method performs a local search
    """
    # Logging
    if puzzle.epoch is 0:
        puzzle.logging()
    # Copy
    _puzzle = copy.deepcopy(puzzle)
    if show is True:
        print(">>> Interim solution")
        _puzzle.show(_puzzle.cell)
    goalEpoch = _puzzle.epoch + epoch
    for ep in range(epoch):
        _puzzle.epoch += 1
        print(f">>> Epoch {_puzzle.epoch}/{goalEpoch}")
        # Get neighbor solution by drop->kick->add
        newPuzzle = self.getNeighborSolution(_puzzle)
        
        # Repeat if the score is high
        for funcNum in range(len(_puzzle.objFunc)):
            prevScore = _puzzle.objFunc.getScore(_puzzle, funcNum)
            newScore = newPuzzle.objFunc.getScore(newPuzzle, funcNum)
            if newScore > prevScore:
                print(f"    - Improved: {_puzzle.objFunc.getScore(_puzzle, all=True)} --> {newPuzzle.objFunc.getScore(newPuzzle, all=True)}")
                _puzzle = copy.deepcopy(newPuzzle)
                _puzzle.logging()
                if show is True:
                    _puzzle.show(_puzzle.cell)
                break
            if newScore < prevScore:
                _puzzle.logging()
                print(f"    - Stayed: {_puzzle.objFunc.getScore(_puzzle, all=True)}")
                break
        else:
            _puzzle = copy.deepcopy(newPuzzle)
            _puzzle.logging()
            print(f"    - Replaced(same score): {_puzzle.objFunc.getScore(_puzzle, all=True)} -> {newPuzzle.objFunc.getScore(newPuzzle, all=True)}")
            if show is True:
                _puzzle.show(_puzzle.cell)
    # Update previous puzzle
    puzzle.totalWeight = copy.deepcopy(_puzzle.totalWeight)
    puzzle.enable = copy.deepcopy(_puzzle.enable)
    puzzle.cell = copy.deepcopy(_puzzle.cell)
    puzzle.cover = copy.deepcopy(_puzzle.cover)
    puzzle.coverDFS = copy.deepcopy(_puzzle.coverDFS)
    puzzle.usedWords = copy.deepcopy(_puzzle.usedWords)
    puzzle.usedPlcIdx = copy.deepcopy(_puzzle.usedPlcIdx)
    puzzle.solSize = copy.deepcopy(_puzzle.solSize)
    puzzle.history = copy.deepcopy(_puzzle.history)
    puzzle.baseHistory = copy.deepcopy(_puzzle.baseHistory)
    puzzle.log = copy.deepcopy(_puzzle.log)
    puzzle.epoch = copy.deepcopy(_puzzle.epoch)
    puzzle.initSol = copy.deepcopy(_puzzle.initSol)
    puzzle.initSeed = copy.deepcopy(_puzzle.initSeed)
    puzzle.dic = copy.deepcopy(_puzzle.dic)
    puzzle.plc = copy.deepcopy(_puzzle.plc)
setattr(Optimizer, "localSearch", localSearch)


# 局所探索法をメソッド化できたので、これを`Optimizer`クラスにセットする`setMethod`メソッドを実装します。

def setMethod(self, methodName, msg=True):
    """
    This method sets the optimization method on the instance
    """
    if methodName not in self.methodList:
        raise ValueError(f"Optimizer doesn't have '{methodName}' method")
    if msg is True:
        print(f" - '{methodName}' method has registered.")
    self.method = methodName
setattr(Optimizer, "setMethod", setMethod)

optimizer.setMethod("localSearch")


# これで、`Optimizer`オブジェクトに`localSearch`メソッドがセットされました。
#
# これまで目的関数クラス(`ObjectiveFunction`)、最適化関数クラス(`Optimizer`)クラスをそれぞれ実装してきました。  
# これらはより良いものにするための機能ですので、それらの情報をパズル本体に教えておく必要があります。
# 本プログラムではこの工程を「コンパイル」と呼びます。  
# 上記４つのクラスを`Puzzle`オブジェクトにコンパイルするための`compile`メソッドを`Puzzle`クラスに実装しましょう：

def compile(self, objFunc, optimizer, msg=True):
    """
    This method compiles the objective function and optimization method into the Puzzle instance
    """
    self.objFunc = objFunc
    self.optimizer = optimizer
    
    if msg is True:
        print("compile succeeded.")
        print(" --- objective functions:")
        for funcNum in range(len(objFunc)):
            print(f"  |-> {funcNum} {objFunc.registeredFuncs[funcNum]}")
        print(f" --- optimizer: {optimizer.method}")
setattr(Puzzle, "compile", compile)

sample_puzzle.compile(objFunc=objFunc, optimizer=optimizer)


# 局所探索法による解の改善を実行する準備が完全に整いました。  
# それでは、これを行う`solve`メソッドを実装し、エポック数を指定して解が改善されていく様子を見てみましょう！

def solve(self, epoch):
    """
    This method repeats the solution improvement by the specified number of epochs
    """
    if self.initSol is False:
        raise RuntimeError("'firstSolve' method has not called")
    if epoch is 0:
        raise ValueError("'epoch' must be lather than 0")
    exec(f"self.optimizer.{self.optimizer.method}(self, {epoch})")
    print(" --- done")
setattr(Puzzle, "solve", solve)

sample_puzzle.solve(epoch=10)

# 最後に表示された解が局所最適解です。  
# 初期解に比べ、解が目的関数に沿って改善されていれば成功です。  
# もしまだ未熟な解だと思えば、`epoch`数を増やしてさらに実行してみてください。  
# 以上が局所探索法です。引数の`move`をオプションとして`True`で指定すると、「反復局所探索法」(未実装)になります。
#
# ここで、解の改善過程における目的関数値の推移を可視化してみましょう。  
# ただし、`epoch=0`は初期解の目的関数値を表します。

sample_puzzle.log


def showLog(self, title="Objective Function's time series", grid=True, figsize=None):
    """
    This method shows log of objective functions
    """
    if self.log is None:
        raise RuntimeError("Puzzle has no log")
    return self.log.plot(subplots=True, title=title, grid=grid, figsize=figsize)
setattr(Puzzle, "showLog", showLog)

sample_puzzle.showLog(figsize=(7,6))


# ***
#
# ## 解の唯一性
# せっかくのパズルも複数の解が存在すると正しく解くことが出来なくなってしまいます。  
# 例えば、「アメリカ」という単語が「リ」のみでクロスしていて、別の場所に「ソマリア」という単語が同じく「リ」のみでクロスしている場合、これらの単語は入れ替え可能となり、解が唯一に定まりません。  
# そこで、パズルの解が唯一であるかを判定する`isSimpleSol`メソッドを実装します。

def isSimpleSol(self):
    """
    This method determines whether it is the simple solution
    """
    rtnBool = True

    # Get word1
    for s, p in enumerate(self.usedPlcIdx[:self.solSize]):
        i = self.plc.i[p]
        j = self.plc.j[p]
        word1 = self.usedWords[s]
        if self.plc.div[p] == 0:
            crossIdx1 = np.where(self.cover[i:i+len(word1),j] == 2)[0]
        elif self.plc.div[p] == 1:
            crossIdx1 = np.where(self.cover[i,j:j+len(word1)] == 2)[0]
        # Get word2
        for t, q in enumerate(self.usedPlcIdx[s+1:self.solSize]):
            i = self.plc.i[q]
            j = self.plc.j[q]
            word2 = self.usedWords[s+t+1]
            if len(word1) != len(word2): # If word1 and word2 have different lengths, they can not be replaced
                continue
            if self.plc.div[q] == 0:
                crossIdx2 = np.where(self.cover[i:i+len(word2),j] == 2)[0]
            if self.plc.div[q] == 1:
                crossIdx2 = np.where(self.cover[i,j:j+len(word2)] == 2)[0]
            replaceable = True
            # Check cross part from word1
            for w1idx in crossIdx1:
                if word1[w1idx] != word2[w1idx]:
                    replaceable = False
                    break
            # Check cross part from word2
            if replaceable is True:
                for w2idx in crossIdx2:
                    if word2[w2idx] != word1[w2idx]:
                        replaceable = False
                        break
            # If word1 and word2 are replaceable, this puzzle doesn't have a simple solution -> return False
            if replaceable is True:
                print(f" - words '{word1}' and '{word2}' are replaceable")
                rtnBool = False
    return rtnBool
setattr(Puzzle, "isSimpleSol", isSimpleSol)

# それでは、パズルが複数の解を持たないことをチェックしてみましょう。  
# 唯一解であれば`True`、複数解を持てば`False`が返ります。  
# 複数解を持つ場合は、どの単語が入れ替え可能であるかも全パターン表示します。

sample_puzzle.isSimpleSol()


# ***
#
# ## パズルの画像化
# 最後に、生成されたパズルを画像として出力してみましょう。パズルを他人と共有する際に便利なツールです。

# +
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
setattr(Puzzle, "saveImage", saveImage)

def saveProblemImage(self, fpath="problem.png", list_label="[Word List]", dpi=100):
    """
    This method generates and returns a puzzle problem with a word list
    """
    data = np.full(self.width*self.height, "", dtype="unicode").reshape(self.height,self.width)
    self.saveImage(data, fpath, list_label, dpi)
setattr(Puzzle, "saveProblemImage", saveProblemImage)
    
def saveAnswerImage(self, fpath="answer.png", list_label="[Word List]", dpi=100):
    """
    This method generates and returns a puzzle answer with a word list.
    """
    data = self.cell
    self.saveImage(data, fpath, list_label, dpi)
setattr(Puzzle, "saveAnswerImage", saveAnswerImage)
# -

# ### 問題として画像化

madeTime = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
sample_puzzle.saveProblemImage(f"fig/puzzle/{madeTime}_{str(sample_dic)}_{width}_{height}_{seed}_{sample_puzzle.epoch}_problem.png", list_label="【単語リスト】")

# ### 解答として画像化

madeTime = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
sample_puzzle.saveAnswerImage(f"fig/puzzle/{madeTime}_{str(sample_dic)}_{width}_{height}_{seed}_{sample_puzzle.epoch}_answer.png", list_label="【単語リスト】")

# ***
#
# ## パズルの巻き戻し・早送り
# Puzzleオブジェクトにはパズルの単語増減の履歴が保持されています。まずは履歴を確認してみましょう。

sample_puzzle.history


# この履歴上の指定した位置に相当する`Puzzle`オブジェクトを返すメソッドを作成します。各種メソッドの機能は以下の通りです：
#  * jump：history番号を指定して、そこに解を移動させる
#  * getPrev：一つ前の履歴番号に解を移動させる
#  * getNext：一つ後の履歴番号に解を移動させる
#  * getLatest：最新の履歴番号に解を移動させる

# +
def jump(self, idx):
    tmp_puzzle = Puzzle(self.width, self.height, self.title, msg=False)
    tmp_puzzle.dic = copy.deepcopy(self.dic)
    tmp_puzzle.plc = Placeable(tmp_puzzle.width, tmp_puzzle.height, tmp_puzzle.dic, msg=False)
    tmp_puzzle.optimizer = copy.deepcopy(self.optimizer)
    tmp_puzzle.objFunc = copy.deepcopy(self.objFunc)
    tmp_puzzle.baseHistory = copy.deepcopy(self.baseHistory)
    
    if set(self.history).issubset(self.baseHistory) is False:
        if idx <= len(self.history):
            tmp_puzzle.baseHistory = copy.deepcopy(self.history)
        else:
            raise RuntimeError('This puzzle is up to date')

    for code, k, div, i, j in tmp_puzzle.baseHistory[:idx]:
        if code == 1:
            tmp_puzzle._add(div, i, j, k)
        elif code == 2:
            tmp_puzzle._drop(div, i, j, k, isKick=False)
        elif code == 3:
            tmp_puzzle._drop(div, i, j, k, isKick=True)
    tmp_puzzle.initSol = True
    return tmp_puzzle
setattr(Puzzle, "jump", jump)

def getPrev(self, n=1):
    if len(self.history) - n < 0:
        return self.jump(0)
    return self.jump(len(self.history) - n)
setattr(Puzzle, "getPrev", getPrev)

def getNext(self, n=1):
    if len(self.history) + n > len(self.baseHistory):
        return self.getLatest()
    return self.jump(len(self.history) + n)
setattr(Puzzle, "getNext", getNext)

def getLatest(self):
    return self.jump(len(self.baseHistory))
setattr(Puzzle, "getLatest", getLatest)
# -

# これらを利用することで、`Puzzle`オブジェクトの状態を自由に移動させることができます。

tmp_puzzle = sample_puzzle.jump(10)
tmp_puzzle.show()
tmp_puzzle = tmp_puzzle.getLatest()
tmp_puzzle.show()


# ***
#
# ## Puzzleオブジェクトの保存
# 最後に、`Puzzle`オブジェクトを`Pickle`ライブラリを用いてバイナリ形式で保存します。  
# このファイルを読み込むことで、過去に生成したオブジェクトを再度読み込むことができます。

def toPickle(self, name=None, msg=True):
    """
    This method saves Puzzle object as a binary file
    """
    now = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
    name = name or f"{now}_{self.dic.name}_{self.width}_{self.height}_{self.initSeed}_{self.epoch}.pickle"
    with open(name, mode="wb") as f:
        pickle.dump(self, f)
    if msg is True:
        print(f"Puzzle has pickled to the path '{name}'")
setattr(Puzzle, "toPickle", toPickle)

import glob
sample_puzzle.toPickle(name=None) # sample.pickleを作る場合の引数：name="pickle/sample.pickle"
for pickle_file in glob.glob("*.pickle"):
    shutil.move(pickle_file, "pickle/")

# こうして保存したパズルデータは、`Pickle`ライブラリの仕様に従ってロードすることができます。  
# 詳しくは拡張機能について詳しく解説した`CrosswordExtension.ipynb`をご覧ください。
#
# ***
# ### 最後に
# これで、クロスワード自動生成ツールの紹介は終わりになります。  
# ここからは目的関数をさらに定義したり、最適化手法を追加したりして、自由に拡張してください。  
# また、ここで紹介したものの他にも、様々な拡張機能を用意しておりますので、その説明は`CrosswordExtension.ipynb`をご覧ください。

e_time = time.time() - start
print (f"e_time: {format(e_time)} s")


