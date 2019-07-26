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
from sample_package import Puzzle, Dictionary, ObjectiveFunction, Optimizer
from src import utils


# -

# ## Puzzle

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
        self.dic = Dictionary(msg=False)
        self.plc = Placeable(self.width, self.height, self.dic, msg=False)
        self.objFunc = None
        self.optimizer = None
        #self.fp = os.path.get_path()
        ## Message
        if msg is True:
            print("Puzzle object has made.")
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

    def importDict(self, dictionary, msg=True):
        self.dic = dictionary
        self.plc = Placeable(self.width, self.height, self.dic, msg=msg)

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
    def drop(self, word=None, divij=None):
        if word is None and divij is None:
            raise ValueError()
        if word is not None: 
            if type(word) is int:
                k = word
            elif type(word) is str:
                k = self.dic.word.index(word)
            else:
                raise TypeError()
            for p in self.usedPlcIdx:
                if self.plc.k[p] == k:
                    div = self.plc.div[p]
                    i = self.plc.i[p]
                    j = self.plc.j[p]
                    break
        else:
            if type(divij) not in(list, tuple):
                raise TypeError()
            if len(divij) is not 3:
                raise TypeError()
            div,i,j = divij
            print(div, i, j)
            for p in self.usedPlcIdx:
                _div = self.plc.div[p]
                _i = self.plc.i[p]
                _j = self.plc.j[p]
                if _div == divij[0] and _i == divij[1] and _j == divij[2]:
                    k = puzzle.plc.k[p]
                    break
        self._drop(div, i, j, k)

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

    def showLog(self, title="Objective Function's time series", grid=True, figsize=None):
        """
        This method shows log of objective functions
        """
        if self.log is None:
            raise RuntimeError("Puzzle has no log")
        return self.log.plot(subplots=True, title=title, grid=grid, figsize=figsize)

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

    def saveProblemImage(self, fpath="problem.png", list_label="[Word List]", dpi=100):
        """
        This method generates and returns a puzzle problem with a word list
        """
        data = np.full(self.width*self.height, "", dtype="unicode").reshape(self.height,self.width)
        self.saveImage(data, fpath, list_label, dpi)

    def saveAnswerImage(self, fpath="answer.png", list_label="[Word List]", dpi=100):
        """
        This method generates and returns a puzzle answer with a word list.
        """
        data = self.cell
        self.saveImage(data, fpath, list_label, dpi)
    
    def jump(self, idx):
        tmp_puzzle = Puzzle(self.width, self.height, self.title, msg=False)
        tmp_puzzle.dic = copy.deepcopy(self.dic)
        tmp_puzzle.plc = Placeable(tmp_puzzle, tmp_puzzle.dic, msg=False)
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
            elif code in (2,3):
                tmp_puzzle._drop(div, i, j, k)
        tmp_puzzle.initSol = True
        return tmp_puzzle

    def getPrev(self, n=1):
        if len(self.history) - n < 0:
            return self.jump(0)
        return self.jump(len(self.history) - n)

    def getNext(self, n=1):
        if len(self.history) + n > len(self.baseHistory):
            return self.getLatest()
        return self.jump(len(self.history) + n)

    def getLatest(self):
        return self.jump(len(self.baseHistory))

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
    
    def getRect(self):
        rows = np.any(self.cover, axis=1)
        cols = np.any(self.cover, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

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
            n2limit = {1:rmin, 2:self.height-(rmax+1), 3:cmin, 4:self.width-(cmax+1)}
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
            if cmin < n:
                raise RuntimeError()
            self.cell = np.roll(self.cell, -n, axis=1)
            self.cover = np.roll(self.cover, -n, axis=1)
            self.coverDFS = np.roll(self.coverDFS, -n, axis=1)
            self.enable = np.roll(self.enable, -n, axis=1)
            for i,p in enumerate(self.usedPlcIdx[:self.solSize]):
                self.usedPlcIdx[i] = self.plc.invP[self.plc.div[p], self.plc.i[p], self.plc.j[p]-n, self.plc.k[p]]
        if direction is 4:
            if self.width-(cmax+1) < n:
                raise RuntimeError()
            self.cell = np.roll(self.cell, n, axis=1)
            self.cover = np.roll(self.cover, n, axis=1)
            self.coverDFS = np.roll(self.coverDFS, n, axis=1)
            self.enable = np.roll(self.enable, n, axis=1)
            for i,p in enumerate(self.usedPlcIdx[:self.solSize]):
                self.usedPlcIdx[i] = self.plc.invP[self.plc.div[p], self.plc.i[p], self.plc.j[p]+n, self.plc.k[p]]

        self.history.append((4, direction, n))


# ## Dictionary

class Dictionary:
    def __init__(self, fpath=None, msg=True):
        self.fpath = fpath
        self.size = 0
        self.name = ''
        self.word = []
        self.weight = []
        self.wLen = []
        self.removedWords = []
        if fpath is not None:
            self.name = os.path.basename(fpath)[:-4]
            self.read(fpath)

        # Message
        if msg is True:
            print("Dictionary object has made.")
            print(f" - file path         : {self.fpath}")
            print(f" - dictionary size   : {self.size}")
            if self.size > 0:
                print(f" - top of dictionary : {self[0]}")

    def __getitem__(self, key):
        return {'word': self.word[key], 'weight': self.weight[key], 'len': self.wLen[key]}
    
    def __str__(self):
        return self.name
    
    def __len__(self):
        return self.size

    def getK(self, word):
        return np.where(self.word == word)[0][0]
    
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

    def read(self, fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            data = f.readlines()

        # Remove "\n"
        def removeNewLineCode(word):
            line = word.rstrip("\n").split(" ")
            if len(line) == 1:
                line.append(0)
            line[1] = int(line[1])
            return line

        dic_list = list(map(removeNewLineCode, data))
        word = [d[0] for d in dic_list]
        weight = [d[1] for d in dic_list]
        self.add(word, weight)

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


# ## Placeable

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


# ## ObjectFunction

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

    def solSize(self, puzzle):
        """
        This method returns the number of words used in the solution
        """
        return puzzle.solSize

    def crossCount(self, puzzle):
        """
        This method returns the number of crosses of a word
        """
        return np.sum(puzzle.cover == 2)

    def fillCount(self, puzzle):
        """
        This method returns the number of character cells in the puzzle
        """
        return np.sum(puzzle.cover >= 1)

    def totalWeight(self, puzzle):
        """
        This method returns the sum of the word weights used for the solution
        """
        return puzzle.totalWeight

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


# ## Optimizer

class Optimizer:
    def __init__(self, msg=True):
        self.methodList = ["localSearch", "iteratedLocalSearch"]
        self.method = ""
        if msg is True:
            print("Optimizer object has made.")

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

    def setMethod(self, methodName, msg=True):
        """
        This method sets the optimization method on the instance
        """
        if methodName not in self.methodList:
            raise ValueError(f"Optimizer doesn't have '{methodName}' method")
        if msg is True:
            print(f" - '{methodName}' method has registered.")
        self.method = methodName


# ### フォント設定
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
puzzle.saveAnswerImage(f"fig/{dic.name}_w{width}_h{height}_r{seed}.png")

e_time = time.time() - start
print (f"e_time: {format(e_time)} s")


