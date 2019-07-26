import itertools
import copy
import numpy as np
from src import utils
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import pickle
import pandas as pd
import math
import datetime

from sample_package.Placeable import Placeable

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
