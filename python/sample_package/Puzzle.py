import datetime
import pandas as pd
import copy
from matplotlib.font_manager import FontProperties
import itertools
import pickle
import numpy as np
from IPython.display import display, HTML

from sample_package.Placeable import Placeable

class Puzzle:
    def __init__(self, width, height, puzzleTitle="スケルトンパズル", msg=True):
        self.width = width
        self.height = height
        self.totalWeight = 0
        self.puzzleTitle = puzzleTitle
        self.cell = np.full(width * height, "", dtype="unicode").reshape(height, width)
        self.cover = np.zeros(width * height, dtype="int64").reshape(height, width)
        self.coverDFS = np.zeros(width * height, dtype="int64").reshape(height, width)
        self.enable = np.ones(width * height, dtype="bool").reshape(height, width)
        self.usedWords = np.full(width * height, "", dtype="U%d" % max(width, height))
        self.usedPlcIdx = np.full(width * height, -1, dtype="int64")
        self.solSize = 0
        self.history = []
        self.historyIdx = 0
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
            print(f" - title       : {self.puzzleTitle}")
            print(f" - width       : {self.width}")
            print(f" - height      : {self.height}")
            print(f" - cell' shape : (width, height) = ({self.cell.shape[0]},{self.cell.shape[1]})")
    def __str__(self):
        return self.puzzleTitle
    def reinit(self, all=False):
        if all is True:
            self.dic = None
            self.plc = None
            self.objFunc = None
            self.optimizer = None
        self.totalWeight = 0
        self.enable = np.ones(self.width*self.height, dtype="bool").reshape(self.height, self.width)
        self.cell = np.full(self.width*self.height, "", dtype="unicode").reshape(self.height, self.width)
        self.cover = np.zeros(self.width*self.height, dtype="int64").reshape(self.height, self.width)
        self.coverDFS = np.zeros(self.width*self.height, dtype="int64").reshape(self.height, self.width)
        self.enable = np.ones(self.width*self.height, dtype="bool").reshape(self.height, self.width)
        self.usedWords = np.full(self.width*self.height, "", dtype="U%d" % max(self.width, self.height))
        self.usedPlcIdx = np.full(self.width*self.height, -1, dtype="int64")
        self.solSize = 0
        self.history = []
        self.historyIdx = 0
        self.log = None
        self.epoch = 0
        self.initSol = False
        self.initSeed = None
    def resetHistory(self):
        self.history = self.history[:self.historyIdx]
    def importDict(self, dictionary, msg=True):
        self.dic = dictionary
        self.plc = Placeable(self, self.dic, msg=msg)
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
    def add(self, div, i, j, k):
        """
        This method places a word at arbitrary positions. If it can not be arranged, nothing is done.
        """
        word = self.dic.word[k]
        weight = self.dic.weight[k]
        wLen = self.dic.wLen[k]
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
                code = self.add(self.plc.div[r], self.plc.i[r], self.plc.j[r], self.plc.k[r])
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
    def show(self, ndarray=None, stdout=False):
        """
        This method displays a puzzle
        """
        if ndarray is None:
            ndarray = self.cell
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
        if stdout is False:
            display(df) 
        else:
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
    def drop(self, div, i, j, k, isKick=False):
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
            jall = np.full(where.size, j, dtype="int64")
            self.cell[i+where,jall] = ""
        if div == 1:
            self.cover[i,j:j+wLen] -= 1
            where = np.where(self.cover[i,j:j+wLen] == 0)[0]
            iall = np.full(where.size, i, dtype="int64")
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
        self.historyIdx += 1
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
                    self.drop(div, i, j, k)
            if div == 1:
                if not np.any(np.diff(np.where(self.cover[i,j:j+wLen] == 2)[0]) == 1):
                    self.drop(div, i, j, k)
            
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
                print("  |-> %d. %s" % (funcNum, objFunc.registeredFuncs[funcNum]))
            print(" --- optimizer: %s" % optimizer.method)
    def solve(self, epoch, stdout=False):
        """
        This method repeats the solution improvement by the specified number of epochs
        """
        self.resetHistory()
        if self.initSol is False:
            raise RuntimeError("'firstSolve' method has not called")
        if epoch is 0:
            raise ValueError("'epoch' must be lather than 0")
        exec(f"self.optimizer.{self.optimizer.method}(self, {epoch}, stdout=stdout)")
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
    def saveImage(self, data, fpath, dpi=100):
        """
        This method generates and returns a puzzle image with a word list
        """
        # Generate puzzle image
        collors = np.where(self.cover<1, "#000000", "#FFFFFF")
        df = pd.DataFrame(data)
        fp = FontProperties(fname="../../fonts/SourceHanCodeJP.ttc", size=14)
    def saveProblemImage(self, fpath="problem.png", dpi=100):
        """
        This method generates and returns a puzzle problem with a word list
        """
        data = np.full(self.width*self.height, "", dtype="unicode").reshape(self.height,self.width)
        self.saveImage(data, fpath, dpi)
    def saveAnswerImage(self, fpath="answer.png", dpi=100):
        """
        This method generates and returns a puzzle answer with a word list.
        """
        data = self.cell
        self.saveImage(data, fpath, dpi)
    def jump(self, idx):
        tmp_puzzle = Puzzle(self.width, self.height, self.puzzleTitle, msg=False)
        tmp_puzzle.dic = copy.deepcopy(self.dic)
        tmp_puzzle.plc = Placeable(tmp_puzzle, tmp_puzzle.dic, msg=False)
        tmp_puzzle.optimizer = copy.deepcopy(self.optimizer)
        tmp_puzzle.objFunc = copy.deepcopy(self.objFunc)
        for code, k, div, i, j in self.history[:idx]:
            if code == 1:
                tmp_puzzle.add(div, i, j, k)
            else:
                tmp_puzzle.drop(div, i, j, k)
        tmp_puzzle.initSol = True
        tmp_puzzle.history = copy.deepcopy(self.history)
        return tmp_puzzle
    def getPrev(self, n=1):
        if self.historyIdx-n < 0:
            return self.jump(0)
        return self.jump(self.historyIdx - n)
    def getNext(self, n=1):
        if self.historyIdx+n > len(self.history):
            return self.getLatest()
        return self.jump(self.historyIdx + n)
    def getLatest(self):
        return self.jump(len(self.history))
    def toPickle(self, fpath=None, msg=True):
        """
        This method saves Puzzle object as a binary file
        """
        now = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
        fpath = fpath or f"../pickle/{now}_{self.dic.name}_{self.width}_{self.height}_{self.initSeed}_{self.epoch}.pickle"
        with open(fpath, mode="wb") as f:
            pickle.dump(self, f)
        if msg is True:
            print(f"Puzzle has pickled to the path '{fpath}'")
