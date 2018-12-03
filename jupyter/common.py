import numpy as np
from numpy.random import *
import pandas as pd
import unicodedata
import itertools
import sys
import copy

class Puzzle():
    def __init__(self, width, height, msg=True):
        self.width = width
        self.height = height
        self.enable = np.ones(width*height, dtype = "bool").reshape(height,width)
        self.cell = np.full(width*height, "", dtype = "unicode").reshape(height,width)
        self.cover = np.zeros(width*height, dtype = "int64").reshape(height,width)
        self.coverDFS = np.zeros(width*height, dtype = "int64").reshape(height,width)
        self.usedWords = np.full(width*height, "", dtype = "U%d" % max(width,height))
        self.usedPlcIdx = np.full(width*height, -1, dtype = "int64")
        self.solSize = 0
        self.initSol = False
        self.dictType = None

        ## Message
        if msg == True:
            print("Puzzle object has made.")
            print(" - width       : %d" % self.width)
            print(" - height      : %d" % self.height)
            print(" - cell' shape : (width, height) = (%d,%d)" % (self.cell.shape[0], self.cell.shape[1]))

    def copy(self, puzzle):
        self.width = copy.deepcopy(puzzle.width)
        self.height = copy.deepcopy(puzzle.height)
        self.enable = copy.deepcopy(puzzle.enable)
        self.cell = copy.deepcopy(puzzle.cell)
        self.cover = copy.deepcopy(puzzle.cover)
        self.coverDFS = copy.deepcopy(puzzle.coverDFS)
        self.usedWords = copy.deepcopy(puzzle.usedWords)
        self.usedPlcIdx = copy.deepcopy(puzzle.usedPlcIdx)
        self.solSize = copy.deepcopy(puzzle.solSize)
        self.initSol = copy.deepcopy(puzzle.initSol)

    ### isEnabledAdd
    def isEnabledAdd(self, div, i, j, word, wLen):
        # If 0 words used, return True
        if self.solSize == 0:
            return True

        # If the same word is in use, return False
        if np.any(self.usedWords == word):
            return False

        # If the word does not fit in the puzzle, return False
        if div == 0 and i+wLen > self.height:
            return False
        if div == 1 and j+wLen > self.width:
            return False

        # US/USA, DOMINICA/DOMINICAN probrem
        if div == 0:
            emptys = self.cell[i:i+wLen,j] == ""
            if np.all(emptys == True) or np.any(self.enable[i:i+wLen,j] == False) or np.all(emptys == False):
                return False
        if div == 1:
            emptys = self.cell[i,j:j+wLen] == ""
            if np.all(emptys == True) or np.any(self.enable[i,j:j+wLen] == False) or np.all(emptys == False):
                return False

        # Judge whether correct intersection
        where = np.where(emptys == False)[0]
        if div == 0:
            jall = np.full(where.size, j, dtype = "int64")
            if np.any(self.cell[where+i,jall] != np.array(list(word))[where]):
                return False
        if div == 1:
            iall = np.full(where.size, i, dtype = "int64")
            if np.any(self.cell[iall,where+j] != np.array(list(word))[where]):
                return False

        # If neighbor cells are filled except at the intersection, return False
        where = np.where(emptys == True)[0]
        if div == 0:
            jall = np.full(where.size, j, dtype = "int64")
            # Left side
            if j > 0 and np.any(self.cell[where+i,jall-1] != ""):
                return False
            # Right side
            if j < self.width-1 and np.any(self.cell[where+i,jall+1] != ""):
                return False
        if div == 1:
            iall = np.full(where.size, i, dtype = "int64")
            # Upper
            if i > 0 and np.any(self.cell[iall-1,where+j] != ""):
                return False
            # Lower
            if i < self.height-1 and np.any(self.cell[iall+1,where+j] != ""):
                return False

        # If the preceding and succeeding cells are already filled
        if div == 0:
            if i > 0 and self.cell[i-1,j] != "":
                return False
            if i+wLen < self.height and self.cell[i+wLen,j] != "":
                return False
        if div == 1:
            if j > 0 and self.cell[i,j-1] != "":
                return False
            if j+wLen < self.width and self.cell[i,j+wLen] != "":
                return False

        # If Break through the all barrier, return True
        return True

    ### add
    def add(self, dic, plc, div, i, j, word):
        # Get the word length
        wLen = len(word)

        # Judge whether adding is enabled
        if self.isEnabledAdd(div, i, j, word, wLen) == False:
            return

        # Put the word to puzzle
        if div == 0:
            self.cell[i:i+wLen,j] = list(word)[0:wLen]
        if div == 1:
            self.cell[i,j:j+wLen] = list(word)[0:wLen]

        # Set the prohibited cell before and after placed word
        if div == 0:
            if i > 0:
                self.enable[i-1,j] = False
            if i+wLen < self.height:
                self.enable[i+wLen,j] = False
        if div == 1:
            if j > 0:
                self.enable[i,j-1] = False
            if j+wLen < self.width:
                self.enable[i,j+wLen] = False

        # Update cover array
        if div == 0:
            self.cover[i:i+wLen,j] += 1
        if div == 1:
            self.cover[i,j:j+wLen] += 1

        # Update properties
        self.usedPlcIdx[self.solSize] = plc.invP[div, i, j, dic.data.index(word)]
        self.usedWords[self.solSize] = word
        self.solSize += 1
        return

    ### firstSolve
    def firstSolve(self, dic, plc):
        # Check the initSol
        if self.initSol:
            sys.stderr.write("error: 'firstSolve' method has already called.")
            return

        # Make a random index of plc
        randomIndex = np.arange(plc.size)
        shuffle(randomIndex)

        # Add as much as possible
        solSizeTmp = -1
        while self.solSize != solSizeTmp:
            solSizeTmp = self.solSize
            for t in randomIndex:
                self.add(dic, plc, plc.div[t], plc.i[t], plc.j[t], dic.data[plc.k[t]])
        self.initSol = True

    ### show
    def show(self, ndarray):
        if self.dictType == "English":
            ndarr = np.where(ndarray == "", "-", ndarray)
        elif self.dictType == "Japanese" or self.dictType == "Kanji":
            ndarr = np.where(ndarray == "", "⬜︎", ndarray)
        print("\n".join([" ".join(map(str, l)) for l in ndarr]))

    ### DFS
    def DFS(self, i, j, ccl):
        self.coverDFS[i,j] = ccl
        if i>0 and self.coverDFS[i-1,j] == 1:
            self.DFS(i-1, j, ccl)
        if i<self.height-1 and self.coverDFS[i+1,j] == 1:
            self.DFS(i+1, j, ccl)
        if j>0 and self.coverDFS[i,j-1] == 1:
            self.DFS(i, j-1, ccl)
        if j<self.width-1 and self.coverDFS[i,j+1] == 1:
            self.DFS(i, j+1, ccl)

    ### compile
    def compile(self, dictionary=None, placeable=None, objFunc=None, optimizer=None, msg=True):
        if dictionary is None or placeable is None or objFunc is None or optimizer is None:
            sys.stderr.write("error: usage .compile(dictionary, placeable, objFunc, optimizer)")
            sys.exit()
        self.dictType = dictionary.dictType
        self.objFunc = objFunc
        self.optimizer = optimizer
        self.optimizer.puzzle = self
        self.optimizer.tmpPuzzle = Puzzle(self.width, self.height, msg=False)
        self.optimizer.dic = dictionary
        self.optimizer.plc = placeable
        self.optimizer.objFunc = objFunc

        if msg:
            print("compile succeeded.")
            print(" --- objective functions:")
            for funcNum in range(len(objFunc.registeredFuncs)):
                print("  |-> %d. %s" % (funcNum, objFunc.registeredFuncs[funcNum]))
            print(" --- optimizer: %s" % optimizer.method)

    ### solve
    def solve(self, *, epock=0):
        if epock == 0:
            sys.stderr.write("error: usage(example) .solve(epock=10)")
            sys.exit()
        self.optimizer.localSearch(epock)

    ### isSimpleSol
    def isSimpleSol(self, plc):
        rtnBool = True
        for s, p in enumerate(self.usedPlcIdx[:self.solSize-1]):
            i = plc.i[p]
            j = plc.j[p]
            word1 = self.usedWords[s]
            if plc.div[p] == 0:
                crossIdx1 = np.where(self.cover[i:i+len(word1),j] == 2)[0]
            elif plc.div[p] == 1:
                crossIdx1 = np.where(self.cover[i,j:j+len(word1)] == 2)[0]
            for t, q in enumerate(self.usedPlcIdx[s+1:self.solSize]):
                i = plc.i[q]
                j = plc.j[q]
                word2 = self.usedWords[s+t+1]
                if len(word1) != len(word2):
                    break
                if plc.div[q] == 0:
                    crossIdx2 = np.where(self.cover[i:i+len(word2),j] == 2)[0]
                if plc.div[q] == 1:
                    crossIdx2 = np.where(self.cover[i,j:j+len(word2)] == 2)[0]
                if crossIdx1.size == crossIdx2.size and np.all(crossIdx1 == crossIdx2):
                    if np.all(np.array(list(word1))[crossIdx1] == np.array(list(word2))[crossIdx2]):
                        print(" - words '%s' and '%s' are replaceable" % (word1, word2))
                        rtnBool = False
        return rtnBool

class Dictionary():
    def __init__(self, fpath, msg=True):
        self.fpath = fpath
        print("Dictionary object has made.")

        ## Read
        print(" - READING DICTIONARY...")
        file = open(self.fpath, 'r', encoding='utf-8')
        self.data = file.readlines()
        file.close()

        # Get a size of dictionary
        self.size = len(self.data)

        # Check dictionary type(English/Japanese)
        uniName = unicodedata.name(self.data[0][0])[0:10]
        if "HIRAGANA" in uniName or "KATAKANA" in uniName:
            self.dictType = "Japanese"
        elif "LATIN" in uniName:
            self.dictType = "English"
            #self.data = [s.upper() for s in self.data]
        elif "CJK" in uniName:
            self.dictType = "Kanji"

        # Remove "\n"
        def removeNewLineCode(word):
            return word.rstrip("\n")
        self.data = list(map(removeNewLineCode, self.data))

        ## Message
        if msg == True:
            print(" - file path         : %s" % self.fpath)
            print(" - dictionary size   : %d" % self.size)
            print(" - dictionary type   : %s" % self.dictType)
            print(" - top of dictionary : %s" % self.data[1])

class Placeable():
    def __init__(self, puzzle, dic, msg=True):
        self.size = 0
        self.width = puzzle.width
        self.height = puzzle.height
        self.div = np.zeros(2*dic.size*self.width*self.height, dtype='int64')
        self.k = np.zeros(2*dic.size*self.width*self.height, dtype='int64')
        self.i = np.zeros(2*dic.size*self.width*self.height, dtype='int64')
        self.j = np.zeros(2*dic.size*self.width*self.height, dtype='int64')
        self.invP = np.zeros(2*dic.size*self.width*self.height, dtype='int64').reshape(2,self.height,self.width,dic.size)
        dicSize = dic.size

        for div in range(2):
            for k in range(dicSize):
                if div == 0:
                    iMax = self.height - len(dic.data[k]) + 1
                    jMax = self.width
                elif div == 1:
                    iMax = self.height
                    jMax = self.width - len(dic.data[k]) + 1
                for i in range(iMax):
                    for j in range(jMax):
                        self.div[self.size] = div
                        self.k[self.size] = k
                        self.i[self.size] = i
                        self.j[self.size] = j
                        self.invP[div,i,j,k] = self.size
                        self.size += 1
        if msg == True:
            print("Placeable object has made.")
            print(" - placeable size : %d/%d(max shape)" % (self.size, self.div.size))

class ObjectiveFunction():
    def __init__(self, puzzle, msg=True):
        self.puzzle = puzzle
        self.flist = ["solSize",
                      "crossCount",
                      "fillCount",
                      "maxConnectedEmptys"]
        self.registeredFuncs = []
        if msg == True:
            print("ObjectiveFunction object has made.")
    ### solSize
    def solSize(self):
        return self.puzzle.solSize

    ### crossCount
    def crossCount(self):
        return np.sum(self.puzzle.cover == 2)

    ### fillCount
    def fillCount(self):
        return np.sum(self.puzzle.cover >= 1)

    ### maxConnectedEmptys
    def maxConnectedEmptys(self):
        ccl = 2
        self.puzzle.coverDFS = np.where(self.puzzle.cover == 0, 1, 0)
        for i, j in itertools.product(range(self.puzzle.height), range(self.puzzle.width)):
            if self.puzzle.coverDFS[i,j] == 1:
                self.puzzle.DFS(i, j, ccl)
                ccl += 1
        score = self.puzzle.width*self.puzzle.height - np.max(np.bincount(self.puzzle.coverDFS.flatten())[1:])
        return score

    ### register
    def register(self, funcNames, msg=True):
        for funcName in funcNames:
            if funcName not in self.flist:
                sys.stderr.write("error: ObjectiveFunction class doesn't have '%s' function." % funcName)
                sys.exit()
            if msg == True:
                print(" - '%s' function has registered." % funcName)
        self.registeredFuncs = funcNames
        return

    ### getScore
    def getScore(self, i, all=False):
        if all:
            scores=np.zeros(len(self.registeredFuncs), dtype="int64")
            for n in range(scores.size):
                scores[n] = eval("self.%s()" % self.registeredFuncs[n])
            return scores
        return eval("self.%s()" % self.registeredFuncs[i])

class Optimizer():
    def __init__(self, msg=True):
        self.methodList = ["localSearch", "iterativeLocalSearch"]
        self.method = ""
        if msg == True:
            print("Opimizer object has made.")

    ### drop
    def drop(self, puzzle, p, indexOfUsedPlcIdx):
        # Get div, i, j, wLen
        div = self.plc.div[p]
        i = self.plc.i[p]
        j = self.plc.j[p]
        wLen = len(self.dic.data[self.plc.k[p]])

        # If '2' is aligned in the cover array, continue to next loop
        if div == 0:
            if np.any(np.diff(np.where(puzzle.cover[i:i+wLen,j] == 2)[0]) == 1):
                return
        if div == 1:
            if np.any(np.diff(np.where(puzzle.cover[i,j:j+wLen] == 2)[0])==1):
                return
        # Pull out a word
        if div == 0:
            puzzle.cover[i:i+wLen,j] -= 1
            where = np.where(puzzle.cover[i:i+wLen,j] == 0)[0]
            jall = np.full(where.size, j, dtype="int64")
            puzzle.cell[i+where,jall] = ""
        if div == 1:
            puzzle.cover[i,j:j+wLen] -= 1
            where = np.where(puzzle.cover[i,j:j+wLen] == 0)[0]
            iall = np.full(where.size, i, dtype="int64")
            puzzle.cell[iall,j+where] = ""
        # Update usedPlcIdx and solSize
        puzzle.usedWords[indexOfUsedPlcIdx] = ""
        puzzle.usedPlcIdx[indexOfUsedPlcIdx] = -1
        puzzle.solSize -= 1
        # Release prohibited cells
        removeFlag = True
        if div == 0:
            if i > 0:
                if i > 2 and np.all(puzzle.cell[[i-3,i-2],[j,j]] != ""):
                    removeFlag = False
                if j > 2 and np.all(puzzle.cell[[i-1,i-1],[j-2,j-1]] != ""):
                    removeFlag = False
                if j < puzzle.width-2 and np.all(puzzle.cell[[i-1,i-1],[j+1,j+2]] != ""):
                    removeFlag = False
                if removeFlag == True:
                    puzzle.enable[i-1,j] = True
            if i+wLen < puzzle.height:
                if i+wLen < puzzle.height-2 and np.all(puzzle.cell[[i+wLen+1,i+wLen+2],[j,j]] != ""):
                    removeFlag = False
                if j > 2 and np.all(puzzle.cell[[i+wLen,i+wLen],[j-2,j-1]] != ""):
                    removeFlag = False
                if j < puzzle.width-2 and np.all(puzzle.cell[[i+wLen,i+wLen],[j+1,j+2]] != ""):
                      removeFlag = False
                if removeFlag == True:
                    puzzle.enable[i+wLen,j] = True
        if div == 1:
            if j > 0:
                if j > 2 and np.all(puzzle.cell[[i,i],[j-3,j-2]] != ""):
                    removeFlag = False
                if i > 2 and np.all(puzzle.cell[[i-2,i-1],[j-1,j-1]] != ""):
                    removeFlag = False
                if i < puzzle.height-2 and np.all(puzzle.cell[[i+1,i+2],[j-1,j-1]] != ""):
                    removeFlag = False
                if removeFlag == True:
                    puzzle.enable[i,j-1] = True
            if j+wLen < puzzle.width:
                if j+wLen < puzzle.width-2 and np.all(puzzle.cell[[i,i],[j+wLen+1,j+wLen+2]] != ""):
                    removeFlag = False
                if i > 2 and np.all(puzzle.cell[[i-2,i-1],[j+wLen,j+wLen]] != ""):
                    removeFlag = False
                if i < puzzle.height-2 and np.all(puzzle.cell[[i+1,i+2],[j+wLen,j+wLen]] != ""):
                    removeFlag = False
                if removeFlag == True:
                    puzzle.enable[i,j+wLen] = True

    ### getNeighborSolution
    def getNeighborSolution(self, tmpPuzzle):
        # If solSize = 0, return
        if tmpPuzzle.solSize == 0:
            return

        # Make a random index of solSize
        randomIndex = np.arange(tmpPuzzle.solSize)
        shuffle(randomIndex)

        # Drop words until connectivity collapses
        for r, p in enumerate(tmpPuzzle.usedPlcIdx[randomIndex]):
            self.drop(tmpPuzzle, p, randomIndex[r])
            # End with connectivity breakdown
            tmpPuzzle.coverDFS = np.where(tmpPuzzle.cover >= 1, 1, 0)
            ccl = 2
            for i, j in itertools.product(range(tmpPuzzle.height), range(tmpPuzzle.width)):
                if tmpPuzzle.coverDFS[i,j] == 1:
                    tmpPuzzle.DFS(i, j, ccl)
                    ccl += 1
            if ccl-2 >= 2:
                break
        # Stuff the deleted index
        whereDelete = np.where(tmpPuzzle.usedPlcIdx == -1)[0]
        for p in whereDelete:
            tmpPuzzle.usedWords = np.append(np.delete(tmpPuzzle.usedWords, p), "")
            tmpPuzzle.usedPlcIdx = np.append(np.delete(tmpPuzzle.usedPlcIdx, p), -1)
            whereDelete -= 1

        # Kick
        # If solSize = 0 after droping, return
        if tmpPuzzle.solSize == 0:
            return

        # Define 'largestCCL' witch has the largest score(fillCount+crossCount)
        cclScores = np.zeros(ccl-2, dtype="int64")
        for c in range(ccl-2):
            cclScores[c] = np.sum(np.where(tmpPuzzle.coverDFS == c+2, tmpPuzzle.cover, 0))
        largestCCL = np.argmax(cclScores) + 2

        # Erase elements except CCL ('kick' in C-program)
        for idx, p in enumerate(tmpPuzzle.usedPlcIdx[:self.puzzle.solSize]):
            if p == -1:
                continue
            if tmpPuzzle.coverDFS[self.plc.i[p],self.plc.j[p]] != largestCCL:
                self.drop(tmpPuzzle, p, idx)

        # Stuff the deleted index
        whereDelete = np.where(tmpPuzzle.usedPlcIdx == -1)[0]
        for p in whereDelete:
            tmpPuzzle.usedWords = np.append(np.delete(tmpPuzzle.usedWords, p), "")
            tmpPuzzle.usedPlcIdx = np.append(np.delete(tmpPuzzle.usedPlcIdx, p), -1)
            whereDelete -= 1

        # Make a random index of plc
        randomIndex = np.arange(self.plc.size)
        shuffle(randomIndex)

        # Add as much as possible
        solSizeTmp = -1
        while tmpPuzzle.solSize != solSizeTmp:
            solSizeTmp = tmpPuzzle.solSize
            for t in randomIndex:
                tmpPuzzle.add(self.dic, self.plc, self.plc.div[t], self.plc.i[t], self.plc.j[t], self.dic.data[self.plc.k[t]])
    ### localSearch
    def localSearch(self, epock, show=True, move=False):
        if show:
            print("Interim solution:")
            self.puzzle.show(self.puzzle.cell)
        for ep in range(1,epock+1):
            print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
            print("Epock %d/%d" % (ep, epock))
            ## Obtain a neighbor solution
            # Copy the Puzzle object and objective function
            self.tmpPuzzle.copy(self.puzzle)
            tmpObjFunc = ObjectiveFunction(self.tmpPuzzle, msg=False)
            tmpObjFunc.register(self.objFunc.registeredFuncs, msg=False)

            # Get neigh solution by drop->kick->add
            self.getNeighborSolution(self.puzzle)

            # Repeat if the score is high
            for funcNum in range(len(tmpObjFunc.registeredFuncs)):
                puzzleScore = self.objFunc.getScore(funcNum)
                tmpPuzzleScore = tmpObjFunc.getScore(funcNum)
                if puzzleScore > tmpPuzzleScore:
                    print(" Improved - score[%d]: %d" % (funcNum,puzzleScore))
                    if show:
                        self.puzzle.show(self.puzzle.cell)
                    break
                if puzzleScore < tmpPuzzleScore:
                    self.puzzle.copy(self.tmpPuzzle)
                    print(" Stayed")
                    break
            else:
                print(" Replaced(same score) - score[%d]: %d" % (funcNum,puzzleScore))
                if show:
                    self.puzzle.show(self.puzzle.cell)
        return
    ### setMethod
    def setMethod(self, methodName, msg=True):
        if methodName not in self.methodList:
            sys.stderr.write("error: Optimizer class doesn't have '%s' method." % methodName)
            sys.exit()
        if msg:
            print(" - '%s' method has registered." % methodName)
        self.method = methodName
