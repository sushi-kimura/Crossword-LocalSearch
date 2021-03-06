import itertools
import numpy as np

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
                print(" - '%s' function has registered." % funcName)
        self.registeredFuncs = funcNames
        return

    def getScore(self, puzzle, i=0, func=None, all=False):
        """
        This method returns any objective function value
        """
        if all is True:
            scores = np.zeros(len(self.registeredFuncs), dtype="int64")
            for n in range(scores.size):
                scores[n] = eval(f"self.{self.registeredFuncs[n]}(puzzle)")
            return scores
        if func is None:
            func = self.registeredFuncs[i]
        return eval(f"self.{func}(puzzle)")
