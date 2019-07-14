import copy

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

    def localSearch(self, puzzle, epoch, show=True, move=False, stdout=False):
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
            _puzzle.show(_puzzle.cell, stdout=stdout)
        goalEpoch = _puzzle.epoch + epoch
        for ep in range(epoch):
            _puzzle.epoch += 1
            print(">>> Epoch %d/%d" % (_puzzle.epoch, goalEpoch))
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
                        _puzzle.show(_puzzle.cell, stdout=stdout)
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
                    _puzzle.show(_puzzle.cell, stdout=stdout)
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
        puzzle.historyIdx = copy.deepcopy(_puzzle.historyIdx)
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
            raise ValueError(f"Optimizer does not have '{methodName}' method")
        if msg:
            print(" - '%s' method has registered." % methodName)
        self.method = methodName
