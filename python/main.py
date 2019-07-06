# coding: utf-8
"""
Crossword Local Search
"""
# In[]
# import os
import numpy as np
from matplotlib.font_manager import FontProperties

# os.chdir("/Users/taiga/Crossword-LocalSearch/Python")
import src

# In[]
# Set variables
fpath = "../dict/pokemon.txt"  # countries hokkaido animals kotowaza birds dinosaurs fishes sports
width = 15
height = 15
randomSeed = 1
withweight = False

fp = FontProperties(fname="../jupyter/fonts/SourceHanCodeJP.ttc", size=14)
np.random.seed(seed=randomSeed)

# In[]
# Make instances
puzzle = src.Puzzle(width, height)
dic = src.Dictionary(fpath)
puzzle.importDict(dic)

objFunc = src.ObjectiveFunction()
optimizer = src.Optimizer()
print("------------------------------------------------------------------")

# In[]
# Register and set method and compile
objFunc.register(["totalWeight","solSize", "crossCount", "fillCount", "maxConnectedEmpties"])
optimizer.setMethod("localSearch")
puzzle.compile(objFunc=objFunc, optimizer=optimizer)
print("------------------------------------------------------------------")

# In[]
# Solve
puzzle.firstSolve()
puzzle.solve(epoch=10)
print("SimpleSolution: %s" % puzzle.isSimpleSol())
puzzle.saveAnswerImage("test.png", fp=fp)
print("==================================================================")
