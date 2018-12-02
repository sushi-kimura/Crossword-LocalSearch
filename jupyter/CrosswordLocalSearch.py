
# coding: utf-8
"""
Crossword Local Search
"""
import numpy as np
from numpy.random import *
import pandas as pd
import unicodedata
import itertools
import sys
import copy

from common import *

# Set variables
fpath = "../dict/countries.txt" # countries hokkaido animals kotowaza birds dinosaurs fishes sports
width = 10
height = 10
randomSeed = 10
withweight = False
takemove = True

# Set a seed
seed(seed = randomSeed)

# Make instances
puzzle = Puzzle(width, height)
dic = Dictionary(fpath)
plc = Placeable(puzzle, dic)
objFunc = ObjectiveFunction(puzzle)
optimizer = Optimizer()
print("------------------------------------------------------------------")

# Register and set method and compile
objFunc.register(["solSize", "crossCount", "fillCount", "maxConnectedEmptys"])
optimizer.setMethod("localSearch")
puzzle.compile(dictionary=dic, placeable=plc, objFunc=objFunc, optimizer=optimizer)
print("------------------------------------------------------------------")

# Solve
puzzle.firstSolve(dic, plc)
puzzle.solve(epock=10)
print("SimpleSolution: %s" % puzzle.isSimpleSol(plc))
print("==================================================================")
