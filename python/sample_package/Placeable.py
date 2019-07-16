import numpy as np


class Placeable:
    def __init__(self, puzzle, dic, msg=True):
        self.size = 0
        self.width = puzzle.width
        self.height = puzzle.height
        self.div, self.i, self.j, self.k = [], [], [], []
        self.invP = np.full((2, self.height, self.width, dic.size), np.nan, dtype='uint16')
        
        for div in (0,1):
            for k in range(dic.size):
                if div == 0:
                    iMax = self.height - dic.wLen[k] + 1
                    jMax = self.width
                elif div == 1:
                    iMax = self.height
                    jMax = self.width - dic.wLen[k] + 1
                for i in range(iMax):
                    for j in range(jMax):
                        self.div.append(div)
                        self.i.append(i)
                        self.j.append(j)
                        self.k.append(k)
                        self.invP[div,i,j,k] = self.size
                        self.size += 1
        self.div = np.array(self.div, dtype="uint8")
        self.i = np.array(self.i, dtype="uint8")
        self.j = np.array(self.j, dtype="uint8")
        self.k = np.array(self.k, dtype="uint8")
        if msg is True:
            print(f"Imported Dictionary name: `{dic.name}`, size: {dic.size}")
            print(f"Placeable size : {self.size}")
            
    def __len__(self):
        return self.size

    def __getitem__(self, key):
        if type(key) in (int, np.int64):
            return {"div": self.div[key], "i": self.i[key], "j": self.j[key], "k": self.k[key]}
        if type(key) is str:
            return eval(f"self.{key}")


