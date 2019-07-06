import numpy as np

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

# display(Image.open("fig/puzzles.png"))

# これらの情報は次のフォーマットで整理されます：
#   * k : 単語番号(辞書内の何番目の単語か)
#   * div : 単語を置く向き(0: 縦, 1: 横)
#   * j : 単語の先頭文字のx座標
#   * i : 単語の先頭文字のy座標

# display(Image.open("fig/sample_placeable.png"))


# メンバ変数は以下の通り：
#   * size : Placeableオブジェクトの大きさ
#   * width : 引数のパズルの横幅
#   * height : 引数のパズルの縦幅
#   * div : Placeable成分の文字列の方向
#   * k : Placeable成分の単語番号
#   * i : Placeable成分のy方向の座標
#   * j : Placeable成分のx方向の座標
#   * invP : Placeableオブジェクトの逆写像

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
                        self.div[self.size] = div
                        self.k[self.size] = k
                        self.i[self.size] = i
                        self.j[self.size] = j
                        self.invP[div,i,j,k] = self.size
                        self.size += 1
        if msg == True:
            print(f"Imported Dictionary name: `{dic.name}`, size: {dic.size}")
            print(f"Placeable size : {self.size}/{self.div.size}(max shape)")
    def __len__(self):
        return self.size
    def __getitem__(self, key):
        if type(key) in (int, np.int64):
            return {"div":self.div[key], "i":self.i[key], "j":self.j[key], "k":self.k[key]}
        if type(key) is str:
            return eval(f"self.{key}")
