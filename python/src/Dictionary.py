import os
import unicodedata
import collections
import numpy as np

# ### Dictionary クラス
# 入力した単語リストを整理して保持するクラス。
# メンバ変数は以下の通り：
#   * fpath : 入力データのファイルパス
#   * size : 辞書の大きさ(単語数)
#   * dictType : 辞書のタイプ("English"/"Japanese")
#   * word : 単語配列
#   * weight : 重み配列
#   * wLen : 単語長配列
#   * removedWords : 削除された単語配列


class Dictionary:
    def __init__(self, fpath, msg=True):
        self.fpath = fpath
        self.name = os.path.basename(fpath)[:-4]
        # Read
        file = open(self.fpath, 'r', encoding='utf-8')
        data = file.readlines()
        file.close()
        self.word = [d[0] for d in data]
        self.weight = [d[1] for d in data]
        self.wLen = [len(w) for w in self.word]
        self.removedWords = []
        # Get a size of dictionary
        self.size = len(data)
        # Check dictionary type(English/Japanese/'Kanji')
        uniName = unicodedata.name(data[0][0])[0:10]
        if "HIRAGANA" in uniName or "KATAKANA" in uniName:
            self.dictType = "Japanese"
        elif "LATIN" in uniName:
            self.dictType = "English"
        elif "CJK" in uniName:
            self.dictType = "Kanji"

        # Remove "\n"
        def removeNewLineCode(word):
            line = word.rstrip("\n").split(" ")
            if len(line) == 1:
                line.append(0)
            line[1] = int(line[1])
            return line
        dic_list = list(map(removeNewLineCode, data))
        self.word = [d[0] for d in dic_list]
        self.weight = [d[1] for d in dic_list]
        self.wLen = [len(w) for w in self.word]

        # Message
        if msg is True:
            print("Dictionary object has made.")
            print(f" - file path         : {self.fpath}")
            print(f" - dictionary size   : {self.size}")
            print(f" - dictionary type   : {self.dictType}")
            print(f" - top of dictionary : {self[0]}")

    def __getitem__(self, key):
        return {'word': self.word[key], 'weight': self.weight[key], 'len': self.wLen[key]}

    def __str__(self):
        return self.name

    def __len__(self):
        return self.size

    def getK(self, word):
        return np.where(self.word == word)[0][0]

    # 無駄な計算を減らすため、他のどの単語とも接続(クロス)できない単語は辞書からあらかじめ削除しておきます。
    # `Dictionary`クラスに`deleteUnusableWords`メソッドを実装します：

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
                if msg:
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

        if msg:
            print("All weights are calculated.")
            print("TOP 5 characters:")
            print(counts.most_common()[:5])
            idx = sorted(range(self.size), key=lambda k: self.weight[k], reverse=True)[:5]
            print("TOP 5 words:")
            print(np.array(self.word)[idx])
