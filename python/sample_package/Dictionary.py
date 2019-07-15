import unicodedata
import os
import collections


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
                if msg is True:
                    print(f"'{w}' can not cross with any other words")
    def calcWeight(self, msg=True):
        """
        Calculate word weights in the dictionary.
        """
        mergedWords = "".join(self.word)
        counts = collections.Counter(mergedWords)
