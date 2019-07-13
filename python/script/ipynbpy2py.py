"""
ipynbファイルから変換したpyファイルを, 指定したパッケージ名のもと, クラス毎に分割する.
コマンドライン引数は
 1. ipynbから得たpyファイル
 2. パッケージ名（-nオプションで指定. デフォルトは'src'）

 実行例：
 python ipynbpy2py.py jupyter/CrosswordLocalSearch.py test_package
"""

import os
import argparse

# In[]
parser = argparse.ArgumentParser(description="convert ipynb.py to .py with given package_name")
parser.add_argument("ipynbpy", type=str,
                    help="python file made by jupytext or ipynb")
parser.add_argument("-n", "--name", type=str, default="src",
                    help="name of package, default=src")
args = parser.parse_args()

# settings
ipynbpy = args.ipynbpy
package_name = args.name

# open
with open(ipynbpy, encoding='utf-8') as f:
    lines = f.readlines()

# read import and class
imports, classes = [], []
for i, line in enumerate(lines):
    if line[:7] == "import " or line[:5] == "from ":
        imports.append(line)
    if line[:6] == "class ":
        classes.append(line)
imports = list(set(imports))
class_names = list(map(lambda c: c[6:-2], classes))

# set class line box, if 5 class is loaded, class_lines = [[],[],[],[],[]]
import_table, import_lines, class_lines = [], [], []
for _ in range(len(classes)):
    import_table.append([])
    import_lines.append([])
    class_lines.append([])
# get class lines written by class Hoge(): area
class_flag = False
class_num = -1
for line in lines:
    if line[:6] == "class ":
        class_flag = True
        class_num += 1
    elif line[0] not in (" ", os.linesep):
        class_flag = False
    if class_flag is True:
        class_lines[class_num].append(line)
# get class lines written by def fuga ... setattr(Hoge, "fuga", fuga)
def_flag = False
def_end_flag = False
for line in lines:
    if line[:4] == "def ":
        def_lines = []
        def_flag = True
    elif line[0] not in (" ", os.linesep, "setattr"):
        def_flag = False
    if def_flag is True:
        def_lines.append(line)
    if line[:7] == "setattr":
        def_end_flag = True
        setattr_class = line.split(",")[0][8:]
    if def_end_flag is True:
        class_num = class_names.index(setattr_class)
        for def_line in def_lines:
            if def_line[0] not in (os.linesep):
                def_line = "    " + def_line
            class_lines[class_num].append(def_line)
        def_end_flag = False

# 'import' arrangement
import_names = []
for import_line in imports:
    if "as" in import_line:
        import_names.append(import_line.split("as ")[-1].rstrip().lstrip())
    else:
        name = import_line.split("import ")[-1]
        if len(name.split(",")) is 1:
            import_names.append(name.rstrip().lstrip())
        else:
            for name in name.split(","):
                import_names.append(name.rstrip().lstrip())
for class_num, class_line in enumerate(class_lines):
    for line in class_line:
        for import_name in import_names:
            if ","+import_name+"." in line or ","+import_name+"(" in line or " "+import_name+"." in line or " "+import_name+"(" in line:
                import_table[class_num].append(import_name)
        for class_name in class_names:
            if ","+class_name+"." in line or ","+class_name+"(" in line or " "+class_name+"." in line or " "+class_name+"(" in line:
                import_table[class_num].append(class_name)
    import_table[class_num] = list(set(import_table[class_num]))
for class_num, class_name in enumerate(class_names):
    if class_name in import_table[class_num]:
        import_table[class_num].remove(class_name)
for class_num in range(len(class_names)):
    for import_name in import_table[class_num]:
        for import_line in imports:
            if import_name in import_line:
                import_lines[class_num].append(import_line)
    import_lines[class_num] = list(set(import_lines[class_num]))
    import_lines[class_num].append(os.linesep)
for class_num in range(len(class_names)):
    for import_name in import_table[class_num]:
        for class_name in class_names:
            if import_name in class_name:
                import_line = f"from {package_name}.{class_name} import {class_name}{os.linesep}"
                import_lines[class_num].append(import_line)
    import_lines[class_num].append(os.linesep)

## output
os.makedirs(package_name, exist_ok=True)
# __init__.py
with open(f'{package_name}/__init__.py', 'w', encoding='utf-8') as of:
    for class_name in class_names:
        of.write(f"from {package_name}.{class_name} import {class_name}{os.linesep}")
# class_name.py
for class_num, class_name in enumerate(class_names):
    with open(f'{package_name}/{class_name}.py', 'w', encoding='utf-8') as of:
        for import_line in import_lines[class_num]:
            of.write(import_line)
        for class_line in class_lines[class_num]:
            of.write(class_line)
