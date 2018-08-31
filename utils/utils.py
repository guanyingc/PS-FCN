import os

def makeFile(f):
    if not os.path.exists(f):
        os.makedirs(f)

def makeFiles(f_list):
    for f in f_list:
        makeFile(f)

def dictToString(dicts, start='\t', end='\n'):
    strs = '' 
    for k, v in sorted(dicts.items()):
        strs += '%s%s: %s%s' % (start, str(k), str(v), end) 
    return strs

def checkIfInList(list1, list2):
    contains = []
    for l1 in list1:
        for l2 in list2:
            if l1 in l2.lower():
                contains.append(l1)
                break
    return contains
