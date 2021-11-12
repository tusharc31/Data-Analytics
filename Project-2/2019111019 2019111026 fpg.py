from fpgrowth_py import fpgrowth
from collections import defaultdict 
import time


def getData(filename):
    with open(filename, 'r') as file:
        data = file.read().replace('\n', '')
    file.close()
    data = data.split('-2')
    for i in range(len(data)):
        data[i] = data[i].split('-1')
        data[i] = data[i][:-1]
        data[i] = list(map(int, data[i]))
    sm=0
    for i in data: sm = sm + len(i)
    # print(f'\nThe dataset has {len(data)} transactions with average {sm/len(data)} items per transaction.\n')
    return data


class treeNode:
    def __init__(self, name, count, parent):
        self.children = {} 
        self.name = name
        self.count = count
        self.nodeLink = None
        self.parent = parent
    def disp(self, ind=1):
        print (ind*'  ', self.name, ' ', self.count)
        for i in range(len(self.children.values())):
            self.children.values()[i].disp(1+ind)
    def inc(self, count):
        self.count = self.count + count

            
def ordered(transactions,minSupport):
    headerTable=defaultdict(lambda : [0,None])
    for i in range(len(transactions)):
        for j in transactions[i][0]:
            headerTable[j][0] = headerTable[j][0]+transactions[i][1]
    for i in range(len(transactions)):
        trans=sorted(transactions[i][0], key=lambda x: headerTable[x][0],reverse=1)
        transactions[i][0]=trans.copy()
        trans = []
        for item in transactions[i][0]:
            if headerTable[item][0]>=minSupport:
                trans.append(item)      

        transactions[i][0]=trans.copy()
    ht = list(headerTable)
    for k in range(len(ht)):
        if headerTable[ht[k]][0]<minSupport:
            del(headerTable[ht[k]])
    return transactions,headerTable


def updateTree(items, fpTree, headerTable, count):
    ite = items[0]
    if ite in fpTree.children:
        fpTree.children[ite].inc(count)
    else:  
        fpTree.children[ite] = treeNode(ite, count, fpTree)
        if not (headerTable[ite][1] == None):
            nodeToTest=headerTable[ite][1]
            while not (nodeToTest.nodeLink == None):    
                nodeToTest = nodeToTest.nodeLink
            nodeToTest.nodeLink = fpTree.children[ite]
        else:
            headerTable[ite][1] = fpTree.children[ite]
    lo = len(items) 
    if lo > 1:
        updateTree(items[1::], fpTree.children[ite], headerTable, count)

def ascendTree(pathNode):
    prefixPath=[]
    while not (pathNode.parent == None):
        prefixPath.append(pathNode.name)
        pathNode=pathNode.parent
    return prefixPath
    
def bottomUpTechnique(treeNode):
    condPats = []
    while not (treeNode == None):
        prefixPath = ascendTree(treeNode)
        lo = len(prefixPath)
        ap = [prefixPath[1:],treeNode.count]
        if lo > 1: 
            condPats.append(ap)
        treeNode = treeNode.nodeLink
    return condPats


def topDownTechnique(treeNode):
    condPats = []
    while not (treeNode == None):
        out =dfs(treeNode,[])
        if len(out[1]) > 0: 
            condPats= condPats + out[1]
        treeNode = treeNode.nodeLink
    return condPats


def buildTree(transactions,minSupport):
    out = ordered(transactions,minSupport)
    trans = out[0]
    headerTable = out[1]
    fpTree = treeNode('Null', 1, None)
    for i in trans:
        lo = len(i[0])
        if lo>0:
            updateTree(i[0], fpTree, headerTable, i[1])
    return fpTree, headerTable






def dfs(pathNode,path):
    kids=pathNode.children
    ans=[]
    cnt=pathNode.count
    for i in kids:
        path.append(i)
        out =dfs(kids[i],path)
        sub = out[0]
        add = out[1]
        cnt= cnt-sub
        ans=ans+add
        path.pop()
    
    if cnt==0:
        pass
    else:
        ans.append([path.copy(),cnt])
    return pathNode.count,ans





def mineFPtree(inTree, headerTable, minSup, preFix,flag):
    reverseHeaderTable = []
    for v in sorted(headerTable.items(), key=lambda p:p[1][0] ,reverse=1):
        reverseHeaderTable.append(v[0])
    if flag:
        reverseHeaderTable = reverseHeaderTable[::-1]
    freqItemList=[]
    for i in range(len(reverseHeaderTable)): 
        
        newFreqSet = preFix.copy()
        newFreqSet.append(reverseHeaderTable[i])
        freqItemList.append(newFreqSet)

        if not flag:
            conditionalPatternBase = topDownTechnique(headerTable[reverseHeaderTable[i]][1])
        else:
            conditionalPatternBase = bottomUpTechnique(headerTable[reverseHeaderTable[i]][1])

        conditionalTree,conditionalHeaderTable=buildTree(conditionalPatternBase,minSupport)
        if not (conditionalHeaderTable == None):
            freqItemList= freqItemList + mineFPtree(conditionalTree, conditionalHeaderTable, minSup, newFreqSet,flag)
    
    return freqItemList

# flag=0 means topDownTechnique
# flag=1 means bottomUpTechnique

_minsup = [600, 580, 560, 540, 520, 500]

### RUNNING FOR TOP DOWN
print('RUNNING FOR TOP DOWN')
flag = 0
for minSupport in _minsup:

    last = time.time()

    transactions=getData("SIGN.txt")
    # transactions=[[i,1] for i in transactions]  
    trans = []
    for i in transactions:
        trans.append([i,1])

    fpTree, headerTable = buildTree(trans,minSupport)    
    freqItemList = mineFPtree(fpTree, headerTable, minSupport, [],flag)
    
    freqItemList.sort(key=len)    
    # uncomment to print the item set
    # print(freqItemList)
    print(f"MINSUP: {minSupport}    TIME: {time.time()-last} seconds")


### RUNNING FOR BOTTOM UP
print('RUNNING FOR BOTTOM UP')
flag = 1
for minSupport in _minsup:

    last = time.time()

    transactions=getData("SIGN.txt")
    # transactions=[[i,1] for i in transactions]  
    trans = []
    for i in transactions:
        trans.append([i,1])

    fpTree, headerTable = buildTree(trans,minSupport)    
    freqItemList = mineFPtree(fpTree, headerTable, minSupport, [],flag)
    
    freqItemList.sort(key=len)    
    # uncomment to print the item set
    # print(freqItemList)
    print(f"MINSUP: {minSupport}    TIME: {time.time()-last} seconds")


### RUNNING USING INBUILT LIBRARY
print('RUNNING USING INBUILT LIBRARY')
for minsup in _minsup:
    start_time = time.time()
    freqItemSet, rules = fpgrowth(transactions, minSupRatio=minsup/len(transactions), minConf=0.0)
    print(f'MINSUP: {minsup}    TIME: {time.time()-start_time} seconds')
