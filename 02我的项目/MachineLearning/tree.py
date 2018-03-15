from math import log
import matplotlib.pyplot as plt

def calcShannonEnt(dataSet):
    _num_entries = len(dataSet)
    _label_counts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in _label_counts:
            _label_counts[currentLabel] = 0
        _label_counts[currentLabel] += 1
    _shannon_ent = 0.0
    for key in _label_counts:
        prob = float(_label_counts[key])/_num_entries
        _shannon_ent -= prob * log(prob, 2)
    return _shannon_ent


def creatDataSet():
    _data_set = [
        [1, 1, 0, 'maybe'],
        [1, 1, 1, 'yes'],
        [1, 0, 0, 'no'],
        [0, 1, 1, 'no'],
        [0, 1, 0, 'no']
    ]
    _labels = ['no surfacing', 'flippers', 'XXXX']
    return _data_set, _labels


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[axis+1:]
            # reduceFeatVec.extend(reduceFeatVec[axis:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    num_feature = len(dataSet[0]) -1
    base_entropy = calcShannonEnt(dataSet)
    best_info_gain = 0.0
    _best_feature = -1
    for i in range(num_feature):
        feat_list = [example[i] for example in dataSet]
        unique_value = set(feat_list)
        new_entropy = 0.0
        for value in unique_value:
            sub_dataset = splitDataSet(dataSet, i, value)
            prob = len(sub_dataset)/float(len(dataSet))
            new_entropy += prob * calcShannonEnt(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            _best_feature = i
    return _best_feature


def majorityCnt(class_list):
    import operator
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys(): class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def getNumberLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumberLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDept = 0
    firstStr = list(myTree.keys())[0]
    seconfDict = myTree[firstStr]
    for key in seconfDict.keys():
        if type(seconfDict[key]).__name__ == 'dict':
            thisDept = 1 + getTreeDepth(seconfDict[key])
        else:
            thisDept = 1
        if thisDept > maxDept:
            maxDept = thisDept
    return maxDept


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumberLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumberLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':
    dataSet, labels = creatDataSet()
    # shannonEnt = calcShannonEnt(dataSet)
    # print(shannonEnt)
    # print(splitDataSet(dataSet, 0, 0))
    # best_feature = chooseBestFeatureToSplit(dataSet)
    myTree = createTree(dataSet, labels)
    print(myTree)
    # createPlot(myTree)

    from sklearn.datasets import load_iris
    iris = load_iris()
    print(calcShannonEnt(iris.data))

