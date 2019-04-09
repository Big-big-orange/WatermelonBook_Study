# -*- coding: utf-8 -*-
# Code source: bigrao


###############################################################
from numpy import *
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']# 显示中文标签
plt.rcParams['axes.unicode_minus']= False

decisionNode=dict(boxstyle="sawtooth",fc="0.8")  	#定义分支点的样式
leafNode=dict(boxstyle="round4",fc="0.8")  			#定义叶节点的样式
arrow_args=dict(arrowstyle="<-")  					#定义箭头标识样式

# 西瓜数据数据集2.0
def createDataSet():
    """
    创建测试的数据集
    :return:
    """
    dataSet = [
        # 1
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 2
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 3
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 4
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 5
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 6
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        # 7
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        # 8
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],

        # ----------------------------------------------------
        # 9
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        # 10
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        # 11
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        # 12
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        # 13
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 14
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 15
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        # 16
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        # 17
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ]

    # 特征值列表
    features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']

    # 特征对应的所有可能的情况
    features_full = {}

    for i in range(len(features)):
        labelList = [example[i] for example in dataSet]
        uniqueLabel = set(labelList)
        features_full[features[i]] = uniqueLabel

    return dataSet, features, features_full


# 计算数据集的基尼值(gini)
def calcgini(dataSet):  # 本题中Label即好or坏瓜		#dataSet每一列是一个属性(列末是Label)
    numEntries = len(dataSet)  # 每一行是一个样本
    labelCounts = {}  # 给所有可能的分类创建字典labelCounts
    for featVec in dataSet:  # 按行循环：即rowVev取遍了数据集中的每一行,featVec只是自定义的一个变量
        # print(featVec)  # 可以尝试打印观察数据
        currentLabel = featVec[-1]  # 故featVec[-1]取遍每行最后一个值即Label
        if currentLabel not in labelCounts.keys():  # 如果当前的Label在字典中还没有
            labelCounts[currentLabel] = 0  # 则先赋值0来创建这个词
        labelCounts[currentLabel] += 1  # 计数, 统计每类Label数量(这行不受if限制)
        # labelCounts的输出类似如下：{'坏瓜': 9, '好瓜': 8}
    gini = 0.0
    for key in labelCounts:  # 遍历每类Label
        prob = float(labelCounts[key]) / numEntries  # 各类Label熵累加
        prob = prob * (1 - prob)
        gini = gini + prob    # 注意，这里只是求子集的基尼值，还没有求出基尼系数
    return gini


# 对于离散特征: 取出该特征取值为value的所有样本
def splitDiscreteDataSet(dataSet, axis, value):  # dataSet是当前结点(待划分)集合
    # axis指示划分所依据的属性
    # value该属性用于划分的取值
    retDataSet = []  # 为return Data Set分配一个列表用来储存
    for featVec in dataSet:
        # print(featVec[axis])
        if featVec[axis] == value:
            # 实际上就是找到带有该特征的样本，但是在样本子集中去掉该特征，形成新的样本
            reducedFeatVec = featVec[:axis]  # 该特征之前的特征仍保留在样本dataSet中,[:axis]]就是[0:axis]
            reducedFeatVec.extend(featVec[axis + 1:])  # 该特征之后的特征仍保留在样本dataSet中
            retDataSet.append(reducedFeatVec)  # 把这个样本加到list中
    return retDataSet

# 对于连续特征: 返回特征取值大于value的所有样本(以value为阈值将集合分成两部分)
def splitContinuousDataSet(dataSet, axis, value):
    retDataSetG = []  # 将储存取值大于value的样本
    retDataSetL = []  # 将储存取值小于value的样本
    for featVec in dataSet:
        if featVec[axis] > value:
            reducedFeatVecG = featVec[:axis]
            reducedFeatVecG.extend(featVec[axis + 1:])
            retDataSetG.append(reducedFeatVecG)
        else:
            reducedFeatVecL = featVec[:axis]
            reducedFeatVecL.extend(featVec[axis + 1:])
            retDataSetL.append(reducedFeatVecL)
    return retDataSetG, retDataSetL  # 返回两个集合, 是含2个元素的tuple形式

# 根据gini值选择当前最好的划分特征(以及对于连续变量还要选择以什么值划分)
def chooseBestFeatureToSplit(dataSet, labels):
    numFeatures = len(dataSet[0]) - 1
    # baseGini = calcgini(dataSet) #这里调用了计算基尼值的函数，来计算当前数据集的基尼值
    bestGini = 0.0
    bestFeature = -1
    bestSplitDict = {}
    for i in range(numFeatures):
        # 遍历所有特征：下面这句是取每一行的第i个, 即得当前集合所有样本第i个feature的值
        featList = [example[i] for example in dataSet]
        # 判断是否为离散特征
        if not (type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int'):
            ### 对于离散特征：求若以该特征划分的熵增
            uniqueVals = set(featList)  # 从列表中创建集合set(得列表唯一元素值),比如色泽特征的取值：‘青绿’，‘乌黑’，‘浅白’
            newGini = 0.0
            for value in uniqueVals:  # 遍历该离散特征每个取值
                subDataSet = splitDiscreteDataSet(dataSet, i, value)  # 比如得到色泽为青绿的西瓜的数据集
                prob = len(subDataSet) / float(len(dataSet))
                newGini += prob * calcgini(subDataSet)  # 各取值的基尼值累加
            Gini_index = newGini  # 得到以该特征划分的基尼指数
        ### 对于连续特征：求若以该特征划分的熵增(区别：n个数据则需添n-1个候选划分点, 并选最佳划分点)
        else:
            # 产生n-1个候选划分点
            sortfeatList = sorted(featList)
            splitList = []
            for j in range(len(sortfeatList) - 1):  # 产生n-1个候选划分点
                splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2.0)
            bestSplitEntropy = 10000  # 设定一个很大的熵值(之后用)
            # 遍历n-1个候选划分点: 求选第j个候选划分点划分时的熵增, 并选出最佳划分点
            for j in range(len(splitList)):
                value = splitList[j]
                newEntropy = 0.0
                DataSet = splitContinuousDataSet(dataSet, i, value)
                subDataSetG = DataSet[0]
                subDataSetL = DataSet[1]
                probG = len(subDataSetG) / float(len(dataSet))
                newEntropy += probG * calcgini(subDataSetG)
                probL = len(subDataSetL) / float(len(dataSet))
                newEntropy += probL * calcgini(subDataSetL)
                if newEntropy < bestSplitEntropy:
                    bestSplitEntropy = newEntropy
                    bestSplit = j
            bestSplitDict[labels[i]] = splitList[bestSplit]  # 字典记录当前连续属性的最佳划分点
            Gini_index = bestSplitEntropy  # 计算以该节点划分的熵增
        ### 在所有属性(包括连续和离散)中选择可以获得最大熵增的属性
        if Gini_index > bestGini:
            bestGini = Gini_index
            bestFeature = i
            # 若当前节点的最佳划分特征为连续特征，则需根据“是否小于等于其最佳划分点”进行二值化处理
    # 即将该特征改为“是否小于等于bestSplitValue”, 例如将“密度”变为“密度<=0.3815”
    # 注意：以下这段直接操作了原dataSet数据, 之前的那些float型的值相应变为0和1
    # 【为何这样做?】在函数createTree()末尾将看到解释
    if type(dataSet[0][bestFeature]).__name__ == 'float' or \
            type(dataSet[0][bestFeature]).__name__ == 'int':
        bestSplitValue = bestSplitDict[labels[bestFeature]]
        labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
        for i in range(shape(dataSet)[0]):
            if dataSet[i][bestFeature] <= bestSplitValue:
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0
    return bestFeature
# 若特征已经划分完，节点下的样本还没有统一取值，则需要进行投票：计算每类Label个数, 返回个数最多的那个label
def majorityCnt(classList):
    classCount = {}  # 将创建键值为Label类型的字典
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0  # 第一次出现的Label加入字典
        classCount[vote] += 1  # 计数
    # print(classCount)   返回值是类似 {'Y': 4, 'N': 3}
    return max(classCount)



def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]  # 返回标签 ['N', 'N', 'Y', 'Y', 'Y', 'N', 'Y']
    if classList.count(classList[0]) == len(classList):
        # 递归停止条件1：当前节点所有样本属于同一类；(注：count()方法统计某元素在列表中出现的次数)
        return classList[0]
    if len(dataSet[0]) == 1:
        # 递归停止条件2：所有可用于划分的特征均使用过了，则调用majorityCnt()投票定Label；
        return majorityCnt(classList)
    # 进行基于基尼系数的决策树划分
    bestFeat = chooseBestFeatureToSplit(dataSet,labels)  # 此时返回的是特征的序号
    bestFeatLabel = labels[bestFeat]
    # 多重字典构建树  例如{'outlook': {0: 'N'}}
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])        #划分完后, 即当前特征已经使用过了, 故将其从“待划分特征集”中删去
    featValues = [example[bestFeat] for example in dataSet]  # 返回的是具体的特征值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]    #['temperature', 'humidity', 'windy']
        myTree[bestFeatLabel][value] = createTree(splitDiscreteDataSet(dataSet, bestFeat, value), subLabels)
            # 划分数据，为下一层计算准备
    return myTree



### 计算树的叶子节点数量
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


### 计算树的最大深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


### 画出节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', \
                            xytext=centerPt, textcoords='axes fraction', va="center", ha="center", \
                            bbox=nodeType, arrowprops=arrow_args)


### 标箭头上的文字
def plotMidText(cntrPt, parentPt, txtString):
    lens = len(txtString)
    xMid = (parentPt[0] + cntrPt[0]) / 2.0 - lens * 0.002
    yMid = (parentPt[1] + cntrPt[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.x0ff + \
              (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.y0ff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.y0ff = plotTree.y0ff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.x0ff = plotTree.x0ff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], \
                     (plotTree.x0ff, plotTree.y0ff), cntrPt, leafNode)
            plotMidText((plotTree.x0ff, plotTree.y0ff) \
                        , cntrPt, str(key))
    plotTree.y0ff = plotTree.y0ff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.x0ff = -0.5 / plotTree.totalW
    plotTree.y0ff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


dataSet, features, features_full=createDataSet()
gini = calcgini(dataSet)
myTree=createTree(dataSet,features)
createPlot(myTree)

print(gini)
exit(0)








# def PrePurn(df_train, df_test):
#     '''
#     pre-purning to generating a decision tree
#
#     @param df_train: dataframe, the training set to generating a tree
#     @param df_test: dataframe, the testing set for purning decision
#     @return root: Node, root of the tree using purning
#     '''
#     # generating a new root node
#     new_node = Node(None, None, {})
#     label_arr = df_train[df_train.columns[-1]]
#
#     label_count = NodeLabel(label_arr)
#     if label_count:  # assert the label_count isn's empty
#         new_node.label = max(label_count, key=label_count.get)
#
#         # end if there is only 1 class in current node data
#         # end if attribution array is empty
#         if len(label_count) == 1 or len(label_arr) == 0:
#             return new_node
#
#         # calculating the test accuracy up to current node
#         a0 = PredictAccuracy(new_node, df_test)
#
#         # get the optimal attribution for a new branching
#         new_node.attr, div_value = OptAttr_Gini(df_train)  # via Gini index
#
#         # get the new branch
#         if div_value == 0:  # categoric variable
#             value_count = ValueCount(df_train[new_node.attr])
#             for value in value_count:
#                 df_v = df_train[df_train[new_node.attr].isin([value])]  # get sub set
#                 df_v = df_v.drop(new_node.attr, 1)
#                 # for child node
#                 new_node_child = Node(None, None, {})
#                 label_arr_child = df_train[df_v.columns[-1]]
#                 label_count_child = NodeLabel(label_arr_child)
#                 new_node_child.label = max(label_count_child, key=label_count_child.get)
#                 new_node.attr_down[value] = new_node_child
#
#             # calculating to check whether need further branching
#             a1 = PredictAccuracy(new_node, df_test)
#             if a1 > a0:  # need branching
#                 for value in value_count:
#                     df_v = df_train[df_train[new_node.attr].isin([value])]  # get sub set
#                     df_v = df_v.drop(new_node.attr, 1)
#                     new_node.attr_down[value] = TreeGenerate(df_v)
#             else:
#                 new_node.attr = None
#                 new_node.attr_down = {}
#
#         else:  # continuous variable # left and right child
#             value_l = "<=%.3f" % div_value
#             value_r = ">%.3f" % div_value
#             df_v_l = df_train[df_train[new_node.attr] <= div_value]  # get sub set
#             df_v_r = df_train[df_train[new_node.attr] > div_value]
#
#             # for child node
#             new_node_l = Node(None, None, {})
#             new_node_r = Node(None, None, {})
#             label_count_l = NodeLabel(df_v_l[df_v_r.columns[-1]])
#             label_count_r = NodeLabel(df_v_r[df_v_r.columns[-1]])
#             new_node_l.label = max(label_count_l, key=label_count_l.get)
#             new_node_r.label = max(label_count_r, key=label_count_r.get)
#             new_node.attr_down[value_l] = new_node_l
#             new_node.attr_down[value_r] = new_node_r
#
#             # calculating to check whether need further branching
#             a1 = PredictAccuracy(new_node, df_test)
#             if a1 > a0:  # need branching
#                 new_node.attr_down[value_l] = TreeGenerate(df_v_l)
#                 new_node.attr_down[value_r] = TreeGenerate(df_v_r)
#             else:
#                 new_node.attr = None
#                 new_node.attr_down = {}
#
#     return new_node
#
#
# def PostPurn(root, df_test):
#     '''
#     pre-purning to generating a decision tree
#
#     @param root: Node, root of the tree
#     @param df_test: dataframe, the testing set for purning decision
#     @return accuracy score through traversal the tree
#     '''
#     # leaf node
#     if root.attr == None:
#         return PredictAccuracy(root, df_test)
#
#     # calculating the test accuracy on children node
#     a1 = 0
#     value_count = ValueCount(df_test[root.attr])
#     for value in list(value_count):
#         df_test_v = df_test[df_test[root.attr].isin([value])]  # get sub set
#         if value in root.attr_down:  # root has the value
#             a1_v = PostPurn(root.attr_down[value], df_test_v)
#         else:  # root doesn't have value
#             a1_v = PredictAccuracy(root, df_test_v)
#         if a1_v == -1:  # -1 means no pruning back from this child
#             return -1
#         else:
#             a1 += a1_v * len(df_test_v.index) / len(df_test.index)
#
#     # calculating the test accuracy on this node
#     node = Node(None, root.label, {})
#     a0 = PredictAccuracy(node, df_test)
#
#     # check if need pruning
#     if a0 >= a1:
#         root.attr = None
#         root.attr_down = {}
#         return a0
#     else:
#         return -1