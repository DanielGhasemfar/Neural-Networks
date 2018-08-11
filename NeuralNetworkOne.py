import random

def createDataSet(sizeOfSet):
    TrainingSet = []
    for n in xrange(0,sizeOfSet-1):
        value = round(random.random())
        label = value
        data = [value,label]
        TrainingSet.append(data)
    return TrainingSet

def createSetOfWeights(nextLayer):
    weights = []
    for i in xrange(0,nextLayer):
        weight = random.random()
        weights.append(weight)
    return weights

def createNeuralNetwork(arrayOfLayers):
    NeuralNetwork = []
    for x in xrange(0,len(arrayOfLayers)):
        layer = []
        for y in xrange(0,arrayOfLayers[x]):
            if x == 0:
                # node = [value, bias]
                bias = 0.0
                node = [0.0, bias]
                layer.append(node)
            else:
                # node = [value, bias, nextWeights[]]
                bias = random.random()
                node = [0.0, bias, createSetOfWeights(arrayOfLayers[x-1])]
                layer.append(node)
        NeuralNetwork.append(layer)
    return NeuralNetwork

def drawNeuralNetwork(NN):
    print ""
    for layer in xrange(0,len(NN)):
        print "LAYER NUMBER "+str(layer+1)
        for node in xrange(0,len(NN[layer])):
            print "Node Value:"+str(NN[layer][node][0])

def sigmoid(x):
    y = 1/(1+2.718281828**(-1*x))
    return y

def ReLu(x):
    return max(0,x)

def singleCalculation(NN, data):
    for node in xrange(0,len(NN[0])):
        NN[0][node][0] = data[0]
    for layer in xrange(1,len(NN)):
        for node in xrange(0,len(NN[layer])):
            nodeValue = NN[layer][node][0]
            for prevNode in xrange(0,len(NN[layer-1])):
                # nodeValue += Sigmoid( Sum( Weight * value ) + bias )
                nodeValue += NN[layer][node][2][prevNode] * NN[layer-1][prevNode][0]
            nodeValue += NN[layer][node][1]
            NN[layer][node][0] = sigmoid(nodeValue)

# This function is specific to this data set.
def cost(NN, data):
    lastLayer = len(NN)-1
    cost = (data[0]-NN[lastLayer][0][0])**2
    return cost

def runDataSet(NN,dataSet):
    averageCost = 0.0
    num = 0.0
    for i in xrange(0,len(dataSet)-1):
        singleCalculation(NN, dataSet[i])
        averageCost += cost(NN, dataSet[i])
        num += 1
        drawNeuralNetwork(NN)
        # CLEAR NN VALUES
    averageCost = averageCost/num
    print "Average error: "+str(averageCost)

TrainingSet = createDataSet(100)
TestSet = createDataSet(50)
# Each data point is either 1 or 0 and the label is equal to the actual data
# in this case but this is not always the case (e.g with a picture of a lion
# and the label "lion")
NeuralNetwork = createNeuralNetwork([1,2,1])
# The input is the number of neurons in each layer. The output is a 2d array
# where the first index is the layer number and the second is the nueron number.
# Each nueron is ordered like so [value, bias, nextWeights]. The next weights
# are the weights of the next set of neurons (first weight connects to first neuron)
runDataSet(NeuralNetwork, TrainingSet)

