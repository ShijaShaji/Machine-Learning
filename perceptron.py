#Perceptron implementation by Shija Shaji

import pandas as pd
import numpy as np
import warnings

MaxIter = 20

trainingData = pd.read_csv("train.data", header = None)
trainData = trainingData.to_numpy()

dftestData = pd.read_csv("test.data", header= None)
testData = dftestData.to_numpy()

warnings.filterwarnings("ignore")

def classFilter(data,positive,negative):

    """
    This method performs the basic filteration of the given dataset by removing a third set of class 

   Parameters:
      data: numpy array - original dataset which needs to be filtered
      positive : string - label of the positive class 
      negative : string - label of the negative class 
   Returns:
      numpy array representing the cleaned dataset
   """
    
    mask = ((data[:,4] == positive) | (data[:,4] == negative))
    cleanedDataset = data[mask]
    return cleanedDataset
    
def perceptronTraining(positiveClass,negativeClass):
   
    """
    Implements Perceptron training algorithm

   Parameters:     
      positive : string - label of the positive class 
      negative : string - label of the negative class 
   Returns:
       b,W : float, numpy array  - bias and weight of Perceptron
   """

    cleanedDataset = classFilter(trainData,positiveClass,negativeClass)      
    b = 0
    W = np.zeros(4)   

    for i in range(1,MaxIter+1):             
        np.random.seed(5)                                      #permuting the dataset in each iteration for better results
        dataset = np.random.permutation(cleanedDataset)       

        for x in range(dataset.shape[0]):           
            X = dataset[x,:4]
            if dataset[x][4] == positiveClass:         #convert output class labels to +1 and -1 
                y = 1
            else: 
                y = -1   

            a = (np.dot(W.T,X)) +  b                   #calculating activation score

            #update rule when misclassified     
            if (a*y <= 0):
                W = np.add(W,(y*X))
                b += y

    return b,W

def perceptronTest(dataset,positiveClass,negativeClass):
    
    """
    Implements Perceptron testing algorithm

   Parameters:   
      dataset  : numpy array - dataset which needs to be classfified   
      positive : string - label of the positive class 
      negative : string - label of the negative class 
   Returns:
      accuracy : float - accuracy of Perceptron classification
   """

    cleanedTestData = classFilter(dataset,positiveClass,negativeClass)
    d = cleanedTestData.shape[0]

    predOutput = []
    bias,weights = perceptronTraining(positiveClass,negativeClass)

    for i in range(0,d):
        inputVector = cleanedTestData[i]
        a = (np.dot(weights.T,inputVector[:4])) +  bias      #calculating activation score
        
        if (a > 0):
           predOutput.append(1)
        else:
            predOutput.append(-1)

    predOutput = np.array(predOutput)        
    trueOutput = np.where(cleanedTestData[:,4] == positiveClass,1,-1)         #convert output class labels to +1 and -1
    
    positiveMask = (trueOutput == 1)
    filteredPredOp1 = predOutput[positiveMask]
    tp = np.count_nonzero(filteredPredOp1==1)
    fn = np.count_nonzero(filteredPredOp1 == -1)

    negativeMask = (trueOutput == -1)
    filteredPredOp2 = predOutput[negativeMask]
    tn = np.count_nonzero(filteredPredOp2== -1)
    fp = np.count_nonzero(filteredPredOp2 == 1)
   
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    return accuracy*100
    
def multiclassTraining(positiveClass, L2coefficent):

    """
    Implements multiclass Perceptron training algorithm

   Parameters:     
      positiveClass : string - label of the positive class 
      L2coefficent: float - L2 regularisation coeffecient
   Returns:
       b,W : float, numpy array  - bias and weight of Perceptron
   """

    b = 0   
    W = np.zeros(4, dtype= object) 

    for i in range(1,MaxIter+1):      
        np.random.seed(10) 
        dataset = np.random.permutation(trainData)

        for x in range(dataset.shape[0]):
            X = dataset[x,:4]
            if dataset[x][4] == positiveClass:
                y = 1
            else: 
                y = -1      

            a = (np.dot(W.T,X)) +  b

            #update rule      
            if (a*y <= 0):
                #check if L2 regularisation coeffecient is None or not and update weights accordingly
                
                if L2coefficent == None:                      
                    W = np.add(W,(y*X))
                else:
                    L2 = (1-2*L2coefficent)
                    regularised = L2*W
                    W = np.add(regularised,(y*X))
                b += y

    return b,W

def multiclassPerceptron(data,L2 = None): 

    """
    Implements multiclass Perceptron testing algorithm

   Parameters:   
      data  : numpy array - dataset which needs to be classfified   
      L2: float - L2 regularisation coeffecient (None by default)
   Returns:
      accuracy : float - accuracy of Perceptron classification
   """

    b1,W1 = multiclassTraining('class-1',L2)
    b2,W2 = multiclassTraining('class-2',L2)
    b3,W3 = multiclassTraining('class-3',L2)
    
    accuracyCount = 0
    totalSamples = data.shape[0]
    for sample in data:

        #computes the activation score using each prediction model and output is the class label of the model which gives highest score 
        X = sample[:4]
        numericScores = []    
        score1 = np.dot((W1.T), X) + b1
        numericScores.append(score1)
        score2 = np.dot((W2.T), X) + b2
        numericScores.append(score2)
        score3 = np.dot((W3.T), X) + b3
        numericScores.append(score3)

        maxScore = max(numericScores)
        classIndex = numericScores.index(maxScore) + 1

        y = sample[4]
        
        if y == 'class-1' and classIndex == 1:
            accuracyCount+=1
        elif y == 'class-2' and classIndex == 2:
            accuracyCount+=1 
        elif y == 'class-3' and classIndex == 3:
            accuracyCount+=1    

    accuracy = accuracyCount/totalSamples
    return accuracy*100


print("------------------Class-1 and Class-2------------------")
output1 = perceptronTest(trainData,'class-1','class-2')
output2 = perceptronTest(testData,'class-1','class-2')
print("Accuracy on Train Data:", output1, "%")
print("Accuracy on Test Data:", output2, "%")
print("\n")

print("------------------Class-2 and Class-3------------------")
output3 = perceptronTest(trainData,'class-2','class-3')
output4 = perceptronTest(testData,'class-2','class-3')
print("Accuracy on Train Data:", output3,"%")
print("Accuracy on Test Data:", output4,"%")
print("\n")

print("------------------Class-1 and Class-3------------------")
output5 = perceptronTest(trainData,'class-1','class-3')
output6 = perceptronTest(testData,'class-1','class-3')
print("Accuracy on Train Data:", output5,"%")
print("Accuracy on Test Data:", output6,"%")
print("\n")

print("------------------Multiclass Classifier------------------")
output7 = multiclassPerceptron(trainData)
output8 = multiclassPerceptron(testData)
print("Accuracy on Train Data: %0.2f" % output7 ,"%")
print("Accuracy on Test Data: %0.2f" %output8,"%")
print("\n")

print("-------------Multiclass Classifier with L2 Regularisation------------------")
lstCoeff = [0.01, 0.10, 1.0, 10.0, 100.0]
for coefficient in lstCoeff:        
    print("Regularisation coefficient : ", coefficient)
    output9 = multiclassPerceptron(trainData,coefficient)
    output10 = multiclassPerceptron(testData,coefficient)
    print("Accuracy on Train Data: %0.2f" % output9,"%")
    print("Accuracy on Test Data: %0.2f" % output10,"%")
    print("\n")






