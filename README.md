# Machine-Learning

Perceptron is a supervised learning algorithm for binary classification and can be considered as a model of a single neuron. 

Psuedocode 

PerceptronTrainingAlgorithm(Dataset D)
Set wi = 0 for all i = 1....d
Set b = 0 
for iteration 1… MaxIter do:
             Shuffle(D)
	for all (x1, ...xd, y) do:
		a = WTX + b 
                             if (y.a) ≤ 0 then
		   W = W + y.X 
		    b = b + y
return b, W 
PerceptronTest(b, W, X ∈ D ) 
 a = WTX + b
return sign(a)

The first four values in the train data and test data (separated by commas) are feature values for four features. The last element is the class label (class-1, class-2 or class-3). A Binary perceptron has been used to train classifiers to discriminate between 
• class 1 and class 2     • class 2 and class 3          • class 1 and class 3. 
Accuracy has been calculated for the classifiers

Later, the binary perceptron implemented has been extended to perform multi-class classification using the 1-vs-rest approach and accuracy has been calculated. 
A L2 regularisation term has been added to the multi-class classifier.. The classifier was trained for 20 iterations with each of the five regularisation coefficients (0.01, 0.1, 1.0, 10.0, 100.0) and accuracy has been calculated.


