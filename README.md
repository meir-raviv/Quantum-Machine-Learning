# Quantum-Machine-Learning
This project represents a 'Classification and data re-upload in parametric quantum circuits (PQC)' - Comparison of the performance of the classifiers presented in the papers 'Data Re-uploading for a Universal Quantum Classifier' and 'Distance-based Classifier with a Quantum Interference Circuit' by implementing and testing both approaches.
Data re-uploading and Interference quantum circuits

# Data re-uploading and Interference quantum circuits

### Abstract:

This article introduces quantum analysis of 2 different methods used in Quantum machine learning. It compares them and shows results of implementations of them both on the known Iris data-set.

### •	Methodology:

We based our work mainly on the articles Data re-uploading for a universal quantum classifier (Barcelona) and Implementing a distance-based classifier with a quantum interference circuit (South Africa). In addition, we managed to absorb a lot of information from the lectures Quantum Machine Learning given in Bar-Ilan University by Professor Adi Makmal.
The implementations of the articles are in Jupiter notebooks, and they contain use of Pennylane and Qis-kit libraries. We run them on real quantum hardware provided freely by IBM (in Santiago).
Both articles suggest efficient ways to solve the machine learning classification task, taking advantage of the prosperity of the quantum computation techniques.

### •	Discussions – what can we take from this paper?

We hope to give the reader a good background on the classification problem of machine learning, and provide 2 different approaches to solve it quantumly. We implement and analyze both approaches and discuss their advantages and disadvantages.

Introduction
Machine learning has taken a big role in our life in the recent years. It has grown exponentially and achieved high accuracy in many different Artificial Intelligence tasks. Today, classical machine learning is a really diverse, constantly improving field. Hence, implementations of it within the quantum computing domain are inevitable. In this article we present two different approaches implemented in quantum machine learning. One is data re-uploading for a universal quantum classifier and the other is distance-based classifier with a quantum interference circuit.
The data re-uploading method is based on the well-known classical neural-network system which is based on the structure of the biological neurons. The distance-based method is based on kernel functions systems rather than classical machine learning algorithm. Later on, we will explain each method more widely.
The idea of machine learning is giving the computing unit large set of training data and using unique mathematical tools it gets to analyze new test data properly. In our task we are supposed to execute classification. Classification means that each data tuple is labeled with some class label, and the computer should give the correct label for the test data. 

The Iris data set
In this article we show and compare implementation of both methods. Our implementation was made on the famous Iris data set which we explain in a few words and is reachable easily through any python platform.
The Iris data set consists of 150 tuples where each one belongs to one out of three species of iris, and it was introduced by Ronald Fisher:

![image](https://user-images.githubusercontent.com/95282051/144718424-b0eaedbe-7b01-41b3-8f87-a0dfaac31478.png)

The samples were taken in the same day and same place by the same person. Each tuple has 4 attributes: Sepal length, Sepal width, Petal length, Petal width. Originally, Fisher made LDA (linear discriminant analysis) process on the data set. The data set is easily accessible through python platforms. After analyzing the data, we discovered that there are 3 different classes (3 different species of the Iris flower), and that 2 classes are more similar while the third one is much different. It can be shown from the graph we produced:

![image](https://user-images.githubusercontent.com/95282051/144718451-cb9059bf-d0de-4ad3-bfaa-907b37d4f8aa.png)


The red class is more isolated than the other two.
It is important to notice that the Iris data set is very small, and classification made on it might be a little biased; usually machine learning algorithms are conducted on very large data sets.
Each article will give us different implementation of classification algorithm that classify the different classes in the Iris data set.

### Research questions:

	Which model will perform better – the neural-network model or the distance-based one?
	What is the expected quantum gain of each model?
	How much is each model sensitive to quantum noise?

### Answers:

  •	We assumed that the distance-based model will perform better since it might avoid barren plateaus and it is very simple. In fact, it did perform better when the classes were more separable, but when the classes were more overlapped it had trouble separating them and finding the true closest class.
	
  •	The quantum gain of the classifiers is mainly the number of qubits required to make classification. The neural network demands only 1 qubit and the distance classifier requires only 4. Compared to classical machine learning classifiers - there we need hundreds and thousands of bits in order to represent the circuit (neural network or the distance function). The step of optimization is same in the classical part of the neural network. On the other hand, the process of getting the hidden layer outputs which can be made in parallel classically, is calculated in sequent in the data re-uploading classifier. We lose part of the gain by the process of data uploading into the circuit which takes more time to encode, but with Q-RAM it can take only O(log⁡N). In terms of performance, we see that the quantum classifiers are not falling from the classical ones, and even get better results.
	
  •	As we discuss later, we find the distance model much more sensitive to noise than the data re-uploading classifier.
  
### Discussion

The classification problem is a wide and well-studied one, with a lot of classical solutions for it. Trying to approach it from a quantum perspective might bring us better solutions and performances. In our opinion, we believe that quantum computation might bring us ways to operate classification in very low time and space complexity, and the results will be much more accurate. It will get better as the number of qubits provided by quantum machines will rise, and when the noise of these machines will strive to zero. Unfortunately, today we are still in the beginning of quantum computation technology, and we believe that in about 15 years the hardware and software of the quantum computation field will reach our estimations.



## Presentation of both methods
### Data re-uploading for a universal quantum classifier

The data re-uploading circuit belongs to the section of variational quantum circuits, in which the computation is divided into 2 parts:
	•	The quantum circuit in which the data is computed
	•	A classical part in which parameters are optimized
The process alternates between both parts and optimizes the parameters until a good result is achieved (minimum of a cost function).
 
#### A brief explanation on neural network:
In classical machine learning, one advanced and well-developed method is the Neural-Network method. It is based on the biological neurons structure. The neurons in nature have several inputs and outputs. They are connected between them, and each neuron receives several pulses from its inputs and decides accordingly if to send a pulse in its output. Similar to this idea we can construct a graph where each node has the role of a neuron, and each edge connects 2 neurons. Each node has several inputs and several outputs, and its output is the result of an activation function which receives its input from the edges that go into the node. Common activation functions are the Sigmoid and ReLU. The neural network is structured like a graph with layers, where each layer has several neurons:

![image](https://user-images.githubusercontent.com/95282051/144718551-3a22fe21-9740-436b-be3d-167cbe23c646.png)

The input layer gets the data as input, and the hidden layers get the result of the previous layer as an input, while each neuron has its own weight vector such that it gets weighted (sum) input. The w vector components are called the parameters of the network, and that’s what we want to bring to the ideal state, and what will help us classify future data. The output layer in the end has output which gives us the classification of the input vector (for example, with the Sigmoid we classify class {-1} for result < 0.5 and we classify class {1} for result > 0.5). The learning of the network is often using the Gradient Decent Method which uses back propagation. It means that we find any partial deviation of the weight vector parameters using the chain rule and optimize each parameter by going slowly in the direction of the gradient (towards the minimum of the function we optimize). The function of the network which we optimize is called the Cost Function. It is usually declared with the weights and inputs, and our mission is to minimize it. For example, we can use mean square error as loss (cost) function. One main disadvantage of the neural network is that	it might have nonconvex cost function and we can get stuck in local minima. 
From classical to quantum neural network
The idea of data re-uploading method is taking the classical neural-network idea and turning it into the quantum computation world. In our case, we discuss a classical neural network with only one hidden layer. Let’s assume that this layer has N neurons. In this quantum circuit we use only one qubit. The structure of the circuit is by taking the N neurons hidden layer, and each neuron is turning into Rotation gate (U) which its inputs are the weights and the input vector:

![image](https://user-images.githubusercontent.com/95282051/144718610-b97e1b32-37cf-4e91-b56d-4abc76797a1a.png)

In our case, X is the input vector and ∅ are the weights. Each layer here is originally a neuron in the classical neural network. As we can see the encoding of the data in our case is by the Rotation gate (using the angle of the vector), which has 3 angles as input. If the input vector has more than 3 features, we can split it and load it in quantum of 3 features each time. So, for N neurons network we will need to have 2N gates. In order to improve that we will load in each layer X and ∅ together and need only N gates:

![image](https://user-images.githubusercontent.com/95282051/144718617-764a4fdc-dea3-4d92-b448-59e252311c59.png)

Where,

![image](https://user-images.githubusercontent.com/95282051/144718626-50cbe680-9463-4933-91fd-0aaa5594570b.png)

Using the Hadamard product - 
![image](https://user-images.githubusercontent.com/95282051/144718638-b053957f-4a2d-4acb-afbd-66d47a5960b2.png)

θ is the ‘bias’ and ω are the weights.
Summing up all we get:

![image](https://user-images.githubusercontent.com/95282051/144718657-58aef103-11f9-41e2-8af4-9ef3a2f1c1cb.png)

A big main problem is that we need to load the data into the circuit what might influence the performance. Another problem is an outcome of the no cloning theorem which says that it is impossible to clone the quantum state of the data X (encoded), so we need to re-upload it again in every layer.
Classification
After presenting the structure of the circuit, we want to define the classification of it. There are 2 options of classification: one is simple and is dividing the range of the probabilities of the |0> qubit state’s result into |C| parts, each one defining a class. The second one which is preferable by the authors of the article is such that we declare |C| different states as the number of classes, while choosing them to be as more as orthogonal and separable in the Bloch Sphere (in the 2-class case we have [0, 1] and [1, 0]). Examining the state we get at the end of the circuit we check to which state we collapse with the highest fidelity, and this is the classification result for the input. For example, in 4 or 6-classes case we can use:

![image](https://user-images.githubusercontent.com/95282051/144718811-4ed5ab9f-49c3-4fe1-aa60-26bcf91e9de1.png)

Learning process and cost function
The learning process is very similar to the classical neural network, using gradient decent. We used Adam Optimizer (adaptive moment) as our optimizer, which worked well. The cost function we used is very simple – we checked the state collapse at the end of the circuit for all the inputs, each input considered with its correct class state:

![image](https://user-images.githubusercontent.com/95282051/144718817-24ab455a-6234-4a99-bcb0-10bba833c3c7.png)

When μ runs on the inputs, and psi-s is the correct class state for the input. The highest fidelity is being sought when optimizing. In the quantum computation terms we define barren plateaus as local minima.

### Getting higher accuracy
Furthermore, in the article they offer more options of using 2 and even 4 qubits, and using weighted fidelity cost function, but we didn’t need that, and we achieved high accuracy anyway.

### Distance-based classifier with a quantum interference circuit
As mentioned above, the distance-based classifier uses a kernel function in order to make classification. It means that we have a function that finds the distance between our test vector from each of 2 classes and it classifies it as the closest class. The kernel classification function is:

![image](https://user-images.githubusercontent.com/95282051/144718843-55f66c95-1276-4b1b-b958-30f1fac09d9c.png)

Where x^m is the m^th train vectors input, y^th is its label, M is the total number of train inputs, x ̃  is the test vector and y ̃ is the classification.
In this classifier we encode the data with the amplitude encoding method, which gives us exponential efficiency. We encode the features of the data to hold in the amplitudes of the quantum state (of |0> and|1>). In order to encode data into the amplitudes we use RY gate (rotate in y angle) which gets theta:

![image](https://user-images.githubusercontent.com/95282051/144718851-55a13abb-a46b-4c81-87b6-e01501247d4b.png)

The circuit has 4 qubits:

|m> is the index register which runs on the train vector.

|a> is an ancilla qubit that is entangled twice with the third qubit 

|i> - once with the training input state and once with the test state. 

|c>  - is encoded with class true state (0 or 1).

The circuit has 2 parts, in which the amplitude encoding represents the kernel function. When calculating the probability of the |0> qubit at |i> at the end of the circuit, we get the expression of the kernel distance function (![image](https://user-images.githubusercontent.com/95282051/144718914-e9c98b8c-e781-46c4-9424-a3012259dc58.png)
).
First, we have a preparation state:

![image](https://user-images.githubusercontent.com/95282051/144718933-ebb4ded7-b612-4ccb-9006-7ca575275e0e.png)

And after that we activate Hadamard and measures, and achieve:

![image](https://user-images.githubusercontent.com/95282051/144718950-7c3a7a92-2cbd-4a21-9218-1c511028b899.png)

Soon we will show the structure of the circuit implementing these states. But first we show necessary preparations that are required to be done before loading data into the circuit.
Our data is very biased and not centralized and therefore before using it we conduct 2 operations:
Standardization - which means we shift the data to have zero mean and unit vector variance, and Normalization - which is having each data to have unit vector size (with the same angle):

![image](https://user-images.githubusercontent.com/95282051/144718958-01ecbf52-c039-4b7d-a2fb-25388d5e84b0.png)

Later we produce these graphs.
Circuit Implementation
The circuit implementation is:

![image](https://user-images.githubusercontent.com/95282051/144718966-6eab6eb4-a998-4902-8d06-c8dedfc2c513.png)

Where stages A-E are the state preparation, and stage F is the measures.
|c> is conditional measured – only if qubit |a> was 0.
This figure is taken from the example given in the article and we made the changes appropriately in the general circuit for each new data point. 

In this classifier we load each time 2 training points – one for each class and one test point. Due to hardware constraints we had to use only test points of class {-1}. Each vector has only 2 attributes, so we had to split the Iris data set each time and use only 2 features. A wider explanation will be given later.

It is important to notice that in this classifier we load the data only once, and no changes in the circuit are made – we only conduct a major number of measurements to achieve correct distribution.



### Circuit implementation
####   Explanation of the comparison process
Our main mission was to implement both methods suggested by the articles on the Iris data set and compare them to find the best performing one. In order to make an appropriate comparison we have decided to act as follows:
The distance-based classifier requires a 2-D input vector. So, we have split the data set into sets of 2-D vectors accordingly:
Each tuple has 4 features, which means we have 6 (=3!) different permutations of couples (with no returns). Each time we made classification between 2 classes. For each ‘contest’ we conducted classification 6 times – for each permutation. Each classification gained 60% of the data as training set and 40% as test set. In the data re-uploading algorithm, we used ‘mini-batches and epochs’ method. We trained the data 10 times. In the distance classifier we made 50 permutations which in each one we randomly picked 2 training points and one test point. Each permutation we ran on the Aer simulator for 8192 times to get appropriate distribution. We calculated the averaged accuracy of the 6 different classifications.
 In the bottom line, we made the same classification task for both classifiers. In addition, we provide examples of running on real IBM quantum hardware, and results that match the examples in the article are produced.
In the Appendix we add more results of the multi-class classification and more graphs that we produced.




## Results
First, we add some tables we produced to show implementations similar to those the articles suggest.
The distance classifier first requires data pre-processing as we showed above. In our implementation:

![image](https://user-images.githubusercontent.com/95282051/144719010-27ad675c-0645-4a96-a797-c0b00ed3d24d.png)

Then we provide the circuit:

![image](https://user-images.githubusercontent.com/95282051/144719016-e05c8e2f-fffd-4a78-978e-df65df7a808e.png)

And the simulator results:

![image](https://user-images.githubusercontent.com/95282051/144719029-ef352ed2-a240-4e29-acfd-2738ce3efdd2.png)

The IBMQ hardware results:

![image](https://user-images.githubusercontent.com/95282051/144719035-a78e1838-4ddb-483a-9eb5-3c0f06793afb.png)

These results match in high accuracy the ones of the article (first line is IBMQ hardware and second line is simulation):

![image](https://user-images.githubusercontent.com/95282051/144719041-b830fea8-719d-4243-a25a-f59ff5b6e011.png)



### •	Outcomes
In both classifiers we managed to achieve very high performances when comparing 2 far classes. Only with class 2 vs class 3 we had more difficulties. Our results:

#### Data re-uploading classifier:
•	Class 1 vs 2: 98% train accuracy and 98% test accuracy
•	Class 1 vs 3: 96% train accuracy and 96% test accuracy
•	Class 2 vs 3: only 85% train accuracy and 85% test accuracy
#### Distance based classifier:
•	Class 1 vs Class 2 achieved total of 97% accuracy int the classification
•	Class 1 vs Class 3 achieved total of 100% accuracy in the classification
•	Class 2 vs Class 3 achieved total of 81% accuracy in the classification

As we can see, when class 1 was involved, we achieved much higher accuracy. But between class 2 and 3 we had more difficulties classifying them in both implementations. When the classes are far the distance classifier performed very well. But with classes 2 and 3 it had more difficulties separating them.


### Comparison between both methods

In order to compare both methods we want to assert that the better algorithm to use is highly reliable on the data we use. 

•	Sensitivity to barren plateaus

The data re-uploading method seemed to be very influenced by the initialization of the parameters. Some of the suns we made just got ‘stuck’ in local minima. This is a huge disadvantage. However, the distance classifier has no parameters dependency, and it did very well with the first attempts without any need to find the best initialization conditions. 

•	Sensitivity to the ‘class relations’

In this case the distance classifier was much more reliable on the data behavior. If the classes were highly separable it managed to get even to 100% of classification. But when they didn’t - it classified poorly. The data re-uploading algorithm performed very high in this case, even relatively high when the classes overlapped.

•	Space complexity

Here, the data re-uploading algorithm gave us the best improvement – only 1 qubit needed!
The distance classifier required 4 qubits.

•	Time complexity of each circuit

The data re-uploading algorithm time is O(N) where N is the number of layers (neurons). We chose N = 3, and higher N can achieve better performances. Of course, the learning process is slow and repeats itself for a high number of iterations.
The distance classifier time complexity is O(N) where N = 28, meaning it has 28 gates in each layer (some of them parallel). Here N is constant for all the circuits. 

•	Noise resistance

It is better to use as few qubits as we can in order to avoid noises of the real quantum machine. So here also the data re-uploading classifier was more resistant to the noises than the distance classifier.

## Conclusions
In general, we can conclude that the data re-uploading classifier has more advantages, and it performs better. Indeed, we will need to examine each case for itself and decide which classifier is better to use.
We believe that the quantum machine learning world is going to ascent high and bring us a lot of new algorithms and perspectives on the region of machine learning, and computation in general. Observing the huge advantage of the classification task we made, we estimate that in the very near future, the quantum computation field will find a solution to the highly discussed question of P and NP problems, and the world widely used RSA encryption algorithm will need a new quantum version since the classical one will permeable using quantum computers.


### Acknowledgment
This work was made during a course of Quantum machine learning given in Bar-Ilan university in Ramat-Gan, Israel. 

### References

[1] A. Perez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, J. I. Latorre, Data re-uploading for a universal quantum classifier

[2] M. Schuld, M. Fingerhuth & F. Petruccione, Implementing a distance-based classifier with a quantum interference circuit

[3] A. Makmal, Quantum machine learning course given in Bar-Ilan University

[4] Wikipedia, Iris flower data set

Appendix
We add here some graphs we produced to show the implementation of the code.
The data re-uploading classifier learning process:

![image](https://user-images.githubusercontent.com/95282051/144719100-02d7c8e8-3ce1-4347-b858-3d8c3fa2a015.png)


And cost function:

![image](https://user-images.githubusercontent.com/95282051/144719107-3c969187-d811-4437-afa6-41ea63095248.png)

The distance-based classifier classification process:

![image](https://user-images.githubusercontent.com/95282051/144719117-78db053b-a857-44c4-bf24-7dbad497397b.png)

![image](https://user-images.githubusercontent.com/95282051/144719118-203f2c6f-e34a-479e-b8bc-521b41d4d787.png)

Multi class classifier
In addition we provide classification task made with the data re-uploading classifier with the 3-classes and all 4 attributes altogether. The circuit as suggested in the article consists of double the size of the layers. Each tuple was split into two set of 2 attributes padded with a zero, and each layer has 2 Rotation gates – one for each set of the tuple. 
The states of the classes are 3:
{ [1, 0], [0.5,√3/2], [0.5,-√3/2]}
In this task we didn’t seek for all the permutations of couples of features since we classified them together.
Everything else remains the same. 
Here are the results.
The circuit structure:

![image](https://user-images.githubusercontent.com/95282051/144719132-83215db7-feaa-45f8-b85e-505ed9fd8f58.png)

The learning and fidelity cost function:

![image](https://user-images.githubusercontent.com/95282051/144719139-b2d324cd-af0d-4f00-9f8e-fe8baa971e33.png)

We achieved here also high accuracy (95% on the test set) even though this task is much difficult, since we need to classify between 3 classes this time. 






