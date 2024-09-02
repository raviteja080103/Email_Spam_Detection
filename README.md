                                                              ARTIFICIAL INTELLIGENCE FOR EFFICIENT SPAM AND PHISHING EMAIL CLASSIFICATION

                                                                                                  ABSTRACT
                                                                                                  
Email has become one of the most important forms of communication. In 2014, there are estimated to be 4.1 billion email accounts worldwide, and about 196 billion emails are sent each day worldwide. Spam is one of the major threats posed to email users. In 2013, 69.6% of all email flows were spam. Links in spam emails may lead to users to websites with malware or phishing schemes, which can access and disrupt the receiver’s computer system. These sites can also gather sensitive information from. Additionally, spam costs businesses around $2000 per employee per year due to decreased productivity. Therefore, an effective spam filtering technology is a significant contribution to the sustainability of the cyberspace and to our society. Current spam techniques could be paired with content-based spam filtering methods to increase effectiveness. Content-based methods analyze the content of the email to determine if the email is spam. 
Therefore, this project employs artificial neural networks to detect SPAM, HAM, and Phishing emails by applying features selection algorithm called PCA (principal component analysis). All existing algorithms detected only SPAM and HAM emails, but proposed algorithm designed to detect 3 different classes called SPAM, HAM, and Phishing. To implement this project, we have combined three different datasets called UCI, CSDMC and SPAM ASSASSIN dataset, where UCI and CSDMC datasets provided SPAM and HAM emails and Spam Assassin dataset provided Phishing emails. All these emails were processed to extract important features used in spam and phishing emails such as JAVA SCRIPTS, HTML tags and other alluring URLS to attract users.


                                                                                             EXISTING SYSTEM
                                                                                             
Support Vector Machine Algorithm (SVM):
Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning. The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.
SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine. Consider the below diagram in which there are two different categories that are classified using a decision boundary or hyperplane:
 
Applications:-
	Face recognition
	Weather prediction
	Medical diagnosis
	Spam detection
	Age/gender identification
	Language identification
	Sentimental analysis
	Authorship identification
	News classification

Disadvantages of SVM:-
	Support vector machine algorithm is not acceptable for large data sets.
	It does not execute very well when the data set has more sound i.e. target classes are overlapping.
	In cases where the number of properties for each data point outstrips the number of training data specimens, the support vector machine will underperform.
	As the support vector classifier works by placing data points, above and below the classifying hyperplane there is no probabilistic clarification for the classification.
 
Naïve bayes:
What is the Naive Bayes algorithm?
Naive Bayes algorithm is a probabilistic learning method that is mostly used in Natural Language Processing (NLP). The algorithm is based on the Bayes theorem and predicts the tag of a text such as a piece of email or newspaper article. It calculates the probability of each tag for a given sample and then gives the tag with the highest probability as output.
Naive Bayes classifier is a collection of many algorithms where all the algorithms share one common principle, and that is each feature being classified is not related to any other feature. The presence or absence of a feature does not affect the presence or absence of the other feature.
Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems. ... Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions.
How Naive Bayes works?
Naive Bayes is a powerful algorithm that is used for text data analysis and with problems with multiple classes. To understand Naive Bayes theorem’s working, it is important to understand the Bayes theorem concept first as it is based on the latter.
Bayes theorem, formulated by Thomas Bayes, calculates the probability of an event occurring based on the prior knowledge of conditions related to an event. It is based on the following formula:
P(A|B) = P(A) * P(B|A)/P(B)
Where we are calculating the probability of class A when predictor B is already provided.
P(B) = prior probability of B
P(A) = prior probability of class A
P(B|A) = occurrence of predictor B given class A probability

Disadvantages of Naïve bayes:-
The Naive Bayes algorithm has the following disadvantages:
	The prediction accuracy of this algorithm is lower than the other probability algorithms.
	It is not suitable for regression. Naive Bayes algorithm is only used for textual data classification and cannot be used to predict numeric values.

AdaBoost Algorithm:
What is the AdaBoost Algorithm?
AdaBoost also called Adaptive Boosting is a technique in Machine Learning used as an Ensemble Method. The most common algorithm used with AdaBoost is decision trees with one level that means with Decision trees with only 1 split. These trees are also called Decision Stumps.
 
It is a one of ensemble boosting classifier proposed by Yoav Freund and Robert Schapire in 1996. It combines multiple classifiers to increase the accuracy of classifiers. AdaBoost is an iterative ensemble method. AdaBoost classifier builds a strong classifier by combining multiple poorly performing classifiers so that you will get high accuracy strong classifier. The basic concept behind Adaboost is to set the weights of classifiers and training the data sample in each iteration such that it ensures the accurate predictions of unusual observations. Any machine learning algorithm can be used as base classifier if it accepts weights on the training set. Adaboost should meet two conditions:
	The classifier should be trained interactively on various weighed training examples.
	In each iteration, it tries to provide an excellent fit for these examples by minimizing training error.
 
How does the AdaBoost algorithm work?
It works in the following steps:
	Initially, Adaboost selects a training subset randomly.
	It iteratively trains the AdaBoost machine learning model by selecting the training set based on the accurate prediction of the last training.
	It assigns the higher weight to wrong classified observations so that in the next iteration these observations will get the high probability for classification.
	Also, It assigns the weight to the trained classifier in each iteration according to the accuracy of the classifier. The more accurate classifier will get high weight.
	This process iterates until the complete training data fits without any error or until reached to the specified maximum number of estimators.
	To classify, perform a "vote" across all the learning algorithms you built.

Disadvantages of Adaboost:-
	AdaBoost is sensitive to noise data. It is highly affected by outliers because it tries to fit each point perfectly.


                                                                                              PROPOSED SYSTEM
                                                                                              
This project employs artificial neural networks to detect SPAM, HAM, and Phishing emails by applying features selection algorithm called PCA (principal component analysis). To implement this project, we have combined three different datasets called UCI, CSDMC and SPAM ASSASSIN dataset, where UCI and CSDMC datasets provided SPAM and HAM emails and Spam Assassin dataset provided Phishing emails. All these emails were processed to extract important features used in spam and phishing emails such as JAVA SCRIPTS, HTML tags and other alluring URLS to attract users.

Pre-processing:-
Data pre-processing is a process of preparing the raw data and making it suitable for a machine learning model. It is the first and crucial step while creating a machine learning model.
When creating a project, it is not always a case that we come across the clean and formatted data. And while doing any operation with data, it is mandatory to clean it and put in a formatted way. So, for this, we use data pre-processing task.

    Why do we need Data Pre-processing?
A real-world data generally contains noises, missing values, and maybe in an unusable format which cannot be directly used for machine learning models. Data pre-processing is required tasks for cleaning the data and making it suitable for a machine learning model which also increases the accuracy and efficiency of a machine learning model.
•	Getting the dataset
•	Importing libraries
•	Importing datasets
•	Finding Missing Data
•	Encoding Categorical Data
•	Splitting dataset into training and test set
•	Feature scaling

    Splitting the Dataset into the Training set and Test set:-
In machine learning data pre-processing, we divide our dataset into a training set and test set. This is one of the crucial steps of data pre-processing as by doing this, we can enhance the performance of our machine learning model.
Supposeif we have given training to our machine learning model by a dataset and we test it by a completely different dataset. Then, it will create difficulties for our model to understand the correlations between the models.
If we train our model very well and its training accuracy is also very high, but we provide a new dataset to it, then it will decrease the performance. So we always try to make a machine learning model which performs well with the training set and also with the test dataset. Here, we can define these datasets as:


    Training Set:
A subset of dataset to train the machine learning model, and we already know the output.
Test set: A subset of dataset to test the machine learning model, and by using the test set, model predicts the output.

    Principal Component Analysis (PCA)
Principal component analysis is an approach of machine learning which is utilized to reduce the dimensionality. It utilizes simple operations of matrices from statistics and linear algebra to compute a projection of source data into the similar count or lesser dimensions. PCA can be thought of a projection approach where data with m-columns or features are projected into a subspace by m or even lesser columns while preserving the most vital part of source data. Let I be a source image matrix with a size of n * m and results in J which is a projection of I. The primary step is to compute the value of mean for every column. Next, the values in every column are centered by subtracting the value of mean column. Now, covariance of the centered matrix is computed. At last, compute the eigenvalue decomposition of every covariance matrix, which gives the list of eigenvalues or eigenvectors. These eigenvectors constitute the directions or components for the reduced subspace of J, whereas the peak amplitudes for the directions are represented by these eigenvectors. Now, these vectors can be sorted by the eigenvalues in descending order to render a ranking of elements or axes of the new subspace for I. Generally, k eigenvectors will be selected which are referred principal components or features

    Particle Swarm Optimization (PSO)
Feature selection method is used for generating an optimal number of features to be used for a certain task like classification. Particle Swarm Optimization (PSO) is an algorithm influenced by the habit of bird flocking or fish schooling. PSO is best used to find the maximum or minimum of a function defined on a multidimensional vector space. PSO has a main advantage of having fewer parameters to tune. PSO obtains the best solution from particles' interaction, but through high-dimensional search space, it converges at a very slow speed towards the global optimum. Moreover, regarding complex and large datasets, it shows poor-quality results. This algorithm is that it is easy to fall into local optimum in high-dimensional space and has a low convergence rate in the iterative process.

    CNN Classifier:
According to the facts, training and testing of CNN involves in allowing every source data via a succession of convolution layers by a kernel or filter, rectified linear unit (ReLU), max pooling, fully connected layer and utilize SoftMax layer with classification layer to categorize the objects with probabilistic values ranging from. 
Convolution layer is the primary layer to extract the features from a source image and maintains the relationship between pixels by learning the features of image by employing tiny blocks of source data. It’s a mathematical function which considers two inputs like source image I(x,y,d)  where x and y denotes the spatial coordinates i.e., number of rows and columns. d is denoted as dimension of an image (here d=3 since the source image is RGB) and a filter or kernel with similar size of input image and can be denoted as F(k_x,k_y,d)..
 
    Representation of convolution layer process:
The output obtained from convolution process of input image and filter has a size of C((x-k_x+1),( y-k_y+1),1), which is referred as feature map. Let us assume an input image with a size of 5×5 and the filter having the size of 3×3. The feature map of input image is obtained by multiplying the input image values with the filter values.
 

    ReLU layer:
Networks those utilizes the rectifier operation for the hidden layers are cited as rectified linear unit (ReLU). This ReLU function G(∙) is a simple computation that returns the value given as input directly if the value of input is greater than zero else returns zero. This can be represented as mathematically using the function max(∙)  over the set of 0 and the input x as follows:
G(x)=max⁡{0,x}
Max pooing layer
This layer mitigates the number of parameters when there are larger size images. This can be called as subsampling or down sampling that mitigates the dimensionality of every feature map by preserving the important information. Max pooling considers the maximum element form the rectified feature map.

    Advantages of proposed system:
CNNs do not require human supervision for the task of identifying important features.
They are very accurate at image recognition and classification.
Weight sharing is another major advantage of CNNs.
Convolutional neural networks also minimize computation in comparison with a regular neural network.
CNNs make use of the same knowledge across all image locations.


