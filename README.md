1. Overview of the analysis
This project includes Jupyter Notebook files that build, train, test, and optimize a deep neural network that models charity success from nine features in a loan application data set. 
We employ the TensorFlow Keras Sequential model with Dense hidden layers and a binary classification output layer and optimize this model by varying the following parameters:
-Training duration (in epochs)
-Hidden layer activation functions
-Hidden layer architecture
-Number of input features through categorical variable bucketing
-Learning rate
-Batch size

2. Resources
* Data Source:
-charity_data.csv
* Software:
-Python 3.7.6
-scikit-learn 0.22.1
-pandas 1.0.1
-TensorFlow 2.4.1
-NumPy 1.19.5
-Matplotlib 3.1.3
-Jupyter Notebook 1.0.0

3. Results
First preprocess our data set charity_data.csv by reading our data and noting the following target, feature, and identification variables:
* Target Variable: IS_SUCCESSFUL
-Feature Variables: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT
-Identification Variables (to be removed): EIN, NAME
-We then encode categorical variables using sklearn.preprocessing.OneHotEncoder after bucketing noisy features APPLICATION_TYPE and CLASSIFICATION with many unique values. 
-After one hot encoding, we split our data into the target and features, split further into training and testing sets, 
and scale our training and testing data using sklearn.preprocessing.StandardScaler.

4. Compiling, Training, and Evaluating the Model
-With our data preprocessed, we build the base model defined in AlphabetSoupCharity.ipynb 
-using tensorflow.keras.models.Sequential and tensorflow.keras.layers.Dense with the following parameters:
-Number of Hidden Layers : 2 layer "Deep neural network is necessary for complex data, good starting point with low computation time."
-Architecture (hidden_nodes1, hidden_nodes2) : (80,30) "First layer has roughly two times the number of inputs (43), smaller second layer offers shorter computation time."
-Hidden Layer Activation Function : relu "Simple choice for inexpensive training with generally good performance."
-Number of Output Nodes : 1 "Model is a binary classifier and should therefore have one output predicting if IS_SUCCESSFUL is True or False."
Output Layer Activation Function : sigmoid "Provides a probability output (value between 0 and 1) for the classification of IS_SUCCESSFUL."
-This yields the model summary shown in Base Model Summary. 
-We then compile and train the model using the binary_crossentropy loss function, adam optimizer, and accuracy metric to obtain the training results shown in Base Model Training. 
-Verifying with the testing set, we obtain the following results:
Loss: 0.559
Accuracy: 0.729
-We next optimize the previous model by adjusting the parameters shown above and more in AlphabetSoupCharity_Optimization.ipynb, initially making the following single changes:
-Training Duration (epochs) : Increase from 100 to 200 "Longer training time could result in more trends learned." Loss 0.582 Accuracy 0.728


