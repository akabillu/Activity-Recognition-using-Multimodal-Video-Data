# Activity-Recognition-using-Multimodal-Video-Data

Introduction:
Human activity recognition (HAR) is a research domain that aims to identify and classify human activities from diverse sources of data, such as video, audio, inertial sensors, or wearable devices. HAR has many applications in health care, security, sports, smart environments, and entertainment. HAR has been studied extensively in the past decades. However, most of the studies focused on single-modal data, which may not capture the complementary information of human activities. Therefore, multiple methods proposed to use multimodal data, which can fuse different types of data to enhance the performance and robustness of HAR. In this project, we attempt to recognize human activities using a simple 1D-convolution neural network and have achieved reasonable performance.

## Data Preprocessing:
The data collection was done by using the VIDIMU dataset, which is a publicly available dataset that contains multimodal data of 13 daily life activities performed by 54 subjects, recorded using a commodity camera and five inertial sensors. 
The data preprocessing was done by using pandas library. The steps were:
1.	Loading and concatenating the processed data files for each subject and activity into a single dataframe using pandas.read_csv() and pandas.concat().
2.	Filtering out subjects S43 and S45 due to unavailability in the dataset.
3.	Splitting the dataframe into features (X) and labels (Y) using pandas.drop().
4.	Reshaping the features to fit the input shape of the CNN model using .reshape().
5.	Converting the labels to one-hot vectors using keras.utils.to_categorical().
6.	The data is split into training and testing sets, with 80% for training and 20% for testing.

## Model Architecture and training:
The model building was done by using Keras. The hyperparameters of the model are fined tuned after conducting number of experiments. The steps are:
1.	Defining a CNN model using keras.Sequential().
2.	Adding two convolutional layers with relu activation using keras.layers.Conv1D().
3.	Adding two max pooling layers using keras.layers.MaxPooling1D().
4.	Adding a flatten layer using keras.layers.Flatten().
5.	Adding a dropout layer with 0.2 rate using keras.layers.Dropout().
6.	Adding a dense layer with softmax activation using keras.layers.Dense().
7.	Compiling the model with categorical crossentropy loss, Adam optimizer with 0.001 learning rate.
8.	The training data is used to train the model over a set number of epochs, with a specified batch size. Training progress is displayed, showing the loss and accuracy values at each epoch.

## Model Evaluation:
The model evaluation was done by using scikit-learn.
1.	Splitting the data into training and testing sets with 0.2 test size and 42 random state using sklearn.model_selection.train_test_split().
2.	Fitting the model on the model on the training data using keras.Model.fit().
3.	Predicting the labels for the testing data using keras.Model.predict().
4.	Converting the one-hot vectors to integers using .argmax().
5.	Calculating performance metrics, such as accuracy, precision, recall, and F1-score using sklearn.metrics.accuracy_score(), sklearn.metrics.precision_score(), sklearn.metrics.recall_score() , and sklearn.metrics.f1_score().

## Results:
The 1D_CNN.ipynb jupyterlab file is attached with this project report. The trained CNN model achieved the following performance metrics on the testing data:

### Evaluation Metric	Performance/Score
Accuracy	66.14%
Precision	73.04%
Recall	69.79%
F1-score	69.79%

## Conclusion and future work:
The project used the VIDIMU dataset to build and evaluate a CNN model for human activity recognition. The results demonstrates that the CNN model was able to recognize most of the activities in the VIDIMU dataset with reasonable accuracy. The project also demonstrated how to use Python tools for data preprocessing, data analysis, model building and model evaluation. The project contributed to the learning of data analysis and machine learning skills, as well as to the exploration of the potential of multimodal data for HAR. The project also contributed to the development of affordable and effective patient tracking solutions and diagnosis.
The VIDIMU dataset is a relatively small dataset, with only 54 subjects and 13 activities. Using more data can help to increase the diversity and variability of human activities, as well as to improve the generalization and robustness of the model. Furthermore, in our experiments we used a simple CNN model with two convolutional layers, two max pooling layers, a dropout layer, and a dense layer. Using more advanced methods and models, such as RNNs, attention mechanisms, or transfer learning, can help to learn more complex and dynamic patterns of human activities, as well as to achieve higher performance.

