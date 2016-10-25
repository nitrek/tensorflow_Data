import tensorflow as tf
import numpy as np

#Data sets
training_data = "trainX_new.csv"
testing_data = "testX_new.csv"

#Load datasets
training_set = tf.contrib.learn.datasets.base.load_csv(filename=training_data,target_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=testing_data,target_dtype=np.float32)
# print dir(tf.contrib.learn.datasets.base.load_csv)
print training_set
print test_set

# Build 3 layer DNN with 10, 20, 10 units respectively.
#classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=4)
# Fit model
#classifier.fit(x=x_train, y=y_train, steps=200)
#accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
#print('Accuracy: {0:f}'.format(accuracy_score))

