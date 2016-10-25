import numpy as np
import tensorflow as tf
import tarfile
import os

def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def import_data():
    print("loading training data")
    trainX = csv_to_numpy_array("../tensorflow/trainX1.csv", delimiter="\t")
    trainY = csv_to_numpy_array("../tensorflow/trainY.csv", delimiter="\t")
    print("loading test data")
    testX = csv_to_numpy_array("../tensorflow/testX1.csv", delimiter="\t")
    testY = csv_to_numpy_array("../tensorflow/testY.csv", delimiter="\t")
    return trainX,trainY,testX,testY


trainX,trainY,testX,testY = import_data()


numFeatures = trainX.shape[1]
numLabels = trainY.shape[1]
numNodesH1 = 2000


#create a tensorflow session
sess = tf.Session()


x = tf.placeholder('float')
y = tf.placeholder('float')

def model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([numFeatures,numNodesH1])),
    'bias':tf.Variable(tf.random_normal([numNodesH1]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([numNodesH1,numLabels])),
    'bias':tf.Variable(tf.random_normal([numLabels]))}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    output = tf.matmul(l1,output_layer['weights']) + output_layer['bias']

    return output

init_OP = tf.initialize_all_variables()

prediction = model(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
optimizer = tf.train.AdamOptimizer().minimize(cost)



correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float' ))


sess.run(init_OP)
print testX.shape

saver = tf.train.Saver()
saver.restore(sess, "trained_model.ckpt")


def labelToString(label):
    print label, np.argmax(label)
    if np.argmax(label) == 0:
        return "public"
    elif np.argmax(label) == 1:
        return "internal"
    elif np.argmax(label) == 2:
        return "restricted"
    else:
        return "highly restricted"


if __name__ == "__main__":

    evaluation = sess.run(accuracy, feed_dict={x: testX, y: testY})
    prediction = sess.run(prediction,  feed_dict={x: testX, y: testY})

    for i in range(len(testX)):
        print("regression predicts %s to be %s and is actually %s" %(str(i + 1), labelToString(prediction[i]), labelToString(testY[i])))
    print("overall accuracy of dataset: %s percent" %str(evaluation))
