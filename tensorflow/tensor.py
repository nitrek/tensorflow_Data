import tensorflow as tf
import numpy as np

def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def import_data():
    print("loading training data")
    trainX = csv_to_numpy_array("trainX.csv", delimiter=",")
    trainY = csv_to_numpy_array("trainY.csv", delimiter=",")
    print("loading test data")
    testX = csv_to_numpy_array("testX.csv", delimiter="\t")
    testY = csv_to_numpy_array("testY.csv", delimiter="\t")
    return trainX,trainY,testX,testY

trainX,trainY,testX,testY = import_data()


numFeatures = trainX.shape[1]

numLabels = trainY.shape[1]

numEpochs = 27000

learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step= 1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)


X = tf.placeholder(tf.float32, [None, numFeatures])

yGold = tf.placeholder(tf.float32, [None, numLabels])

weights = tf.Variable(tf.random_normal([numFeatures,numLabels],
                                       mean=0,
                                       stddev=(np.sqrt(6/numFeatures+
                                                         numLabels+1)),
                                       name="weights"))

bias = tf.Variable(tf.random_normal([1,numLabels],
                                    mean=0,
                                    stddev=(np.sqrt(6/numFeatures+numLabels+1)),
                                    name="bias"))

init_OP = tf.initialize_all_variables()

apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")

training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)
