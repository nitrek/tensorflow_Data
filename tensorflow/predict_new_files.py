import tensorflow as tf
import os
import numpy as np

def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def import_data():

    print("loading training data")
    trainX = csv_to_numpy_array("trainX.csv", delimiter="\t")
    trainY = csv_to_numpy_array("trainY.csv", delimiter="\t")
    # print("loading test data")
    testX = csv_to_numpy_array("to_predict.csv", delimiter="\t")
    # testY = csv_to_numpy_array("testY.csv", delimiter="\t")
    return trainX,trainY,testX

def labelToString(label):
    # print label, np.argmax(label)
    if np.argmax(label) == 0:
        return "public"
    elif np.argmax(label) == 1:
        return "internal"
    elif np.argmax(label) == 2:
        return "restricted"
    else:
        return "highly restricted"

if __name__ == "__main__":

	test_dir = "source_files_to_test"

	filenames = os.listdir(test_dir)

	trainX,trainY,testX = import_data()

	numFeatures = trainX.shape[1]
	numLabels = trainY.shape[1]


	sess = tf.Session()
	X = tf.placeholder(tf.float32, [None, numFeatures])
	# yGold = tf.placeholder(tf.float32, [None, numLabels])

	tf.set_random_seed(22)
	weights = tf.Variable(tf.random_normal([numFeatures,numLabels],
                                       mean=0,
                                       stddev=(np.sqrt(6/numFeatures+
                                                         numLabels+1)),
                                       name="weights"))
	tf.set_random_seed(22)
	bias = tf.Variable(tf.random_normal([1,numLabels],
                                    mean=0,
                                    stddev=(np.sqrt(6/numFeatures+numLabels+1)),
                                    name="bias"))
	init_OP = tf.initialize_all_variables()

	activation_OP = tf.nn.softmax(tf.matmul(X, weights) + bias)
	sess.run(init_OP)
	saver = tf.train.Saver()
	saver.restore(sess, "trained_variables.ckpt")

	prediction = sess.run(activation_OP, feed_dict={X: testX})
	# print prediction
	print "Regression predicts:\n"
	for i in range(len(testX)):
		# print("regression predicts %s to be %s" %(str(i + 1), labelToString(prediction[i])))
		print("%s to be %s" %(filenames[i], labelToString(prediction[i])))
	sess.close()