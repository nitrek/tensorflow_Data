import tensorflow as tf
import os
import numpy as np

def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def import_data():

    print("loading training data")
    trainX = csv_to_numpy_array("../tensorflow/trainX1.csv", delimiter="\t")
    trainY = csv_to_numpy_array("../tensorflow/trainY.csv", delimiter="\t")
    # print("loading test data")
    testX = csv_to_numpy_array("../tensorflow/to_predict.csv", delimiter="\t")
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

def model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([numFeatures,numNodesH1])),
    'bias':tf.Variable(tf.random_normal([numNodesH1]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([numNodesH1,numLabels])),
    'bias':tf.Variable(tf.random_normal([numLabels]))}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    output = tf.matmul(l1,output_layer['weights']) + output_layer['bias']

    return output


if __name__ == "__main__":

	test_dir = "../tensorflow/source_files_to_test"

	filenames = os.listdir(test_dir)

	trainX,trainY,testX = import_data()

	numFeatures = trainX.shape[1]
	numLabels = trainY.shape[1]
   	numNodesH1 = 2000


	sess = tf.Session()
	x = tf.placeholder(tf.float32, [None, numFeatures])
	y = tf.placeholder('float')



   	init_OP = tf.initialize_all_variables()

	prediction = model(x)
    	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
   	optimizer = tf.train.AdamOptimizer().minimize(cost)

    	sess.run(init_OP)

	saver = tf.train.Saver()
	saver.restore(sess, "trained_model.ckpt")

	#prediction = sess.run(optimizer, feed_dict={x: testX})
	prediction = sess.run(prediction, feed_dict={x:testX})
	# print prediction
	print "model predicts:\n"
	for i in range(len(testX)):
		print("%s to be %s" %(filenames[i], labelToString(prediction[i])))
	sess.close()
