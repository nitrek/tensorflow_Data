import tensorflow as tf
import numpy as np

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

epochs = 1000

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

def train(x):
    prediction = model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range (epochs):
            print 'step: ',epoch
            c = sess.run([optimizer], feed_dict={x:trainX, y:trainY})


        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float' ))
        print('accuracy:', accuracy.eval(session=sess, feed_dict= {x:testX, y:testY}))

        saver = tf.train.Saver()
        saver.save(sess, "trained_model.ckpt")



train(x)
