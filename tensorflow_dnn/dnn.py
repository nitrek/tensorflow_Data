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
print numFeatures
numLabels = trainY.shape[1]

numNodesH1 = 2000

numNodesH2 = 600

numNodesH3 = 80

epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

def model(data):
    tf.set_random_seed(1)
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([numFeatures,numNodesH1])),
    'bias':tf.Variable(tf.random_normal([numNodesH1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([numNodesH1,numNodesH2])),
    'bias':tf.Variable(tf.random_normal([numNodesH2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([numNodesH2,numNodesH3])),
    'bias':tf.Variable(tf.random_normal([numNodesH3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([numNodesH3,numLabels])),
    'bias':tf.Variable(tf.random_normal([numLabels]))}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    #print sess.run(hidden_2_layer['weights'])
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)
    #print sess.run(hidden_3_layer['weights'])
    output = tf.matmul(l3,output_layer['weights']) + output_layer['bias']

#     return output

# def train(x):
    prediction = output
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range (epochs):
            #print sess.run(hidden_1_layer['weights'])
            #print sess.run(hidden_2_layer['weights'])
            print sess.run(hidden_3_layer['weights'])
            
            print 'step: ',epoch
            c = sess.run([optimizer], feed_dict={x:trainX, y:trainY})
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float' ))
            #print('test accuracy:', accuracy.eval(session=sess, feed_dict= {x:testX, y:testY}))
            #print('train accuracy:', accuracy.eval(session=sess, feed_dict= {x:trainX, y:trainY}))

        saver = tf.train.Saver()
        saver.save(sess, "trained_model.ckpt")



model(x)
