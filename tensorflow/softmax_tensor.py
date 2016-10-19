from __future__ import division
import tensorflow as tf
import numpy as np
import tarfile
import os
import matplotlib.pyplot as plt
import time

def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def import_data():
    print("loading training data")
    trainX = csv_to_numpy_array("trainX1.csv", delimiter="\t")
    trainY = csv_to_numpy_array("trainY.csv", delimiter="\t")
    print("loading test data")
    testX = csv_to_numpy_array("testX1.csv", delimiter="\t")
    testY = csv_to_numpy_array("testY.csv", delimiter="\t")
    return trainX,trainY,testX,testY

trainX,trainY,testX,testY = import_data()


numFeatures = trainX.shape[1]

numLabels = trainY.shape[1]

numEpochs = 2000

# learningRate = tf.train.exponential_decay(learning_rate=0.0008,
#                                           global_step= 1,
#                                           decay_steps=trainX.shape[0],
#                                           decay_rate= 0.95,
#                                           staircase=True)


X = tf.placeholder(tf.float32, [None, numFeatures])

yGold = tf.placeholder(tf.float32, [None, numLabels])
tf.set_random_seed(100)
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

# apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
# add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
# activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

activation_OP = tf.nn.softmax(tf.matmul(X, weights) + bias)

# cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")

cost_OP = tf.reduce_mean(-tf.reduce_sum(yGold * tf.log(activation_OP), reduction_indices = [1]))

training_OP = tf.train.GradientDescentOptimizer(0.05).minimize(cost_OP)

# training_OP = tf.train.GradientDescentOptimizer(0.5).minimize(cost_OP)


epoch_values=[]
accuracy_values=[]
cost_values=[]
# Turn on interactive plotting
plt.ion()
# Create the main, super plot
fig = plt.figure()
# Create two subplots on their own axes and give titles
ax1 = plt.subplot("211")
ax1.set_title("TRAINING ACCURACY", fontsize=18)
ax2 = plt.subplot("212")
ax2.set_title("TRAINING COST", fontsize=18)
plt.tight_layout()

sess = tf.Session()
sess.run(init_OP)
print testX.shape, testY.shape
## Ops for vizualization
# argmax(activation_OP, 1) gives the label our model thought was most likely
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))

accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

# Summary op for regression output
activation_summary_OP = tf.histogram_summary("output", activation_OP)

# Summary op for accuracy
accuracy_summary_OP = tf.scalar_summary("accuracy", accuracy_OP)

# Summary op for cost
cost_summary_OP = tf.scalar_summary("cost", cost_OP)

# Summary ops to check how variables (W, b) are updating after each iteration
weightSummary = tf.histogram_summary("weights", weights.eval(session=sess))
biasSummary = tf.histogram_summary("biases", bias.eval(session=sess))

# Merge all summaries
all_summary_OPS = tf.merge_all_summaries()

# Summary writer
writer = tf.train.SummaryWriter("summary_logs", sess.graph_def)

cost = 0
diff = 1

# Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence."%diff)
        break
    else:
        step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
        if i % 10 == 0:
            epoch_values.append(i)
            summary_results, train_accuracy, newCost = sess.run(
                [all_summary_OPS, accuracy_OP, cost_OP], 
                feed_dict={X: trainX, yGold: trainY}
            )
            accuracy_values.append(train_accuracy)
            cost_values.append(newCost)
            writer.add_summary(summary_results, i)
            diff = abs(newCost - cost)
            cost = newCost
            print("step %d, training accuracy %g"%(i, train_accuracy))
            print("step %d, cost %g"%(i, newCost))
            print("step %d, change in cost %g"%(i, diff))

            accuracyLine, = ax1.plot(epoch_values, accuracy_values)
            costLine, = ax2.plot(epoch_values, cost_values)
            fig.canvas.draw()
            # plt.plot(accuracyLine, costLine)
            # time.sleep(1)

# fig.show()
fig.savefig("plots.png")
# print("final accuracy on test set: %s" %str(sess.run(accuracy_OP, feed_dict={X: testX, yGold: testY})))

saver = tf.train.Saver()
saver.save(sess, "trained_variables.ckpt")
sess.close()