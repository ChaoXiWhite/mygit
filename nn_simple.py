import getdataFromFile
# train.data train.label test.data test.label
data = getdataFromFile.read_data_sets("./tfdata/0611ab.tfrecords")

import tensorflow as tf
import numpy as np
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(fvecs, labels, test_size=0.33, random_state=42)

x = tf.placeholder("float", [None, 15])#zhanweifu
W = tf.Variable(tf.zeros([15,1]))
b = tf.Variable(tf.zeros([1]))
matm=tf.matmul(x,W)
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,1])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# for batch_xs,batch_ys in (data.train.rcdata,data.train.labels):
# 	sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print sess.run(accuracy, feed_dict={x: data.test.rcdata, y_: data.test.labels})

for i in range(100):
 print i
 batch_xs, batch_ys = data.train.next_batch(100)
 # print len(batch_xs),len(batch_ys)
 # print ("batch_xs:{}, batch_ys:{}".format(batch_xs,batch_ys))
 batch_ys = np.reshape(batch_ys,(100,1))
 sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
 correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
 accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
 print sess.run(accuracy, feed_dict={x: data.test.rcdata, y_: np.reshape(data.test.labels,(1000,1))})
 