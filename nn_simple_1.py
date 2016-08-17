import time
start = time.time()
import getdataFromFile
# train.data train.label test.data test.label
# data = getdataFromFile.read_data_sets("./tfdata/0611ab.tfrecords")

import pprint, pickle
pkl_file = open('data.pkl','rb')
data_train_rcdata = pickle.load(pkl_file)
data_train_labels = pickle.load(pkl_file)
data_test_rcdata = pickle.load(pkl_file)
data_test_labels = pickle.load(pkl_file)

print "read data needs: %d s" % (time.time()-start)

start = time.time()

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

for i in range(10):
	for batch_xs,batch_ys in zip(data_train_rcdata,data_train_labels):
		# print ("batch_xs:{}, batch_ys:{}".format(batch_xs,batch_ys))
		batch_xs = np.reshape(batch_xs,(1,15))
		batch_ys = np.reshape(batch_ys,(1,1))
		sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
print "train data needs: %d s" % (time.time()-start)

start = time.time()
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict={x: data_test_rcdata, y_: np.reshape(data_test_labels,(1000,1))})
print "predict data needs: %d s" % (time.time()- start)

# for i in range(100):
#  # print i
#  batch_xs, batch_ys = data.train.next_batch(100)
#  # print len(batch_xs),len(batch_ys)
#  # print ("batch_xs:{}, batch_ys:{}".format(batch_xs,batch_ys))
#  batch_ys = np.reshape(batch_ys,(100,1))
#  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#  print sess.run(accuracy, feed_dict={x: data.test.rcdata, y_: np.reshape(data.test.labels,(1000,1))})