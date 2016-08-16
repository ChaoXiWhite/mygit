import numpy as np
import tensorflow as tf

from sklearn.cross_validation import train_test_split

# import time
# start = time.time()

def extrac_data(filename):
	out = np.loadtxt(filename,delimiter = ',')
	labels = out[:,0]
	labels = labels.reshape(labels.size,1)
	fvecs = out[:,1:]
	return fvecs,labels

def readFromTFrecord(target):
    # print "check 3"
    rcdata = []
    rclabel = []
    # i = 0
    for serialized_example in tf.python_io.tf_record_iterator(target):
        # tmp = []
        # print "check 4"
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        all_count = example.features.feature["all_count"].float_list.value
        user_count = example.features.feature["user_count"].float_list.value
        user_succ = example.features.feature["user_succ"].float_list.value
        fail_count = example.features.feature["fail_count"].float_list.value
        user_fail = example.features.feature["user_fail"].float_list.value
        device_count = example.features.feature["device_count"].float_list.value
        ua_count = example.features.feature["ua_count"].float_list.value
        entry_count = example.features.feature["entry_count"].float_list.value
        pwd_login = example.features.feature["pwd_login"].float_list.value
        pwd_succ = example.features.feature["pwd_succ"].float_list.value
        pst_succ = example.features.feature["pst_succ"].float_list.value
        remote_login = example.features.feature["remote_login"].float_list.value
        new_device = example.features.feature["new_device"].float_list.value
        no_user = example.features.feature["no_user"].float_list.value
        pwd_wron = example.features.feature["pwd_wron"].float_list.value
        label = example.features.feature["label"].float_list.value
        # print("all_count: {}, user_count: {}, label: {}".format(all_count, user_count, label))
        # print("user_succ: {}, user_fail: {}, device_count: {}".format(user_succ, user_fail, device_count))
        # print("ua_count: {}, entry_count: {}, pwd_login: {}".format(ua_count, entry_count, pwd_login))
        # print("pwd_succ: {}, pst_succ: {}, remote_login: {}".format(pwd_succ, pst_succ, remote_login))
        # print("new_device: {}, no_user: {}, pwd_wron: {}".format(new_device, no_user, pwd_wron))
        tmp = [all_count[0],user_count[0],user_succ[0],user_fail[0],device_count[0],ua_count[0],entry_count[0],
        pwd_login[0],pwd_succ[0],pwd_succ[0],pst_succ[0],remote_login[0],new_device[0],no_user[0],pwd_wron[0]]
        # print tmp
        rcdata.append(tmp)
        rclabel.append(label[0])
    #     print rcdata
    #     print rclabel
    #     print "check 5"
    #     i += 1
    #     if i == 3:
    #     	break
    # print "check 6"
    return np.array(rcdata), rclabel

class DataSet(object):
	def __init__(self,rcdata,labels):
		self._rcdata = np.array(rcdata)
		self._labels = np.array(labels)
		self._epochs_completed = 0
		self._index_in_epoch = 0
		self._num_examples = rcdata.shape[0]
	@property
	def rcdata(self):
		return self._rcdata
	@property
	def labels(self):
		return self._labels
	@property
	def num_examples(self):
		return self._num_examples
	@property
	def epochs_completed(self):
		return self._epochs_completed
	def next_batch(self, batch_size):
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			self._epochs_completed += 1
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			print "perm:", perm
			self._rcdata = self._rcdata[perm]
			self._labels = self._labels[perm]
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._rcdata[start:end], self._labels[start:end]

def read_data_sets(filename):
	class DataSets(object):
		pass
	data_sets = DataSets()
	# print "check 1"
	data,labels = readFromTFrecord(filename)
	# print "check 2"
	X_train, X_test, y_train, y_test = train_test_split(data, labels,test_size=0.1, random_state=42)
	# print X_train.shape
	# print X_test.shape
	# print y_train.shape
	# print y_test.shape
	data_sets.train = DataSet(X_train,y_train)
	data_sets.test = DataSet(X_test,y_test)
	return data_sets

# def main():
# 	rc_datas = read_data_sets('./tfdata/0611ab.tfrecords')
# 	print rc_datas
# 	print rc_datas.train
# 	print rc_datas.test
# 	# print labels
# 	# train = DataSet(data, labels)
# 	# print train

# if __name__ == '__main__':
# 	main()

# end = time.time()
# print "run time: %d s" % (end - start)