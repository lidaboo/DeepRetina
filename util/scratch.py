# from collections import OrderedDict

# D = {"hello":1, "why":2}
# print(D.items())

# import tensorflow as tf

# a = tf.stack([1,3,4,1])
# b = tf.unstack(a)
# with tf.Session() as sess:
# 	a,b = sess.run([a,b])
# 	print(a)

def hello(a= 1, b= 2):
	print(a)
	print(b)


def second(c = 5 ,**kwargs):
	print(c)
	hello(**kwargs)
c= {'a': 10,'b':20}
second(a = 10, c= 100, b= 20)