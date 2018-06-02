# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 18:07:30 2018

@author: Yousef Essam
"""

import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


#Plot simple Line

#matplotlib inline
plt.rcParams['figure.figsize'] = (10, 6)

X = np.arange(0.0, 5.0, 0.1)

a=1
b=0

Y= a*X + b 

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()


#Linear Regression with TensorFlow

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 3 + 2

myFunc = lambda y: y + np.random.normal(loc=0.0, scale=0.1)
myVectFun = np.vectorize(myFunc) 
y_data = myVectFun(y_data)
#zip(x_data,y_data) [0:5]
# add some noise

a = tf.Variable(1.0)
b = tf.Variable(0.2)
y = a * x_data + b

plt.scatter(x_data,y_data) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

loss = tf.reduce_mean(tf.square(y - y_data)) # loss function 
optimizer = tf.train.GradientDescentOptimizer(0.5) # learning rate = 0.5
train = optimizer.minimize(loss) # we want optimizer to minize losses
init = tf.global_variables_initializer()
error = []
with tf.Session() as ss:
    ss.run(init)
    train_data = []
    for step in range(100):
        error.append(ss.run(loss))
        evals = ss.run([train,a,b])[1:]
        if step % 5 == 0:
            print(step, evals)
            train_data.append(evals)
        
converter = plt.colors
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    myModel = lambda x: a*x + b
    f_y = myModel(x_data)
    line = plt.plot(x_data, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(x_data, y_data, 'ro')


green_line = mpatches.Patch(color='red', label='Data Points')

plt.legend(handles=[green_line])

plt.show()

 

plt.plot(list(range(1,101,1)),error)

plt.show()       
