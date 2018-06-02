# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:39:57 2018

@author: Yousef Essam
"""

import tensorflow as tf


#Defining multidimensional arrays using TensorFlow

Scalar = tf.constant([2])
Vector = tf.constant([5,6,2])
Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )

with tf.Session() as session:
    result = session.run(Scalar)
    print("Scalar (1 entry):\n %s \n" % result)
    result = session.run(Vector)
    print("Vector (3 entries) :\n %s \n" % result)
    result = session.run(Matrix)
    print("Matrix (3x3 entries):\n %s \n" % result)
    result = session.run(Tensor)
    print("Tensor (3x3x3 entries) :\n %s \n" % result)
    

#Element wise multiplication and matrix multiplication
    
Matrix_one = tf.constant([[2,3],[3,4]])
Matrix_two = tf.constant([[2,3],[3,4]])

first_operation = tf.multiply(Matrix_one, Matrix_two)
second_operation = tf.matmul(Matrix_one, Matrix_two)
with tf.Session() as session:
    result = session.run(first_operation)
    result2 = session.run(second_operation)
    print("Defined using tensorflow function :")
    print(result)    
    print(result2)    
    
    




#Usage of Placeholders
    

name = tf.placeholder(tf.string)
b = "welcome " + name
dict_array = {name: "yousef"}
init_op = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init_op)
    result = session.run(b,dict_array)
    print(result)


#Operations
    
a = tf.constant([5])
b = tf.constant([2])
c = tf.add(a,b)
d = tf.subtract(a,b)

with tf.Session() as session:
    result = session.run(c)
    print('c =: %s' % result)
    result = session.run(d)
    print('d =: %s' % result)    
    
    
    
    
    
    
    
    