import tensorflow as tf
#Tensorflow Hello example
a = tf.constant([5])
b = tf.constant([2])
c = a+b
d = a*b
with tf.Session() as session:
    result = session.run(c)
    result2 = session.run(d)
    print("The addition of this two constants is: {0} {1}".format(result,result2))
   
# Multiply two matrics    
matrixA = tf.constant([[2,3],[3,4]])
matrixB = tf.constant([[2,3],[3,4]])
with tf.Session() as session:
    res= tf.multiply(matrixA, matrixB)
    res2 = tf.matmul(matrixA,matrixB)
    print(session.run(res))
    print(session.run(res2))


#Use variable , initialize it
    
a=tf.constant(1000)
b=tf.Variable(0)
update = tf.assign(b,a)
init_op = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init_op) 
    session.run(update)
    print(session.run(b))
    
#Fibonacci sequence - method1
myarray = [tf.constant(1),tf.constant(1)]
for i in range(2,10):   
    myarray.append(tf.add(myarray[i-1],myarray[i-2]))

with tf.Session() as session:
    result = session.run(myarray)
    print(result)


#Fibonacci sequence - method2
a = tf.Variable(1)
b = tf.Variable(1)
c = tf.add(a,b)
temp  = tf.Variable(0)
init_op = tf.global_variables_initializer()
update1 = tf.assign(temp,c)
update2 = tf.assign(a,b)
update3 = tf.assign(b,temp)

with tf.Session() as session:
    session.run(init_op)
    for i in range(10):
        print(session.run(a))
        session.run(update1)
        session.run(update2)
        session.run(update3)
    

#Factorial using Tensorflow - for loop
number = tf.Variable(4,dtype=tf.int32)
result= tf.Variable(1)
updateNumber = tf.assign(number, tf.subtract(number,1))
calcFactorial = tf.assign(result,tf.multiply(result,number))
init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    for i in range(session.run(number),1,-1):
        session.run(calcFactorial)
        session.run(updateNumber)
    print(session.run(result))

#Tensorflow simple lambdas
t1 = tf.constant(1)
t2 = tf.constant(2)

def f1(): return t1+t2
def f2(): return t1-t2

res = tf.cond(tf.less(t1, t2), f1,f2)

with tf.Session() as sess:
    print(sess.run(res))

#parametrised lambdas
def f1(p):
    def res(): return t1+t2+p
    return res

def f2(p):
    def res(): return t1-t2+p
    return res

t1 = tf.constant(1)
t2 = tf.constant(2)
p1 = tf.constant(3)
p2 = tf.constant(4)

res = tf.cond(tf.less(t1, t2), f1(p1), f2(p2))

with tf.Session() as sess:
    print(sess.run(res))
    

#While loop tensor basics
    
def cond(t1, t2):
    return tf.less(t1, t2)

def body(t1, t2):
    return [tf.add(t1, 1), t2]

t1 = tf.constant(1)
t2 = tf.constant(5)

res = tf.while_loop(cond, body, [t1, t2])

with tf.Session() as sess:
    print(sess.run(res))
   
    
#Factorial using Tensorflow - while loop
def cond(number,res):
    return tf.greater(number, 1)    

def body(number,res):
    return [number-1,res*number]

number = tf.constant(4)
res = tf.constant(1)

res = tf.while_loop(cond, body, [number,res])

with tf.Session() as sess:
    print(sess.run(res[1]))
  
#Tensorflow While loop example
a,b = tf.while_loop(lambda a,b: a < 30, lambda a,b: (a*2 , b*3),(1,2))

with tf.Session() as session:
    result = session.run([a,b])
    print(result)


#Factorial using Recursion Python
    
def fact(number):
    if(number == 1):
        return 1
    return  number * fact(number-1)    

number = 4
res = 1

print(fact(number))


#WHile loop fixed number of iterations

def cond(t1, t2, i, iters):
    return tf.less(i, iters)

def body(t1, t2, i, iters):
    return [tf.add(t1, 1), t2+1, tf.add(i, 1), iters]

t1 = tf.constant(1)
t2 = tf.constant(5)
iters = tf.constant(3)

res = tf.while_loop(cond, body, [t1, t2, 0, iters])

with tf.Session() as sess:
    print(sess.run(res))
    
    
#while loop conditional break
def cond_loop(t1, t2, iters):
    
    def cond(t1, t2, i):
        return tf.less(i, iters)

    def body(t1, t2, i):
        def increment(t1, t2):
            def f1(): return tf.add(t1, 1), tf.add(t2, 1)
            return f1

        def swap(t1, t2):
            def f2(): return t2, t1
            return f2

        t1, t2 = tf.cond(tf.less(i+1, iters),
                         increment(t1, t2),
                         swap(t1, t2))

        return [t1, t2, tf.add(i, 1)]

    return tf.while_loop(cond, body, [t1, t2, 0])

t1 = tf.constant(1)
t2 = tf.constant(5)
iters = tf.constant(3)

with tf.Session() as sess:
    loop = cond_loop(t1, t2, iters)
    print(sess.run(loop))    