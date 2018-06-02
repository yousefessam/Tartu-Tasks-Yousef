# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 18:19:57 2018

@author: Yousef Essam
"""

#Lamda funtion with one parameter
f = lambda x: 2*x
print(f)
print(f(2))


#Lamda funtion with Two parameter
f = lambda x,y: x + y
print(f)
print(f(2,3))

#Lamda funtion with Two parameter AND default parameter
f = lambda x,y,z=3: x + y + z
print(f)
print(f(2,3))
print(f(2,3,10))



#Lamda with anu number of operators

import operator
from functools import reduce
f = lambda *x: reduce(operator.add, x)
print(f)

print(f(1))
print(f(1, 2))
print(f(1, 2, 3))


#Lamda return dict

f = lambda **anyThing: anyThing
print(f(a=1, b=3))


#Lamda Funtions all type of parameters together

f = lambda a, b=4, *mydata, **anyThing: (a, b, mydata, anyThing)

print(f('required', 3, 'optional-positional', g=4))
print(f('required', 3, 'optional-positional', name="yousef",g=4))

result = f('required', 3, 'optional-positional', name="yousef",age=4)
result[0]
result[3]['age']
result[3]['name']



# Applications of lambda functions
#Lambda functions are used in places where you need a function, 
#but may not want to define one using def. 
#For example, say you want to solve the nonlinear equation x−−√=2.5.
from scipy.optimize import fsolve
import numpy as np

sol = fsolve(lambda x: 2.5 - np.sqrt(x), 8)
print(sol)



#application 2


from scipy.optimize import fsolve
import numpy as np

def func(x, a):
    return a * np.sqrt(x) - 4.0

sol, = fsolve(lambda x: func(x, 3.2), 0)
print(sol)


# Use lambda with reduce

myArray = [0, 1, 2, 3, 4]
print(reduce(lambda x, y: x + y,myArray ))


#Lambda for integration
#We can evaluate the integral ∫ x2 dx from [0 to 2] with a lambda function.
from scipy.integrate import quad

print(quad(lambda x: x**2, 0, 2))

print(quad(lambda x: 2*x, 0, 2))


#Vectorize Function

def myfunc(a, b):
     "Return a-b if a>b, otherwise return a+b"
     if a > b:
         return a - b
     else:
         return a + b

vfunc = np.vectorize(myfunc)

vfunc([1, 2, 3, 4], 2)
