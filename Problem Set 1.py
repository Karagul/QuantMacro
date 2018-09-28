
# coding: utf-8

# In[2]:


import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#---------------------------------------------------
x = sy.Symbol('x')
f = x**0.321

#-----------Factorial Function----------------------
def factorial(n):

    k = 1
    for i in range(n):
        k = k * (i + 1)
    return k

#----------Taylor Series Function-------------------
def taylor(function,x0,n):
    i = 0
    p = 0
    while i <= n:
        p = p + (function.diff(x,i).subs(x,x0))/(factorial(i))*(x-x0)**i
        i += 1
    return p


#--------Plotting the function----------------------
def plot():
    x_lims = [0,4]
    y_lims = [0,4]
    x1 = np.linspace(x_lims[0],x_lims[1],1000)
    y1 = []
    r = [1, 2, 5, 20]
    for j in r:
        func = taylor(f,1,j)
        print('Taylor expansion at n='+str(j),func)
        for k in x1:
            y1.append(func.subs(x,k))
        plt.plot(x1,y1,label='order '+str(j))
        y1 = []
    plt.plot(x1, x1**0.321, label="Function")
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('Taylor series approximation')
    plt.show()

plot()


# In[2]:


import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import math

plt.style.use("ggplot")
#---------------------------------------------------
x = sy.Symbol('x', real=True)
f1 = (x + abs(x))/2

#-----------Factorial Function----------------------
def factorial(n):

    k = 1
    for i in range(n):
        k = k * (i + 1)
    return k

#----------Taylor Series Function-------------------
def taylor(function,x0,n):
    i = 0
    p = 0
    while i <= n:
        p = p + (function.diff(x,i).subs(x,x0))/(factorial(i))*(x-x0)**i
        i += 1
    return p


#--------Plotting the function----------------------
def plot():
    x_lims = [-2,6]
    y_lims = [-2,6]
    x1 = np.linspace(x_lims[0],x_lims[1],1000)
    y1 = []
    r = [1, 2, 5, 20]
    for j in r:
        func = taylor(f1,2,j)
        print('Taylor expansion at n='+str(j),func)
        for k in x1:
            y1.append(func.subs(x,k))
        plt.plot(x1,y1,label='order '+str(j))
        y1 = []
    plt.plot(x1, (x1+ abs(x1))/2 ,label="Function")
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('Taylor series approximation')
    plt.show()

plot()


# In[40]:


import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
from math import exp

#Defining the number of nodes required to interpolate the function.

#Defining the function
def f(x):
    f = np.exp(1/x)
    return f

domain = np.linspace(-1,1,100)

x = np.linspace(-1,1,12) #Number of nodes = 12
#Polyfit allows us to calculate our theta coefficients. It takes as inputs the number of nodes, specified by x, the function - given by y and the order of interpolation. 
z1 = np.polyfit(x,f(x), 3)
z2 = np.polyfit(x,f(x),5)
z3 = np.polyfit(x,f(x),10)

#With the polyval function I need to specify my domain and the polyfit function. Polyval gives you the function evaluated at the nodes. 
val3 = np.polyval(z1, domain)
val5 = np.polyval (z2, domain)
val10 = np.polyval(z3, domain)

#Calculating the error between the true function and the interpolated function
e1= f(domain)-val3
e2 = f(domain)-val5
e3 = f(domain)-val10

plt.figure(1)

plt.subplot(121)
#plotting the domain and polyval function. 
plt.plot(domain, f(domain), 'y',label='Function')
plt.plot(domain, val3, 'b', label='Cubic')
plt.plot(domain, val5, 'r', label='Order 5')
plt.plot(domain, val10, 'g', label='Order 10')
plt.ylim([-20000,100000])
plt.legend(loc='best')
plt.title('Exponential Function Interpolation')

plt.subplot(122)
#plotting the domain and polyval function. 
plt.plot(domain, e1, 'b', label='Cubic error')
plt.plot(domain, e2, 'r', label='Order 5 error')
plt.plot(domain, e3, 'g', label='Order 10 error')
plt.ylim([-20000,100000])
plt.legend(loc='best')
plt.title('Errors associated with the Exponential Function Interpolation')

plt.subplots_adjust(top=1, bottom=0.08, left=0, right=2, hspace=0.25,
                    wspace=0.35)

plt.show()


# In[99]:


#Runge Function - 1/1+25*x^2
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np

#Specifying the domain of the function
domain = np.linspace(-1, 1, 100)

#Defining the Runge function
def f(x):
    y = 1/(1+25*(x**2))
    return y

domain = np.linspace(-1,1,100)

x = np.linspace(-1,1,12) #Number of nodes = 12
#Polyfit allows us to calculate our theta coefficients. It takes as inputs the number of nodes, specified by x, the function - given by y and the order of interpolation. 
z1 = np.polyfit(x,f(x), 3)
z2 = np.polyfit(x,f(x),5)

#With the polyval function I need to specify my domain and the polyfit function. Polyval gives you the function evaluated at the nodes. 
val3 = np.polyval(z1, domain)
val5 = np.polyval (z2, domain)

# Get the order 10 monomial
x = np.asarray(np.linspace(-1,1,10))
y = 1/(1+25*(x**2))

# Get matrix of exponents of x values => A
A = np.zeros([10, 10])
for i in range(10):
    A[::,i] = np.power(x.T,i)
b = y

# Solve Ax=b linear eq. system to get 
s = np.linalg.solve(A, b)
# where x denotes coeffs of polynomial in reverse order
# Flip polynomial coeffs
s = np.flip(s,axis=0)
# Print polynomial coeffs
print(np.poly1d(s))

# Evaluate polynomial at X axis and plot result
val10 = np.polyval(s, domain)

#Calculating the error between the true function and the interpolated function
e1= f(domain)- val3
e2 = f(domain)- val5
e3 = f(domain)- val10

plt.figure(2)

#Plot everything together
plt.subplot(121)
plt.plot(domain, f(domain), 'y', label='Runge function', lw=3)
plt.plot(domain, val3, 'g', label='cubic')
plt.plot(domain, val5,'r', label='deg 5')
plt.plot(domain, val10, label= 'deg 10')
plt.ylim([-0.5,1.1])
plt.legend(['Runge', 'cubic', 'deg 5', 'deg 10'], loc='best')
plt.title('Runge Function Interpolation')

plt.subplot(122)
#plotting the domain and polyval function. 
plt.plot(domain, e1, 'b', label='Cubic error')
plt.plot(domain, e2, 'r', label='Order 5 error')
plt.plot(domain, e3, 'g', label='Order 10 error')
plt.ylim([-0.2,0.75])
plt.legend(loc='best')
plt.title('Errors associated with the Runge Function Interpolation')
plt.subplots_adjust(top=1, bottom=0.08, left=0, right=2, hspace=0.25, wspace=0.35)
plt.show()


# In[106]:


#Ramp Function - x + |x|/2
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
import math


#Specifying the domain of the function
domain = np.linspace(-1, 1, 100)

#Defining the Runge function
def f(x):
    y = (x + abs(x))/2
    return y

domain = np.linspace(-1,1,100)

x = np.linspace(-1,1,12) #Number of nodes = 12
#Polyfit allows us to calculate our theta coefficients. It takes as inputs the number of nodes, specified by x, the function - given by y and the order of interpolation. 
z1 = np.polyfit(x,f(x), 3)
z2 = np.polyfit(x,f(x),5)

#With the polyval function I need to specify my domain and the polyfit function. Polyval gives you the function evaluated at the nodes. 
val3 = np.polyval(z1, domain)
val5 = np.polyval (z2, domain)

# Get the order 10 monomial
x = np.asarray(np.linspace(-1,1,10))
y = (x + abs(x))/2

# Get matrix of exponents of x values => A
A = np.zeros([10, 10])
for i in range(10):
    A[::,i] = np.power(x.T,i)
b = y

# Solve Ax=b linear eq. system to get 
s = np.linalg.solve(A, b)
# where x denotes coeffs of polynomial in reverse order
# Flip polynomial coeffs
s = np.flip(s,axis=0)
# Print polynomial coeffs
print(np.poly1d(s))

# Evaluate polynomial at X axis and plot result
val10 = np.polyval(s, domain)

#Calculating the error between the true function and the interpolated function
e1= abs(f(domain)- val3)
e2 = abs(f(domain)- val5)
e3 = abs(f(domain)- val10)

plt.figure(2)

#Plot everything together
plt.subplot(121)
plt.plot(domain, f(domain), 'y', label='Runge function', lw=3)
plt.plot(domain, val3, 'g', label='cubic')
plt.plot(domain, val5,'r', label='deg 5')
plt.plot(domain, val10, label= 'deg 10')
plt.ylim([-0.5,1.1])
plt.legend(['Runge', 'cubic', 'deg 5', 'deg 10'], loc='best')
plt.title('Ramp Function Interpolation')

plt.subplot(122)
#plotting the domain and polyval function. 
plt.plot(domain, e1, 'b', label='Cubic error')
plt.plot(domain, e2, 'r', label='Order 5 error')
plt.plot(domain, e3, 'g', label='Order 10 error')
plt.ylim([-0.05,0.2])
plt.legend(loc='best')
plt.title('Errors associated with the Ramp Function Interpolation')
plt.subplots_adjust(top=1, bottom=0.08, left=0, right=2, hspace=0.25, wspace=0.35)
plt.show()


# In[193]:


import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
from math import exp
import warnings
from math import cos
warnings.simplefilter('ignore', np.RankWarning)
#Defining the function
def f(x):
    f = np.exp(1/x)
    return f

#Defining the number of evaluation points
domain = np.linspace(-1,1,100)

#Defining the Chebyshev nodes
x_cb=[]
for i in range(10):
    q= cos((((2*i)-1)/20)*np.pi)
    x_cb.append(q)

x=np.asarray(x_cb)
#Polyfit allows us to calculate our theta coefficients. It takes as inputs the number of nodes, specified by x, the function - given by y and the order of interpolation. 
z1 = np.polyfit(x,f(x),3)
z2 = np.polyfit(x,f(x),5)
z3 = np.polyfit(x,f(x),10)

#With the polyval function I need to specify my domain and the polyfit function. Polyval gives you the function evaluated at the nodes. 
val3 = np.polyval(z1, domain)
val5 = np.polyval (z2, domain)
val10 = np.polyval(z3, domain)

#Calculating the error between the true function and the interpolated function
e1= f(domain)-val3
e2 = f(domain)-val5
e3 = f(domain)-val10

plt.figure(1)

plt.subplot(121)
#plotting the domain and polyval function. 
plt.plot(domain, f(domain), 'y',label='Function')
plt.plot(domain, val3, 'b', label='Cubic')
plt.plot(domain, val5, 'r', label='Order 5')
plt.plot(domain, val10, 'g', label='Order 10')
plt.ylim([-200,1000])
plt.legend(loc='best')
plt.title('Exponential Function Interpolation - Cbehyshev Nodes')

plt.subplot(122)
#plotting the domain and polyval function. 
plt.plot(domain, e1, 'b', label='Cubic error')
plt.plot(domain, e2, 'r', label='Order 5 error')
plt.plot(domain, e3, 'g', label='Order 10 error')
plt.ylim([-400,1000])
plt.legend(loc='best')
plt.title('Errors associated with the Exponential Function Interpolation - Cbehyshev Nodes')

plt.subplots_adjust(top=1, bottom=0.08, left=0, right=2, hspace=0.25,
                    wspace=0.35)
plt.show()


# In[191]:


import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
from math import exp
import warnings
warnings.simplefilter('ignore', np.RankWarning)
from math import cos

#Defining the function
def f(x):
    f = 1/(1+25*(x**2))
    return f

#Defining the points for the evaluation of the function
domain = np.linspace(-1,1,100)

#Defining the Chebyshev nodes
x_cb=[]
for i in range(10):
    q=cos((((2*i)-1)/20)*np.pi)
    x_cb.append(q)

x=np.asarray(x_cb)
#Polyfit allows us to calculate our theta coefficients. It takes as inputs the number of nodes, specified by x, the function - given by y and the order of interpolation. 
z1 = np.polyfit(x,f(x),3)
z2 = np.polyfit(x,f(x),5)
z3 = np.polyfit(x,f(x),10)

#With the polyval function I need to specify my domain and the polyfit function. Polyval gives you the function evaluated at the nodes. 
val3 = np.polyval(z1, domain)
val5 = np.polyval (z2, domain)
val10 = np.polyval(z3, domain)

#Calculating the error between the true function and the interpolated function
e1= f(domain)-val3
e2 = f(domain)-val5
e3 = f(domain)-val10

plt.figure(1)

plt.subplot(121)
#plotting the domain and polyval function. 
plt.plot(domain, f(domain), 'y',label='Function')
plt.plot(domain, val3, 'b', label='Cubic')
plt.plot(domain, val5, 'r', label='Order 5')
plt.plot(domain, val10, 'g', label='Order 10')
plt.ylim([-0.5,1])
plt.legend(loc='best')
plt.title('Runge Function Interpolation - Cbehyshev Nodes')

plt.subplot(122)
#plotting the domain and polyval function. 
plt.plot(domain, e1, 'b', label='Cubic error')
plt.plot(domain, e2, 'r', label='Order 5 error')
plt.plot(domain, e3, 'g', label='Order 10 error')
plt.ylim([-0.5,1])
plt.legend(loc='best')
plt.title('Errors associated with the Runge Function Interpolation - Cbehyshev Nodes')

plt.subplots_adjust(top=1, bottom=0.08, left=0, right=2, hspace=0.25,
                    wspace=0.35)
plt.show()


# In[192]:


import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
from math import exp
import warnings
warnings.simplefilter('ignore', np.RankWarning)
from math import cos

#Defining the function
def f(x):
    f = (x+abs(x))/2
    return f

#Defining the points for the evaluation of the function
domain = np.linspace(-1,1,100)

#Defining the Chebyshev nodes
x_cb=[]
for i in range(10):
    q=cos((((2*i)-1)/20)*np.pi)
    x_cb.append(q)
x=np.asarray(x_cb)

#Polyfit allows us to calculate our theta coefficients. It takes as inputs the number of nodes, specified by x, the function - given by y and the order of interpolation. 
z1 = np.polyfit(x,f(x),3)
z2 = np.polyfit(x,f(x),5)
z3 = np.polyfit(x,f(x),10)

#With the polyval function I need to specify my domain and the polyfit function. Polyval gives you the function evaluated at the nodes. 
val3 = np.polyval(z1, domain)
val5 = np.polyval (z2, domain)
val10 = np.polyval(z3, domain)

#Calculating the error between the true function and the interpolated function
e1= abs(f(domain)-val3)
e2 = abs(f(domain)-val5)
e3 = abs(f(domain)-val10)

plt.figure(1)

plt.subplot(121)
#plotting the domain and polyval function. 
plt.plot(domain, f(domain), 'y',label='Function')
plt.plot(domain, val3, 'b', label='Cubic')
plt.plot(domain, val5, 'r', label='Order 5')
plt.plot(domain, val10, 'g', label='Order 10')
plt.ylim([-0.5,1])
plt.legend(loc='best')
plt.title('Ramp Function Interpolation - Cbehyshev Nodes')

plt.subplot(122)
#plotting the domain and polyval function. 
plt.plot(domain, e1, 'b', label='Cubic error')
plt.plot(domain, e2, 'r', label='Order 5 error')
plt.plot(domain, e3, 'g', label='Order 10 error')
plt.ylim([0,0.2])
plt.legend(loc='best')
plt.title('Errors associated with the Ramp Function Interpolation - Cbehyshev Nodes')

plt.subplots_adjust(top=1, bottom=0.08, left=0, right=2, hspace=0.25,
                    wspace=0.35)
plt.show()


# In[71]:


import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
from math import exp
from math import cos

#Defining the number of nodes required to interpolate the function.

#Defining the function
def func(x):
    f = np.exp(1/x)
    return f

domain = np.linspace(-1,1,100)
x = np.linspace(-1,1,12) #Number of nodes = 12
#Polyfit allows us to calculate our theta coefficients. It takes as inputs the number of nodes, specified by x, the function - given by y and the order of interpolation. 
z1 = np.polynomial.chebyshev.chebfit(x,func(x),3)
z2 = np.polynomial.chebyshev.chebfit(x,func(x),5)
z3 = np.polynomial.chebyshev.chebfit(x,func(x),10)

#With the polyval function I need to specify my domain and the polyfit function. Polyval gives you the function evaluated at the nodes. 
val3 = np.polynomial.chebyshev.chebval(domain, z1)
val5 = np.polynomial.chebyshev.chebval(domain, z2)
val10 = np.polynomial.chebyshev.chebval(domain, z3)

#Calculating the error between the true function and the interpolated function
e1= func(domain)-val3
e2 = func(domain)-val5
e3 = func(domain)-val10

plt.figure(1)

plt.subplot(121)
#plotting the domain and polyval function. 
plt.plot(domain, func(domain), 'y',label='Function')
plt.plot(domain, val3, 'b', label='Cubic')
plt.plot(domain, val5, 'r', label='Order 5')
plt.plot(domain, val10, 'g', label='Order 10')
plt.ylim([-20000,100000])
plt.legend(loc='best')
plt.title('Exponential Function Interpolation - Chebyshev')

plt.subplot(122)
#plotting the domain and polyval function. 
plt.plot(domain, e1, 'b', label='Cubic error')
plt.plot(domain, e2, 'r', label='Order 5 error')
plt.plot(domain, e3, 'g', label='Order 10 error')
plt.ylim([-20000,100000])
plt.legend(loc='best')
plt.title('Errors associated with the Exponential Function Interpolation - Chebyshev')

plt.subplots_adjust(top=1, bottom=0.08, left=0, right=2, hspace=0.25,
                    wspace=0.35)
plt.show()


# In[69]:


import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
from math import exp
from math import cos

#Defining the number of nodes required to interpolate the function.

#Defining the function
def func(x):
    f = 1/(1+25*(x**2))
    return f

domain = np.linspace(-1,1,100)
x = np.linspace(-1,1,12) #Number of nodes = 12
#Polyfit allows us to calculate our theta coefficients. It takes as inputs the number of nodes, specified by x, the function - given by y and the order of interpolation. 
z1 = np.polynomial.chebyshev.chebfit(x,func(x),3)
z2 = np.polynomial.chebyshev.chebfit(x,func(x),5)
z3 = np.polynomial.chebyshev.chebfit(x,func(x),10)

#With the polyval function I need to specify my domain and the polyfit function. Polyval gives you the function evaluated at the nodes. 
val3 = np.polynomial.chebyshev.chebval(domain, z1)
val5 = np.polynomial.chebyshev.chebval(domain, z2)
val10 = np.polynomial.chebyshev.chebval(domain, z3)

#Calculating the error between the true function and the interpolated function
e1= func(domain)-val3
e2 = func(domain)-val5
e3 = func(domain)-val10

plt.figure(1)

plt.subplot(121)
#plotting the domain and polyval function. 
plt.plot(domain, func(domain), 'y',label='Function')
plt.plot(domain, val3, 'b', label='Cubic')
plt.plot(domain, val5, 'r', label='Order 5')
plt.plot(domain, val10, 'g', label='Order 10')
plt.ylim([-0.2,1])
plt.legend(loc='best')
plt.title('Runge Function Interpolation - Chebyshev')

plt.subplot(122)
#plotting the domain and polyval function. 
plt.plot(domain, e1, 'b', label='Cubic error')
plt.plot(domain, e2, 'r', label='Order 5 error')
plt.plot(domain, e3, 'g', label='Order 10 error')
plt.ylim([-0.2,0.6])
plt.legend(loc='best')
plt.title('Errors associated with the Runge Function Interpolation - Chebyshev')

plt.subplots_adjust(top=1, bottom=0.08, left=0, right=2, hspace=0.25,
                    wspace=0.35)
plt.show()


# In[123]:


import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
from math import exp
from math import cos

#Defining the number of nodes required to interpolate the function.

#Defining the function
def func(x):
    f = (x + abs(x))/2
    return f

domain = np.linspace(-1,1,100)
x = np.linspace(-1,1,12) #Number of nodes = 12
#Polyfit allows us to calculate our theta coefficients. It takes as inputs the number of nodes, specified by x, the function - given by y and the order of interpolation. 
z1 = np.polynomial.chebyshev.chebfit(x,func(x),3)
z2 = np.polynomial.chebyshev.chebfit(x,func(x),5)
z3 = np.polynomial.chebyshev.chebfit(x,func(x),10)

#With the polyval function I need to specify my domain and the polyfit function. Polyval gives you the function evaluated at the nodes. 
val3 = np.polynomial.chebyshev.chebval(domain, z1)
val5 = np.polynomial.chebyshev.chebval(domain, z2)
val10 = np.polynomial.chebyshev.chebval(domain, z3)

#Calculating the error between the true function and the interpolated function
e1= abs(func(domain)-val3)
e2 = abs(func(domain)-val5)
e3 = abs(func(domain)-val10)

plt.figure(1)

plt.subplot(121)
#plotting the domain and polyval function. 
plt.plot(domain, func(domain), 'y',label='Function')
plt.plot(domain, val3, 'b', label='Cubic')
plt.plot(domain, val5, 'r', label='Order 5')
plt.plot(domain, val10, 'g', label='Order 10')
plt.ylim([-0.2,1])
plt.legend(loc='best')
plt.title('Ramp Function Interpolation - Chebyshev')

plt.subplot(122)
#plotting the domain and polyval function. 
plt.plot(domain, e1, 'b', label='Cubic error')
plt.plot(domain, e2, 'r', label='Order 5 error')
plt.plot(domain, e3, 'g', label='Order 10 error')
plt.ylim([0,0.1])
plt.legend(loc='best')
plt.title('Errors associated with the Ramp Function Interpolation - Chebyshev')

plt.subplots_adjust(top=1, bottom=0.08, left=0, right=2, hspace=0.25,
                    wspace=0.35)
plt.show()


# In[96]:


import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
from math import exp
from math import cos

domain = np.linspace(0,10,100)

#Defining the function
def func(x):
    f = (np.exp(-x))/(5 + (0.01*np.exp(-x)))
    return f

#Defining the number of nodes required to interpolate the function.
xl=[]
for i in range(10):
    q=cos((((2*i)-1)/20)*np.pi)
    xl.append(q)

x=np.asarray(xl)

#Polyfit allows us to calculate our theta coefficients. It takes as inputs the number of nodes, specified by x, the function - given by y and the order of interpolation. 
z1 = np.polynomial.chebyshev.chebfit(x,func(x),3)
z2 = np.polynomial.chebyshev.chebfit(x,func(x),5)
z3 = np.polynomial.chebyshev.chebfit(x,func(x),8)

#With the polyval function I need to specify my domain and the polyfit function. Polyval gives you the function evaluated at the nodes. 
val3 = np.polynomial.chebyshev.chebval(domain, z1)
val5 = np.polynomial.chebyshev.chebval(domain, z2)
val10 = np.polynomial.chebyshev.chebval(domain, z3)

#Calculating the error between the true function and the interpolated function
e1= func(domain)-val3
e2 = func(domain)-val5
e3 = func(domain)-val10

plt.figure(1)

plt.subplot(121)
#plotting the domain and polyval function. 
plt.plot(domain, func(domain), 'y',label='Function')
plt.plot(domain, val3, 'b', label='Cubic')
plt.plot(domain, val5, 'r', label='Order 5')
plt.plot(domain, val10, 'g', label='Order 8')
plt.ylim([0,0.3])
plt.xlim([0,3])
plt.legend(loc='best')
plt.title('Probability Function Interpolation with ρ=5')

plt.subplot(122)
#plotting the domain and polyval function. 
plt.plot(domain, e1, 'b', label='Cubic error')
plt.plot(domain, e2, 'r', label='Order 5 error')
plt.plot(domain, e3, 'g', label='Order 8 error')
plt.ylim([0,0.3])
plt.xlim([0,3])
plt.legend(loc='best')
plt.title('Errors of the Probability Function Interpolation with ρ=5')

plt.subplots_adjust(top=1, bottom=0.08, left=0, right=2, hspace=0.25,
                    wspace=0.35)
plt.show()


# In[94]:


#With different rho1
#Define the function we want to interpolate:
domain = np.linspace(0, 10, 1000)
def f(x):
    f = np.exp(-x)/(4+(0.01*np.exp(-x)))
    return f

#Get the Chebyshev nodes (10 nodes) 
xl=[]
for i in range(10):
    q=cos((((2*i)-1)/20)*np.pi)
    xl.append(q)

x=np.asarray(xl)

#Interpolation of polinomials using a Chebyshev way. 
z1=np.polynomial.chebyshev.chebfit(x,f(x),3)
z2=np.polynomial.chebyshev.chebfit(x,f(x),5)
z3=np.polynomial.chebyshev.chebfit(x,f(x),8)


#Evaluating the Chebyshev Polynomials
val1=np.polynomial.chebyshev.chebval(domain,z1)
val2=np.polynomial.chebyshev.chebval(domain, z2)
val3=np.polynomial.chebyshev.chebval(domain, z3) 

#Approximation errors. 
error11= abs(f(domain)-val1)
error21= abs(f(domain)-val2)
error31= abs(f(domain)-val3)

#Plot
plt.figure(1)

plt.subplot(121)
plt.plot(domain,f(domain), label='Probability', lw=5)
plt.plot(domain,val1, label='Cubic')
plt.plot(domain,val2, label='Order 5')
plt.plot(domain,val3, label = 'Order 8')
plt.ylim([0,0.3])
plt.xlim([0,3])
plt.legend(loc=1)
plt.title('Probability function approximation with ρ=4')

plt.subplot(122)
plt.plot(domain,error11, label='Cubic')
plt.plot(domain,error21, label='Order 5')
plt.plot(domain,error31, label= 'Order 8')
plt.legend(loc='best')
plt.ylim([0,0.4])
plt.xlim([0,3])
plt.title('Probability approx errors with ρ=4')
plt.subplots_adjust(top=0.9, bottom=0.1, left=0, right=1.5, hspace=0.5, wspace=0.5)
plt.show()


# In[209]:


#Code worked on with the help of Elena and Pau
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.tri as mtri
from math import cos 
from matplotlib import cm

N  = 20
nodes = -(np.cos((2*np.arange(1,N+1)-1)/(2*N)*np.pi))

k = []
for i in nodes:
    a=0
    b=10
    m= ((i+1)*(b-a)/2) + a
    k.append(m)
k =np.asarray(k)

h = []
for i in nodes:
    a=0
    b=10
    m= ((i+1)*(b-a)/2) + a
    h.append(m)
    
h =np.asarray(h)

#Define the CES Function
def f(alpha,rho,k,h):
    return np.power((1-alpha)*k**rho + alpha*h**rho,1/rho)

#Evaluating the functions at the nodes
alpha = 0.5
sigma = 0.999
rho= (sigma -1)/sigma
K,H = np.meshgrid(k,h)
wk = f(alpha,rho,K,H)

#Creating the Polynomials for the Chebyshev approximation using the recursive formulation
def T(deg,x): #Defining n to be the degree 
	poly = []
	poly.append(np.ones(len(x)))
	poly.append(x)
	for i in range (1,deg):
		p = 2* x *poly[i-1] - poly[i-2]
		poly.append(p)
	poly_mat = np.matrix(poly[deg])
	return poly_mat # Returning the matrix of the polynomials.

def coeff(enodes,fnodes,d):
    theta=np.empty((d+1)*(d+1))
    theta.shape = (d+1,d+1)
    for i in range(d+1):
        for j in range(d+1):
            theta[i,j] = (np.sum(np.array(fnodes)*np.array(((T(i,enodes).T @ T(j,enodes)))))
                          /np.array((T(i,enodes)*T(i,enodes).T)*(T(j,enodes)*T(j,enodes).T)))
    return theta

# Approximation of the function
def f_a(x,y,theta,d):
    f = []
    val1 = ((2*((x-a)/(b-a)))-1)
    val2 = ((2*((y-a)/(b-a)))-1)
    for u in range(d):
        for v in range(d):
                f.append(np.array(theta[u,v])*np.array(((T(u,val1).T @ T(v,val2)))))
    f_sum = sum(f)
    return f_sum

#########Order 3
theta3 = coeff(nodes, wk, 2)
Z13 = f_a(k,h, theta3, 2)
Z= f(alpha,rho,K,H)
error3 = abs(Z-Z13)

fig = plt.figure(figsize=plt.figaspect(0.25))
fig.suptitle('σ=1, Order 3 plots', fontsize=16)

#Using subplots
# Actual Function at the nodes
ax = fig.add_subplot(131, projection='3d')
ax.set_title("Function Evaluated at the nodes, order = 3")
X,Y = np.meshgrid(k, h)
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Ouput')
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(132, projection='3d')
ax.set_title("Approximated Function, order = 3")
X,Y = np.meshgrid(k, h)
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, Z13, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(133, projection='3d')
ax.set_title("Approximated Function, order = 3")
X,Y = np.meshgrid(k, h)

# Put axis labels
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, error3, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)
plt.show()

#########Order 15
theta15 = coeff(nodes, wk, 15)
Z115 = f_a(k,h, theta15, 15)
Z= f(alpha,rho,K,H)
error15 = abs(Z-Z115)

fig = plt.figure(figsize=plt.figaspect(0.25))
fig.suptitle('σ=1, Order 15 plots', fontsize=16)

#Using subplots
# Actual Function at the nodes
ax = fig.add_subplot(131, projection='3d')
ax.set_title("Function Evaluated at the nodes, order = 15")
X,Y = np.meshgrid(k, h)
Z = f(alpha,rho,X,Y)

# Put axis labels
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Ouput')
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)
plt.show()

# Approximated function at the nodes
ax = fig.add_subplot(132, projection='3d')
ax.set_title("Approximated Function, order = 15")
X,Y = np.meshgrid(k, h)
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, Z115, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(133, projection='3d')
ax.set_title("Approximated Function, order = 15")
X,Y = np.meshgrid(k, h)

# Put axis labels
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, error15, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

#Creating the Isoquants 
pct_true = [5, 10, 25, 50, 75, 90, 95]
pct_approx = [5, 10, 25, 50, 75, 90, 95]

#Creating the percentiles for production
for i in pct_true:
    pert = np.asarray(np.percentile(Z, pct_true))
       
for i in pct_approx:
    pera = np.asarray(np.percentile(Z13, pct_approx))

#Creating the Contour lines 
X,Y = np.meshgrid(k, h)
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, pert)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('CES Function')
plt.show()

#Creating Contour lines for the approximated function 
X1,Y1 = np.meshgrid(k, h)
fig, ax = plt.subplots()
CS = ax.contour(X1, Y1, Z13, pera)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Approximated function')
plt.show()


# In[213]:


#Code worked on with the help of Elena and Pau
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.tri as mtri
from math import cos 
from matplotlib import cm

N  = 20
nodes = -(np.cos((2*np.arange(1,N+1)-1)/(2*N)*np.pi))

k = []
for i in nodes:
    a=0
    b=10
    m= ((i+1)*(b-a)/2) + a
    k.append(m)
k =np.asarray(k)

h = []
for i in nodes:
    a=0
    b=10
    m= ((i+1)*(b-a)/2) + a
    h.append(m)
    
h =np.asarray(h)

#Define the CES Function
def f(alpha,rho,k,h):
    return np.power((1-alpha)*k**rho + alpha*h**rho,1/rho)

#Evaluating the functions at the nodes
alpha = .5
sigma = .25
rho= (sigma -1)/sigma
K,H = np.meshgrid(k,h)
wk = f(alpha,rho,K,H)

#Creating the Polynomials for the Chebyshev approximation using the recursive formulation
def T(deg,x): #Defining n to be the degree 
	poly = []
	poly.append(np.ones(len(x)))
	poly.append(x)
	for i in range (1,deg):
		p = 2* x *poly[i-1] - poly[i-2]
		poly.append(p)
	poly_mat = np.matrix(poly[deg])
	return poly_mat # Returning the matrix of the polynomials.

def coeff(enodes,fnodes,d):
    theta=np.empty((d+1)*(d+1))
    theta.shape = (d+1,d+1)
    for i in range(d+1):
        for j in range(d+1):
            theta[i,j] = (np.sum(np.array(fnodes)*np.array(((T(i,enodes).T @ T(j,enodes)))))
                          /np.array((T(i,enodes)*T(i,enodes).T)*(T(j,enodes)*T(j,enodes).T)))
    return theta

# Approximation of the function
def f_a(x,y,theta,d):
    f = []
    val1 = ((2*((x-a)/(b-a)))-1)
    val2 = ((2*((y-a)/(b-a)))-1)
    for u in range(d):
        for v in range(d):
                f.append(np.array(theta[u,v])*np.array(((T(u,val1).T @ T(v,val2)))))
    f_sum = sum(f)
    return f_sum

#########Order 3
theta3 = coeff(nodes, wk, 2)
Z13 = f_a(k,h, theta3, 2)
Z= f(alpha,rho,K,H)
error3 = abs(Z-Z13)

fig = plt.figure(figsize=plt.figaspect(0.25))
fig.suptitle('σ=0.25, Order 3 plots', fontsize=16)

#Using subplots
# Actual Function at the nodes
ax = fig.add_subplot(131, projection='3d')
ax.set_title("Function Evaluated at the nodes, order = 3")
X,Y = np.meshgrid(k, h)
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Ouput')
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(132, projection='3d')
ax.set_title("Approximated Function, order = 3")
X,Y = np.meshgrid(k, h)
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, Z13, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(133, projection='3d')
ax.set_title("Approximated Function, order = 3")
X,Y = np.meshgrid(k, h)

# Put axis labels
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, error3, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)
plt.show()

#########Order 5
theta5 = coeff(nodes, wk, 5)
Z15 = f_a(k,h, theta5, 5)
Z= f(alpha,rho,K,H)
error5 = abs(Z-Z15)

fig = plt.figure(figsize=plt.figaspect(0.25))
fig.suptitle('σ=0.25, Order 5 plots', fontsize=16)

#Using subplots
# Actual Function at the nodes
ax = fig.add_subplot(131, projection='3d')
ax.set_title("Function Evaluated at the nodes, order = 5")
X,Y = np.meshgrid(k, h)
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Ouput')
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(132, projection='3d')
ax.set_title("Approximated Function, order = 5")
X,Y = np.meshgrid(k, h)
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, Z15, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(133, projection='3d')
ax.set_title("Approximated Function, order = 5")
X,Y = np.meshgrid(k, h)
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, error5, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)
plt.show()

#########Order 7
theta7 = coeff(nodes, wk, 7)
Z17 = f_a(k,h, theta7, 7)
Z= f(alpha,rho,K,H)
error7 = abs(Z-Z17)

fig = plt.figure(figsize=plt.figaspect(0.25))
fig.suptitle('σ=0.25, Order 7 plots', fontsize=16)

#Using subplots
# Actual Function at the nodes
ax = fig.add_subplot(131, projection='3d')
ax.set_title("Function Evaluated at the nodes, order = 7")
X,Y = np.meshgrid(k, h)
Z = f(alpha,rho,X,Y)

# Put axis labels
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Ouput')
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(132, projection='3d')
ax.set_title("Approximated Function, order = 7")
X,Y = np.meshgrid(k, h)
Z1 = f_a(k,h,theta7,7)

# Put axis labels
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, Z17, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(133, projection='3d')
ax.set_title("Approximated Function, order = 7")
X,Y = np.meshgrid(k, h)

# Put axis labels
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, error7, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)
plt.show()

#########Order 15
theta15 = coeff(nodes, wk, 15)
Z115 = f_a(k,h, theta15, 15)
Z= f(alpha,rho,K,H)
error15 = abs(Z-Z115)

fig = plt.figure(figsize=plt.figaspect(0.25))
fig.suptitle('σ=0.25, Order 15 plots', fontsize=16)

#Using subplots
# Actual Function at the nodes
ax = fig.add_subplot(131, projection='3d')
ax.set_title("Function Evaluated at the nodes, order = 15")
X,Y = np.meshgrid(k, h)
Z = f(alpha,rho,X,Y)

# Put axis labels
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Ouput')
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(132, projection='3d')
ax.set_title("Approximated Function, order = 15")
X,Y = np.meshgrid(k, h)
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, Z115, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(133, projection='3d')
ax.set_title("Approximated Function, order = 15")
X,Y = np.meshgrid(k, h)

# Put axis labels
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, error15, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)
plt.show()

#Creating the Isoquants 
pct_true = [5, 10, 25, 50, 75, 90, 95]
pct_approx = [5, 10, 25, 50, 75, 90, 95]

#Creating the percentiles for production
for i in pct_true:
    pert = np.asarray(np.percentile(Z, pct_true))
       
for i in pct_approx:
    pera = np.asarray(np.percentile(Z13, pct_approx))

#Creating the Contour lines 
X,Y = np.meshgrid(k, h)
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, pert)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('CES Function')
plt.show()

#Creating Contour lines for the approximated function 
X1,Y1 = np.meshgrid(k, h)
fig, ax = plt.subplots()
CS = ax.contour(X1, Y1, Z13, pera)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Approximated function')
plt.show()


# In[212]:


#Code worked on with the help of Elena and Pau
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.tri as mtri
from math import cos 
from matplotlib import cm

N  = 20
nodes = -(np.cos((2*np.arange(1,N+1)-1)/(2*N)*np.pi))

k = []
for i in nodes:
    a=0
    b=10
    m= ((i+1)*(b-a)/2) + a
    k.append(m)
k =np.asarray(k)

h = []
for i in nodes:
    a=0
    b=10
    m= ((i+1)*(b-a)/2) + a
    h.append(m)
    
h =np.asarray(h)

#Define the CES Function
def f(alpha,rho,k,h):
    return np.power((1-alpha)*k**rho + alpha*h**rho,1/rho)

#Evaluating the functions at the nodes
alpha = 0.5
sigma = 5
rho= (sigma -1)/sigma
K,H = np.meshgrid(k,h)
wk = f(alpha,rho,K,H)

#Creating the Polynomials for the Chebyshev approximation using the recursive formulation
def T(deg,x): #Defining n to be the degree 
	poly = []
	poly.append(np.ones(len(x)))
	poly.append(x)
	for i in range (1,deg):
		p = 2* x *poly[i-1] - poly[i-2]
		poly.append(p)
	poly_mat = np.matrix(poly[deg])
	return poly_mat # Returning the matrix of the polynomials.

def coeff(enodes,fnodes,d):
    theta=np.empty((d+1)*(d+1))
    theta.shape = (d+1,d+1)
    for i in range(d+1):
        for j in range(d+1):
            theta[i,j] = (np.sum(np.array(fnodes)*np.array(((T(i,enodes).T @ T(j,enodes)))))
                          /np.array((T(i,enodes)*T(i,enodes).T)*(T(j,enodes)*T(j,enodes).T)))
    return theta

# Approximation of the function
def f_a(x,y,theta,d):
    f = []
    val1 = ((2*((x-a)/(b-a)))-1)
    val2 = ((2*((y-a)/(b-a)))-1)
    for u in range(d):
        for v in range(d):
                f.append(np.array(theta[u,v])*np.array(((T(u,val1).T @ T(v,val2)))))
    f_sum = sum(f)
    return f_sum

#########Order 3
theta3 = coeff(nodes, wk, 2)
Z13 = f_a(k,h, theta3, 2)
Z= f(alpha,rho,K,H)
error3 = abs(Z-Z13)

fig = plt.figure(figsize=plt.figaspect(0.25))
fig.suptitle('σ=5, Order 3 plots', fontsize=16)

#Using subplots
# Actual Function at the nodes
ax = fig.add_subplot(131, projection='3d')
ax.set_title("Function Evaluated at the nodes, order = 3")
X,Y = np.meshgrid(k, h)
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Ouput')
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(132, projection='3d')
ax.set_title("Approximated Function, order = 3")
X,Y = np.meshgrid(k, h)
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, Z13, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(133, projection='3d')
ax.set_title("Approximated Function, order = 3")
X,Y = np.meshgrid(k, h)

# Put axis labels
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, error3, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)
plt.show()

#########Order 15
theta15 = coeff(nodes, wk, 15)
Z115 = f_a(k,h, theta15, 15)
Z= f(alpha,rho,K,H)
error15 = abs(Z-Z115)

fig = plt.figure(figsize=plt.figaspect(0.25))
fig.suptitle('σ=5, Order 15 plots', fontsize=16)

#Using subplots
# Actual Function at the nodes
ax = fig.add_subplot(131, projection='3d')
ax.set_title("Function Evaluated at the nodes, order = 15")
X,Y = np.meshgrid(k, h)
Z = f(alpha,rho,X,Y)

# Put axis labels
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Ouput')
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(132, projection='3d')
ax.set_title("Approximated Function, order = 15")
X,Y = np.meshgrid(k, h)
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, Z115, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

# Approximated function at the nodes
ax = fig.add_subplot(133, projection='3d')
ax.set_title("Approximated Function, order = 15")
X,Y = np.meshgrid(k, h)

# Put axis labels
ax.set_xlabel('Capital')
ax.set_ylabel('Labour')
ax.set_zlabel('Output')
ax.plot_surface(X, Y, error15, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)
plt.show()

#Creating the Isoquants 
pct_true = [5, 10, 25, 50, 75, 90, 95]
pct_approx = [5, 10, 25, 50, 75, 90, 95]

#Creating the percentiles for production
for i in pct_true:
    pert = np.asarray(np.percentile(Z, pct_true))
       
for i in pct_approx:
    pera = np.asarray(np.percentile(Z13, pct_approx))

#Creating the Contour lines 
X,Y = np.meshgrid(k, h)
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, pert)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('CES Function')
plt.show()

#Creating Contour lines for the approximated function 
X1,Y1 = np.meshgrid(k, h)
fig, ax = plt.subplots()
CS = ax.contour(X1, Y1, Z13, pera)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Approximated function')
plt.show()

