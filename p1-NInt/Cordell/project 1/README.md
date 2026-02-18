# Introduction

This project demonstrates and explores several numeric integration techniques.

The first technique explored is the trapezoid rule. The trapezoid rule takes the average of the leftpoint and rightpoint rules. These methods incorporate Reimann sums which split the area under the curve into N rectangles and adds up the area of each rectangle. The height of each rectangle is typically taken at the left, right, or middle of the rectangle--- hence the names of the rules: leftpoint, rightpoint and midpoint. As one might imagine, the estimate becomes more accurate at higher values of N. Specifically, as N increases the error of leftpoint or rightpoint decreases proportionally to $\frac{1}{N}$ and the error of midpoint or trapezoid decreases proportionally to $\frac{1}{N^{2}}$.

Simpson's method is a weighted average of the midpoint and trapezoid rule where midpoint is weighted by a factor of $\frac{2}{3}$ and the trapezoid is weighted by a factor of $\frac{1}{3}$. It is the most accurate out of the methods so far. As N increases, the error reduces by a factor of $\frac{1}{N^{4}}$.

The Guassian quadrature method is a more refined and much more interesting method than any of the previous methods that revolve around Reimann sums. It utilizes Legendre polynomials to represent the integrand. Recall that Legendre polynomials are an orthonormal set of polynomials, and so they can be used to represent any polynomial exactly. Furthermore, if the integrand isn't a polynomial, Legendre polynomials can still be utilized to approximate it. The Guassian quadrature method approximates the integrand using Legendre polynomials and then integrates the constructed polynomial exactly. This is expressed as: 

```math
\int_{-1}^{1} \mathrm{d}x\, f(x) \approx \sum_{i=1}^N c_{N,i} f\left(x_{N,i}\right)                 (1)
```

where $`x_{N,i}`$ are the sample points chosen at the roots of the Legendre polynomials and the weights are given by

```math
c_{i,n}=\frac{1}{P_n^{\prime}(x_{N,i})}\int_{-1}^1\frac{P_n(x)}{x-x_{N,i}} \mathrm{d}x                  (2)
```

Code.py is a script that demonstrates the numerical integration techniques discussed above.

To run the code.py script open the command prompt from the directory you have the it in:
```cmd
conda activate [the name of your environment]
python code.py
```
It outputs the estimated result of the integral 

```math
I = \int_0^2 \mathrm{d}x\, \sin^2\left(\sqrt{100x}\right)                   (3)
```

using the trapezoid rule along with the error and number of subintervals. Then, it outputs the estimated result using the Simpson rule. Lastly, it outputs these results using the Gaussian quadrature.

# Procedure
Here are some highlight-worthy segments of code:

```python
def leftpoint(f,a,b,N):
    mysum = 0
    h = f
    x_array = np.linspace(a, b, N)
    w = (b-a)/(N-1)
    A_array = h(x_array)*w
    mysum = np.sum(A_array)
    mysum = mysum - A_array[N-1]      
    return mysum

def rightpoint(f,a,b,N):
    mysum = 0
    h = f
    x_array = np.linspace(a, b, N)
    w = (b-a)/(N-1)
    A_array = h(x_array)*w
    mysum = np.sum(A_array)
    mysum = mysum - A_array[0]     
    return mysum 

def trapezoid(f, a, b, N):
    return 0.5* (leftpoint(f, a, b, N) + rightpoint(f, a, b, N))
```

The leftpoint and rightpoint functions segment the region being integrated over into N-1 equal width rectangles. The height is calculated using the value of the integrand when x equals the left or right end of each rectangle. The area of each rectangle is calculated and summed to approximate the integral.

```python
def quad(a, b, N):
    roots, weights = sp.special.roots_legendre(N)
    x = ((b-a)*roots/2)+(a+b)/2
    dx_over_du = 2/(b-a)
    return dx_over_du* np.sum(weights*sin(x))
```

The Gaussian quadrature function selects the specific sample points and their relative weights using the scipy special functions library. The integral is parameterized from -1 to 1 because that is where the Legendre polynomials are defined. Then the results are summed over following Eq. (1).

# Analysis
Table 1 compares the approximated results of Integral (3) using the trapezoid rule and the Gaussian quadrature at subintervals incremented by powers of 2.