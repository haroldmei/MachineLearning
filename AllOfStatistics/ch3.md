## Chapter 3

# 3.1 Expectation of a r.v.  
D3.1 Expected value: $ \mathbb{E}(X)=\int x dF(x)=\mu=\mu_X $ is defined as following. If $\mathbb{E}(X)$ exists, then $\int_x |x|dF(x)<\infty$   
$$ 
\mathbb{E}(X) = \int x dF(x) = 
\begin{cases}
  \sum_x x f(x)     & \text{if X is discrete } \\
  \int x f(x)dx     & \text{if X is continuous }
\end{cases}
$$

E3.2 Let X ~ Bernoulli(p). $\mathbb{E}(X)=\sum_{x=0}^1 xf(x) = 0 \times (1-p) + 1 \times p = p$  
E3.3 Let X ~ Binomial(n,p), $\mathbb{E}(X)=\sum_{x=0}^n xf(x) = C_n^0 p^0 (1-p)^n + C_n^1 p^1 (1-p)^{n-1} + C_n^2 p^2 (1-p)^{n-2} + ... + C_n^n p^n (1-p)^0$, using binomial expansion the result is $(p + 1 - p)^n = 1$. Example in the book is just a special case where n = 2 and p = .5   
E3.4 Let X ~ Uniform(-1,3), $\mathbb{E}(X) = \int_{-1}^{3} xf(x)dx = \int_{-1}^{3} \frac{1}{4} xdx = 1 $  
E3.5 Let X ~ Cauchy. $f_X(x) = \frac{1}{\pi (1+x^2)}$. In D3.1 we know how to check if $\mathbb{E}(X)$ exists. Let's check Cauchy distribution   
When you see $\int \frac{1}{1+x^2}dx$, you should know that this integral is just arctan(x). Using integration by parts we have:   
$\int |x|dF(x)=\frac{2}{\pi} \int_0^{\infty} \frac{x}{1+x^2}dx = [x arctan(x)]_0^{\infty}-\int_0^{\infty}arctan(x)dx = \infty$. Which means the expectation of Cauchy doesn't exist.   
From $\int \frac{1}{1+x^2}dx$ to arctan(x): $x=tan(v), v=arctan(x) \Rightarrow \frac{dx}{dv}=1+tan^2(v) = 1+x^2 \Rightarrow \frac{dv}{dx} = \frac{1}{1+x^2}$   

T3.6 The Rule of the Lazy Statistician. Let Y=r(X), then $\mathbb{E}(Y)=\mathbb{E}(r(X))=\int r(x)dF(x)$  
E3.7 Let X ~ Unif(0,1). Let $ Y=r(X)=e^X $. Then $\mathbb{E}(Y)=\int_0^1 e^xf(x)dx=\int_0^1e^xdx=e^x bracevert_{0}^{1} = e-1$  
E3.8 Break a unit length stick at random and let Y be the longer piece, What is the mean of Y? If X is the break point then X ~ Unif(0,1); Y=r(X)=max{X, 1-X} hence:   
$\mathbb{E}(Y)=\int r(x)dF(x)=\int_0^{1/2}(1-x)dx+\int_{1/2}^1xdx = \frac{3}{4}$   
E3.9 Let (X,Y) have a jointly uniform distribution on the unit square, let $Z=r(X,Y)=X^2+Y^2$, then  
$\mathbb{E}(Z) = \iint r(x,y)dF(x,y) = \iint_{0,0}^{1,1}(x^2+y^2)dxdy=\int_0^1 x^2dx + \int_0^1 y^2dy = \frac{2}{3}$  

THE KTH MOMENT of X is defined to be $\mathbb{E}(X^k)$ assuming that $\mathbb{E}(|X|^k)<\infty$  
T3.10 If the kth moment exists and if j < k then the jth moment exists.  
PROOF:   

# 3.2 Properties of Expectations  


