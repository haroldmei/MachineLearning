
## 6.1 Introduction
Statistical inference or "learning" as it is called in CS, is the process of using data to infer the distribution that generated the data.

## 6.2 Parametric and Nonparametric Models
A statistical model $\mathfrak{F}$ is a set of distributions/densities/regression functions. A parametric model is a set $\mathfrak{F}$ that can be parameterized by a finite number of parameters. For example, if we assume that the data is from a Normal distribution, it's model is:  
$\mathfrak{F}=\{f(x;\mu,\delta)-\frac{1}{\delta \sqrt{2\pi}}exp\{-\frac{1}{2\delta^2}(x-\mu)^2 \}, \mu \in \mathbb{R}, \delta > 0 \} \tag{6.1}$  
In general a parametric model takes the form:  
$\mathfrak{F}=\{f(x;\mu,\theta): \theta \in \Theta \}$ where $\Theta$ is the parameter space.  
If $\theta$ is a vector and we only interested in one compoment of it, the remaining part is called 'nuisance parameters'.  
A nonparametric model is a set $\mathfrak{F}$ that cannot be parameterized by a finite number of parameters. For example $\mathfrak{F}_{ALL}={all CDF} $ is nonparametric.   

E6.1 (One-dimensional Parametric Estimation). Let iid $X_i$~Bernoulli(p), the problem is to estimate the parameter p.

E6.2 (Two-dimensional Parametric Estimation). Let iid $X_i$~F and we assume that the PDF $f \in \mathfrak{F}$ given in (6.1). There are two parameters $\mu,\delta$, the goal is to estimate the two parameters from the data.   

E6.3 (Nonparametric estimation of the CDF). Let iid $X_i$ be independent observations from CDF F. The problem is to estimate F assuming only that $F \in \mathfrak{F}$. In chapter 7 the empirical distribution function $\hat{F}$ is one example.    

E6.4 (Nonparametric density estimation). Let $X_i$ be iid from CDF F and leet f=F' be the PDF. Suppose we want to estimate PDF. It is not possible to estimate f assuming only that $F\in \mathfrak{F}$. We need to assume some smoothness on f. For example we might assume that $f \in \mathfrak{F} =\mathfrak{F}_{DENS} \cap \mathfrak{F}_{SOB}$ where $\mathfrak{F}_{DENS}$ is the set of all probability density functions and $\mathfrak{F}_{SOB}=\{ f:\int(f''(x))^2dx < \infty \}$. The class $\mathfrak{F}_{SOB}$ is called a Sobolev space; it is the set of functions that are not too 'wiggly'.  

E6.5 (Nonparametric estimation of functionals). Let iid $X_i$ ~ F. A Statistical Functional is a function of F, denoted as T(F). For example the mean $\mu=T(F)=\int x dF(x)$ is a statistical functional. Other examples include variance, skewness, coorelation, etc.    

E6.6 (Regression, prediction and classification). Suppose we observe pairs of data ($X_i, Y_i$). Perhaps $X_i$ is the blood pressure of subject i and $Y_i$ is how long they live.  
X: predictor, regressor, feature, independent variable;  
Y: outcome, response, dependent variable.  
$r(x)=\mathbb{E}(Y|X=x)$ is called regression function. Think about linear regression. The cost function of LR is $J(\theta)=\frac{1}{2}\sum_{i=1}^m (h_{\theta}(x^{(i)})-y^{(i)})^2$ and the task is to estimate $h_{\theta}$ given the data. But how is it associated with the 'regression function' we mentioned here? If we interpret LR in probabilistic way, and assume the response variable satisfies: $Y|X \sim N(\theta^T X, \delta^2)$ where $\delta$ is some variance, then the regression function is just $r(x)=\mathbb{E}(Y|X=x)$ based on the distribution, and in LR case it is just the mean of Normal: $r(x) = h_{\theta} = \theta^T x$.   
If $r\in\mathfrak{F}$ where $\mathfrak{F}$ is finite dimensional, the set of straight lines for example, then we have a parametric regression model. If $\mathfrak{F}$ is not finite dimensional then we have a nonparametric regression model. The goal of predicting Y for a given X is called Prediction. If Y is discrete then the prediction is called classification. The goal of estimation the function r is called regression or curve estimation.    
The regression model is also written as $Y=r(X)+\epsilon$ where $\mathbb{E}(\epsilon)=0$.   
PROOF: $\mathbb{E}(Y)=\mathbb{E}(r(X)+\epsilon)=\mathbb{E}[\mathbb{E}(Y|X)]+\mathbb{E}(\epsilon)=\mathbb{E}(Y)+\mathbb{E}(\epsilon)$ which gives $\mathbb{E}(\epsilon)=0$   

## 6.3 Fundamental Concepts in Inference  
### 6.3.1 Point estimation
Point estimation refers to providing a single "best guess" of some quantity of interest. The quantity of interest could be a parameter in a parametric model, a CDF F, a probability density function f, a regression function r, or a prediction for a future value Y of some random variable.  
A point estimation of $\theta$ is denoted by $\hat{\theta}$ or $\hat{\theta_n}$. $\theta$ is a fixed unknown quantity in the universe, it is also called the 'true value'.    
More generally a point estimator $\hat\theta_n$ of a parameter $\theta$ is some function of $X_i$: $\hat\theta_n=g(X_1,X_2,...,X_n)$. Since the iid data points are random variables, its dependent $\hat\theta_n$ is also a r.v. and we can find out its distribution based on X.  
The bias of an estimator is defined as: $bias(\hat{\theta_n})=\mathbb{E}_{\theta}(\hat{\theta_n})-\theta$. If $\mathbb{E}(\hat{\theta_n})=\theta$ we say that the estimation is unbiased. Although many of the estimators in use are biased, a reasonable requirement is it should converge to the true value as we collect more and more data.   
D6.7 A point estimator $\hat\theta_n$ of a parameter $\theta$ is consistent if $\hat\theta_n \overset{p}{\longrightarrow} \theta$  
Again since $\hat\theta_n$ is a r.v. and has its own distribution, its distribution is called sampling distribution. The standard deviation of $\hat\theta_n$ is called standard error denoted by: $$se=se(\hat\theta_n)=\sqrt{\mathbb{V}(\hat\theta_n)}$$.  
Often, the standard error depends on the unknown F. In those cases, se is an unknown quantity but we usually can estimate it. The estimated se is denoted by $\hat{se}$.   

E6.8 Let iid X~Bernoulli(p) and let $\hat p_n = \sum X_i /n$. Then $\mathbb{E}(\hat p_n) = n^{-1} \sum \mathbb{E}(X_i) = p$ so $\hat p_n$ is unbiased; The standard error $se=\sqrt{\mathbb{V}(\hat p_n)}=\sqrt{p(1-p)/n}$, the estimated standard error is $\hat{se}=\sqrt{\hat p_n (1 - \hat p_n)/n}$    
The quality of a point estimation is sometimes assessed by MSE: Mean Squre Error: $MSE=\mathbb{E}_\theta (\hat\theta_n - \theta)^2$   
T6.9 The MSE can be written as $MSE=bias^2(\hat\theta_n)+\mathbb{V}(\hat\theta_n)$  
PROOF: $bias^2+\mathbb{V} \\ = \mathbb{E}_\theta^2(\hat\theta_n) - 2\theta \mathbb{E}_\theta(\hat\theta_n) + \theta^2 + \mathbb{E}_\theta(\hat\theta_n^2)-\mathbb{E}_\theta^2(\hat\theta_n) \\ = \mathbb{E}_\theta(\hat\theta_n^2) - 2\theta \mathbb{E}_\theta(\hat\theta_n) + \theta^2 \\ = \mathbb{E}_\theta^2(\hat\theta_n - \theta)$   

T6.10 If $bias \longrightarrow 0$ and $se \longrightarrow 0$ as $n \longrightarrow \infty$ then $\hat\theta_n$ is consistent, that is, $\hat\theta_n \overset{p}{\longrightarrow} \theta $.   
PROOF: Since $bias \longrightarrow 0$ and $se \longrightarrow 0$, by definition we have $MSE \longrightarrow 0$, which means $\hat\theta_n \overset{qm}{\longrightarrow} \theta$, this implies that $\hat\theta_n \overset{p}{\longrightarrow} \theta $.     

E6.11 Return to coin flipping example, we have $\mathbb{E}_p(\hat\theta_n)=p$, so the bias is 0; we also have $se=\sqrt{p(1-p)/n} \longrightarrow 0$, hence we have $\hat p_n \overset{p}{\longrightarrow} p $, $\hat p_n$ is a consistent estimator.   

D6.12 An estimator is asymptotically Normal if $$\frac{\hat\theta_n-\theta}{se} \rightsquigarrow N(0,1)$$.    

### 6.3.2 Confidence Sets
A 1-$\alpha$ confidence interval for a parameter $\theta$ is an interval $C_n=(a,b)$ where $a=a(X_1,X_2,...,X_n)$ and $b=b(X_1,X_2,...,X_n)$ are functions of the data such that $$\mathbb{P}(\theta \in C_n) \ge 1-\alpha, for all \theta \in \Theta$$.   
A confidence interval is not a probability statement about $\theta$ since $\theta$ is a fixed quantity, not a r.v. It is actually a probability statement about its estimation $\hat \theta_n$ since the estimation is a r.v. and can be trapped within (a,b) with 95% probability.   

E6.13 Everyday newspapers report opinion polls. For example, they might say that "83 percent of the population favor arming pilots with guns." Usually, you will see a statement like "this poll is accurate to within 4 points 95 percent of the time." They are saying that 84 +/- 4 is a 95 percent confidence interval for the true but unknown proportion p of people who favor arming pilots with guns.  

E6.15 In the coin flipping setting. Let $C_n=(\hat p_n - \epsilon_n, \hat p_n + \epsilon_n$ where $\epsilon_n^2=log(2/\alpha)/(2n)$. From Hoeffding's inequality(4.4) it follows that && \mathbb{P}(p \in \C_n) \ge 1-\alpha &&  
From T4.5:   
$\mathbb{P}(|\overline{X_n}-p|>\epsilon) \le 2exp(-2n\epsilon^2) \\ 
\Rightarrow \mathbb{P}(|\hat p_n - p| < \epsilon) \ge 1 - 2exp(-2n\epsilon^2) \\ 
\Rightarrow 1-\alpha = 1-2exp(-2n\epsilon^2) \\
\Rightarrow \epsilon^2 = log(2/\alpha)/(2n)$   

T6.16 (Normal-based Confidence Interval). Suppose that $\hat \theta_n \approx N(\theta, \hat {se}^2)$. Let $\Phi$ be the CDF of a standard Normal and let $z_{\alpha/2} = \Phi^{-1}(1 - \alpha/2)$, that is, $\mathbb{P}(Z > z_{\alpha/2})$ and $\mathbb{P}(-z_{\alpha/2} < Z < z_{\alpha/2}) = 1 - \alpha$ where Z ~ N(0,1).  Let $$ C_n=(\hat \theta_n - z_{\alpha/2}\hat{se}, \hat\theta_n + z_{\alpha/2}\hat{se}) \tag{6.10} $$  
Then $$  \mathbb{P}(\theta \in C_n) \longrightarrow 1-\alpha  $$   
PROOF


## 6.3.3 Hypothesis testing



Exercises
1. Let X~Poiss($\lambda$) and let $\hat \lambda = n^{-1}\sum_1^n X_i$. Find the bias, se and MSE of this estimator.  
$bias(\hat\lambda)=\mathbb{E}_\lambda(\hat\lambda)-\lambda = \lambda - \lambda = 0$  
$se = \sqrt{\mathbb{V}(\hat\lambda)}=\sqrt{\lambda/n}$  
$MSE=\mathbb{E}_\lambda(\hat\lambda - \lambda)^2 = bias(\hat\lambda)^2 + se(\hat\lambda)^2 = \lambda/n \longrightarrow 0$  

2. Let X~U(0,$\theta$) and let $\theta=max(X_1,X_2,...,X_n)$. Find the bias,se and MSE of the estimator.  
$bias(\hat\theta)=\mathbb{E}_\theta(\hat\theta)-\theta \\ 
= \mathbb{E}_\theta(max(X_1,X_2,...,X_n))-\theta \\ 
= max(\mathbb{E}_\theta(X_1),\mathbb{E}_\theta(X_2),...,\mathbb{E}_\theta(X_n))-\theta  \\
= \frac{\theta}{2}$   
$se = \sqrt{\mathbb{V}(\hat\theta)} \\
=\sqrt{\mathbb{V}(max(X_1,X_2,...,X_n))} \\
=\sqrt{max(\mathbb{V}(X_1),\mathbb{V}(X_2),...,\mathbb{V}(X_n))} \\
=\sqrt{\theta_2/12} = \frac{\theta}{2\sqrt{3}}$   
$MSE = bias(\hat\theta)^2 + se(\hat\theta)^2 = \frac{\theta^2}{3}$   

3. Let X~U(0,$\theta$) and let $\theta=2\overline(X_n)$. Find the bias,se and MSE of this estimator.  
$\mathbb{E}(\hat\theta) = 2\mathbb{E}(\overline{X_n})=\frac{2}{n}\sum_1^n \mathbb{E}(X_i) = 2 * \frac{\theta}{2} = \theta$  
$bias(\hat\theta)=\mathbb{E}_\theta(\hat\theta)-\theta = \theta - \theta = 0$, it is unbiased;   
$se = \sqrt{\mathbb{V}(\hat\theta)}=\sqrt{\frac{4}{n^2} \mathbb{V}(\sum_1^n X_i)} = \sqrt{\frac{4}{n^2} \sum_1^n \mathbb{V}(X_i)} = \sqrt{\frac{4}{n} \mathbb{V}(X)} = \sqrt{\frac{4}{n} \frac{\theta^2}{12}} = \frac{\theta}{\sqrt{3n}} \longrightarrow 0 $    
$MSE = bias(\hat\theta)^2 + se(\hat\theta)^2 = \theta_2/(3n) \longrightarrow 0$   
