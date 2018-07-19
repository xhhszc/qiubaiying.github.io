---
layout:     post
title:      Hoeffding霍夫丁不等式及其在集成学习理论的应用
subtitle:   The Application of Hoeffding Inequality in Ensemble Learning
date:       2018-04-17
author:     xhhszc
catalog: true
tags:
    - Machine Learning
    - Ensemble Learning
    - Hoeffding Inequality
---

# Hoeffding霍夫丁不等式及其在集成学习理论的应用
------

Hoeffding霍夫丁不等式
------------------------
机器学习中，算法的泛化能力往往是通过研究泛化误差的概率上界所进行的，这个就称为泛化误差上界。直观的说，在有限的训练数据中得到的规律，则认为真实的总体数据中也是近似这个规律的。比如一个大罐子里装满了红球和白球，各一半，我随手抓了一把，然后根据这些红球白球的比例预测整个罐子也是这样的比例，这样做不一定很准确，但结果总是近似的，而且如果抓出的球越多，预测结果也就越可信。

对于两种不同的学习方法，通常比较他们的误差上界来决定他们的优劣。hoeffding不等式于1963年被Wassily Hoeffding提出并证明，用于计算随机变量的和与其期望值偏差的概率上限。下面我们理清hoeffding 不等式的来龙去脉。

# 1.伯努利随机变量的特例
我们假定一个硬币A面朝上的概率为$p$，则B面朝上的概率为$1-p$。抛n次硬币，A面朝上次数的期望值为$n*p$。则A面朝上的次数不超过k次的概率为：
\begin{equation}
P(H(n)\leq k)=\sum_{i=0}^kC_n^ip^i(1-p)^{n-i}\\
=\sum_{i=0}^k\frac{n!}{i!(n-i)!}p^i(1-p)^{n-i}
\end{equation}
其中$H(n)$为抛n次硬币A面朝上的次数。

对某一$\varepsilon>0$当$k=(p-\varepsilon)n$时，有Hoeffding不等式
\begin{equation}
P(H(n)\leq (p-\varepsilon)n) \leq e^{-2\varepsilon^2n}
\end{equation}
对应的，当$k=(p+\varepsilon)n$时，
\begin{equation}
P(H(n)\geq (p+\varepsilon)n) \leq e^{-2\varepsilon^2n}
\end{equation}
由此我们可以推导出
\begin{equation}
P((p-\varepsilon)n\leq H(n)\leq (p+\varepsilon)n) \geq 1-2e^{-2\varepsilon^2n}
\end{equation}
特别的，当$\varepsilon=\sqrt{\frac{\ln n}{n}}$时，
\begin{equation}
P(|H(n)-pn|\leq \sqrt{n\ln n}) \geq 1-2e^{-2\ln n}=1-\frac{2}{n^2}
\end{equation}

# 2.伯努利随机变量的一般情况
令独立同分布随机变量$X_1,X_2,...,X_n$，其中$X_i\in[a_i,b_i]$，则这些变量的经验均值为：$\bar{X}=\frac{X_1+X_2+,...,+X_n}{n}$
对于任意$t>0$有
\begin{equation}
P(|\bar X-E(\bar X)|\geq t) \leq 2e^{-\frac{2n^2t^2}{\sum_{i=1}^n(b_i-a_i)^2}}
\end{equation}
或$S_n = X_1+X_2+,...,+X_n$
\begin{equation}
P(|S_n-E(S_n)|\geq t) \leq 2e^{-\frac{2t^2}{\sum_{i=1}^n(b_i-a_i)^2}}
\end{equation}

> 证明如下：
> 霍夫丁引理：假设X为均值为0的随机变量且满足$P(X\in[a,b])=1$，有以下不等式成立：
> $$E(e^{sX})\leq e^{\frac{s^2(b-a)^2}{8}}$$
> 则对于独立随机变量$X_1,X_2,...,X_n$满足$P(X_i\in[a_i,b_i])=1$，对于$t>0$：
> $$P(S_n-E(S_n)\geq t) =P(e^{s(S_n-E(S_n))}\geq e^{st})\\
\leq e^{-st}E(e^{s(S_n-E(S_n))})\\
=e^{-st}\prod_{i=1}^n E(e^{s(X_i-E(X_i))})\\
\leq e^{-st}\prod_{i=1}^n E(e^{\frac{s^2(b_i-a_i)^2}{8}})\\
=exp(-st+0.125s^2\sum_{i=1}^n(b_i-a_i)^2)$$
>令$g(s)= -st+0.125s^2\sum_{i=1}^n(b_i-a_i)^2$，则$g(s)$为二次函数，当$s = \frac{4t}{\sum_{i=1}^n(b_i-a_i)^2}$时函数获得最小值。因此：
>$$P(S_n-E(S_n)\geq t) \leq e^{-\frac{2t^2}{\sum_{i=1}^n(b_i-a_i)^2}}$$

# 3.集成学习的错误率上界
类似于抛硬币的例子，对于集成学习中基学习器的错误率$\epsilon$,
\begin{equation}
P(H(n)\leq k)=\sum_{i=0}^kC_n^i(1-\epsilon)^i\epsilon^{n-i}
\end{equation}
表示n个基学习器中分类正确的个数小于k的概率。若假定集成通过简单投票法结合n个分类器，超过半数的基学习器正确，则集成分类就正确，即$k=n/2=(1-\epsilon-\varepsilon)n$：
$$P(集成分类错误)=P(H(n)\leq \frac{n}{2})\\
=\sum_{i=0}^{\frac{n}{2}}C_n^i(1-\epsilon)^i\epsilon^{n-i}\\
\leq exp(-\frac{n}{2}(1-2\epsilon^2))\quad(由\varepsilon=\frac{1}{2}-\epsilon 可得)$$
其中，$\varepsilon=\frac{1}{2}-\epsilon>0$，也就是说，当错误率$\varepsilon<0.5$时，随着集成中基学习器的数目n的增大，集成的错误率将指数级下降，最终趋向于0。而当错误率$\varepsilon\geq0.5$时，以上式子不成立。


---------------------------------------

参考文章：

【1】[机器学习数学原理（8）——霍夫丁不等式](https://blog.csdn.net/z_x_1996/article/details/73564926)

【2】[Hoeffding不等式的认识以及泛化误差上界的证明](https://www.jianshu.com/p/f0a053e85f42)
