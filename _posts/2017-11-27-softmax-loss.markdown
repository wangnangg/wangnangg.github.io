---
layout: post
title:  "CS231n: Softmax Loss Computation"
date:   2017-11-27
categories: note
---

# Value of Loss

Let $$l$$ be the loss, $$r$$ be the regularization strength, $$S = X \cdot W$$:


$$
\begin{split}
S_{i,j} &= X_{i, 1} W_{1, j} + X_{i, 2} W_{2, j} + ... + X_{i, D}W_{D, j} \\
L_{i,j} &= \frac{e^{S_{i,j}}}{\sum_k e^{S_{i,k}}} = \frac{e^{S_{i,j} - \max \{S_{i,k}\} }}{\sum_k e^{S_{i,k}- \max \{S_{i,k}\} }} \\
l &= -\sum_{i}\log(L_{i,y[i]})/N +r\sum_{i, j}W_{i,j}W_{i,j}
\end{split}
$$


Code:

```python
N = X.shape[0]
S = np.dot(X, W)
Sm =S -  np.max(S, axis=1).reshape(N, 1)
L = np.exp(Sm)
L /= np.sum(L, axis=1).reshape(N, 1)
loss = -np.mean(np.log(L[np.arange(N), y])) + reg *np.sum(W * W)
```

# Gradient of Loss

Consider the derivative of $$W_{m, n}$$:


$$
l'= -\sum_{i}\frac{1}{L_{i, y[i]}}L'_{i,y[i]}/N +2rW_{m,n}
$$


If $$y[i] \neq n$$:


$$
\begin{split}
L'_{i, y[i]} = -\frac{e^{S_{i,y[i]}} e^{S_{i,n}}X_{i,m}}{(\sum_ke^{S_{i,k}})^2} = -L^2_{i, y[i]}e^{S_{i,n}-S_{i, y[i]}}X_{i,m}
\end{split}
$$


If $$y[i] = n$$:


$$
L'_{i, y[i]} = L_{i,n}X_{i,m} - L^2_{i, n}X_{i,m} = (L_{i, n} - L^2_{i,n})X_{i,m}
$$




Code:

```python
N = X.shape[0]
Se = S - S[np.arange(N), y].reshape(N, 1)
Le = -np.square(L[np.arange(N), y]).reshape(N, 1) * np.exp(Se)
Le[np.arange(N), y] = L[np.arange(N), y] - np.square(L[np.arange(N), y])
Le /= -L[np.arange(N), y].reshape(N, 1)
dW = np.dot(X.T, Le) / N + 2 * reg * W
```

