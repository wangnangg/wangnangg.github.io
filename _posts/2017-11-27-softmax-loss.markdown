---
layout: post
title:  "CS231n: Softmax Loss Computation"
date:   2017-11-27
categories: note
---

Let $$l$$ be the loss, $$r$$ be the regularization strength, $$S = X \cdot W$$:


$$
\begin{split}
S_{i,j} &= X_{i, 1} W_{1, j} + X_{i, 1} W_{1, j} + ... + X_{i, D}W_{D, j} \\
L_{i,j} &= \frac{e^{S_{i,j}}}{\sum_k e^{S_{i,k}}} = \frac{e^{S_{i,j} - \max \{S_{i,k}\} }}{\sum_k e^{S_{i,k}- \max \{S_{i,k}\} }} \\
l &= -\sum_{i}\log(L_{i,y[i]})/N +r\sum_{i, j}W_{i,j}W_{i,j}
\end{split}
$$


Code:

```python
N = X.shape[0]
S = np.dot(X, W)
Sm -= np.max(S, axis=1).reshape(N, 1)
L = np.exp(Sm)
loss = -np.mean(np.log(L[np.arange(N), y])) + reg *np.sum(W * W)
```

