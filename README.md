# CM_Project
Here's how you can format this text as Markdown for GitHub:

---

# Project 19

**(M)** is so-called **extreme learning**, i.e., a neural network with one hidden layer, $$y = w\sigma(W_1 x)$$, where the weight matrix for the hidden layer $$W_1$$ is a fixed random matrix, $$\sigma(\cdot)$$ is an elementwise activation function of your choice, and the output weight vector $$w$$ is chosen by solving a $$L_2$$ minimization problem:

$$ \min_w f(w) $$

with

$$ f(w) = \| X w - y \|_2 $$

---

**(A1)** is your own implementation of the QR factorization technique, which must obtain linear cost in the largest dimension.

---

**(A2)** is incremental QR, that is, a strategy in which you update the QR factorization after the addition of a new random feature column $$x_{n+1}$$to the feature matrix $$X$$. More formally, you are required to write a function that takes as an input the factors of an already-computed factorization $$X = QR$$ where $$Q$$ is stored either directly or via its Householder vectors, at your choice) of $$X \in \mathbb{R}^{m \times n}$$, and uses them to compute a factorization $$\hat{X} = \hat{Q} \hat{R}$$ of the matrix $$\hat{X} = [X | x_{n+1}] \in \mathbb{R}^{m \times (n+1)}$$, reusing the work already done. This update should have a cost lower than $$O(m n^2)$$.

---

No off-the-shelf solvers allowed.
