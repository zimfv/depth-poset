# Random Abstract Complex
For the given vectors $n = (n_0, n_1, ..., n_d)$ and $\beta = (\beta_0, \beta_1, ..., \beta_d)$, where $n$ is the generated complex numbers of cells of given dimension and $\beta$ is Betti numbers, we can find the vector of ranks $r = (r_1, r_2, ..., r_d)$ satysfyng the system
$$
    A\cdot r = n - \beta
$$
where
$$
    A = 
    \begin{pmatrix}
        1 & 0 & 0 & \cdots & 0 & 0 \\
        1 & 1 & 0 & \cdots & 0 & 0 \\
        0 & 1 & 1 & \cdots & 0 & 0 \\
        \cdots & \cdots & \cdots & \cdots & \cdots & \cdots \\
        0 & 0 & 0 & \cdots & 1 & 1 \\
        0 & 0 & 0 & \cdots & 0 & 1 \\
    \end{pmatrix} \in \{0, 1\}^{n-1\times n}
$$

To construct the boundary matrix of the abstract Lefschetz complex we aim to generate $d-1$ matrices $\Delta_1, \Delta_2, ..., \Delta_d$, such that $\Delta_k \in \mathbb{F}_2^{n_k\times n_{k-1}}$ has rank $r_k$ and $\Delta_k\times\Delta_{k-1} = 0$.

Let $\Delta_0$ be the only matrix from $\mathbb{F}_2^{n_0\times 0}$. Then for each $k$ we can generate the $\Delta_k$ from $\Delta_{k-1}$: Let $K_k$ be the basis of $\ker\Delta_k$, and $A_k, B_k$ be the uniformly distributed matrices over $\mathbb{F}_2$ sizes $(\dim\ker\Delta_{k-1}, r_k)$ and $(r_k, n_k - r_k)$, and ranks $\min(\dim\ker\Delta_{k-1}, r_k)$ and $\min(r_k, n_k - r_k)$. 

Then we compute
$$
    \Delta_k' = (K_{k-1}\cdot A_k, K_{k-1}\cdot A_k\cdot B_k)
$$
and define $\Delta_k$ from $\Delta_k'$ by random column permutation.