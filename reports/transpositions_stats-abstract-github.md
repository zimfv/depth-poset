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

To construct the boundary matrix of the abstract Lefschetz complex we aim to generate $d-1$ matrices $\Delta_1, \Delta_2, ..., \Delta_d$, such that $`\Delta_k \in \mathbb{F}_2^{n_k\times n_{k-1}}`$ has rank $r_k$ and $`\Delta_k\times\Delta_{k-1} = 0`$.

Let $\Delta_0$ be the only matrix from $`\mathbb{F}_2^{n_0\times 0}`$. Then for each $k$ we can generate the $\Delta_k$ from $\Delta_{k-1}$: Let $K_k$ be the basis of $\ker\Delta_k$, and $A_k, B_k$ be the uniformly distributed matrices over $`\mathbb{F}_2`$ sizes $`(\dim\ker\Delta_{k-1}, r_k)`$ and $(r_k, n_k - r_k)$, and ranks $`\min(\dim\ker\Delta_{k-1}, r_k)`$ and $`\min(r_k, n_k - r_k)$`. 

Then we compute

$$
\Delta_k' = (K_{k-1}\cdot A_k, K_{k-1}\cdot A_k\cdot B_k)
$$

and define $\Delta_k$ from $\Delta_k'$ by random column permutation.




# Data 
We have 1200 complexes.

|     |   Dimension |   Cells | $n$-vector       | Betti-vector   |   Nodes in Depth Poset |   Complexes |
|----:|------------:|--------:|:-----------------|:---------------|-----------------------:|------------:|
|   0 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 0, 0)$ |                     20 |          10 |
|   1 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 0, 1)$ |                     20 |          10 |
|   2 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 0, 2)$ |                     20 |          10 |
|   3 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 0, 3)$ |                     20 |          10 |
|   4 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 1, 0)$ |                     19 |          10 |
|   5 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 1, 1)$ |                     19 |          10 |
|   6 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 1, 2)$ |                     19 |          10 |
|   7 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 1, 3)$ |                     19 |          10 |
|   8 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 2, 0)$ |                     18 |          10 |
|   9 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 2, 1)$ |                     18 |          10 |
|  10 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 2, 2)$ |                     18 |          10 |
|  11 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 2, 3)$ |                     18 |          10 |
|  12 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 3, 0)$ |                     17 |          10 |
|  13 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 3, 1)$ |                     17 |          10 |
|  14 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 3, 2)$ |                     17 |          10 |
|  15 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 0, 3, 3)$ |                     17 |          10 |
|  16 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 1, 1, 0)$ |                     19 |          10 |
|  17 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 1, 1, 1)$ |                     19 |          10 |
|  18 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 1, 1, 2)$ |                     19 |          10 |
|  19 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 1, 1, 3)$ |                     19 |          10 |
|  20 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 1, 2, 0)$ |                     18 |          10 |
|  21 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 1, 2, 1)$ |                     18 |          10 |
|  22 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 1, 2, 2)$ |                     18 |          10 |
|  23 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 1, 2, 3)$ |                     18 |          10 |
|  24 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 1, 3, 0)$ |                     17 |          10 |
|  25 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 1, 3, 1)$ |                     17 |          10 |
|  26 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 1, 3, 2)$ |                     17 |          10 |
|  27 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 1, 3, 3)$ |                     17 |          10 |
|  28 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 2, 2, 0)$ |                     18 |          10 |
|  29 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 2, 2, 1)$ |                     18 |          10 |
|  30 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 2, 2, 2)$ |                     18 |          10 |
|  31 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 2, 2, 3)$ |                     18 |          10 |
|  32 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 2, 3, 0)$ |                     17 |          10 |
|  33 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 2, 3, 1)$ |                     17 |          10 |
|  34 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 2, 3, 2)$ |                     17 |          10 |
|  35 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 2, 3, 3)$ |                     17 |          10 |
|  36 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 3, 3, 0)$ |                     17 |          10 |
|  37 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 3, 3, 1)$ |                     17 |          10 |
|  38 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 3, 3, 2)$ |                     17 |          10 |
|  39 |           3 |      42 | $(8, 13, 13, 8)$ | $(1, 3, 3, 3)$ |                     17 |          10 |
|  40 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 0, 1)$ |                     23 |          15 |
|  41 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 0, 2)$ |                     23 |          15 |
|  42 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 0, 3)$ |                     23 |          10 |
|  43 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 1, 0)$ |                     22 |          10 |
|  44 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 1, 1)$ |                     22 |          10 |
|  45 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 1, 2)$ |                     22 |          10 |
|  46 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 1, 3)$ |                     22 |          10 |
|  47 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 2, 0)$ |                     21 |          10 |
|  48 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 2, 1)$ |                     21 |          10 |
|  49 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 2, 2)$ |                     21 |          10 |
|  50 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 2, 3)$ |                     21 |          10 |
|  51 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 3, 0)$ |                     20 |          10 |
|  52 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 3, 1)$ |                     20 |          10 |
|  53 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 3, 2)$ |                     20 |          10 |
|  54 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 3, 3)$ |                     20 |          10 |
|  55 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 1, 1, 0)$ |                     22 |          10 |
|  56 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 1, 1, 1)$ |                     22 |          10 |
|  57 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 1, 1, 2)$ |                     22 |          10 |
|  58 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 1, 1, 3)$ |                     22 |          10 |
|  59 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 1, 2, 0)$ |                     21 |          10 |
|  60 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 1, 2, 1)$ |                     21 |          10 |
|  61 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 1, 2, 2)$ |                     21 |          10 |
|  62 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 1, 2, 3)$ |                     21 |          10 |
|  63 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 1, 3, 0)$ |                     20 |          10 |
|  64 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 1, 3, 1)$ |                     20 |          10 |
|  65 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 1, 3, 2)$ |                     20 |          10 |
|  66 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 1, 3, 3)$ |                     20 |          10 |
|  67 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 2, 2, 0)$ |                     21 |          10 |
|  68 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 2, 2, 1)$ |                     21 |          10 |
|  69 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 2, 2, 2)$ |                     21 |          10 |
|  70 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 2, 2, 3)$ |                     21 |          10 |
|  71 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 2, 3, 0)$ |                     20 |          10 |
|  72 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 2, 3, 1)$ |                     20 |          10 |
|  73 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 2, 3, 2)$ |                     20 |          10 |
|  74 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 2, 3, 3)$ |                     20 |          10 |
|  75 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 3, 3, 0)$ |                     20 |          10 |
|  76 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 3, 3, 1)$ |                     20 |          10 |
|  77 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 3, 3, 2)$ |                     20 |          10 |
|  78 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 3, 3, 3)$ |                     20 |          10 |
|  79 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 0, 0)$ |                     23 |          10 |
|  80 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 0, 1)$ |                     23 |          10 |
|  81 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 0, 2)$ |                     23 |          10 |
|  82 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 0, 3)$ |                     23 |          10 |
|  83 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 1, 0)$ |                     22 |          10 |
|  84 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 1, 1)$ |                     22 |          10 |
|  85 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 1, 2)$ |                     22 |          10 |
|  86 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 1, 3)$ |                     22 |          10 |
|  87 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 2, 0)$ |                     21 |          10 |
|  88 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 2, 1)$ |                     21 |          10 |
|  89 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 2, 2)$ |                     21 |          10 |
|  90 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 2, 3)$ |                     21 |          10 |
|  91 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 3, 0)$ |                     20 |          10 |
|  92 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 3, 1)$ |                     20 |          10 |
|  93 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 3, 2)$ |                     20 |          10 |
|  94 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 0, 3, 3)$ |                     20 |          10 |
|  95 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 1, 1, 0)$ |                     22 |          10 |
|  96 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 1, 1, 1)$ |                     22 |          10 |
|  97 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 1, 1, 2)$ |                     22 |          10 |
|  98 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 1, 1, 3)$ |                     22 |          10 |
|  99 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 1, 2, 0)$ |                     21 |          10 |
| 100 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 1, 2, 1)$ |                     21 |          10 |
| 101 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 1, 2, 2)$ |                     21 |          10 |
| 102 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 1, 2, 3)$ |                     21 |          10 |
| 103 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 1, 3, 0)$ |                     20 |          10 |
| 104 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 1, 3, 1)$ |                     20 |          10 |
| 105 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 1, 3, 2)$ |                     20 |          10 |
| 106 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 1, 3, 3)$ |                     20 |          10 |
| 107 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 2, 2, 0)$ |                     21 |          10 |
| 108 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 2, 2, 1)$ |                     21 |          10 |
| 109 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 2, 2, 2)$ |                     21 |          10 |
| 110 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 2, 2, 3)$ |                     21 |          10 |
| 111 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 2, 3, 0)$ |                     20 |          10 |
| 112 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 2, 3, 1)$ |                     20 |          10 |
| 113 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 2, 3, 2)$ |                     20 |          10 |
| 114 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 2, 3, 3)$ |                     20 |          10 |
| 115 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 3, 3, 0)$ |                     20 |          10 |
| 116 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 3, 3, 1)$ |                     20 |          10 |
| 117 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 3, 3, 2)$ |                     20 |          10 |
| 118 |           3 |      48 | $(9, 19, 15, 5)$ | $(1, 3, 3, 3)$ |                     20 |          10 |

# Transpositions
In this data we got 39293 by transposing consecutive pairs. The distribution of the transposition types is given in the table:

| transposition_type   |   no switch (nested) |   no switch (not nested) |   switch backward |   switch forward |
|:---------------------|---------------------:|-------------------------:|------------------:|-----------------:|
| birth-birth          |                 5889 |                     6025 |              2861 |             3010 |
| birth-death          |                    0 |                     2427 |                 0 |             1342 |
| death-death          |                 6010 |                     5854 |              2937 |             2938 |


# Equations Detailed
We checked 48 equations. And we can see the distribution of transposiions, satysfying these equations in the tables:

## Birth-Birth Transpositions
| Name      | Transposition Type   | Switch Type            | Order           | Formula                                                                                                                                   | Correct (WC)   | Correct (WOC)   | Correct, % (WC)   | Correct, % (WOC)   |
|:----------|:---------------------|:-----------------------|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------|:---------------|:----------------|:------------------|:-------------------|
| Eq08      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Succ}_1^\text{at}(a, y) = \text{Succ}_1^\text{bt}(x, y) \oplus \{(x, b)\} \oplus \text{Succ}_1^\text{bt}(a, b)$                    | 3010/3010      | 3010/3010       | 100.00%           | 100.00%            |
| Eq09      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Succ}_1^\text{at}(x, b) = \text{Succ}_1^\text{bt}(a, b)$                                                                           | 3010/3010      | 3010/3010       | 100.00%           | 100.00%            |
| Eq10      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Pred}_1^\text{at}(a, y) = \text{Pred}_1^\text{bt}(x, y)$                                                                           | 3010/3010      | 3010/3010       | 100.00%           | 100.00%            |
| Eq11      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Pred}_1^\text{at}(x, b) = \text{Pred}_1^\text{bt}(a, b) \oplus \{(a, y)\}$                                                         | 3010/3010      | 3010/3010       | 100.00%           | 100.00%            |
| Eq12      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Succ}_2^\text{at}(a, y) = \text{Succ}_2^\text{bt}(x, y) \oplus \{(x, b), (a, b)\}$                                                 | 3010/3010      | 3010/3010       | 100.00%           | 100.00%            |
| Eq13      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Succ}_2^\text{at}(x, b) = \text{Succ}_2^\text{bt}(a, b)$                                                                           | 1225/3010      | 1225/3010       | 40.70%            | 40.70%             |
| Eq14      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Pred}_2^\text{at}(a, y) = [\text{Pred}_2^\text{bt}(a, b) \cap \mathcal{L}]$                                                        | 3010/3010      | 3010/3010       | 100.00%           | 100.00%            |
| Eq15      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Pred}_2^\text{at}(x, b) = \text{Pred}_2^\text{bt}(x, y) \oplus \{(a, y)\} \oplus [\text{Pred}_2^\text{bt}(a, b) \cap \mathcal{M}]$ | 3010/3010      | 3010/3010       | 100.00%           | 100.00%            |
| Eq16      | birth-birth          | no switch (nested)     | $a < x < b < y$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y) \oplus \{(a, b)\} \oplus \text{Succ}_1^\text{bt}(a, b)$                    | 0/2935         | 0/2935          | 0.00%             | 0.00%              |
| Eq16      | birth-birth          | no switch (nested)     | $a < x < y < b$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y) \oplus \{(a, b)\} \oplus \text{Succ}_1^\text{bt}(a, b)$                    | 2954/2954      | 2954/2954       | 100.00%           | 100.00%            |
| Eq16s     | birth-birth          | no switch (nested)     | $a < x < b < y$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b) \oplus \{(x, y)\} \oplus \text{Succ}_1^\text{bt}(x, y)$                    | 2935/2935      | 2935/2935       | 100.00%           | 100.00%            |
| Eq16s     | birth-birth          | no switch (nested)     | $a < x < y < b$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b) \oplus \{(x, y)\} \oplus \text{Succ}_1^\text{bt}(x, y)$                    | 0/2954         | 0/2954          | 0.00%             | 0.00%              |
| Eq17      | birth-birth          | no switch (nested)     | $a < x < b < y$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b) \oplus \{(x, y)\}$                                                         | 0/2935         | 0/2935          | 0.00%             | 0.00%              |
| Eq17      | birth-birth          | no switch (nested)     | $a < x < y < b$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b) \oplus \{(x, y)\}$                                                         | 2954/2954      | 2954/2954       | 100.00%           | 100.00%            |
| Eq17s     | birth-birth          | no switch (nested)     | $a < x < b < y$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y) \oplus \{(a, b)\}$                                                         | 2935/2935      | 2935/2935       | 100.00%           | 100.00%            |
| Eq17s     | birth-birth          | no switch (nested)     | $a < x < y < b$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y) \oplus \{(a, b)\}$                                                         | 0/2954         | 0/2954          | 0.00%             | 0.00%              |
| Eq26      | birth-birth          | switch backward        | $a < x < y < b$ | $\text{Pred}_2^\text{bt}(a, b) = \text{Pred}_2^\text{at}(a, y) \oplus \{(x, y)\} \oplus [\text{Pred}_2^\text{at}(x, b) \cap \mathcal{M}]$ | 2861/2861      | 2861/2861       | 100.00%           | 100.00%            |
| Eq27      | birth-birth          | switch backward        | $a < x < y < b$ | $\text{Pred}_2^\text{bt}(x, y) = [\text{Pred}_2^\text{at}(x, b) \cap \mathcal{L}]$                                                        | 2861/2861      | 2861/2861       | 100.00%           | 100.00%            |
| EqPred1ab | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b)$                                                                           | 3033/3033      | 3033/3033       | 100.00%           | 100.00%            |
| EqPred1ab | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b)$                                                                           | 2992/2992      | 2992/2992       | 100.00%           | 100.00%            |
| EqPred1xy | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y)$                                                                           | 3033/3033      | 3033/3033       | 100.00%           | 100.00%            |
| EqPred1xy | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y)$                                                                           | 2992/2992      | 2992/2992       | 100.00%           | 100.00%            |
| EqPred2ab | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b)$                                                                           | 3033/3033      | 3033/3033       | 100.00%           | 100.00%            |
| EqPred2ab | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b)$                                                                           | 2992/2992      | 2992/2992       | 100.00%           | 100.00%            |
| EqPred2xy | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y)$                                                                           | 3033/3033      | 3033/3033       | 100.00%           | 100.00%            |
| EqPred2xy | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y)$                                                                           | 2992/2992      | 2992/2992       | 100.00%           | 100.00%            |
| EqSucc1ab | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b)$                                                                           | 3033/3033      | 3033/3033       | 100.00%           | 100.00%            |
| EqSucc1ab | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b)$                                                                           | 2992/2992      | 2992/2992       | 100.00%           | 100.00%            |
| EqSucc1xy | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y)$                                                                           | 3033/3033      | 3033/3033       | 100.00%           | 100.00%            |
| EqSucc1xy | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y)$                                                                           | 2992/2992      | 2992/2992       | 100.00%           | 100.00%            |
| EqSucc2ab | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b)$                                                                           | 3033/3033      | 3033/3033       | 100.00%           | 100.00%            |
| EqSucc2ab | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b)$                                                                           | 2992/2992      | 2992/2992       | 100.00%           | 100.00%            |
| EqSucc2xy | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y)$                                                                           | 3033/3033      | 3033/3033       | 100.00%           | 100.00%            |
| EqSucc2xy | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y)$                                                                           | 2992/2992      | 2992/2992       | 100.00%           | 100.00%            |

## Death-Death Transpositions
| Name      | Transposition Type   | Switch Type            | Order           | Formula                                                                                                                                   | Correct (WC)   | Correct (WOC)   | Correct, % (WC)   | Correct, % (WOC)   |
|:----------|:---------------------|:-----------------------|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------|:---------------|:----------------|:------------------|:-------------------|
| Eq28      | death-death          | switch forward         | $a < x < y < b$ | $\text{Succ}_1^\text{at}(x, b) = \text{Succ}_1^\text{bt}(x, y) \oplus \{(a, y), (a, b)\}$                                                 | 2938/2938      | 2938/2938       | 100.00%           | 100.00%            |
| Eq29      | death-death          | switch forward         | $a < x < y < b$ | $\text{Succ}_1^\text{at}(a, y) = \text{Succ}_1^\text{bt}(a, b)$                                                                           | 1211/2938      | 1211/2938       | 41.22%            | 41.22%             |
| Eq30      | death-death          | switch forward         | $a < x < y < b$ | $\text{Pred}_1^\text{at}(x, b) = [\text{Pred}_1^\text{bt}(a, b) \cap \mathcal{B}]$                                                        | 2938/2938      | 2938/2938       | 100.00%           | 100.00%            |
| Eq31      | death-death          | switch forward         | $a < x < y < b$ | $\text{Pred}_1^\text{at}(a, y) = \text{Pred}_1^\text{bt}(x, y) \oplus \{(x, b)\} \oplus[\text{Pred}_1^\text{bt}(a, b) \cap \mathcal{N}]$  | 2938/2938      | 2938/2938       | 100.00%           | 100.00%            |
| Eq32      | death-death          | switch forward         | $a < x < y < b$ | $\text{Succ}_2^\text{at}(x, b) = \text{Succ}_2^\text{bt}(x, y) \oplus \{(a, y)\} \oplus \text{Succ}_2^\text{bt}(a, b)$                    | 2938/2938      | 2938/2938       | 100.00%           | 100.00%            |
| Eq33      | death-death          | switch forward         | $a < x < y < b$ | $\text{Succ}_2^\text{at}(a, y) = \text{Succ}_2^\text{bt}(a, b)$                                                                           | 2938/2938      | 2938/2938       | 100.00%           | 100.00%            |
| Eq34      | death-death          | switch forward         | $a < x < y < b$ | $\text{Pred}_2^\text{at}(x, b) = \text{Pred}_2^\text{bt}(x, y)$                                                                           | 2938/2938      | 2938/2938       | 100.00%           | 100.00%            |
| Eq35      | death-death          | switch forward         | $a < x < y < b$ | $\text{Pred}_2^\text{at}(a, y) = \text{Pred}_2^\text{bt}(a, b) \oplus \{(x, b)\}$                                                         | 2938/2938      | 2938/2938       | 100.00%           | 100.00%            |
| Eq36      | death-death          | no switch (nested)     | $a < x < y < b$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y) \oplus \{(a, b)\} \oplus \text{Succ}_2^\text{bt}(a, b)$                    | 3003/3003      | 3003/3003       | 100.00%           | 100.00%            |
| Eq36      | death-death          | no switch (nested)     | $x < a < y < b$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y) \oplus \{(a, b)\} \oplus \text{Succ}_2^\text{bt}(a, b)$                    | 0/3007         | 0/3007          | 0.00%             | 0.00%              |
| Eq36s     | death-death          | no switch (nested)     | $a < x < y < b$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b) \oplus \{(x, y)\} \oplus \text{Succ}_2^\text{bt}(x, y)$                    | 0/3003         | 0/3003          | 0.00%             | 0.00%              |
| Eq36s     | death-death          | no switch (nested)     | $x < a < y < b$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b) \oplus \{(x, y)\} \oplus \text{Succ}_2^\text{bt}(x, y)$                    | 3007/3007      | 3007/3007       | 100.00%           | 100.00%            |
| Eq37      | death-death          | no switch (nested)     | $a < x < y < b$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b) \oplus \{(x, y)\}$                                                         | 3003/3003      | 3003/3003       | 100.00%           | 100.00%            |
| Eq37      | death-death          | no switch (nested)     | $x < a < y < b$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b) \oplus \{(x, y)\}$                                                         | 0/3007         | 0/3007          | 0.00%             | 0.00%              |
| Eq37s     | death-death          | no switch (nested)     | $a < x < y < b$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y) \oplus \{(a, b)\}$                                                         | 0/3003         | 0/3003          | 0.00%             | 0.00%              |
| Eq37s     | death-death          | no switch (nested)     | $x < a < y < b$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y) \oplus \{(a, b)\}$                                                         | 3007/3007      | 3007/3007       | 100.00%           | 100.00%            |
| Eq46      | death-death          | switch backward        | $a < x < y < b$ | $\text{Pred}_1^\text{bt}(a, b) = \text{Pred}_1^\text{at}(x, b) \oplus \{(x, y)\} \oplus [\text{Pred}_1^\text{at}(a, y) \cap \mathcal{N}]$ | 2937/2937      | 2937/2937       | 100.00%           | 100.00%            |
| Eq47      | death-death          | switch backward        | $a < x < y < b$ | $\text{Pred}_1^\text{bt}(x, y) = [\text{Pred}_1^\text{at}(a, y) \cap \mathcal{B}]$                                                        | 2937/2937      | 2937/2937       | 100.00%           | 100.00%            |
| EqPred1ab | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b)$                                                                           | 2905/2905      | 2905/2905       | 100.00%           | 100.00%            |
| EqPred1ab | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b)$                                                                           | 2949/2949      | 2949/2949       | 100.00%           | 100.00%            |
| EqPred1xy | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y)$                                                                           | 2905/2905      | 2905/2905       | 100.00%           | 100.00%            |
| EqPred1xy | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y)$                                                                           | 2949/2949      | 2949/2949       | 100.00%           | 100.00%            |
| EqPred2ab | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b)$                                                                           | 2905/2905      | 2905/2905       | 100.00%           | 100.00%            |
| EqPred2ab | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b)$                                                                           | 2949/2949      | 2949/2949       | 100.00%           | 100.00%            |
| EqPred2xy | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y)$                                                                           | 2905/2905      | 2905/2905       | 100.00%           | 100.00%            |
| EqPred2xy | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y)$                                                                           | 2949/2949      | 2949/2949       | 100.00%           | 100.00%            |
| EqSucc1ab | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b)$                                                                           | 2905/2905      | 2905/2905       | 100.00%           | 100.00%            |
| EqSucc1ab | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b)$                                                                           | 2949/2949      | 2949/2949       | 100.00%           | 100.00%            |
| EqSucc1xy | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y)$                                                                           | 2905/2905      | 2905/2905       | 100.00%           | 100.00%            |
| EqSucc1xy | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y)$                                                                           | 2949/2949      | 2949/2949       | 100.00%           | 100.00%            |
| EqSucc2ab | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b)$                                                                           | 2905/2905      | 2905/2905       | 100.00%           | 100.00%            |
| EqSucc2ab | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b)$                                                                           | 2949/2949      | 2949/2949       | 100.00%           | 100.00%            |
| EqSucc2xy | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y)$                                                                           | 2905/2905      | 2905/2905       | 100.00%           | 100.00%            |
| EqSucc2xy | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y)$                                                                           | 2949/2949      | 2949/2949       | 100.00%           | 100.00%            |

## Birth-Death Transpositions
| Name      | Transposition Type   | Switch Type            | Order           | Formula                                                                              | Correct (WC)   | Correct (WOC)   | Correct, % (WC)   | Correct, % (WOC)   |
|:----------|:---------------------|:-----------------------|:----------------|:-------------------------------------------------------------------------------------|:---------------|:----------------|:------------------|:-------------------|
| Eq48      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Succ}_1^\text{at}(a, x) = \text{Succ}_1^\text{bt}(a, b)$                      | 1342/1342      | 1342/1342       | 100.00%           | 100.00%            |
| Eq49      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Succ}_1^\text{at}(b, y) = \text{Succ}_1^\text{bt}(x, y)$                      | 1342/1342      | 1342/1342       | 100.00%           | 100.00%            |
| Eq50      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_1^\text{at}(a, x) = \{(s, t)\in \text{BD}:\; U_1^\text{bt}[t, x] = 1\}$ | 165/1342       | 0/1342          | 12.30%            | 0.00%              |
| Eq50a     | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_1^\text{at}(a, x) = A_1^\text{bt} \cap B_1^\text{bt}$                   | 165/1342       | 1342/1342       | 12.30%            | 100.00%            |
| Eq50b     | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_1^\text{at}(a, x) = A_1^\text{bt} \cap B_1^\text{at}$                   | 1342/1342      | 1342/1342       | 100.00%           | 100.00%            |
| Eq51      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_1^\text{at}(b, y) = \text{Pred}_1^\text{bt}(x, y)$                      | 1342/1342      | 1342/1342       | 100.00%           | 100.00%            |
| Eq52      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Succ}_2^\text{at}(a, x) = \text{Succ}_2^\text{bt}(a, b)$                      | 1342/1342      | 1342/1342       | 100.00%           | 100.00%            |
| Eq53      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Succ}_2^\text{at}(b, y) = \text{Succ}_2^\text{bt}(x, y)$                      | 1342/1342      | 1342/1342       | 100.00%           | 100.00%            |
| Eq54      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_2^\text{at}(a, x) = \text{Pred}_2^\text{bt}(a, b)$                      | 1342/1342      | 1342/1342       | 100.00%           | 100.00%            |
| Eq55      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_2^\text{at}(b, y) = \{(s, t)\in \text{BD}:\; U_2^\text{bt}[b, s] = 1\}$ | 175/1342       | 175/1342        | 13.04%            | 13.04%             |
| Eq55a     | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_2^\text{at}(b, y) = A_2^\text{bt} \cap B_2^\text{bt}$                   | 175/1342       | 1342/1342       | 13.04%            | 100.00%            |
| Eq55b     | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_2^\text{at}(b, y) = A_2^\text{bt} \cap B_2^\text{at}$                   | 1342/1342      | 1342/1342       | 100.00%           | 100.00%            |
| EqPred1ab | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b)$                      | 1216/1216      | 1216/1216       | 100.00%           | 100.00%            |
| EqPred1ab | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b)$                      | 1211/1211      | 1211/1211       | 100.00%           | 100.00%            |
| EqPred1xy | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y)$                      | 1216/1216      | 1216/1216       | 100.00%           | 100.00%            |
| EqPred1xy | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y)$                      | 1211/1211      | 1211/1211       | 100.00%           | 100.00%            |
| EqPred2ab | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b)$                      | 1216/1216      | 1216/1216       | 100.00%           | 100.00%            |
| EqPred2ab | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b)$                      | 1211/1211      | 1211/1211       | 100.00%           | 100.00%            |
| EqPred2xy | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y)$                      | 1216/1216      | 1216/1216       | 100.00%           | 100.00%            |
| EqPred2xy | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y)$                      | 1211/1211      | 1211/1211       | 100.00%           | 100.00%            |
| EqSucc1ab | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b)$                      | 1216/1216      | 1216/1216       | 100.00%           | 100.00%            |
| EqSucc1ab | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b)$                      | 1211/1211      | 1211/1211       | 100.00%           | 100.00%            |
| EqSucc1xy | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y)$                      | 1216/1216      | 1216/1216       | 100.00%           | 100.00%            |
| EqSucc1xy | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y)$                      | 1211/1211      | 1211/1211       | 100.00%           | 100.00%            |
| EqSucc2ab | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b)$                      | 1216/1216      | 1216/1216       | 100.00%           | 100.00%            |
| EqSucc2ab | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b)$                      | 1211/1211      | 1211/1211       | 100.00%           | 100.00%            |
| EqSucc2xy | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y)$                      | 1216/1216      | 1216/1216       | 100.00%           | 100.00%            |
| EqSucc2xy | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y)$                      | 1211/1211      | 1211/1211       | 100.00%           | 100.00%            |

## Equations Correctness Variants 
The Equations Eq08, Eq09, Eq10, Eq11, Eq12, Eq13, Eq14, Eq15 corespond the __birth-birth switch forward case__.
The correctness distribution for such cases is given in the table:
| Eq08   | Eq09   | Eq10   | Eq11   | Eq12   | Eq13   | Eq14   | Eq15   |   count (WC) |   count (WOC) | portion, % (WC)   | portion, % (WOC)   |
|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|-------------:|--------------:|:------------------|:-------------------|
| ✔      | ✔      | ✔      | ✔      | ✔      | ✘      | ✔      | ✔      |         1785 |          1785 | 59.30%            | 59.30%             |
| ✔      | ✔      | ✔      | ✔      | ✔      | ✔      | ✔      | ✔      |         1225 |          1225 | 40.70%            | 40.70%             |

The Equations Eq16, Eq16s, Eq17, Eq17s corespond the __birth-birth no switch nested case__.
The correctness distribution for such cases is given in the table:
| Eq16   | Eq16s   | Eq17   | Eq17s   |   count (WC) |   count (WOC) | portion, % (WC)   | portion, % (WOC)   |
|:-------|:--------|:-------|:--------|-------------:|--------------:|:------------------|:-------------------|
| ✘      | ✔       | ✘      | ✔       |         2935 |          2935 | 49.84%            | 49.84%             |
| ✔      | ✘       | ✔      | ✘       |         2954 |          2954 | 50.16%            | 50.16%             |

The Equations Eq26, Eq27 corespond the __birth-birth switch backward case__.
The correctness distribution for such cases is given in the table:
| Eq26   | Eq27   |   count (WC) |   count (WOC) | portion, % (WC)   | portion, % (WOC)   |
|:-------|:-------|-------------:|--------------:|:------------------|:-------------------|
| ✔      | ✔      |         2861 |          2861 | 100.00%           | 100.00%            |

The Equations Eq28, Eq29, Eq30, Eq31, Eq32, Eq33, Eq34, Eq35 corespond the __death-death switch forward case__.
The correctness distribution for such cases is given in the table:
| Eq28   | Eq29   | Eq30   | Eq31   | Eq32   | Eq33   | Eq34   | Eq35   |   count (WC) |   count (WOC) | portion, % (WC)   | portion, % (WOC)   |
|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|-------------:|--------------:|:------------------|:-------------------|
| ✔      | ✘      | ✔      | ✔      | ✔      | ✔      | ✔      | ✔      |         1727 |          1727 | 58.78%            | 58.78%             |
| ✔      | ✔      | ✔      | ✔      | ✔      | ✔      | ✔      | ✔      |         1211 |          1211 | 41.22%            | 41.22%             |

The Equations Eq36, Eq36s, Eq37, Eq37s corespond the __death-death no switch nested case__.
The correctness distribution for such cases is given in the table:
| Eq36   | Eq36s   | Eq37   | Eq37s   |   count (WC) |   count (WOC) | portion, % (WC)   | portion, % (WOC)   |
|:-------|:--------|:-------|:--------|-------------:|--------------:|:------------------|:-------------------|
| ✘      | ✔       | ✘      | ✔       |         3007 |          3007 | 50.03%            | 50.03%             |
| ✔      | ✘       | ✔      | ✘       |         3003 |          3003 | 49.97%            | 49.97%             |

The Equations Eq46, Eq47 corespond the __death-death switch backward case__.
The correctness distribution for such cases is given in the table:
| Eq46   | Eq47   |   count (WC) |   count (WOC) | portion, % (WC)   | portion, % (WOC)   |
|:-------|:-------|-------------:|--------------:|:------------------|:-------------------|
| ✔      | ✔      |         2937 |          2937 | 100.00%           | 100.00%            |

The Equations Eq48, Eq49, Eq50, Eq50a, Eq50b, Eq51, Eq52, Eq53, Eq54, Eq55, Eq55a, Eq55b corespond the __birth-death switch forward case__.
The correctness distribution for such cases is given in the table:
| Eq48   | Eq49   | Eq50   | Eq50a   | Eq50b   | Eq51   | Eq52   | Eq53   | Eq54   | Eq55   | Eq55a   | Eq55b   |   count (WC) |   count (WOC) | portion, % (WC)   | portion, % (WOC)   |
|:-------|:-------|:-------|:--------|:--------|:-------|:-------|:-------|:-------|:-------|:--------|:--------|-------------:|--------------:|:------------------|:-------------------|
| ✔      | ✔      | ✘      | ✘       | ✔       | ✔      | ✔      | ✔      | ✔      | ✘      | ✘       | ✔       |         1013 |               | 75.48%            |                    |
| ✔      | ✔      | ✘      | ✘       | ✔       | ✔      | ✔      | ✔      | ✔      | ✔      | ✔       | ✔       |          164 |               | 12.22%            |                    |
| ✔      | ✔      | ✔      | ✔       | ✔       | ✔      | ✔      | ✔      | ✔      | ✘      | ✘       | ✔       |          154 |               | 11.48%            |                    |
| ✔      | ✔      | ✔      | ✔       | ✔       | ✔      | ✔      | ✔      | ✔      | ✔      | ✔       | ✔       |           11 |               | 0.82%             |                    |
| ✔      | ✔      | ✘      | ✔       | ✔       | ✔      | ✔      | ✔      | ✔      | ✘      | ✔       | ✔       |              |          1167 |                   | 86.96%             |
| ✔      | ✔      | ✘      | ✔       | ✔       | ✔      | ✔      | ✔      | ✔      | ✔      | ✔       | ✔       |              |           175 |                   | 13.04%             |

The Equations EqPred1ab, EqPred1xy, EqPred2ab, EqPred2xy, EqSucc1ab, EqSucc1xy, EqSucc2ab, EqSucc2xy corespond the __birth-birth no switch not nested, death-death no switch not nested, birth-death no switch nested and birth-death no switch not nested cases__.
The correctness distribution for such cases is given in the table:
| EqPred1ab   | EqPred1xy   | EqPred2ab   | EqPred2xy   | EqSucc1ab   | EqSucc1xy   | EqSucc2ab   | EqSucc2xy   |   count (WC) |   count (WOC) | portion, % (WC)   | portion, % (WOC)   |
|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|-------------:|--------------:|:------------------|:-------------------|
| ✔           | ✔           | ✔           | ✔           | ✔           | ✔           | ✔           | ✔           |        14306 |         14306 | 100.00%           | 100.00%            |




# Incorrect Equations
We have 16742 transpositions, such that some equations are incorrect.
We will list some examples.

## Transposition <2, 3> in complex 0
Here is a __birth-birth__ __no switch__ transposition <2, 3>.

$$
a = 2, \; b = 8, \; x = 3, \; y = 14
$$

$$
a < x < b < y
$$

<img src="transpositions_stats-abstract.assets/image-3cd726f1d27a.png" width="1200">

### Birth Death Pairs
|                          |                                               |
|:-------------------------|:----------------------------------------------|
| Before the transposition | $\{(3, 14), (4, 6), (0, 9), (5, 7), (2, 8)\}$ |
| After the transposition  | $\{(3, 14), (4, 6), (0, 9), (5, 7), (2, 8)\}$ |

### Relations
|                                | Algorithm 1                                            | Algorithm 2                          |
|:-------------------------------|:-------------------------------------------------------|:-------------------------------------|
| Before the transposition (WC)  | $\{(6, 8), (6, 14), (7, 9), (6, 9), (7, 8)\}$          | $\{(5, 2), (5, 0), (2, 0), (4, 3)\}$ |
| Before the transposition (WOC) | $\{(6, 8), (6, 14), (7, 9), (6, 9), (7, 8)\}$          | $\{(5, 2), (5, 0), (2, 0), (4, 3)\}$ |
| After the transposition (WC)   | $\{(8, 14), (6, 8), (6, 14), (7, 9), (6, 9), (7, 8)\}$ | $\{(2, 0), (5, 2), (5, 0), (4, 3)\}$ |
| After the transposition (WOC)  | $\{(8, 14), (6, 8), (6, 14), (7, 9), (6, 9), (7, 8)\}$ | $\{(2, 0), (5, 2), (5, 0), (4, 3)\}$ |

### Successors and Predecessers
|                       | $\text{Set}^\text{bt}(2, 8)$                         | $\text{Set}^\text{bt}(3, 14)$                 | $\text{Set}^\text{at}(2, 8)$                         | $\text{Set}^\text{at}(3, 14)$                         |
|:----------------------|:-----------------------------------------------------|:----------------------------------------------|:-----------------------------------------------------|:------------------------------------------------------|
| $\text{Succ}_1$ (WC)  | $\text{Succ}_1^\text{bt}(2, 8) = \emptyset$          | $\text{Succ}_1^\text{bt}(3, 14) = \emptyset$  | $\text{Succ}_1^\text{at}(2, 8) = \{(3, 14)\}$        | $\text{Succ}_1^\text{at}(3, 14) = \emptyset$          |
| $\text{Succ}_1$ (WOC) | $\text{Succ}_1^\text{bt}(2, 8) = \emptyset$          | $\text{Succ}_1^\text{bt}(3, 14) = \emptyset$  | $\text{Succ}_1^\text{at}(2, 8) = \{(3, 14)\}$        | $\text{Succ}_1^\text{at}(3, 14) = \emptyset$          |
| $\text{Pred}_1$ (WC)  | $\text{Pred}_1^\text{bt}(2, 8) = \{(4, 6), (5, 7)\}$ | $\text{Pred}_1^\text{bt}(3, 14) = \{(4, 6)\}$ | $\text{Pred}_1^\text{at}(2, 8) = \{(4, 6), (5, 7)\}$ | $\text{Pred}_1^\text{at}(3, 14) = \{(4, 6), (2, 8)\}$ |
| $\text{Pred}_1$ (WOC) | $\text{Pred}_1^\text{bt}(2, 8) = \{(4, 6), (5, 7)\}$ | $\text{Pred}_1^\text{bt}(3, 14) = \{(4, 6)\}$ | $\text{Pred}_1^\text{at}(2, 8) = \{(4, 6), (5, 7)\}$ | $\text{Pred}_1^\text{at}(3, 14) = \{(4, 6), (2, 8)\}$ |
| $\text{Succ}_2$ (WC)  | $\text{Succ}_2^\text{bt}(2, 8) = \{(0, 9)\}$         | $\text{Succ}_2^\text{bt}(3, 14) = \emptyset$  | $\text{Succ}_2^\text{at}(2, 8) = \{(0, 9)\}$         | $\text{Succ}_2^\text{at}(3, 14) = \emptyset$          |
| $\text{Succ}_2$ (WOC) | $\text{Succ}_2^\text{bt}(2, 8) = \{(0, 9)\}$         | $\text{Succ}_2^\text{bt}(3, 14) = \emptyset$  | $\text{Succ}_2^\text{at}(2, 8) = \{(0, 9)\}$         | $\text{Succ}_2^\text{at}(3, 14) = \emptyset$          |
| $\text{Pred}_2$ (WC)  | $\text{Pred}_2^\text{bt}(2, 8) = \{(5, 7)\}$         | $\text{Pred}_2^\text{bt}(3, 14) = \{(4, 6)\}$ | $\text{Pred}_2^\text{at}(2, 8) = \{(5, 7)\}$         | $\text{Pred}_2^\text{at}(3, 14) = \{(4, 6)\}$         |
| $\text{Pred}_2$ (WOC) | $\text{Pred}_2^\text{bt}(2, 8) = \{(5, 7)\}$         | $\text{Pred}_2^\text{bt}(3, 14) = \{(4, 6)\}$ | $\text{Pred}_2^\text{at}(2, 8) = \{(5, 7)\}$         | $\text{Pred}_2^\text{at}(3, 14) = \{(4, 6)\}$         |

### Temp Sets
|                                                                     | Before the Transposition             | After the Transposition              |
|:--------------------------------------------------------------------|:-------------------------------------|:-------------------------------------|
| $\mathcal{L} = \{(s, t)\in \text{BD}:\; f(t) < f(y)\}$ (WC)         | $\{(4, 6), (0, 9), (5, 7), (2, 8)\}$ | $\{(4, 6), (0, 9), (5, 7), (2, 8)\}$ |
| $\mathcal{L} = \{(s, t)\in \text{BD}:\; f(t) < f(y)\}$ (WOC)        | $\{(4, 6), (0, 9), (5, 7), (2, 8)\}$ | $\{(4, 6), (0, 9), (5, 7), (2, 8)\}$ |
| $\mathcal{M} = \{(s, t)\in \text{BD}:\; f(y) < f(t) < f(b)\}$ (WC)  | $\emptyset$                          | $\emptyset$                          |
| $\mathcal{M} = \{(s, t)\in \text{BD}:\; f(y) < f(t) < f(b)\}$ (WOC) | $\emptyset$                          | $\emptyset$                          |

### Equations
There are 2 equations are wrong.

| eq    | left (formula)                  | left (paramatrized)              | left (value) (WC)    | left (value) (WOC)   | right (formula)                                                                        | right (paramatrized)                                                                     | right (value) (WC)            | right (value) (WOC)           | correct (WC)   | correct (WOC)   |
|:------|:--------------------------------|:---------------------------------|:---------------------|:---------------------|:---------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------|:------------------------------|:------------------------------|:---------------|:----------------|
| Eq16  | $\text{Succ}_1^\text{at}(x, y)$ | $\text{Succ}_1^\text{at}(3, 14)$ | $\emptyset$          | $\emptyset$          | $\text{Succ}_1^\text{bt}(x, y) \oplus \{(a, b)\} \oplus \text{Succ}_1^\text{bt}(a, b)$ | $\text{Succ}_1^\text{bt}(3, 14) \oplus \{(2, 8)\} \oplus \text{Succ}_1^\text{bt}(2, 8)$  | $\{(2, 8)\}$                  | $\{(2, 8)\}$                  | ✘              | ✘               |
| Eq16s | $\text{Succ}_1^\text{at}(a, b)$ | $\text{Succ}_1^\text{at}(2, 8)$  | $\{(3, 14)\}$        | $\{(3, 14)\}$        | $\text{Succ}_1^\text{bt}(a, b) \oplus \{(x, y)\} \oplus \text{Succ}_1^\text{bt}(x, y)$ | $\text{Succ}_1^\text{bt}(2, 8) \oplus \{(3, 14)\} \oplus \text{Succ}_1^\text{bt}(3, 14)$ | $\{(3, 14)\}$                 | $\{(3, 14)\}$                 | ✔              | ✔               |
| Eq17  | $\text{Pred}_1^\text{at}(a, b)$ | $\text{Pred}_1^\text{at}(2, 8)$  | $\{(4, 6), (5, 7)\}$ | $\{(4, 6), (5, 7)\}$ | $\text{Pred}_1^\text{bt}(a, b) \oplus \{(x, y)\}$                                      | $\text{Pred}_1^\text{bt}(2, 8) \oplus \{(3, 14)\}$                                       | $\{(4, 6), (5, 7), (3, 14)\}$ | $\{(4, 6), (5, 7), (3, 14)\}$ | ✘              | ✘               |
| Eq17s | $\text{Pred}_1^\text{at}(x, y)$ | $\text{Pred}_1^\text{at}(3, 14)$ | $\{(4, 6), (2, 8)\}$ | $\{(4, 6), (2, 8)\}$ | $\text{Pred}_1^\text{bt}(x, y) \oplus \{(a, b)\}$                                      | $\text{Pred}_1^\text{bt}(3, 14) \oplus \{(2, 8)\}$                                       | $\{(4, 6), (2, 8)\}$          | $\{(4, 6), (2, 8)\}$          | ✔              | ✔               |

## Transposition <6, 7> in complex 0
Here is a __death-death__ __no switch__ transposition <6, 7>.

$$
a = 5, \; b = 7, \; x = 4, \; y = 6
$$

$$
x < a < y < b
$$

<img src="transpositions_stats-abstract.assets/image-e90c46e45ecc.png" width="1200">

### Birth Death Pairs
|                          |                                               |
|:-------------------------|:----------------------------------------------|
| Before the transposition | $\{(3, 14), (4, 6), (0, 9), (5, 7), (2, 8)\}$ |
| After the transposition  | $\{(3, 14), (4, 6), (0, 9), (5, 7), (2, 8)\}$ |

### Relations
|                                | Algorithm 1                                   | Algorithm 2                                          |
|:-------------------------------|:----------------------------------------------|:-----------------------------------------------------|
| Before the transposition (WC)  | $\{(6, 8), (6, 14), (7, 9), (6, 9), (7, 8)\}$ | $\{(5, 2), (5, 0), (2, 0), (4, 3)\}$                 |
| Before the transposition (WOC) | $\{(6, 8), (6, 14), (7, 9), (6, 9), (7, 8)\}$ | $\{(5, 2), (5, 0), (2, 0), (4, 3)\}$                 |
| After the transposition (WC)   | $\{(6, 8), (6, 14), (7, 9), (6, 9), (7, 8)\}$ | $\{(4, 3), (5, 4), (2, 0), (5, 0), (5, 3), (5, 2)\}$ |
| After the transposition (WOC)  | $\{(6, 8), (6, 14), (7, 9), (6, 9), (7, 8)\}$ | $\{(4, 3), (5, 4), (2, 0), (5, 0), (5, 3), (5, 2)\}$ |

### Successors and Predecessers
|                       | $\text{Set}^\text{bt}(5, 7)$                         | $\text{Set}^\text{bt}(4, 6)$                                  | $\text{Set}^\text{at}(5, 7)$                                          | $\text{Set}^\text{at}(4, 6)$                                  |
|:----------------------|:-----------------------------------------------------|:--------------------------------------------------------------|:----------------------------------------------------------------------|:--------------------------------------------------------------|
| $\text{Succ}_1$ (WC)  | $\text{Succ}_1^\text{bt}(5, 7) = \{(0, 9), (2, 8)\}$ | $\text{Succ}_1^\text{bt}(4, 6) = \{(3, 14), (0, 9), (2, 8)\}$ | $\text{Succ}_1^\text{at}(5, 7) = \{(0, 9), (2, 8)\}$                  | $\text{Succ}_1^\text{at}(4, 6) = \{(3, 14), (0, 9), (2, 8)\}$ |
| $\text{Succ}_1$ (WOC) | $\text{Succ}_1^\text{bt}(5, 7) = \{(0, 9), (2, 8)\}$ | $\text{Succ}_1^\text{bt}(4, 6) = \{(3, 14), (0, 9), (2, 8)\}$ | $\text{Succ}_1^\text{at}(5, 7) = \{(0, 9), (2, 8)\}$                  | $\text{Succ}_1^\text{at}(4, 6) = \{(3, 14), (0, 9), (2, 8)\}$ |
| $\text{Pred}_1$ (WC)  | $\text{Pred}_1^\text{bt}(5, 7) = \emptyset$          | $\text{Pred}_1^\text{bt}(4, 6) = \emptyset$                   | $\text{Pred}_1^\text{at}(5, 7) = \emptyset$                           | $\text{Pred}_1^\text{at}(4, 6) = \emptyset$                   |
| $\text{Pred}_1$ (WOC) | $\text{Pred}_1^\text{bt}(5, 7) = \emptyset$          | $\text{Pred}_1^\text{bt}(4, 6) = \emptyset$                   | $\text{Pred}_1^\text{at}(5, 7) = \emptyset$                           | $\text{Pred}_1^\text{at}(4, 6) = \emptyset$                   |
| $\text{Succ}_2$ (WC)  | $\text{Succ}_2^\text{bt}(5, 7) = \{(0, 9), (2, 8)\}$ | $\text{Succ}_2^\text{bt}(4, 6) = \{(3, 14)\}$                 | $\text{Succ}_2^\text{at}(5, 7) = \{(3, 14), (4, 6), (0, 9), (2, 8)\}$ | $\text{Succ}_2^\text{at}(4, 6) = \{(3, 14)\}$                 |
| $\text{Succ}_2$ (WOC) | $\text{Succ}_2^\text{bt}(5, 7) = \{(0, 9), (2, 8)\}$ | $\text{Succ}_2^\text{bt}(4, 6) = \{(3, 14)\}$                 | $\text{Succ}_2^\text{at}(5, 7) = \{(3, 14), (4, 6), (0, 9), (2, 8)\}$ | $\text{Succ}_2^\text{at}(4, 6) = \{(3, 14)\}$                 |
| $\text{Pred}_2$ (WC)  | $\text{Pred}_2^\text{bt}(5, 7) = \emptyset$          | $\text{Pred}_2^\text{bt}(4, 6) = \emptyset$                   | $\text{Pred}_2^\text{at}(5, 7) = \emptyset$                           | $\text{Pred}_2^\text{at}(4, 6) = \{(5, 7)\}$                  |
| $\text{Pred}_2$ (WOC) | $\text{Pred}_2^\text{bt}(5, 7) = \emptyset$          | $\text{Pred}_2^\text{bt}(4, 6) = \emptyset$                   | $\text{Pred}_2^\text{at}(5, 7) = \emptyset$                           | $\text{Pred}_2^\text{at}(4, 6) = \{(5, 7)\}$                  |

### Temp Sets
|                                                                     | Before the Transposition   | After the Transposition   |
|:--------------------------------------------------------------------|:---------------------------|:--------------------------|
| $\mathcal{B} = \{(s, t)\in \text{BD}:\; f(s) > f(x)\}$ (WC)         | $\{(5, 7)\}$               | $\{(5, 7)\}$              |
| $\mathcal{B} = \{(s, t)\in \text{BD}:\; f(s) > f(x)\}$ (WOC)        | $\{(5, 7)\}$               | $\{(5, 7)\}$              |
| $\mathcal{N} = \{(s, t)\in \text{BD}:\; f(x) > f(s) > f(a)\}$ (WC)  | $\emptyset$                | $\emptyset$               |
| $\mathcal{N} = \{(s, t)\in \text{BD}:\; f(x) > f(s) > f(a)\}$ (WOC) | $\emptyset$                | $\emptyset$               |

### Equations
There are 2 equations are wrong.

| eq    | left (formula)                  | left (paramatrized)             | left (value) (WC)                     | left (value) (WOC)                    | right (formula)                                                                        | right (paramatrized)                                                                   | right (value) (WC)                    | right (value) (WOC)                   | correct (WC)   | correct (WOC)   |
|:------|:--------------------------------|:--------------------------------|:--------------------------------------|:--------------------------------------|:---------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|:--------------------------------------|:--------------------------------------|:---------------|:----------------|
| Eq36  | $\text{Succ}_2^\text{at}(x, y)$ | $\text{Succ}_2^\text{at}(4, 6)$ | $\{(3, 14)\}$                         | $\{(3, 14)\}$                         | $\text{Succ}_2^\text{bt}(x, y) \oplus \{(a, b)\} \oplus \text{Succ}_2^\text{bt}(a, b)$ | $\text{Succ}_2^\text{bt}(4, 6) \oplus \{(5, 7)\} \oplus \text{Succ}_2^\text{bt}(5, 7)$ | $\{(3, 14), (0, 9), (5, 7), (2, 8)\}$ | $\{(3, 14), (0, 9), (5, 7), (2, 8)\}$ | ✘              | ✘               |
| Eq36s | $\text{Succ}_2^\text{at}(a, b)$ | $\text{Succ}_2^\text{at}(5, 7)$ | $\{(3, 14), (4, 6), (0, 9), (2, 8)\}$ | $\{(3, 14), (4, 6), (0, 9), (2, 8)\}$ | $\text{Succ}_2^\text{bt}(a, b) \oplus \{(x, y)\} \oplus \text{Succ}_2^\text{bt}(x, y)$ | $\text{Succ}_2^\text{bt}(5, 7) \oplus \{(4, 6)\} \oplus \text{Succ}_2^\text{bt}(4, 6)$ | $\{(2, 8), (4, 6), (0, 9), (3, 14)\}$ | $\{(2, 8), (4, 6), (0, 9), (3, 14)\}$ | ✔              | ✔               |
| Eq37  | $\text{Pred}_2^\text{at}(a, b)$ | $\text{Pred}_2^\text{at}(5, 7)$ | $\emptyset$                           | $\emptyset$                           | $\text{Pred}_2^\text{bt}(a, b) \oplus \{(x, y)\}$                                      | $\text{Pred}_2^\text{bt}(5, 7) \oplus \{(4, 6)\}$                                      | $\{(4, 6)\}$                          | $\{(4, 6)\}$                          | ✘              | ✘               |
| Eq37s | $\text{Pred}_2^\text{at}(x, y)$ | $\text{Pred}_2^\text{at}(4, 6)$ | $\{(5, 7)\}$                          | $\{(5, 7)\}$                          | $\text{Pred}_2^\text{bt}(x, y) \oplus \{(a, b)\}$                                      | $\text{Pred}_2^\text{bt}(4, 6) \oplus \{(5, 7)\}$                                      | $\{(5, 7)\}$                          | $\{(5, 7)\}$                          | ✔              | ✔               |

## Transposition <8, 9> in complex 0
Here is a __death-death__ __no switch__ transposition <8, 9>.

$$
a = 0, \; b = 9, \; x = 2, \; y = 8
$$

$$
a < x < y < b
$$

<img src="transpositions_stats-abstract.assets/image-ebc81a95f1c0.png" width="1200">

### Birth Death Pairs
|                          |                                               |
|:-------------------------|:----------------------------------------------|
| Before the transposition | $\{(3, 14), (4, 6), (0, 9), (5, 7), (2, 8)\}$ |
| After the transposition  | $\{(3, 14), (4, 6), (0, 9), (5, 7), (2, 8)\}$ |

### Relations
|                                | Algorithm 1                                   | Algorithm 2                          |
|:-------------------------------|:----------------------------------------------|:-------------------------------------|
| Before the transposition (WC)  | $\{(6, 8), (6, 14), (7, 9), (6, 9), (7, 8)\}$ | $\{(5, 2), (5, 0), (2, 0), (4, 3)\}$ |
| Before the transposition (WOC) | $\{(6, 8), (6, 14), (7, 9), (6, 9), (7, 8)\}$ | $\{(5, 2), (5, 0), (2, 0), (4, 3)\}$ |
| After the transposition (WC)   | $\{(6, 8), (6, 14), (7, 9), (6, 9), (7, 8)\}$ | $\{(5, 2), (5, 0), (4, 3)\}$         |
| After the transposition (WOC)  | $\{(6, 8), (6, 14), (7, 9), (6, 9), (7, 8)\}$ | $\{(5, 2), (5, 0), (4, 3)\}$         |

### Successors and Predecessers
|                       | $\text{Set}^\text{bt}(0, 9)$                         | $\text{Set}^\text{bt}(2, 8)$                         | $\text{Set}^\text{at}(0, 9)$                         | $\text{Set}^\text{at}(2, 8)$                         |
|:----------------------|:-----------------------------------------------------|:-----------------------------------------------------|:-----------------------------------------------------|:-----------------------------------------------------|
| $\text{Succ}_1$ (WC)  | $\text{Succ}_1^\text{bt}(0, 9) = \emptyset$          | $\text{Succ}_1^\text{bt}(2, 8) = \emptyset$          | $\text{Succ}_1^\text{at}(0, 9) = \emptyset$          | $\text{Succ}_1^\text{at}(2, 8) = \emptyset$          |
| $\text{Succ}_1$ (WOC) | $\text{Succ}_1^\text{bt}(0, 9) = \emptyset$          | $\text{Succ}_1^\text{bt}(2, 8) = \emptyset$          | $\text{Succ}_1^\text{at}(0, 9) = \emptyset$          | $\text{Succ}_1^\text{at}(2, 8) = \emptyset$          |
| $\text{Pred}_1$ (WC)  | $\text{Pred}_1^\text{bt}(0, 9) = \{(4, 6), (5, 7)\}$ | $\text{Pred}_1^\text{bt}(2, 8) = \{(4, 6), (5, 7)\}$ | $\text{Pred}_1^\text{at}(0, 9) = \{(4, 6), (5, 7)\}$ | $\text{Pred}_1^\text{at}(2, 8) = \{(4, 6), (5, 7)\}$ |
| $\text{Pred}_1$ (WOC) | $\text{Pred}_1^\text{bt}(0, 9) = \{(4, 6), (5, 7)\}$ | $\text{Pred}_1^\text{bt}(2, 8) = \{(4, 6), (5, 7)\}$ | $\text{Pred}_1^\text{at}(0, 9) = \{(4, 6), (5, 7)\}$ | $\text{Pred}_1^\text{at}(2, 8) = \{(4, 6), (5, 7)\}$ |
| $\text{Succ}_2$ (WC)  | $\text{Succ}_2^\text{bt}(0, 9) = \emptyset$          | $\text{Succ}_2^\text{bt}(2, 8) = \{(0, 9)\}$         | $\text{Succ}_2^\text{at}(0, 9) = \emptyset$          | $\text{Succ}_2^\text{at}(2, 8) = \emptyset$          |
| $\text{Succ}_2$ (WOC) | $\text{Succ}_2^\text{bt}(0, 9) = \emptyset$          | $\text{Succ}_2^\text{bt}(2, 8) = \{(0, 9)\}$         | $\text{Succ}_2^\text{at}(0, 9) = \emptyset$          | $\text{Succ}_2^\text{at}(2, 8) = \emptyset$          |
| $\text{Pred}_2$ (WC)  | $\text{Pred}_2^\text{bt}(0, 9) = \{(5, 7), (2, 8)\}$ | $\text{Pred}_2^\text{bt}(2, 8) = \{(5, 7)\}$         | $\text{Pred}_2^\text{at}(0, 9) = \{(5, 7)\}$         | $\text{Pred}_2^\text{at}(2, 8) = \{(5, 7)\}$         |
| $\text{Pred}_2$ (WOC) | $\text{Pred}_2^\text{bt}(0, 9) = \{(5, 7), (2, 8)\}$ | $\text{Pred}_2^\text{bt}(2, 8) = \{(5, 7)\}$         | $\text{Pred}_2^\text{at}(0, 9) = \{(5, 7)\}$         | $\text{Pred}_2^\text{at}(2, 8) = \{(5, 7)\}$         |

### Temp Sets
|                                                                     | Before the Transposition      | After the Transposition       |
|:--------------------------------------------------------------------|:------------------------------|:------------------------------|
| $\mathcal{B} = \{(s, t)\in \text{BD}:\; f(s) > f(x)\}$ (WC)         | $\{(4, 6), (5, 7), (3, 14)\}$ | $\{(4, 6), (5, 7), (3, 14)\}$ |
| $\mathcal{B} = \{(s, t)\in \text{BD}:\; f(s) > f(x)\}$ (WOC)        | $\{(4, 6), (5, 7), (3, 14)\}$ | $\{(4, 6), (5, 7), (3, 14)\}$ |
| $\mathcal{N} = \{(s, t)\in \text{BD}:\; f(x) > f(s) > f(a)\}$ (WC)  | $\emptyset$                   | $\emptyset$                   |
| $\mathcal{N} = \{(s, t)\in \text{BD}:\; f(x) > f(s) > f(a)\}$ (WOC) | $\emptyset$                   | $\emptyset$                   |

### Equations
There are 2 equations are wrong.

| eq    | left (formula)                  | left (paramatrized)             | left (value) (WC)   | left (value) (WOC)   | right (formula)                                                                        | right (paramatrized)                                                                   | right (value) (WC)   | right (value) (WOC)   | correct (WC)   | correct (WOC)   |
|:------|:--------------------------------|:--------------------------------|:--------------------|:---------------------|:---------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|:---------------------|:----------------------|:---------------|:----------------|
| Eq36  | $\text{Succ}_2^\text{at}(x, y)$ | $\text{Succ}_2^\text{at}(2, 8)$ | $\emptyset$         | $\emptyset$          | $\text{Succ}_2^\text{bt}(x, y) \oplus \{(a, b)\} \oplus \text{Succ}_2^\text{bt}(a, b)$ | $\text{Succ}_2^\text{bt}(2, 8) \oplus \{(0, 9)\} \oplus \text{Succ}_2^\text{bt}(0, 9)$ | $\emptyset$          | $\emptyset$           | ✔              | ✔               |
| Eq36s | $\text{Succ}_2^\text{at}(a, b)$ | $\text{Succ}_2^\text{at}(0, 9)$ | $\emptyset$         | $\emptyset$          | $\text{Succ}_2^\text{bt}(a, b) \oplus \{(x, y)\} \oplus \text{Succ}_2^\text{bt}(x, y)$ | $\text{Succ}_2^\text{bt}(0, 9) \oplus \{(2, 8)\} \oplus \text{Succ}_2^\text{bt}(2, 8)$ | $\{(0, 9), (2, 8)\}$ | $\{(0, 9), (2, 8)\}$  | ✘              | ✘               |
| Eq37  | $\text{Pred}_2^\text{at}(a, b)$ | $\text{Pred}_2^\text{at}(0, 9)$ | $\{(5, 7)\}$        | $\{(5, 7)\}$         | $\text{Pred}_2^\text{bt}(a, b) \oplus \{(x, y)\}$                                      | $\text{Pred}_2^\text{bt}(0, 9) \oplus \{(2, 8)\}$                                      | $\{(5, 7)\}$         | $\{(5, 7)\}$          | ✔              | ✔               |
| Eq37s | $\text{Pred}_2^\text{at}(x, y)$ | $\text{Pred}_2^\text{at}(2, 8)$ | $\{(5, 7)\}$        | $\{(5, 7)\}$         | $\text{Pred}_2^\text{bt}(x, y) \oplus \{(a, b)\}$                                      | $\text{Pred}_2^\text{bt}(2, 8) \oplus \{(0, 9)\}$                                      | $\{(0, 9), (5, 7)\}$ | $\{(0, 9), (5, 7)\}$  | ✘              | ✘               |

## Transposition <14, 15> in complex 0
Here is a __birth-death__ __switch forward__ transposition <14, 15>.

$$
a = 3, \; b = 14, \; x = 15, \; y = 26
$$

$$
a < b < x < y
$$

<img src="transpositions_stats-abstract.assets/image-1a74867e9f22.png" width="1200">

### Birth Death Pairs
|                          |                                                                                                                                                   |
|:-------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition | $\{(17, 23), (3, 14), (2, 8), (10, 27), (4, 6), (11, 32), (0, 9), (13, 29), (15, 26), (5, 7), (16, 25), (19, 22), (12, 31), (18, 24), (20, 21)\}$ |
| After the transposition  | $\{(17, 23), (2, 8), (10, 27), (4, 6), (14, 26), (11, 32), (13, 29), (5, 7), (0, 9), (16, 25), (19, 22), (12, 31), (3, 15), (18, 24), (20, 21)\}$ |

### Relations
|                                | Algorithm 1                                                                                                                                                                                                                                                                                                                                                                                                                 | Algorithm 2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|:-------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition (WC)  | $\{(24, 27), (25, 32), (26, 27), (31, 32), (29, 32), (22, 32), (6, 8), (6, 14), (24, 26), (24, 32), (26, 29), (29, 31), (23, 27), (22, 31), (7, 9), (25, 27), (24, 31), (21, 23), (26, 31), (22, 27), (23, 26), (22, 24), (23, 29), (23, 32), (6, 9), (7, 8)\}$                                                                                                                                                             | $\{(4, 3), (20, 17), (17, 12), (17, 15), (19, 18), (19, 15), (15, 11), (16, 10), (18, 10), (20, 13), (18, 13), (20, 10), (5, 0), (20, 19), (15, 13), (16, 15), (18, 15), (12, 11), (20, 12), (20, 18), (20, 15), (5, 2), (17, 10), (17, 16), (17, 13), (2, 0), (19, 16), (13, 12), (15, 12), (16, 11)\}$                                                                                                                                                                                                                            |
| Before the transposition (WOC) | $\{(6, 12), (24, 27), (25, 32), (7, 20), (26, 27), (14, 16), (31, 32), (29, 32), (14, 19), (22, 32), (9, 11), (7, 10), (7, 16), (6, 14), (6, 8), (24, 26), (7, 19), (24, 32), (26, 29), (29, 31), (14, 15), (8, 11), (23, 27), (9, 16), (22, 31), (9, 19), (8, 20), (7, 9), (6, 16), (25, 27), (7, 15), (24, 31), (26, 31), (21, 23), (22, 27), (23, 26), (22, 24), (23, 29), (23, 32), (8, 13), (8, 19), (6, 9), (7, 8)\}$ | $\{(20, 8), (4, 3), (20, 17), (17, 9), (19, 6), (17, 12), (17, 15), (19, 18), (11, 8), (19, 15), (13, 8), (15, 11), (16, 10), (18, 7), (15, 14), (18, 10), (20, 13), (18, 13), (20, 10), (12, 9), (12, 6), (5, 0), (20, 19), (17, 8), (19, 8), (19, 14), (13, 7), (15, 13), (20, 6), (16, 15), (20, 9), (18, 15), (20, 12), (12, 11), (20, 18), (12, 8), (20, 15), (5, 2), (17, 7), (17, 10), (17, 16), (17, 13), (2, 0), (11, 9), (19, 16), (10, 7), (15, 6), (13, 12), (15, 9), (16, 8), (15, 12), (16, 11), (18, 8), (18, 14)\}$ |
| After the transposition (WC)   | $\{(24, 27), (25, 32), (26, 27), (31, 32), (29, 32), (22, 32), (24, 26), (6, 8), (24, 32), (26, 29), (29, 31), (23, 27), (22, 31), (7, 9), (25, 27), (7, 15), (24, 31), (21, 23), (26, 31), (22, 27), (23, 26), (22, 24), (23, 29), (23, 32), (6, 9), (7, 8)\}$                                                                                                                                                             | $\{(4, 3), (20, 17), (14, 13), (17, 12), (19, 18), (16, 10), (18, 10), (20, 13), (18, 13), (20, 10), (5, 0), (20, 19), (14, 12), (19, 14), (12, 11), (20, 12), (20, 18), (14, 11), (5, 2), (17, 10), (17, 16), (17, 13), (2, 0), (19, 16), (13, 12), (16, 11), (18, 14)\}$                                                                                                                                                                                                                                                          |
| After the transposition (WOC)  | $\{(6, 12), (24, 27), (25, 32), (7, 20), (26, 27), (31, 32), (29, 32), (22, 32), (9, 11), (15, 14), (7, 10), (7, 16), (6, 8), (24, 26), (6, 14), (7, 19), (24, 32), (26, 29), (29, 31), (8, 11), (23, 27), (9, 16), (22, 31), (9, 19), (8, 20), (7, 9), (15, 16), (15, 19), (6, 16), (25, 27), (7, 15), (24, 31), (21, 23), (26, 31), (22, 27), (23, 26), (22, 24), (23, 29), (23, 32), (8, 13), (8, 19), (6, 9), (7, 8)\}$ | $\{(20, 8), (4, 3), (20, 17), (17, 9), (19, 6), (14, 13), (17, 12), (17, 15), (19, 18), (11, 8), (19, 15), (13, 8), (16, 10), (18, 7), (18, 10), (20, 13), (18, 13), (20, 10), (12, 9), (14, 6), (12, 6), (5, 0), (20, 19), (14, 9), (14, 12), (17, 8), (14, 15), (19, 8), (19, 14), (13, 7), (20, 6), (16, 15), (20, 9), (18, 15), (20, 12), (12, 11), (20, 18), (12, 8), (20, 15), (14, 11), (5, 2), (17, 7), (17, 10), (17, 16), (17, 13), (2, 0), (11, 9), (19, 16), (10, 7), (13, 12), (16, 8), (16, 11), (18, 8), (18, 14)\}$ |

### Successors and Predecessers
|                       | $\text{Set}^\text{bt}(3, 14)$                 | $\text{Set}^\text{bt}(15, 26)$                                                           | $\text{Set}^\text{at}(3, 15)$                 | $\text{Set}^\text{at}(14, 26)$                                       |
|:----------------------|:----------------------------------------------|:-----------------------------------------------------------------------------------------|:----------------------------------------------|:---------------------------------------------------------------------|
| $\text{Succ}_1$ (WC)  | $\text{Succ}_1^\text{bt}(3, 14) = \emptyset$  | $\text{Succ}_1^\text{bt}(15, 26) = \{(10, 27), (13, 29), (12, 31)\}$                     | $\text{Succ}_1^\text{at}(3, 15) = \emptyset$  | $\text{Succ}_1^\text{at}(14, 26) = \{(10, 27), (13, 29), (12, 31)\}$ |
| $\text{Succ}_1$ (WOC) | $\text{Succ}_1^\text{bt}(3, 14) = \emptyset$  | $\text{Succ}_1^\text{bt}(15, 26) = \{(10, 27), (13, 29), (12, 31)\}$                     | $\text{Succ}_1^\text{at}(3, 15) = \emptyset$  | $\text{Succ}_1^\text{at}(14, 26) = \{(10, 27), (13, 29), (12, 31)\}$ |
| $\text{Pred}_1$ (WC)  | $\text{Pred}_1^\text{bt}(3, 14) = \{(4, 6)\}$ | $\text{Pred}_1^\text{bt}(15, 26) = \{(17, 23), (18, 24)\}$                               | $\text{Pred}_1^\text{at}(3, 15) = \{(5, 7)\}$ | $\text{Pred}_1^\text{at}(14, 26) = \{(17, 23), (18, 24)\}$           |
| $\text{Pred}_1$ (WOC) | $\text{Pred}_1^\text{bt}(3, 14) = \{(4, 6)\}$ | $\text{Pred}_1^\text{bt}(15, 26) = \{(17, 23), (18, 24)\}$                               | $\text{Pred}_1^\text{at}(3, 15) = \{(5, 7)\}$ | $\text{Pred}_1^\text{at}(14, 26) = \{(17, 23), (18, 24)\}$           |
| $\text{Succ}_2$ (WC)  | $\text{Succ}_2^\text{bt}(3, 14) = \emptyset$  | $\text{Succ}_2^\text{bt}(15, 26) = \{(13, 29), (11, 32), (12, 31)\}$                     | $\text{Succ}_2^\text{at}(3, 15) = \emptyset$  | $\text{Succ}_2^\text{at}(14, 26) = \{(13, 29), (11, 32), (12, 31)\}$ |
| $\text{Succ}_2$ (WOC) | $\text{Succ}_2^\text{bt}(3, 14) = \emptyset$  | $\text{Succ}_2^\text{bt}(15, 26) = \{(13, 29), (11, 32), (12, 31)\}$                     | $\text{Succ}_2^\text{at}(3, 15) = \emptyset$  | $\text{Succ}_2^\text{at}(14, 26) = \{(13, 29), (11, 32), (12, 31)\}$ |
| $\text{Pred}_2$ (WC)  | $\text{Pred}_2^\text{bt}(3, 14) = \{(4, 6)\}$ | $\text{Pred}_2^\text{bt}(15, 26) = \{(17, 23), (16, 25), (19, 22), (18, 24), (20, 21)\}$ | $\text{Pred}_2^\text{at}(3, 15) = \{(4, 6)\}$ | $\text{Pred}_2^\text{at}(14, 26) = \{(19, 22), (18, 24)\}$           |
| $\text{Pred}_2$ (WOC) | $\text{Pred}_2^\text{bt}(3, 14) = \{(4, 6)\}$ | $\text{Pred}_2^\text{bt}(15, 26) = \{(17, 23), (16, 25), (19, 22), (18, 24), (20, 21)\}$ | $\text{Pred}_2^\text{at}(3, 15) = \{(4, 6)\}$ | $\text{Pred}_2^\text{at}(14, 26) = \{(19, 22), (18, 24)\}$           |

### Temp Sets
|                                                                    | Before the Transposition                               | After the Transposition                                |
|:-------------------------------------------------------------------|:-------------------------------------------------------|:-------------------------------------------------------|
| $A_1 = \{(s, t)\in \text{BD}:\; f(a) < f(s) < f(t) < f(x)\}$ (WC)  | $\{(4, 6), (5, 7)\}$                                   | $\{(4, 6), (5, 7)\}$                                   |
| $A_1 = \{(s, t)\in \text{BD}:\; f(a) < f(s) < f(t) < f(x)\}$ (WOC) | $\{(4, 6), (5, 7)\}$                                   | $\{(4, 6), (5, 7)\}$                                   |
| $B_1 = \{(s, t)\in \text{BD}:\; U_1[t, x] = 1\}$ (WC)              | $\emptyset$                                            | $\{(5, 7)\}$                                           |
| $B_1 = \{(s, t)\in \text{BD}:\; U_1[t, x] = 1\}$ (WOC)             | $\{(3, 14), (5, 7)\}$                                  | $\{(5, 7)\}$                                           |
| $A_2 = \{(s, t)\in \text{BD}:\; f(b) < f(s) < f(t) < f(y)\}$ (WC)  | $\{(17, 23), (16, 25), (19, 22), (18, 24), (20, 21)\}$ | $\{(17, 23), (16, 25), (19, 22), (18, 24), (20, 21)\}$ |
| $A_2 = \{(s, t)\in \text{BD}:\; f(b) < f(s) < f(t) < f(y)\}$ (WOC) | $\{(17, 23), (16, 25), (19, 22), (18, 24), (20, 21)\}$ | $\{(17, 23), (16, 25), (19, 22), (18, 24), (20, 21)\}$ |
| $B_2 = \{(s, t)\in \text{BD}:\; U_2[b, s] = 1\}$ (WC)              | $\emptyset$                                            | $\{(19, 22), (18, 24)\}$                               |
| $B_2 = \{(s, t)\in \text{BD}:\; U_2[b, s] = 1\}$ (WOC)             | $\{(19, 22), (18, 24), (15, 26)\}$                     | $\{(19, 22), (18, 24)\}$                               |

### Equations
There are 4 equations are wrong.

| eq    | left (formula)                  | left (paramatrized)               | left (value) (WC)                  | left (value) (WOC)                 | right (formula)                                      | right (paramatrized)                                  | right (value) (WC)                 | right (value) (WOC)                | correct (WC)   | correct (WOC)   |
|:------|:--------------------------------|:----------------------------------|:-----------------------------------|:-----------------------------------|:-----------------------------------------------------|:------------------------------------------------------|:-----------------------------------|:-----------------------------------|:---------------|:----------------|
| Eq48  | $\text{Succ}_1^\text{at}(a, x)$ | $\text{Succ}_1^\text{at}(3, 15)$  | $\emptyset$                        | $\emptyset$                        | $\text{Succ}_1^\text{bt}(a, b)$                      | $\text{Succ}_1^\text{bt}(3, 14)$                      | $\emptyset$                        | $\emptyset$                        | ✔              | ✔               |
| Eq49  | $\text{Succ}_1^\text{at}(b, y)$ | $\text{Succ}_1^\text{at}(14, 26)$ | $\{(10, 27), (13, 29), (12, 31)\}$ | $\{(10, 27), (13, 29), (12, 31)\}$ | $\text{Succ}_1^\text{bt}(x, y)$                      | $\text{Succ}_1^\text{bt}(15, 26)$                     | $\{(10, 27), (13, 29), (12, 31)\}$ | $\{(10, 27), (13, 29), (12, 31)\}$ | ✔              | ✔               |
| Eq50  | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(3, 15)$  | $\{(5, 7)\}$                       | $\{(5, 7)\}$                       | $\{(s, t)\in \text{BD}:\; U_1^\text{bt}[t, x] = 1\}$ | $\{(s, t)\in \text{BD}:\; U_1^\text{bt}[t, 15] = 1\}$ | $\emptyset$                        | $\{(4, 6), (2, 8)\}$               | ✘              | ✘               |
| Eq50a | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(3, 15)$  | $\{(5, 7)\}$                       | $\{(5, 7)\}$                       | $A_1^\text{bt} \cap B_1^\text{bt}$                   | $A_1^\text{bt} \cap B_1^\text{bt}$                    | $\emptyset$                        | $\{(5, 7)\}$                       | ✘              | ✔               |
| Eq50b | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(3, 15)$  | $\{(5, 7)\}$                       | $\{(5, 7)\}$                       | $A_1^\text{bt} \cap B_1^\text{at}$                   | $A_1^\text{bt} \cap B_1^\text{at}$                    | $\{(5, 7)\}$                       | $\{(5, 7)\}$                       | ✔              | ✔               |
| Eq51  | $\text{Pred}_1^\text{at}(b, y)$ | $\text{Pred}_1^\text{at}(14, 26)$ | $\{(17, 23), (18, 24)\}$           | $\{(17, 23), (18, 24)\}$           | $\text{Pred}_1^\text{bt}(x, y)$                      | $\text{Pred}_1^\text{bt}(15, 26)$                     | $\{(17, 23), (18, 24)\}$           | $\{(17, 23), (18, 24)\}$           | ✔              | ✔               |
| Eq52  | $\text{Succ}_2^\text{at}(a, x)$ | $\text{Succ}_2^\text{at}(3, 15)$  | $\emptyset$                        | $\emptyset$                        | $\text{Succ}_2^\text{bt}(a, b)$                      | $\text{Succ}_2^\text{bt}(3, 14)$                      | $\emptyset$                        | $\emptyset$                        | ✔              | ✔               |
| Eq53  | $\text{Succ}_2^\text{at}(b, y)$ | $\text{Succ}_2^\text{at}(14, 26)$ | $\{(13, 29), (11, 32), (12, 31)\}$ | $\{(13, 29), (11, 32), (12, 31)\}$ | $\text{Succ}_2^\text{bt}(x, y)$                      | $\text{Succ}_2^\text{bt}(15, 26)$                     | $\{(13, 29), (11, 32), (12, 31)\}$ | $\{(13, 29), (11, 32), (12, 31)\}$ | ✔              | ✔               |
| Eq54  | $\text{Pred}_2^\text{at}(a, x)$ | $\text{Pred}_2^\text{at}(3, 15)$  | $\{(4, 6)\}$                       | $\{(4, 6)\}$                       | $\text{Pred}_2^\text{bt}(a, b)$                      | $\text{Pred}_2^\text{bt}(3, 14)$                      | $\{(4, 6)\}$                       | $\{(4, 6)\}$                       | ✔              | ✔               |
| Eq55  | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(14, 26)$ | $\{(19, 22), (18, 24)\}$           | $\{(19, 22), (18, 24)\}$           | $\{(s, t)\in \text{BD}:\; U_2^\text{bt}[b, s] = 1\}$ | $\{(s, t)\in \text{BD}:\; U_2^\text{bt}[14, s] = 1\}$ | $\emptyset$                        | $\{(13, 29)\}$                     | ✘              | ✘               |
| Eq55a | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(14, 26)$ | $\{(19, 22), (18, 24)\}$           | $\{(19, 22), (18, 24)\}$           | $A_2^\text{bt} \cap B_2^\text{bt}$                   | $A_2^\text{bt} \cap B_2^\text{bt}$                    | $\emptyset$                        | $\{(18, 24), (19, 22)\}$           | ✘              | ✔               |
| Eq55b | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(14, 26)$ | $\{(19, 22), (18, 24)\}$           | $\{(19, 22), (18, 24)\}$           | $A_2^\text{bt} \cap B_2^\text{at}$                   | $A_2^\text{bt} \cap B_2^\text{at}$                    | $\{(18, 24), (19, 22)\}$           | $\{(18, 24), (19, 22)\}$           | ✔              | ✔               |

## Transposition <15, 16> in complex 0
Here is a __birth-birth__ __switch forward__ transposition <15, 16>.

$$
a = 15, \; b = 26, \; x = 16, \; y = 25
$$

$$
a < x < y < b
$$

<img src="transpositions_stats-abstract.assets/image-95b0b3f656a6.png" width="1200">

### Birth Death Pairs
|                          |                                                                                                          |
|:-------------------------|:---------------------------------------------------------------------------------------------------------|
| Before the transposition | $\{(17, 23), (10, 27), (11, 32), (13, 29), (15, 26), (16, 25), (19, 22), (12, 31), (18, 24), (20, 21)\}$ |
| After the transposition  | $\{(17, 23), (16, 26), (10, 27), (11, 32), (13, 29), (19, 22), (15, 25), (12, 31), (18, 24), (20, 21)\}$ |

### Relations
|                                | Algorithm 1                                                                                                                                                                                                                                | Algorithm 2                                                                                                                                                                                                                                                              |
|:-------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition (WC)  | $\{(24, 27), (25, 32), (26, 27), (31, 32), (29, 32), (22, 32), (24, 26), (24, 32), (26, 29), (29, 31), (23, 27), (22, 31), (25, 27), (24, 31), (21, 23), (26, 31), (22, 27), (23, 26), (22, 24), (23, 29), (23, 32)\}$                     | $\{(20, 17), (17, 12), (17, 15), (19, 18), (19, 15), (15, 11), (16, 10), (18, 10), (20, 13), (18, 13), (20, 10), (20, 19), (15, 13), (16, 15), (18, 15), (12, 11), (20, 18), (20, 12), (20, 15), (17, 10), (17, 16), (17, 13), (19, 16), (13, 12), (15, 12), (16, 11)\}$ |
| Before the transposition (WOC) | $\{(24, 27), (25, 32), (26, 27), (31, 32), (29, 32), (22, 32), (24, 26), (24, 32), (26, 29), (29, 31), (23, 27), (22, 31), (25, 27), (24, 31), (21, 23), (26, 31), (22, 27), (23, 26), (22, 24), (23, 29), (23, 32)\}$                     | $\{(20, 17), (17, 12), (17, 15), (19, 18), (19, 15), (15, 11), (16, 10), (18, 10), (20, 13), (18, 13), (20, 10), (20, 19), (15, 13), (16, 15), (18, 15), (12, 11), (20, 18), (20, 12), (20, 15), (17, 10), (17, 16), (17, 13), (19, 16), (13, 12), (15, 12), (16, 11)\}$ |
| After the transposition (WC)   | $\{(25, 29), (24, 27), (25, 32), (26, 27), (31, 32), (29, 32), (22, 32), (24, 26), (25, 31), (24, 32), (26, 29), (29, 31), (23, 27), (22, 31), (24, 31), (21, 23), (26, 31), (22, 27), (23, 26), (22, 24), (23, 29), (23, 32), (25, 26)\}$ | $\{(20, 17), (17, 12), (17, 15), (19, 18), (19, 15), (15, 11), (16, 10), (16, 13), (18, 10), (20, 13), (18, 13), (20, 10), (20, 19), (16, 12), (15, 10), (15, 16), (18, 15), (12, 11), (20, 12), (20, 18), (20, 15), (17, 10), (17, 16), (17, 13), (19, 16), (13, 12)\}$ |
| After the transposition (WOC)  | $\{(25, 29), (24, 27), (25, 32), (26, 27), (31, 32), (29, 32), (22, 32), (24, 26), (25, 31), (24, 32), (26, 29), (29, 31), (23, 27), (22, 31), (24, 31), (21, 23), (26, 31), (22, 27), (23, 26), (22, 24), (23, 29), (23, 32), (25, 26)\}$ | $\{(20, 17), (17, 12), (17, 15), (19, 18), (19, 15), (15, 11), (16, 10), (16, 13), (18, 10), (20, 13), (18, 13), (20, 10), (20, 19), (16, 12), (15, 10), (15, 16), (18, 15), (12, 11), (20, 12), (20, 18), (20, 15), (17, 10), (17, 16), (17, 13), (19, 16), (13, 12)\}$ |

### Successors and Predecessers
|                       | $\text{Set}^\text{bt}(15, 26)$                                                           | $\text{Set}^\text{bt}(16, 25)$                                       | $\text{Set}^\text{at}(15, 25)$                                                 | $\text{Set}^\text{at}(16, 26)$                                       |
|:----------------------|:-----------------------------------------------------------------------------------------|:---------------------------------------------------------------------|:-------------------------------------------------------------------------------|:---------------------------------------------------------------------|
| $\text{Succ}_1$ (WC)  | $\text{Succ}_1^\text{bt}(15, 26) = \{(10, 27), (13, 29), (12, 31)\}$                     | $\text{Succ}_1^\text{bt}(16, 25) = \{(10, 27), (11, 32)\}$           | $\text{Succ}_1^\text{at}(15, 25) = \{(13, 29), (16, 26), (11, 32), (12, 31)\}$ | $\text{Succ}_1^\text{at}(16, 26) = \{(10, 27), (13, 29), (12, 31)\}$ |
| $\text{Succ}_1$ (WOC) | $\text{Succ}_1^\text{bt}(15, 26) = \{(10, 27), (13, 29), (12, 31)\}$                     | $\text{Succ}_1^\text{bt}(16, 25) = \{(10, 27), (11, 32)\}$           | $\text{Succ}_1^\text{at}(15, 25) = \{(13, 29), (16, 26), (11, 32), (12, 31)\}$ | $\text{Succ}_1^\text{at}(16, 26) = \{(10, 27), (13, 29), (12, 31)\}$ |
| $\text{Pred}_1$ (WC)  | $\text{Pred}_1^\text{bt}(15, 26) = \{(17, 23), (18, 24)\}$                               | $\text{Pred}_1^\text{bt}(16, 25) = \emptyset$                        | $\text{Pred}_1^\text{at}(15, 25) = \emptyset$                                  | $\text{Pred}_1^\text{at}(16, 26) = \{(17, 23), (18, 24), (15, 25)\}$ |
| $\text{Pred}_1$ (WOC) | $\text{Pred}_1^\text{bt}(15, 26) = \{(17, 23), (18, 24)\}$                               | $\text{Pred}_1^\text{bt}(16, 25) = \emptyset$                        | $\text{Pred}_1^\text{at}(15, 25) = \emptyset$                                  | $\text{Pred}_1^\text{at}(16, 26) = \{(17, 23), (18, 24), (15, 25)\}$ |
| $\text{Succ}_2$ (WC)  | $\text{Succ}_2^\text{bt}(15, 26) = \{(13, 29), (11, 32), (12, 31)\}$                     | $\text{Succ}_2^\text{bt}(16, 25) = \{(10, 27), (11, 32), (15, 26)\}$ | $\text{Succ}_2^\text{at}(15, 25) = \{(10, 27), (16, 26), (11, 32)\}$           | $\text{Succ}_2^\text{at}(16, 26) = \{(10, 27), (13, 29), (12, 31)\}$ |
| $\text{Succ}_2$ (WOC) | $\text{Succ}_2^\text{bt}(15, 26) = \{(13, 29), (11, 32), (12, 31)\}$                     | $\text{Succ}_2^\text{bt}(16, 25) = \{(10, 27), (11, 32), (15, 26)\}$ | $\text{Succ}_2^\text{at}(15, 25) = \{(10, 27), (16, 26), (11, 32)\}$           | $\text{Succ}_2^\text{at}(16, 26) = \{(10, 27), (13, 29), (12, 31)\}$ |
| $\text{Pred}_2$ (WC)  | $\text{Pred}_2^\text{bt}(15, 26) = \{(17, 23), (16, 25), (19, 22), (18, 24), (20, 21)\}$ | $\text{Pred}_2^\text{bt}(16, 25) = \{(17, 23), (19, 22)\}$           | $\text{Pred}_2^\text{at}(15, 25) = \{(19, 22), (17, 23), (18, 24), (20, 21)\}$ | $\text{Pred}_2^\text{at}(16, 26) = \{(17, 23), (19, 22), (15, 25)\}$ |
| $\text{Pred}_2$ (WOC) | $\text{Pred}_2^\text{bt}(15, 26) = \{(17, 23), (16, 25), (19, 22), (18, 24), (20, 21)\}$ | $\text{Pred}_2^\text{bt}(16, 25) = \{(17, 23), (19, 22)\}$           | $\text{Pred}_2^\text{at}(15, 25) = \{(19, 22), (17, 23), (18, 24), (20, 21)\}$ | $\text{Pred}_2^\text{at}(16, 26) = \{(17, 23), (19, 22), (15, 25)\}$ |

### Temp Sets
|                                                                     | Before the Transposition                     | After the Transposition                      |
|:--------------------------------------------------------------------|:---------------------------------------------|:---------------------------------------------|
| $\mathcal{L} = \{(s, t)\in \text{BD}:\; f(t) < f(y)\}$ (WC)         | $\{(19, 22), (17, 23), (18, 24), (20, 21)\}$ | $\{(19, 22), (17, 23), (18, 24), (20, 21)\}$ |
| $\mathcal{L} = \{(s, t)\in \text{BD}:\; f(t) < f(y)\}$ (WOC)        | $\{(19, 22), (17, 23), (18, 24), (20, 21)\}$ | $\{(19, 22), (17, 23), (18, 24), (20, 21)\}$ |
| $\mathcal{M} = \{(s, t)\in \text{BD}:\; f(y) < f(t) < f(b)\}$ (WC)  | $\emptyset$                                  | $\emptyset$                                  |
| $\mathcal{M} = \{(s, t)\in \text{BD}:\; f(y) < f(t) < f(b)\}$ (WOC) | $\emptyset$                                  | $\emptyset$                                  |

### Equations
There are 1 equations are wrong.

| eq   | left (formula)                  | left (paramatrized)               | left (value) (WC)                            | left (value) (WOC)                           | right (formula)                                                                                           | right (paramatrized)                                                                                            | right (value) (WC)                           | right (value) (WOC)                          | correct (WC)   | correct (WOC)   |
|:-----|:--------------------------------|:----------------------------------|:---------------------------------------------|:---------------------------------------------|:----------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------|:---------------------------------------------|:---------------------------------------------|:---------------|:----------------|
| Eq08 | $\text{Succ}_1^\text{at}(a, y)$ | $\text{Succ}_1^\text{at}(15, 25)$ | $\{(13, 29), (16, 26), (11, 32), (12, 31)\}$ | $\{(13, 29), (16, 26), (11, 32), (12, 31)\}$ | $\text{Succ}_1^\text{bt}(x, y) \oplus \{(x, b)\} \oplus \text{Succ}_1^\text{bt}(a, b)$                    | $\text{Succ}_1^\text{bt}(16, 25) \oplus \{(16, 26)\} \oplus \text{Succ}_1^\text{bt}(15, 26)$                    | $\{(16, 26), (11, 32), (12, 31), (13, 29)\}$ | $\{(16, 26), (11, 32), (12, 31), (13, 29)\}$ | ✔              | ✔               |
| Eq09 | $\text{Succ}_1^\text{at}(x, b)$ | $\text{Succ}_1^\text{at}(16, 26)$ | $\{(10, 27), (13, 29), (12, 31)\}$           | $\{(10, 27), (13, 29), (12, 31)\}$           | $\text{Succ}_1^\text{bt}(a, b)$                                                                           | $\text{Succ}_1^\text{bt}(15, 26)$                                                                               | $\{(10, 27), (13, 29), (12, 31)\}$           | $\{(10, 27), (13, 29), (12, 31)\}$           | ✔              | ✔               |
| Eq10 | $\text{Pred}_1^\text{at}(a, y)$ | $\text{Pred}_1^\text{at}(15, 25)$ | $\emptyset$                                  | $\emptyset$                                  | $\text{Pred}_1^\text{bt}(x, y)$                                                                           | $\text{Pred}_1^\text{bt}(16, 25)$                                                                               | $\emptyset$                                  | $\emptyset$                                  | ✔              | ✔               |
| Eq11 | $\text{Pred}_1^\text{at}(x, b)$ | $\text{Pred}_1^\text{at}(16, 26)$ | $\{(17, 23), (18, 24), (15, 25)\}$           | $\{(17, 23), (18, 24), (15, 25)\}$           | $\text{Pred}_1^\text{bt}(a, b) \oplus \{(a, y)\}$                                                         | $\text{Pred}_1^\text{bt}(15, 26) \oplus \{(15, 25)\}$                                                           | $\{(17, 23), (18, 24), (15, 25)\}$           | $\{(17, 23), (18, 24), (15, 25)\}$           | ✔              | ✔               |
| Eq12 | $\text{Succ}_2^\text{at}(a, y)$ | $\text{Succ}_2^\text{at}(15, 25)$ | $\{(10, 27), (16, 26), (11, 32)\}$           | $\{(10, 27), (16, 26), (11, 32)\}$           | $\text{Succ}_2^\text{bt}(x, y) \oplus \{(x, b), (a, b)\}$                                                 | $\text{Succ}_2^\text{bt}(16, 25) \oplus \{(16, 26), (15, 26)\}$                                                 | $\{(10, 27), (16, 26), (11, 32)\}$           | $\{(10, 27), (16, 26), (11, 32)\}$           | ✔              | ✔               |
| Eq13 | $\text{Succ}_2^\text{at}(x, b)$ | $\text{Succ}_2^\text{at}(16, 26)$ | $\{(10, 27), (13, 29), (12, 31)\}$           | $\{(10, 27), (13, 29), (12, 31)\}$           | $\text{Succ}_2^\text{bt}(a, b)$                                                                           | $\text{Succ}_2^\text{bt}(15, 26)$                                                                               | $\{(13, 29), (11, 32), (12, 31)\}$           | $\{(13, 29), (11, 32), (12, 31)\}$           | ✘              | ✘               |
| Eq14 | $\text{Pred}_2^\text{at}(a, y)$ | $\text{Pred}_2^\text{at}(15, 25)$ | $\{(19, 22), (17, 23), (18, 24), (20, 21)\}$ | $\{(19, 22), (17, 23), (18, 24), (20, 21)\}$ | $[\text{Pred}_2^\text{bt}(a, b) \cap \mathcal{L}]$                                                        | $[\text{Pred}_2^\text{bt}(15, 26) \cap \mathcal{L}]$                                                            | $\{(18, 24), (17, 23), (19, 22), (20, 21)\}$ | $\{(18, 24), (17, 23), (19, 22), (20, 21)\}$ | ✔              | ✔               |
| Eq15 | $\text{Pred}_2^\text{at}(x, b)$ | $\text{Pred}_2^\text{at}(16, 26)$ | $\{(17, 23), (19, 22), (15, 25)\}$           | $\{(17, 23), (19, 22), (15, 25)\}$           | $\text{Pred}_2^\text{bt}(x, y) \oplus \{(a, y)\} \oplus [\text{Pred}_2^\text{bt}(a, b) \cap \mathcal{M}]$ | $\text{Pred}_2^\text{bt}(16, 25) \oplus \{(15, 25)\} \oplus [\text{Pred}_2^\text{bt}(15, 26) \cap \mathcal{M}]$ | $\{(17, 23), (19, 22), (15, 25)\}$           | $\{(17, 23), (19, 22), (15, 25)\}$           | ✔              | ✔               |

## Transposition <1, 2> in complex 1
Here is a __birth-birth__ __no switch__ transposition <1, 2>.

$$
a = 1, \; b = 11, \; x = 2, \; y = 10
$$

$$
a < x < y < b
$$

<img src="transpositions_stats-abstract.assets/image-0463cd2a1bc8.png" width="1200">

### Birth Death Pairs
|                          |                                                |
|:-------------------------|:-----------------------------------------------|
| Before the transposition | $\{(1, 11), (3, 7), (2, 10), (4, 6), (5, 9)\}$ |
| After the transposition  | $\{(1, 11), (3, 7), (2, 10), (4, 6), (5, 9)\}$ |

### Relations
|                                | Algorithm 1             | Algorithm 2                  |
|:-------------------------------|:------------------------|:-----------------------------|
| Before the transposition (WC)  | $\{(9, 10), (10, 11)\}$ | $\{(3, 1), (4, 2), (5, 2)\}$ |
| Before the transposition (WOC) | $\{(9, 10), (10, 11)\}$ | $\{(3, 1), (4, 2), (5, 2)\}$ |
| After the transposition (WC)   | $\{(9, 10)\}$           | $\{(3, 1), (4, 2), (5, 2)\}$ |
| After the transposition (WOC)  | $\{(9, 10)\}$           | $\{(3, 1), (4, 2), (5, 2)\}$ |

### Successors and Predecessers
|                       | $\text{Set}^\text{bt}(1, 11)$                  | $\text{Set}^\text{bt}(2, 10)$                         | $\text{Set}^\text{at}(1, 11)$                 | $\text{Set}^\text{at}(2, 10)$                         |
|:----------------------|:-----------------------------------------------|:------------------------------------------------------|:----------------------------------------------|:------------------------------------------------------|
| $\text{Succ}_1$ (WC)  | $\text{Succ}_1^\text{bt}(1, 11) = \emptyset$   | $\text{Succ}_1^\text{bt}(2, 10) = \{(1, 11)\}$        | $\text{Succ}_1^\text{at}(1, 11) = \emptyset$  | $\text{Succ}_1^\text{at}(2, 10) = \emptyset$          |
| $\text{Succ}_1$ (WOC) | $\text{Succ}_1^\text{bt}(1, 11) = \emptyset$   | $\text{Succ}_1^\text{bt}(2, 10) = \{(1, 11)\}$        | $\text{Succ}_1^\text{at}(1, 11) = \emptyset$  | $\text{Succ}_1^\text{at}(2, 10) = \emptyset$          |
| $\text{Pred}_1$ (WC)  | $\text{Pred}_1^\text{bt}(1, 11) = \{(2, 10)\}$ | $\text{Pred}_1^\text{bt}(2, 10) = \{(5, 9)\}$         | $\text{Pred}_1^\text{at}(1, 11) = \emptyset$  | $\text{Pred}_1^\text{at}(2, 10) = \{(5, 9)\}$         |
| $\text{Pred}_1$ (WOC) | $\text{Pred}_1^\text{bt}(1, 11) = \{(2, 10)\}$ | $\text{Pred}_1^\text{bt}(2, 10) = \{(5, 9)\}$         | $\text{Pred}_1^\text{at}(1, 11) = \emptyset$  | $\text{Pred}_1^\text{at}(2, 10) = \{(5, 9)\}$         |
| $\text{Succ}_2$ (WC)  | $\text{Succ}_2^\text{bt}(1, 11) = \emptyset$   | $\text{Succ}_2^\text{bt}(2, 10) = \emptyset$          | $\text{Succ}_2^\text{at}(1, 11) = \emptyset$  | $\text{Succ}_2^\text{at}(2, 10) = \emptyset$          |
| $\text{Succ}_2$ (WOC) | $\text{Succ}_2^\text{bt}(1, 11) = \emptyset$   | $\text{Succ}_2^\text{bt}(2, 10) = \emptyset$          | $\text{Succ}_2^\text{at}(1, 11) = \emptyset$  | $\text{Succ}_2^\text{at}(2, 10) = \emptyset$          |
| $\text{Pred}_2$ (WC)  | $\text{Pred}_2^\text{bt}(1, 11) = \{(3, 7)\}$  | $\text{Pred}_2^\text{bt}(2, 10) = \{(5, 9), (4, 6)\}$ | $\text{Pred}_2^\text{at}(1, 11) = \{(3, 7)\}$ | $\text{Pred}_2^\text{at}(2, 10) = \{(5, 9), (4, 6)\}$ |
| $\text{Pred}_2$ (WOC) | $\text{Pred}_2^\text{bt}(1, 11) = \{(3, 7)\}$  | $\text{Pred}_2^\text{bt}(2, 10) = \{(5, 9), (4, 6)\}$ | $\text{Pred}_2^\text{at}(1, 11) = \{(3, 7)\}$ | $\text{Pred}_2^\text{at}(2, 10) = \{(5, 9), (4, 6)\}$ |

### Temp Sets
|                                                                     | Before the Transposition     | After the Transposition      |
|:--------------------------------------------------------------------|:-----------------------------|:-----------------------------|
| $\mathcal{L} = \{(s, t)\in \text{BD}:\; f(t) < f(y)\}$ (WC)         | $\{(5, 9), (4, 6), (3, 7)\}$ | $\{(5, 9), (4, 6), (3, 7)\}$ |
| $\mathcal{L} = \{(s, t)\in \text{BD}:\; f(t) < f(y)\}$ (WOC)        | $\{(5, 9), (4, 6), (3, 7)\}$ | $\{(5, 9), (4, 6), (3, 7)\}$ |
| $\mathcal{M} = \{(s, t)\in \text{BD}:\; f(y) < f(t) < f(b)\}$ (WC)  | $\emptyset$                  | $\emptyset$                  |
| $\mathcal{M} = \{(s, t)\in \text{BD}:\; f(y) < f(t) < f(b)\}$ (WOC) | $\emptyset$                  | $\emptyset$                  |

### Equations
There are 2 equations are wrong.

| eq    | left (formula)                  | left (paramatrized)              | left (value) (WC)   | left (value) (WOC)   | right (formula)                                                                        | right (paramatrized)                                                                      | right (value) (WC)     | right (value) (WOC)    | correct (WC)   | correct (WOC)   |
|:------|:--------------------------------|:---------------------------------|:--------------------|:---------------------|:---------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|:-----------------------|:-----------------------|:---------------|:----------------|
| Eq16  | $\text{Succ}_1^\text{at}(x, y)$ | $\text{Succ}_1^\text{at}(2, 10)$ | $\emptyset$         | $\emptyset$          | $\text{Succ}_1^\text{bt}(x, y) \oplus \{(a, b)\} \oplus \text{Succ}_1^\text{bt}(a, b)$ | $\text{Succ}_1^\text{bt}(2, 10) \oplus \{(1, 11)\} \oplus \text{Succ}_1^\text{bt}(1, 11)$ | $\emptyset$            | $\emptyset$            | ✔              | ✔               |
| Eq16s | $\text{Succ}_1^\text{at}(a, b)$ | $\text{Succ}_1^\text{at}(1, 11)$ | $\emptyset$         | $\emptyset$          | $\text{Succ}_1^\text{bt}(a, b) \oplus \{(x, y)\} \oplus \text{Succ}_1^\text{bt}(x, y)$ | $\text{Succ}_1^\text{bt}(1, 11) \oplus \{(2, 10)\} \oplus \text{Succ}_1^\text{bt}(2, 10)$ | $\{(1, 11), (2, 10)\}$ | $\{(1, 11), (2, 10)\}$ | ✘              | ✘               |
| Eq17  | $\text{Pred}_1^\text{at}(a, b)$ | $\text{Pred}_1^\text{at}(1, 11)$ | $\emptyset$         | $\emptyset$          | $\text{Pred}_1^\text{bt}(a, b) \oplus \{(x, y)\}$                                      | $\text{Pred}_1^\text{bt}(1, 11) \oplus \{(2, 10)\}$                                       | $\emptyset$            | $\emptyset$            | ✔              | ✔               |
| Eq17s | $\text{Pred}_1^\text{at}(x, y)$ | $\text{Pred}_1^\text{at}(2, 10)$ | $\{(5, 9)\}$        | $\{(5, 9)\}$         | $\text{Pred}_1^\text{bt}(x, y) \oplus \{(a, b)\}$                                      | $\text{Pred}_1^\text{bt}(2, 10) \oplus \{(1, 11)\}$                                       | $\{(1, 11), (5, 9)\}$  | $\{(1, 11), (5, 9)\}$  | ✘              | ✘               |

## Transposition <41, 42> in complex 1
Here is a __death-death__ __switch forward__ transposition <41, 42>.

$$
a = 34, \; b = 42, \; x = 36, \; y = 41
$$

$$
a < x < y < b
$$

<img src="transpositions_stats-abstract.assets/image-1dbca2611686.png" width="1200">

### Birth Death Pairs
|                          |                                                                                      |
|:-------------------------|:-------------------------------------------------------------------------------------|
| Before the transposition | $\{(38, 39), (34, 42), (31, 46), (33, 44), (32, 45), (37, 43), (36, 41), (35, 40)\}$ |
| After the transposition  | $\{(38, 39), (31, 46), (36, 42), (33, 44), (32, 45), (34, 41), (37, 43), (35, 40)\}$ |

### Relations
|                                | Algorithm 1                                                                                                                                                | Algorithm 2                                                                                                                            |
|:-------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition (WC)  | $\{(42, 45), (43, 46), (41, 46), (39, 40), (39, 46), (41, 42), (40, 42), (41, 45), (42, 44), (39, 42), (43, 45), (41, 44), (43, 44), (42, 46), (39, 44)\}$ | $\{(38, 36), (36, 33), (37, 32), (38, 35), (35, 32), (36, 32), (37, 31), (38, 31), (33, 31), (37, 33), (36, 31), (38, 33)\}$           |
| Before the transposition (WOC) | $\{(42, 45), (43, 46), (41, 46), (39, 40), (39, 46), (41, 42), (40, 42), (41, 45), (42, 44), (39, 42), (43, 45), (41, 44), (43, 44), (42, 46), (39, 44)\}$ | $\{(38, 36), (36, 33), (37, 32), (38, 35), (35, 32), (36, 32), (37, 31), (38, 31), (33, 31), (37, 33), (36, 31), (38, 33)\}$           |
| After the transposition (WC)   | $\{(42, 45), (40, 46), (43, 46), (39, 40), (39, 46), (40, 45), (42, 41), (42, 44), (40, 44), (39, 42), (43, 45), (40, 41), (43, 44), (42, 46), (39, 44)\}$ | $\{(36, 34), (38, 36), (36, 33), (37, 32), (38, 35), (35, 32), (36, 32), (37, 31), (38, 31), (33, 31), (37, 33), (36, 31), (38, 33)\}$ |
| After the transposition (WOC)  | $\{(42, 45), (40, 46), (43, 46), (39, 40), (39, 46), (40, 45), (42, 41), (42, 44), (40, 44), (39, 42), (43, 45), (40, 41), (43, 44), (42, 46), (39, 44)\}$ | $\{(36, 34), (38, 36), (36, 33), (37, 32), (38, 35), (35, 32), (36, 32), (37, 31), (38, 31), (33, 31), (37, 33), (36, 31), (38, 33)\}$ |

### Successors and Predecessers
|                       | $\text{Set}^\text{bt}(34, 42)$                                       | $\text{Set}^\text{bt}(36, 41)$                                                 | $\text{Set}^\text{at}(36, 42)$                                                 | $\text{Set}^\text{at}(34, 41)$                             |
|:----------------------|:---------------------------------------------------------------------|:-------------------------------------------------------------------------------|:-------------------------------------------------------------------------------|:-----------------------------------------------------------|
| $\text{Succ}_1$ (WC)  | $\text{Succ}_1^\text{bt}(34, 42) = \{(31, 46), (32, 45), (33, 44)\}$ | $\text{Succ}_1^\text{bt}(36, 41) = \{(31, 46), (32, 45), (34, 42), (33, 44)\}$ | $\text{Succ}_1^\text{at}(36, 42) = \{(31, 46), (32, 45), (34, 41), (33, 44)\}$ | $\text{Succ}_1^\text{at}(34, 41) = \emptyset$              |
| $\text{Succ}_1$ (WOC) | $\text{Succ}_1^\text{bt}(34, 42) = \{(31, 46), (32, 45), (33, 44)\}$ | $\text{Succ}_1^\text{bt}(36, 41) = \{(31, 46), (32, 45), (34, 42), (33, 44)\}$ | $\text{Succ}_1^\text{at}(36, 42) = \{(31, 46), (32, 45), (34, 41), (33, 44)\}$ | $\text{Succ}_1^\text{at}(34, 41) = \emptyset$              |
| $\text{Pred}_1$ (WC)  | $\text{Pred}_1^\text{bt}(34, 42) = \{(36, 41), (38, 39), (35, 40)\}$ | $\text{Pred}_1^\text{bt}(36, 41) = \emptyset$                                  | $\text{Pred}_1^\text{at}(36, 42) = \{(38, 39)\}$                               | $\text{Pred}_1^\text{at}(34, 41) = \{(36, 42), (35, 40)\}$ |
| $\text{Pred}_1$ (WOC) | $\text{Pred}_1^\text{bt}(34, 42) = \{(36, 41), (38, 39), (35, 40)\}$ | $\text{Pred}_1^\text{bt}(36, 41) = \emptyset$                                  | $\text{Pred}_1^\text{at}(36, 42) = \{(38, 39)\}$                               | $\text{Pred}_1^\text{at}(34, 41) = \{(36, 42), (35, 40)\}$ |
| $\text{Succ}_2$ (WC)  | $\text{Succ}_2^\text{bt}(34, 42) = \emptyset$                        | $\text{Succ}_2^\text{bt}(36, 41) = \{(31, 46), (32, 45), (33, 44)\}$           | $\text{Succ}_2^\text{at}(36, 42) = \{(31, 46), (32, 45), (34, 41), (33, 44)\}$ | $\text{Succ}_2^\text{at}(34, 41) = \emptyset$              |
| $\text{Succ}_2$ (WOC) | $\text{Succ}_2^\text{bt}(34, 42) = \emptyset$                        | $\text{Succ}_2^\text{bt}(36, 41) = \{(31, 46), (32, 45), (33, 44)\}$           | $\text{Succ}_2^\text{at}(36, 42) = \{(31, 46), (32, 45), (34, 41), (33, 44)\}$ | $\text{Succ}_2^\text{at}(34, 41) = \emptyset$              |
| $\text{Pred}_2$ (WC)  | $\text{Pred}_2^\text{bt}(34, 42) = \emptyset$                        | $\text{Pred}_2^\text{bt}(36, 41) = \{(38, 39)\}$                               | $\text{Pred}_2^\text{at}(36, 42) = \{(38, 39)\}$                               | $\text{Pred}_2^\text{at}(34, 41) = \{(36, 42)\}$           |
| $\text{Pred}_2$ (WOC) | $\text{Pred}_2^\text{bt}(34, 42) = \emptyset$                        | $\text{Pred}_2^\text{bt}(36, 41) = \{(38, 39)\}$                               | $\text{Pred}_2^\text{at}(36, 42) = \{(38, 39)\}$                               | $\text{Pred}_2^\text{at}(34, 41) = \{(36, 42)\}$           |

### Temp Sets
|                                                                     | Before the Transposition   | After the Transposition   |
|:--------------------------------------------------------------------|:---------------------------|:--------------------------|
| $\mathcal{B} = \{(s, t)\in \text{BD}:\; f(s) > f(x)\}$ (WC)         | $\{(37, 43), (38, 39)\}$   | $\{(37, 43), (38, 39)\}$  |
| $\mathcal{B} = \{(s, t)\in \text{BD}:\; f(s) > f(x)\}$ (WOC)        | $\{(37, 43), (38, 39)\}$   | $\{(37, 43), (38, 39)\}$  |
| $\mathcal{N} = \{(s, t)\in \text{BD}:\; f(x) > f(s) > f(a)\}$ (WC)  | $\{(35, 40)\}$             | $\{(35, 40)\}$            |
| $\mathcal{N} = \{(s, t)\in \text{BD}:\; f(x) > f(s) > f(a)\}$ (WOC) | $\{(35, 40)\}$             | $\{(35, 40)\}$            |

### Equations
There are 1 equations are wrong.

| eq   | left (formula)                  | left (paramatrized)               | left (value) (WC)                            | left (value) (WOC)                           | right (formula)                                                                                          | right (paramatrized)                                                                                           | right (value) (WC)                           | right (value) (WOC)                          | correct (WC)   | correct (WOC)   |
|:-----|:--------------------------------|:----------------------------------|:---------------------------------------------|:---------------------------------------------|:---------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|:---------------------------------------------|:---------------------------------------------|:---------------|:----------------|
| Eq28 | $\text{Succ}_1^\text{at}(x, b)$ | $\text{Succ}_1^\text{at}(36, 42)$ | $\{(31, 46), (32, 45), (34, 41), (33, 44)\}$ | $\{(31, 46), (32, 45), (34, 41), (33, 44)\}$ | $\text{Succ}_1^\text{bt}(x, y) \oplus \{(a, y), (a, b)\}$                                                | $\text{Succ}_1^\text{bt}(36, 41) \oplus \{(34, 41), (34, 42)\}$                                                | $\{(31, 46), (32, 45), (34, 41), (33, 44)\}$ | $\{(31, 46), (32, 45), (34, 41), (33, 44)\}$ | ✔              | ✔               |
| Eq29 | $\text{Succ}_1^\text{at}(a, y)$ | $\text{Succ}_1^\text{at}(34, 41)$ | $\emptyset$                                  | $\emptyset$                                  | $\text{Succ}_1^\text{bt}(a, b)$                                                                          | $\text{Succ}_1^\text{bt}(34, 42)$                                                                              | $\{(31, 46), (32, 45), (33, 44)\}$           | $\{(31, 46), (32, 45), (33, 44)\}$           | ✘              | ✘               |
| Eq30 | $\text{Pred}_1^\text{at}(x, b)$ | $\text{Pred}_1^\text{at}(36, 42)$ | $\{(38, 39)\}$                               | $\{(38, 39)\}$                               | $[\text{Pred}_1^\text{bt}(a, b) \cap \mathcal{B}]$                                                       | $[\text{Pred}_1^\text{bt}(34, 42) \cap \mathcal{B}]$                                                           | $\{(38, 39)\}$                               | $\{(38, 39)\}$                               | ✔              | ✔               |
| Eq31 | $\text{Pred}_1^\text{at}(a, y)$ | $\text{Pred}_1^\text{at}(34, 41)$ | $\{(36, 42), (35, 40)\}$                     | $\{(36, 42), (35, 40)\}$                     | $\text{Pred}_1^\text{bt}(x, y) \oplus \{(x, b)\} \oplus[\text{Pred}_1^\text{bt}(a, b) \cap \mathcal{N}]$ | $\text{Pred}_1^\text{bt}(36, 41) \oplus \{(36, 42)\} \oplus[\text{Pred}_1^\text{bt}(34, 42) \cap \mathcal{N}]$ | $\{(36, 42), (35, 40)\}$                     | $\{(36, 42), (35, 40)\}$                     | ✔              | ✔               |
| Eq32 | $\text{Succ}_2^\text{at}(x, b)$ | $\text{Succ}_2^\text{at}(36, 42)$ | $\{(31, 46), (32, 45), (34, 41), (33, 44)\}$ | $\{(31, 46), (32, 45), (34, 41), (33, 44)\}$ | $\text{Succ}_2^\text{bt}(x, y) \oplus \{(a, y)\} \oplus \text{Succ}_2^\text{bt}(a, b)$                   | $\text{Succ}_2^\text{bt}(36, 41) \oplus \{(34, 41)\} \oplus \text{Succ}_2^\text{bt}(34, 42)$                   | $\{(31, 46), (32, 45), (34, 41), (33, 44)\}$ | $\{(31, 46), (32, 45), (34, 41), (33, 44)\}$ | ✔              | ✔               |
| Eq33 | $\text{Succ}_2^\text{at}(a, y)$ | $\text{Succ}_2^\text{at}(34, 41)$ | $\emptyset$                                  | $\emptyset$                                  | $\text{Succ}_2^\text{bt}(a, b)$                                                                          | $\text{Succ}_2^\text{bt}(34, 42)$                                                                              | $\emptyset$                                  | $\emptyset$                                  | ✔              | ✔               |
| Eq34 | $\text{Pred}_2^\text{at}(x, b)$ | $\text{Pred}_2^\text{at}(36, 42)$ | $\{(38, 39)\}$                               | $\{(38, 39)\}$                               | $\text{Pred}_2^\text{bt}(x, y)$                                                                          | $\text{Pred}_2^\text{bt}(36, 41)$                                                                              | $\{(38, 39)\}$                               | $\{(38, 39)\}$                               | ✔              | ✔               |
| Eq35 | $\text{Pred}_2^\text{at}(a, y)$ | $\text{Pred}_2^\text{at}(34, 41)$ | $\{(36, 42)\}$                               | $\{(36, 42)\}$                               | $\text{Pred}_2^\text{bt}(a, b) \oplus \{(x, b)\}$                                                        | $\text{Pred}_2^\text{bt}(34, 42) \oplus \{(36, 42)\}$                                                          | $\{(36, 42)\}$                               | $\{(36, 42)\}$                               | ✔              | ✔               |

## Transposition <7, 8> in complex 3
Here is a __birth-death__ __switch forward__ transposition <7, 8>.

$$
a = 4, \; b = 7, \; x = 8, \; y = 35
$$

$$
a < b < x < y
$$

<img src="transpositions_stats-abstract.assets/image-5180bd6a8bf3.png" width="1200">

### Birth Death Pairs
|                          |                                                                                                                                                   |
|:-------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition | $\{(17, 27), (10, 24), (12, 26), (2, 11), (1, 14), (8, 35), (3, 9), (13, 28), (18, 25), (5, 6), (20, 22), (19, 21), (4, 7), (15, 31), (16, 23)\}$ |
| After the transposition  | $\{(17, 27), (10, 24), (12, 26), (1, 14), (7, 35), (3, 9), (13, 28), (18, 25), (5, 6), (20, 22), (4, 8), (19, 21), (2, 11), (15, 31), (16, 23)\}$ |

### Relations
|                                | Algorithm 1                                                                                                                                                                                                                                                                                                                                                                                                                                   | Algorithm 2                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|:-------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition (WC)  | $\{(25, 35), (21, 25), (21, 31), (31, 35), (23, 31), (9, 14), (11, 14), (27, 28), (6, 14), (25, 31), (24, 35), (26, 35), (21, 24), (21, 27), (22, 28), (22, 31), (7, 9), (6, 7), (21, 23), (22, 27), (23, 35), (7, 11), (6, 9), (25, 26)\}$                                                                                                                                                                                                   | $\{(20, 8), (4, 3), (3, 1), (20, 17), (17, 15), (19, 15), (13, 8), (16, 10), (16, 13), (18, 13), (20, 16), (17, 8), (19, 8), (19, 17), (10, 8), (16, 12), (18, 12), (3, 2), (20, 12), (20, 18), (12, 8), (20, 15), (5, 2), (19, 16), (18, 8)\}$                                                                                                                                                                                                                      |
| Before the transposition (WOC) | $\{(6, 12), (6, 18), (25, 35), (6, 15), (21, 25), (21, 31), (31, 35), (23, 31), (9, 14), (11, 14), (27, 28), (9, 20), (6, 14), (6, 20), (25, 31), (7, 19), (24, 35), (26, 35), (21, 24), (21, 27), (22, 28), (9, 10), (9, 16), (22, 31), (9, 13), (11, 16), (11, 13), (7, 9), (6, 7), (7, 12), (6, 10), (6, 19), (21, 23), (22, 27), (14, 20), (23, 35), (9, 15), (11, 12), (9, 18), (11, 15), (7, 8), (11, 18), (7, 11), (6, 9), (25, 26)\}$ | $\{(20, 11), (20, 8), (12, 7), (20, 14), (4, 3), (3, 1), (20, 17), (17, 9), (17, 6), (17, 15), (8, 6), (10, 9), (19, 15), (13, 11), (13, 8), (16, 7), (15, 11), (16, 10), (18, 7), (16, 13), (18, 13), (20, 16), (17, 11), (17, 8), (19, 8), (19, 17), (13, 7), (10, 8), (16, 9), (16, 12), (18, 6), (18, 12), (20, 9), (20, 12), (3, 2), (20, 18), (12, 8), (20, 15), (12, 11), (5, 2), (17, 7), (8, 7), (19, 16), (13, 9), (15, 9), (18, 11), (16, 11), (18, 8)\}$ |
| After the transposition (WC)   | $\{(25, 35), (21, 25), (21, 31), (8, 9), (31, 35), (23, 31), (9, 14), (11, 14), (27, 28), (6, 14), (25, 31), (24, 35), (26, 35), (21, 24), (21, 27), (8, 11), (22, 28), (22, 31), (21, 23), (22, 27), (23, 35), (6, 9), (25, 26)\}$                                                                                                                                                                                                           | $\{(12, 7), (4, 3), (3, 1), (20, 17), (17, 15), (19, 15), (16, 7), (16, 10), (18, 7), (16, 13), (18, 13), (20, 16), (19, 17), (13, 7), (16, 12), (18, 12), (3, 2), (20, 12), (20, 18), (20, 15), (5, 2), (17, 7), (19, 16)\}$                                                                                                                                                                                                                                        |
| After the transposition (WOC)  | $\{(6, 12), (6, 18), (25, 35), (6, 15), (21, 25), (21, 31), (8, 9), (31, 35), (23, 31), (8, 12), (9, 14), (11, 14), (27, 28), (9, 20), (6, 14), (6, 20), (25, 31), (24, 35), (26, 35), (21, 24), (21, 27), (22, 28), (8, 11), (9, 10), (9, 16), (22, 31), (9, 13), (11, 16), (11, 13), (6, 7), (6, 10), (6, 19), (21, 23), (22, 27), (8, 7), (14, 20), (23, 35), (9, 15), (11, 12), (8, 19), (9, 18), (11, 15), (11, 18), (6, 9), (25, 26)\}$ | $\{(20, 11), (20, 8), (12, 7), (20, 14), (4, 3), (3, 1), (20, 17), (17, 9), (17, 6), (17, 15), (10, 9), (19, 15), (13, 11), (13, 8), (16, 7), (15, 11), (16, 10), (18, 7), (16, 13), (18, 13), (20, 16), (17, 11), (17, 8), (19, 8), (19, 17), (13, 7), (10, 8), (16, 9), (16, 12), (18, 6), (18, 12), (7, 6), (20, 9), (20, 12), (3, 2), (20, 18), (12, 11), (20, 15), (5, 2), (12, 8), (17, 7), (19, 16), (13, 9), (7, 8), (15, 9), (18, 11), (16, 11), (18, 8)\}$ |

### Successors and Predecessers
|                       | $\text{Set}^\text{bt}(4, 7)$                          | $\text{Set}^\text{bt}(8, 35)$                                                                               | $\text{Set}^\text{at}(4, 8)$                          | $\text{Set}^\text{at}(7, 35)$                                                           |
|:----------------------|:------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:------------------------------------------------------|:----------------------------------------------------------------------------------------|
| $\text{Succ}_1$ (WC)  | $\text{Succ}_1^\text{bt}(4, 7) = \{(3, 9), (2, 11)\}$ | $\text{Succ}_1^\text{bt}(8, 35) = \emptyset$                                                                | $\text{Succ}_1^\text{at}(4, 8) = \{(3, 9), (2, 11)\}$ | $\text{Succ}_1^\text{at}(7, 35) = \emptyset$                                            |
| $\text{Succ}_1$ (WOC) | $\text{Succ}_1^\text{bt}(4, 7) = \{(3, 9), (2, 11)\}$ | $\text{Succ}_1^\text{bt}(8, 35) = \emptyset$                                                                | $\text{Succ}_1^\text{at}(4, 8) = \{(3, 9), (2, 11)\}$ | $\text{Succ}_1^\text{at}(7, 35) = \emptyset$                                            |
| $\text{Pred}_1$ (WC)  | $\text{Pred}_1^\text{bt}(4, 7) = \{(5, 6)\}$          | $\text{Pred}_1^\text{bt}(8, 35) = \{(10, 24), (12, 26), (18, 25), (15, 31), (16, 23)\}$                     | $\text{Pred}_1^\text{at}(4, 8) = \emptyset$           | $\text{Pred}_1^\text{at}(7, 35) = \{(10, 24), (12, 26), (18, 25), (15, 31), (16, 23)\}$ |
| $\text{Pred}_1$ (WOC) | $\text{Pred}_1^\text{bt}(4, 7) = \{(5, 6)\}$          | $\text{Pred}_1^\text{bt}(8, 35) = \{(10, 24), (12, 26), (18, 25), (15, 31), (16, 23)\}$                     | $\text{Pred}_1^\text{at}(4, 8) = \emptyset$           | $\text{Pred}_1^\text{at}(7, 35) = \{(10, 24), (12, 26), (18, 25), (15, 31), (16, 23)\}$ |
| $\text{Succ}_2$ (WC)  | $\text{Succ}_2^\text{bt}(4, 7) = \{(3, 9)\}$          | $\text{Succ}_2^\text{bt}(8, 35) = \emptyset$                                                                | $\text{Succ}_2^\text{at}(4, 8) = \{(3, 9)\}$          | $\text{Succ}_2^\text{at}(7, 35) = \emptyset$                                            |
| $\text{Succ}_2$ (WOC) | $\text{Succ}_2^\text{bt}(4, 7) = \{(3, 9)\}$          | $\text{Succ}_2^\text{bt}(8, 35) = \emptyset$                                                                | $\text{Succ}_2^\text{at}(4, 8) = \{(3, 9)\}$          | $\text{Succ}_2^\text{at}(7, 35) = \emptyset$                                            |
| $\text{Pred}_2$ (WC)  | $\text{Pred}_2^\text{bt}(4, 7) = \emptyset$           | $\text{Pred}_2^\text{bt}(8, 35) = \{(10, 24), (12, 26), (13, 28), (18, 25), (20, 22), (19, 21), (17, 27)\}$ | $\text{Pred}_2^\text{at}(4, 8) = \emptyset$           | $\text{Pred}_2^\text{at}(7, 35) = \{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ |
| $\text{Pred}_2$ (WOC) | $\text{Pred}_2^\text{bt}(4, 7) = \emptyset$           | $\text{Pred}_2^\text{bt}(8, 35) = \{(10, 24), (12, 26), (13, 28), (18, 25), (20, 22), (19, 21), (17, 27)\}$ | $\text{Pred}_2^\text{at}(4, 8) = \emptyset$           | $\text{Pred}_2^\text{at}(7, 35) = \{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ |

### Temp Sets
|                                                                    | Before the Transposition                                                                       | After the Transposition                                                                        |
|:-------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
| $A_1 = \{(s, t)\in \text{BD}:\; f(a) < f(s) < f(t) < f(x)\}$ (WC)  | $\{(5, 6)\}$                                                                                   | $\{(5, 6)\}$                                                                                   |
| $A_1 = \{(s, t)\in \text{BD}:\; f(a) < f(s) < f(t) < f(x)\}$ (WOC) | $\{(5, 6)\}$                                                                                   | $\{(5, 6)\}$                                                                                   |
| $B_1 = \{(s, t)\in \text{BD}:\; U_1[t, x] = 1\}$ (WC)              | $\emptyset$                                                                                    | $\emptyset$                                                                                    |
| $B_1 = \{(s, t)\in \text{BD}:\; U_1[t, x] = 1\}$ (WOC)             | $\{(4, 7)\}$                                                                                   | $\emptyset$                                                                                    |
| $A_2 = \{(s, t)\in \text{BD}:\; f(b) < f(s) < f(t) < f(y)\}$ (WC)  | $\{(10, 24), (12, 26), (16, 23), (13, 28), (18, 25), (20, 22), (19, 21), (15, 31), (17, 27)\}$ | $\{(10, 24), (12, 26), (16, 23), (13, 28), (18, 25), (20, 22), (19, 21), (15, 31), (17, 27)\}$ |
| $A_2 = \{(s, t)\in \text{BD}:\; f(b) < f(s) < f(t) < f(y)\}$ (WOC) | $\{(10, 24), (12, 26), (16, 23), (13, 28), (18, 25), (20, 22), (19, 21), (15, 31), (17, 27)\}$ | $\{(10, 24), (12, 26), (16, 23), (13, 28), (18, 25), (20, 22), (19, 21), (15, 31), (17, 27)\}$ |
| $B_2 = \{(s, t)\in \text{BD}:\; U_2[b, s] = 1\}$ (WC)              | $\emptyset$                                                                                    | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$                                         |
| $B_2 = \{(s, t)\in \text{BD}:\; U_2[b, s] = 1\}$ (WOC)             | $\{(12, 26), (8, 35), (16, 23), (13, 28), (18, 25), (17, 27)\}$                                | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$                                         |

### Equations
There are 3 equations are wrong.

| eq    | left (formula)                  | left (paramatrized)              | left (value) (WC)                                      | left (value) (WOC)                                     | right (formula)                                      | right (paramatrized)                                 | right (value) (WC)                                     | right (value) (WOC)                                    | correct (WC)   | correct (WOC)   |
|:------|:--------------------------------|:---------------------------------|:-------------------------------------------------------|:-------------------------------------------------------|:-----------------------------------------------------|:-----------------------------------------------------|:-------------------------------------------------------|:-------------------------------------------------------|:---------------|:----------------|
| Eq48  | $\text{Succ}_1^\text{at}(a, x)$ | $\text{Succ}_1^\text{at}(4, 8)$  | $\{(3, 9), (2, 11)\}$                                  | $\{(3, 9), (2, 11)\}$                                  | $\text{Succ}_1^\text{bt}(a, b)$                      | $\text{Succ}_1^\text{bt}(4, 7)$                      | $\{(3, 9), (2, 11)\}$                                  | $\{(3, 9), (2, 11)\}$                                  | ✔              | ✔               |
| Eq49  | $\text{Succ}_1^\text{at}(b, y)$ | $\text{Succ}_1^\text{at}(7, 35)$ | $\emptyset$                                            | $\emptyset$                                            | $\text{Succ}_1^\text{bt}(x, y)$                      | $\text{Succ}_1^\text{bt}(8, 35)$                     | $\emptyset$                                            | $\emptyset$                                            | ✔              | ✔               |
| Eq50  | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(4, 8)$  | $\emptyset$                                            | $\emptyset$                                            | $\{(s, t)\in \text{BD}:\; U_1^\text{bt}[t, x] = 1\}$ | $\{(s, t)\in \text{BD}:\; U_1^\text{bt}[t, 8] = 1\}$ | $\{(5, 6)\}$                                           | $\{(5, 6)\}$                                           | ✘              | ✘               |
| Eq50a | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(4, 8)$  | $\emptyset$                                            | $\emptyset$                                            | $A_1^\text{bt} \cap B_1^\text{bt}$                   | $A_1^\text{bt} \cap B_1^\text{bt}$                   | $\emptyset$                                            | $\emptyset$                                            | ✔              | ✔               |
| Eq50b | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(4, 8)$  | $\emptyset$                                            | $\emptyset$                                            | $A_1^\text{bt} \cap B_1^\text{at}$                   | $A_1^\text{bt} \cap B_1^\text{at}$                   | $\emptyset$                                            | $\emptyset$                                            | ✔              | ✔               |
| Eq51  | $\text{Pred}_1^\text{at}(b, y)$ | $\text{Pred}_1^\text{at}(7, 35)$ | $\{(10, 24), (12, 26), (18, 25), (15, 31), (16, 23)\}$ | $\{(10, 24), (12, 26), (18, 25), (15, 31), (16, 23)\}$ | $\text{Pred}_1^\text{bt}(x, y)$                      | $\text{Pred}_1^\text{bt}(8, 35)$                     | $\{(10, 24), (12, 26), (18, 25), (15, 31), (16, 23)\}$ | $\{(10, 24), (12, 26), (18, 25), (15, 31), (16, 23)\}$ | ✔              | ✔               |
| Eq52  | $\text{Succ}_2^\text{at}(a, x)$ | $\text{Succ}_2^\text{at}(4, 8)$  | $\{(3, 9)\}$                                           | $\{(3, 9)\}$                                           | $\text{Succ}_2^\text{bt}(a, b)$                      | $\text{Succ}_2^\text{bt}(4, 7)$                      | $\{(3, 9)\}$                                           | $\{(3, 9)\}$                                           | ✔              | ✔               |
| Eq53  | $\text{Succ}_2^\text{at}(b, y)$ | $\text{Succ}_2^\text{at}(7, 35)$ | $\emptyset$                                            | $\emptyset$                                            | $\text{Succ}_2^\text{bt}(x, y)$                      | $\text{Succ}_2^\text{bt}(8, 35)$                     | $\emptyset$                                            | $\emptyset$                                            | ✔              | ✔               |
| Eq54  | $\text{Pred}_2^\text{at}(a, x)$ | $\text{Pred}_2^\text{at}(4, 8)$  | $\emptyset$                                            | $\emptyset$                                            | $\text{Pred}_2^\text{bt}(a, b)$                      | $\text{Pred}_2^\text{bt}(4, 7)$                      | $\emptyset$                                            | $\emptyset$                                            | ✔              | ✔               |
| Eq55  | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(7, 35)$ | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | $\{(s, t)\in \text{BD}:\; U_2^\text{bt}[b, s] = 1\}$ | $\{(s, t)\in \text{BD}:\; U_2^\text{bt}[7, s] = 1\}$ | $\emptyset$                                            | $\emptyset$                                            | ✘              | ✘               |
| Eq55a | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(7, 35)$ | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | $A_2^\text{bt} \cap B_2^\text{bt}$                   | $A_2^\text{bt} \cap B_2^\text{bt}$                   | $\emptyset$                                            | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | ✘              | ✔               |
| Eq55b | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(7, 35)$ | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | $A_2^\text{bt} \cap B_2^\text{at}$                   | $A_2^\text{bt} \cap B_2^\text{at}$                   | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | ✔              | ✔               |

## Transposition <35, 36> in complex 3
Here is a __birth-death__ __switch forward__ transposition <35, 36>.

$$
a = 8, \; b = 35, \; x = 36, \; y = 42
$$

$$
a < b < x < y
$$

<img src="transpositions_stats-abstract.assets/image-e30d43078d34.png" width="1200">

### Birth Death Pairs
|                          |                                                                                                                                                                                         |
|:-------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition | $\{(10, 24), (38, 39), (32, 46), (12, 26), (19, 21), (36, 42), (37, 41), (30, 45), (8, 35), (16, 23), (13, 28), (18, 25), (29, 47), (20, 22), (34, 40), (33, 43), (15, 31), (17, 27)\}$ |
| After the transposition  | $\{(10, 24), (38, 39), (32, 46), (12, 26), (35, 42), (19, 21), (8, 36), (37, 41), (30, 45), (16, 23), (13, 28), (18, 25), (29, 47), (20, 22), (34, 40), (33, 43), (15, 31), (17, 27)\}$ |

### Relations
|                                | Algorithm 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Algorithm 2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|:-------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition (WC)  | $\{(25, 35), (41, 43), (42, 45), (21, 25), (21, 31), (31, 35), (23, 31), (27, 28), (25, 31), (41, 45), (24, 35), (26, 35), (43, 45), (21, 24), (46, 47), (21, 27), (22, 28), (22, 31), (39, 47), (42, 43), (41, 47), (21, 23), (22, 27), (23, 35), (40, 45), (25, 26)\}$                                                                                                                                                                                                                                                                                                                                                                                                                           | $\{(20, 8), (20, 17), (17, 15), (19, 15), (37, 36), (13, 8), (16, 10), (16, 13), (18, 13), (33, 32), (20, 16), (38, 34), (17, 8), (34, 30), (19, 8), (34, 33), (36, 30), (37, 29), (19, 17), (10, 8), (36, 33), (30, 29), (16, 12), (18, 12), (32, 29), (20, 12), (20, 18), (12, 8), (20, 15), (19, 16), (34, 32), (33, 30), (18, 8)\}$                                                                                                                                                                                                                                                                                                                                                                         |
| Before the transposition (WOC) | $\{(25, 29), (25, 35), (41, 43), (26, 30), (25, 32), (25, 38), (35, 36), (24, 33), (42, 45), (26, 36), (21, 25), (21, 31), (31, 35), (22, 32), (21, 34), (23, 31), (23, 37), (22, 29), (28, 30), (22, 38), (27, 28), (28, 33), (28, 36), (27, 34), (25, 31), (41, 45), (26, 32), (24, 35), (26, 29), (26, 35), (43, 45), (25, 37), (21, 24), (26, 38), (46, 47), (21, 27), (22, 28), (21, 36), (22, 31), (22, 37), (39, 47), (22, 34), (28, 38), (27, 36), (42, 43), (35, 37), (41, 47), (25, 36), (24, 34), (21, 23), (26, 37), (26, 34), (22, 27), (21, 29), (31, 33), (22, 30), (21, 32), (21, 38), (31, 36), (23, 29), (23, 35), (22, 33), (40, 45), (27, 32), (28, 37), (27, 38), (25, 26)\}$ | $\{(20, 8), (29, 23), (29, 26), (20, 17), (34, 25), (34, 22), (17, 15), (34, 31), (36, 28), (37, 27), (19, 15), (36, 31), (37, 36), (13, 8), (33, 23), (32, 27), (16, 10), (30, 27), (16, 13), (18, 13), (33, 32), (38, 28), (20, 16), (38, 25), (29, 22), (18, 8), (38, 34), (34, 21), (17, 8), (37, 23), (34, 30), (19, 8), (34, 33), (36, 30), (37, 26), (37, 29), (19, 17), (30, 23), (10, 8), (36, 33), (32, 23), (33, 22), (30, 29), (16, 12), (33, 31), (38, 21), (32, 29), (18, 12), (38, 24), (20, 12), (20, 18), (38, 27), (20, 15), (12, 8), (34, 23), (34, 26), (19, 16), (36, 26), (37, 25), (37, 31), (34, 32), (30, 25), (36, 35), (32, 22), (33, 21), (30, 28), (33, 24), (33, 30), (33, 27)\}$ |
| After the transposition (WC)   | $\{(42, 45), (41, 43), (26, 36), (21, 25), (21, 31), (23, 31), (27, 28), (28, 36), (25, 31), (41, 45), (43, 45), (21, 24), (46, 47), (21, 27), (22, 28), (21, 36), (22, 31), (39, 47), (27, 36), (42, 43), (41, 47), (25, 36), (21, 23), (22, 27), (40, 45), (31, 36), (25, 26)\}$                                                                                                                                                                                                                                                                                                                                                                                                                 | $\{(35, 30), (35, 33), (20, 8), (20, 17), (17, 15), (19, 15), (13, 8), (16, 10), (16, 13), (18, 13), (33, 32), (20, 16), (38, 34), (17, 8), (34, 30), (19, 8), (34, 33), (37, 29), (19, 17), (10, 8), (30, 29), (16, 12), (18, 12), (32, 29), (20, 12), (20, 18), (12, 8), (20, 15), (19, 16), (34, 32), (33, 30), (18, 8)\}$                                                                                                                                                                                                                                                                                                                                                                                   |
| After the transposition (WOC)  | $\{(25, 29), (25, 35), (41, 43), (26, 30), (25, 32), (25, 38), (24, 33), (42, 45), (26, 36), (21, 25), (21, 31), (31, 35), (22, 32), (21, 34), (23, 31), (23, 37), (22, 29), (28, 30), (36, 37), (22, 38), (27, 28), (28, 33), (28, 36), (27, 34), (25, 31), (41, 45), (26, 32), (24, 35), (26, 29), (43, 45), (26, 35), (25, 37), (21, 24), (26, 38), (46, 47), (21, 27), (22, 28), (21, 36), (22, 31), (22, 37), (39, 47), (22, 34), (28, 38), (27, 36), (42, 43), (41, 47), (25, 36), (24, 34), (21, 23), (26, 37), (26, 34), (22, 27), (21, 29), (31, 33), (22, 30), (21, 32), (21, 38), (23, 35), (23, 29), (22, 33), (40, 45), (31, 36), (36, 35), (27, 32), (28, 37), (27, 38), (25, 26)\}$ | $\{(35, 30), (35, 33), (29, 23), (20, 8), (35, 36), (29, 26), (20, 17), (34, 25), (34, 22), (17, 15), (34, 31), (37, 27), (19, 15), (37, 36), (13, 8), (33, 23), (32, 27), (16, 10), (30, 27), (16, 13), (35, 26), (18, 13), (33, 32), (38, 28), (20, 16), (38, 25), (29, 22), (18, 8), (38, 34), (34, 21), (17, 8), (37, 23), (34, 30), (19, 8), (34, 33), (37, 26), (37, 29), (19, 17), (30, 23), (10, 8), (32, 23), (33, 22), (30, 29), (16, 12), (33, 31), (38, 21), (32, 29), (18, 12), (35, 28), (38, 24), (20, 12), (20, 18), (35, 31), (38, 27), (20, 15), (12, 8), (34, 23), (34, 26), (19, 16), (37, 25), (37, 31), (34, 32), (30, 25), (32, 22), (33, 21), (30, 28), (33, 24), (33, 30), (33, 27)\}$ |

### Successors and Predecessers
|                       | $\text{Set}^\text{bt}(8, 35)$                                                                               | $\text{Set}^\text{bt}(36, 42)$                             | $\text{Set}^\text{at}(8, 36)$                                                                               | $\text{Set}^\text{at}(35, 42)$                             |
|:----------------------|:------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------|
| $\text{Succ}_1$ (WC)  | $\text{Succ}_1^\text{bt}(8, 35) = \emptyset$                                                                | $\text{Succ}_1^\text{bt}(36, 42) = \{(33, 43), (30, 45)\}$ | $\text{Succ}_1^\text{at}(8, 36) = \emptyset$                                                                | $\text{Succ}_1^\text{at}(35, 42) = \{(33, 43), (30, 45)\}$ |
| $\text{Succ}_1$ (WOC) | $\text{Succ}_1^\text{bt}(8, 35) = \emptyset$                                                                | $\text{Succ}_1^\text{bt}(36, 42) = \{(33, 43), (30, 45)\}$ | $\text{Succ}_1^\text{at}(8, 36) = \emptyset$                                                                | $\text{Succ}_1^\text{at}(35, 42) = \{(33, 43), (30, 45)\}$ |
| $\text{Pred}_1$ (WC)  | $\text{Pred}_1^\text{bt}(8, 35) = \{(10, 24), (12, 26), (18, 25), (15, 31), (16, 23)\}$                     | $\text{Pred}_1^\text{bt}(36, 42) = \emptyset$              | $\text{Pred}_1^\text{at}(8, 36) = \{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $\text{Pred}_1^\text{at}(35, 42) = \emptyset$              |
| $\text{Pred}_1$ (WOC) | $\text{Pred}_1^\text{bt}(8, 35) = \{(10, 24), (12, 26), (18, 25), (15, 31), (16, 23)\}$                     | $\text{Pred}_1^\text{bt}(36, 42) = \emptyset$              | $\text{Pred}_1^\text{at}(8, 36) = \{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $\text{Pred}_1^\text{at}(35, 42) = \emptyset$              |
| $\text{Succ}_2$ (WC)  | $\text{Succ}_2^\text{bt}(8, 35) = \emptyset$                                                                | $\text{Succ}_2^\text{bt}(36, 42) = \{(33, 43), (30, 45)\}$ | $\text{Succ}_2^\text{at}(8, 36) = \emptyset$                                                                | $\text{Succ}_2^\text{at}(35, 42) = \{(33, 43), (30, 45)\}$ |
| $\text{Succ}_2$ (WOC) | $\text{Succ}_2^\text{bt}(8, 35) = \emptyset$                                                                | $\text{Succ}_2^\text{bt}(36, 42) = \{(33, 43), (30, 45)\}$ | $\text{Succ}_2^\text{at}(8, 36) = \emptyset$                                                                | $\text{Succ}_2^\text{at}(35, 42) = \{(33, 43), (30, 45)\}$ |
| $\text{Pred}_2$ (WC)  | $\text{Pred}_2^\text{bt}(8, 35) = \{(10, 24), (12, 26), (13, 28), (18, 25), (20, 22), (19, 21), (17, 27)\}$ | $\text{Pred}_2^\text{bt}(36, 42) = \{(37, 41)\}$           | $\text{Pred}_2^\text{at}(8, 36) = \{(10, 24), (12, 26), (13, 28), (18, 25), (20, 22), (19, 21), (17, 27)\}$ | $\text{Pred}_2^\text{at}(35, 42) = \emptyset$              |
| $\text{Pred}_2$ (WOC) | $\text{Pred}_2^\text{bt}(8, 35) = \{(10, 24), (12, 26), (13, 28), (18, 25), (20, 22), (19, 21), (17, 27)\}$ | $\text{Pred}_2^\text{bt}(36, 42) = \{(37, 41)\}$           | $\text{Pred}_2^\text{at}(8, 36) = \{(10, 24), (12, 26), (13, 28), (18, 25), (20, 22), (19, 21), (17, 27)\}$ | $\text{Pred}_2^\text{at}(35, 42) = \emptyset$              |

### Temp Sets
|                                                                    | Before the Transposition                                                                       | After the Transposition                                                                        |
|:-------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
| $A_1 = \{(s, t)\in \text{BD}:\; f(a) < f(s) < f(t) < f(x)\}$ (WC)  | $\{(10, 24), (12, 26), (16, 23), (13, 28), (18, 25), (20, 22), (19, 21), (15, 31), (17, 27)\}$ | $\{(10, 24), (12, 26), (16, 23), (13, 28), (18, 25), (20, 22), (19, 21), (15, 31), (17, 27)\}$ |
| $A_1 = \{(s, t)\in \text{BD}:\; f(a) < f(s) < f(t) < f(x)\}$ (WOC) | $\{(10, 24), (12, 26), (16, 23), (13, 28), (18, 25), (20, 22), (19, 21), (15, 31), (17, 27)\}$ | $\{(10, 24), (12, 26), (16, 23), (13, 28), (18, 25), (20, 22), (19, 21), (15, 31), (17, 27)\}$ |
| $B_1 = \{(s, t)\in \text{BD}:\; U_1[t, x] = 1\}$ (WC)              | $\emptyset$                                                                                    | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$                               |
| $B_1 = \{(s, t)\in \text{BD}:\; U_1[t, x] = 1\}$ (WOC)             | $\{(12, 26), (8, 35), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$                      | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$                               |
| $A_2 = \{(s, t)\in \text{BD}:\; f(b) < f(s) < f(t) < f(y)\}$ (WC)  | $\{(38, 39), (37, 41)\}$                                                                       | $\{(38, 39), (37, 41)\}$                                                                       |
| $A_2 = \{(s, t)\in \text{BD}:\; f(b) < f(s) < f(t) < f(y)\}$ (WOC) | $\{(38, 39), (37, 41)\}$                                                                       | $\{(38, 39), (37, 41)\}$                                                                       |
| $B_2 = \{(s, t)\in \text{BD}:\; U_2[b, s] = 1\}$ (WC)              | $\emptyset$                                                                                    | $\emptyset$                                                                                    |
| $B_2 = \{(s, t)\in \text{BD}:\; U_2[b, s] = 1\}$ (WOC)             | $\{(36, 42)\}$                                                                                 | $\emptyset$                                                                                    |

### Equations
There are 3 equations are wrong.

| eq    | left (formula)                  | left (paramatrized)               | left (value) (WC)                                                          | left (value) (WOC)                                                         | right (formula)                                      | right (paramatrized)                                  | right (value) (WC)                                                         | right (value) (WOC)                                                        | correct (WC)   | correct (WOC)   |
|:------|:--------------------------------|:----------------------------------|:---------------------------------------------------------------------------|:---------------------------------------------------------------------------|:-----------------------------------------------------|:------------------------------------------------------|:---------------------------------------------------------------------------|:---------------------------------------------------------------------------|:---------------|:----------------|
| Eq48  | $\text{Succ}_1^\text{at}(a, x)$ | $\text{Succ}_1^\text{at}(8, 36)$  | $\emptyset$                                                                | $\emptyset$                                                                | $\text{Succ}_1^\text{bt}(a, b)$                      | $\text{Succ}_1^\text{bt}(8, 35)$                      | $\emptyset$                                                                | $\emptyset$                                                                | ✔              | ✔               |
| Eq49  | $\text{Succ}_1^\text{at}(b, y)$ | $\text{Succ}_1^\text{at}(35, 42)$ | $\{(33, 43), (30, 45)\}$                                                   | $\{(33, 43), (30, 45)\}$                                                   | $\text{Succ}_1^\text{bt}(x, y)$                      | $\text{Succ}_1^\text{bt}(36, 42)$                     | $\{(33, 43), (30, 45)\}$                                                   | $\{(33, 43), (30, 45)\}$                                                   | ✔              | ✔               |
| Eq50  | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(8, 36)$  | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $\{(s, t)\in \text{BD}:\; U_1^\text{bt}[t, x] = 1\}$ | $\{(s, t)\in \text{BD}:\; U_1^\text{bt}[t, 36] = 1\}$ | $\emptyset$                                                                | $\emptyset$                                                                | ✘              | ✘               |
| Eq50a | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(8, 36)$  | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $A_1^\text{bt} \cap B_1^\text{bt}$                   | $A_1^\text{bt} \cap B_1^\text{bt}$                    | $\emptyset$                                                                | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | ✘              | ✔               |
| Eq50b | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(8, 36)$  | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $A_1^\text{bt} \cap B_1^\text{at}$                   | $A_1^\text{bt} \cap B_1^\text{at}$                    | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | ✔              | ✔               |
| Eq51  | $\text{Pred}_1^\text{at}(b, y)$ | $\text{Pred}_1^\text{at}(35, 42)$ | $\emptyset$                                                                | $\emptyset$                                                                | $\text{Pred}_1^\text{bt}(x, y)$                      | $\text{Pred}_1^\text{bt}(36, 42)$                     | $\emptyset$                                                                | $\emptyset$                                                                | ✔              | ✔               |
| Eq52  | $\text{Succ}_2^\text{at}(a, x)$ | $\text{Succ}_2^\text{at}(8, 36)$  | $\emptyset$                                                                | $\emptyset$                                                                | $\text{Succ}_2^\text{bt}(a, b)$                      | $\text{Succ}_2^\text{bt}(8, 35)$                      | $\emptyset$                                                                | $\emptyset$                                                                | ✔              | ✔               |
| Eq53  | $\text{Succ}_2^\text{at}(b, y)$ | $\text{Succ}_2^\text{at}(35, 42)$ | $\{(33, 43), (30, 45)\}$                                                   | $\{(33, 43), (30, 45)\}$                                                   | $\text{Succ}_2^\text{bt}(x, y)$                      | $\text{Succ}_2^\text{bt}(36, 42)$                     | $\{(33, 43), (30, 45)\}$                                                   | $\{(33, 43), (30, 45)\}$                                                   | ✔              | ✔               |
| Eq54  | $\text{Pred}_2^\text{at}(a, x)$ | $\text{Pred}_2^\text{at}(8, 36)$  | $\{(10, 24), (12, 26), (13, 28), (18, 25), (20, 22), (19, 21), (17, 27)\}$ | $\{(10, 24), (12, 26), (13, 28), (18, 25), (20, 22), (19, 21), (17, 27)\}$ | $\text{Pred}_2^\text{bt}(a, b)$                      | $\text{Pred}_2^\text{bt}(8, 35)$                      | $\{(10, 24), (12, 26), (13, 28), (18, 25), (20, 22), (19, 21), (17, 27)\}$ | $\{(10, 24), (12, 26), (13, 28), (18, 25), (20, 22), (19, 21), (17, 27)\}$ | ✔              | ✔               |
| Eq55  | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(35, 42)$ | $\emptyset$                                                                | $\emptyset$                                                                | $\{(s, t)\in \text{BD}:\; U_2^\text{bt}[b, s] = 1\}$ | $\{(s, t)\in \text{BD}:\; U_2^\text{bt}[35, s] = 1\}$ | $\{(34, 40), (30, 45)\}$                                                   | $\{(34, 40), (30, 45)\}$                                                   | ✘              | ✘               |
| Eq55a | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(35, 42)$ | $\emptyset$                                                                | $\emptyset$                                                                | $A_2^\text{bt} \cap B_2^\text{bt}$                   | $A_2^\text{bt} \cap B_2^\text{bt}$                    | $\emptyset$                                                                | $\emptyset$                                                                | ✔              | ✔               |
| Eq55b | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(35, 42)$ | $\emptyset$                                                                | $\emptyset$                                                                | $A_2^\text{bt} \cap B_2^\text{at}$                   | $A_2^\text{bt} \cap B_2^\text{at}$                    | $\emptyset$                                                                | $\emptyset$                                                                | ✔              | ✔               |

## Transposition <15, 16> in complex 49
Here is a __birth-death__ __switch forward__ transposition <15, 16>.

$$
a = 1, \; b = 15, \; x = 16, \; y = 24
$$

$$
a < b < x < y
$$

<img src="transpositions_stats-abstract.assets/image-613d810dd73c.png" width="1200">

### Birth Death Pairs
|                          |                                                                                                                   |
|:-------------------------|:------------------------------------------------------------------------------------------------------------------|
| Before the transposition | $\{(14, 27), (1, 15), (19, 23), (12, 26), (11, 28), (10, 32), (13, 22), (18, 25), (16, 24), (17, 30), (20, 21)\}$ |
| After the transposition  | $\{(15, 24), (14, 27), (19, 23), (12, 26), (11, 28), (10, 32), (13, 22), (18, 25), (1, 16), (17, 30), (20, 21)\}$ |

### Relations
|                                | Algorithm 1                                                                                                                                                                    | Algorithm 2                                                                                                                                                                                                  |
|:-------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition (WC)  | $\{(21, 26), (23, 26), (25, 28), (30, 32), (21, 28), (23, 32), (26, 32), (23, 28), (22, 26), (27, 32), (24, 28), (23, 30), (21, 23)\}$                                         | $\{(20, 11), (16, 11), (16, 10), (11, 10), (20, 14), (17, 10), (14, 11), (20, 17), (19, 10), (19, 16), (20, 13), (18, 12), (20, 12), (16, 14), (20, 18), (13, 11), (18, 14)\}$                               |
| Before the transposition (WOC) | $\{(21, 26), (23, 26), (25, 28), (30, 32), (21, 28), (15, 20), (23, 32), (26, 32), (15, 16), (23, 28), (22, 26), (27, 32), (24, 28), (15, 19), (15, 18), (23, 30), (21, 23)\}$ | $\{(20, 11), (20, 14), (20, 17), (13, 11), (16, 10), (20, 13), (11, 10), (16, 15), (18, 12), (18, 15), (20, 12), (20, 18), (14, 11), (17, 10), (19, 10), (19, 16), (16, 14), (16, 11), (18, 14)\}$           |
| After the transposition (WC)   | $\{(21, 26), (23, 26), (25, 28), (30, 32), (21, 28), (23, 32), (26, 32), (23, 28), (22, 26), (27, 32), (24, 28), (23, 30), (21, 23)\}$                                         | $\{(20, 11), (15, 11), (11, 10), (15, 14), (17, 10), (20, 14), (20, 17), (18, 10), (19, 10), (20, 13), (18, 11), (15, 10), (18, 12), (20, 12), (20, 18), (13, 11), (14, 11)\}$                               |
| After the transposition (WOC)  | $\{(21, 26), (16, 20), (23, 26), (25, 28), (16, 19), (30, 32), (21, 28), (23, 32), (26, 32), (23, 28), (22, 26), (16, 15), (24, 28), (27, 32), (16, 18), (23, 30), (21, 23)\}$ | $\{(20, 11), (20, 14), (20, 17), (13, 11), (15, 11), (15, 14), (18, 10), (18, 16), (20, 13), (11, 10), (15, 10), (15, 16), (18, 12), (20, 12), (20, 18), (14, 11), (17, 10), (19, 10), (19, 16), (18, 11)\}$ |

### Successors and Predecessers
|                       | $\text{Set}^\text{bt}(1, 15)$                | $\text{Set}^\text{bt}(16, 24)$                                       | $\text{Set}^\text{at}(1, 16)$                | $\text{Set}^\text{at}(15, 24)$                                       |
|:----------------------|:---------------------------------------------|:---------------------------------------------------------------------|:---------------------------------------------|:---------------------------------------------------------------------|
| $\text{Succ}_1$ (WC)  | $\text{Succ}_1^\text{bt}(1, 15) = \emptyset$ | $\text{Succ}_1^\text{bt}(16, 24) = \{(11, 28)\}$                     | $\text{Succ}_1^\text{at}(1, 16) = \emptyset$ | $\text{Succ}_1^\text{at}(15, 24) = \{(11, 28)\}$                     |
| $\text{Succ}_1$ (WOC) | $\text{Succ}_1^\text{bt}(1, 15) = \emptyset$ | $\text{Succ}_1^\text{bt}(16, 24) = \{(11, 28)\}$                     | $\text{Succ}_1^\text{at}(1, 16) = \emptyset$ | $\text{Succ}_1^\text{at}(15, 24) = \{(11, 28)\}$                     |
| $\text{Pred}_1$ (WC)  | $\text{Pred}_1^\text{bt}(1, 15) = \emptyset$ | $\text{Pred}_1^\text{bt}(16, 24) = \emptyset$                        | $\text{Pred}_1^\text{at}(1, 16) = \emptyset$ | $\text{Pred}_1^\text{at}(15, 24) = \emptyset$                        |
| $\text{Pred}_1$ (WOC) | $\text{Pred}_1^\text{bt}(1, 15) = \emptyset$ | $\text{Pred}_1^\text{bt}(16, 24) = \emptyset$                        | $\text{Pred}_1^\text{at}(1, 16) = \emptyset$ | $\text{Pred}_1^\text{at}(15, 24) = \emptyset$                        |
| $\text{Succ}_2$ (WC)  | $\text{Succ}_2^\text{bt}(1, 15) = \emptyset$ | $\text{Succ}_2^\text{bt}(16, 24) = \{(10, 32), (14, 27), (11, 28)\}$ | $\text{Succ}_2^\text{at}(1, 16) = \emptyset$ | $\text{Succ}_2^\text{at}(15, 24) = \{(10, 32), (14, 27), (11, 28)\}$ |
| $\text{Succ}_2$ (WOC) | $\text{Succ}_2^\text{bt}(1, 15) = \emptyset$ | $\text{Succ}_2^\text{bt}(16, 24) = \{(10, 32), (14, 27), (11, 28)\}$ | $\text{Succ}_2^\text{at}(1, 16) = \emptyset$ | $\text{Succ}_2^\text{at}(15, 24) = \{(10, 32), (14, 27), (11, 28)\}$ |
| $\text{Pred}_2$ (WC)  | $\text{Pred}_2^\text{bt}(1, 15) = \emptyset$ | $\text{Pred}_2^\text{bt}(16, 24) = \{(19, 23)\}$                     | $\text{Pred}_2^\text{at}(1, 16) = \emptyset$ | $\text{Pred}_2^\text{at}(15, 24) = \emptyset$                        |
| $\text{Pred}_2$ (WOC) | $\text{Pred}_2^\text{bt}(1, 15) = \emptyset$ | $\text{Pred}_2^\text{bt}(16, 24) = \{(19, 23)\}$                     | $\text{Pred}_2^\text{at}(1, 16) = \emptyset$ | $\text{Pred}_2^\text{at}(15, 24) = \emptyset$                        |

### Temp Sets
|                                                                    | Before the Transposition   | After the Transposition   |
|:-------------------------------------------------------------------|:---------------------------|:--------------------------|
| $A_1 = \{(s, t)\in \text{BD}:\; f(a) < f(s) < f(t) < f(x)\}$ (WC)  | $\emptyset$                | $\emptyset$               |
| $A_1 = \{(s, t)\in \text{BD}:\; f(a) < f(s) < f(t) < f(x)\}$ (WOC) | $\emptyset$                | $\emptyset$               |
| $B_1 = \{(s, t)\in \text{BD}:\; U_1[t, x] = 1\}$ (WC)              | $\emptyset$                | $\emptyset$               |
| $B_1 = \{(s, t)\in \text{BD}:\; U_1[t, x] = 1\}$ (WOC)             | $\{(1, 15)\}$              | $\emptyset$               |
| $A_2 = \{(s, t)\in \text{BD}:\; f(b) < f(s) < f(t) < f(y)\}$ (WC)  | $\{(20, 21), (19, 23)\}$   | $\{(20, 21), (19, 23)\}$  |
| $A_2 = \{(s, t)\in \text{BD}:\; f(b) < f(s) < f(t) < f(y)\}$ (WOC) | $\{(20, 21), (19, 23)\}$   | $\{(20, 21), (19, 23)\}$  |
| $B_2 = \{(s, t)\in \text{BD}:\; U_2[b, s] = 1\}$ (WC)              | $\emptyset$                | $\emptyset$               |
| $B_2 = \{(s, t)\in \text{BD}:\; U_2[b, s] = 1\}$ (WOC)             | $\{(16, 24), (18, 25)\}$   | $\emptyset$               |

### Equations
There are 0 equations are wrong.

| eq    | left (formula)                  | left (paramatrized)               | left (value) (WC)                  | left (value) (WOC)                 | right (formula)                                      | right (paramatrized)                                  | right (value) (WC)                 | right (value) (WOC)                | correct (WC)   | correct (WOC)   |
|:------|:--------------------------------|:----------------------------------|:-----------------------------------|:-----------------------------------|:-----------------------------------------------------|:------------------------------------------------------|:-----------------------------------|:-----------------------------------|:---------------|:----------------|
| Eq48  | $\text{Succ}_1^\text{at}(a, x)$ | $\text{Succ}_1^\text{at}(1, 16)$  | $\emptyset$                        | $\emptyset$                        | $\text{Succ}_1^\text{bt}(a, b)$                      | $\text{Succ}_1^\text{bt}(1, 15)$                      | $\emptyset$                        | $\emptyset$                        | ✔              | ✔               |
| Eq49  | $\text{Succ}_1^\text{at}(b, y)$ | $\text{Succ}_1^\text{at}(15, 24)$ | $\{(11, 28)\}$                     | $\{(11, 28)\}$                     | $\text{Succ}_1^\text{bt}(x, y)$                      | $\text{Succ}_1^\text{bt}(16, 24)$                     | $\{(11, 28)\}$                     | $\{(11, 28)\}$                     | ✔              | ✔               |
| Eq50  | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(1, 16)$  | $\emptyset$                        | $\emptyset$                        | $\{(s, t)\in \text{BD}:\; U_1^\text{bt}[t, x] = 1\}$ | $\{(s, t)\in \text{BD}:\; U_1^\text{bt}[t, 16] = 1\}$ | $\emptyset$                        | $\emptyset$                        | ✔              | ✔               |
| Eq50a | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(1, 16)$  | $\emptyset$                        | $\emptyset$                        | $A_1^\text{bt} \cap B_1^\text{bt}$                   | $A_1^\text{bt} \cap B_1^\text{bt}$                    | $\emptyset$                        | $\emptyset$                        | ✔              | ✔               |
| Eq50b | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(1, 16)$  | $\emptyset$                        | $\emptyset$                        | $A_1^\text{bt} \cap B_1^\text{at}$                   | $A_1^\text{bt} \cap B_1^\text{at}$                    | $\emptyset$                        | $\emptyset$                        | ✔              | ✔               |
| Eq51  | $\text{Pred}_1^\text{at}(b, y)$ | $\text{Pred}_1^\text{at}(15, 24)$ | $\emptyset$                        | $\emptyset$                        | $\text{Pred}_1^\text{bt}(x, y)$                      | $\text{Pred}_1^\text{bt}(16, 24)$                     | $\emptyset$                        | $\emptyset$                        | ✔              | ✔               |
| Eq52  | $\text{Succ}_2^\text{at}(a, x)$ | $\text{Succ}_2^\text{at}(1, 16)$  | $\emptyset$                        | $\emptyset$                        | $\text{Succ}_2^\text{bt}(a, b)$                      | $\text{Succ}_2^\text{bt}(1, 15)$                      | $\emptyset$                        | $\emptyset$                        | ✔              | ✔               |
| Eq53  | $\text{Succ}_2^\text{at}(b, y)$ | $\text{Succ}_2^\text{at}(15, 24)$ | $\{(10, 32), (14, 27), (11, 28)\}$ | $\{(10, 32), (14, 27), (11, 28)\}$ | $\text{Succ}_2^\text{bt}(x, y)$                      | $\text{Succ}_2^\text{bt}(16, 24)$                     | $\{(10, 32), (14, 27), (11, 28)\}$ | $\{(10, 32), (14, 27), (11, 28)\}$ | ✔              | ✔               |
| Eq54  | $\text{Pred}_2^\text{at}(a, x)$ | $\text{Pred}_2^\text{at}(1, 16)$  | $\emptyset$                        | $\emptyset$                        | $\text{Pred}_2^\text{bt}(a, b)$                      | $\text{Pred}_2^\text{bt}(1, 15)$                      | $\emptyset$                        | $\emptyset$                        | ✔              | ✔               |
| Eq55  | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(15, 24)$ | $\emptyset$                        | $\emptyset$                        | $\{(s, t)\in \text{BD}:\; U_2^\text{bt}[b, s] = 1\}$ | $\{(s, t)\in \text{BD}:\; U_2^\text{bt}[15, s] = 1\}$ | $\emptyset$                        | $\emptyset$                        | ✔              | ✔               |
| Eq55a | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(15, 24)$ | $\emptyset$                        | $\emptyset$                        | $A_2^\text{bt} \cap B_2^\text{bt}$                   | $A_2^\text{bt} \cap B_2^\text{bt}$                    | $\emptyset$                        | $\emptyset$                        | ✔              | ✔               |
| Eq55b | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(15, 24)$ | $\emptyset$                        | $\emptyset$                        | $A_2^\text{bt} \cap B_2^\text{at}$                   | $A_2^\text{bt} \cap B_2^\text{at}$                    | $\emptyset$                        | $\emptyset$                        | ✔              | ✔               |