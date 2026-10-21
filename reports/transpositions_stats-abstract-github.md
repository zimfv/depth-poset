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
We have 25 complexes.

|    |   Dimension |   Cells | $n$-vector       | Betti-vector   |   Nodes in Depth Poset |   Complexes |
|---:|------------:|--------:|:-----------------|:---------------|-----------------------:|------------:|
|  0 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 0, 1)$ |                     23 |           5 |
|  1 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 0, 1, 2)$ |                     22 |           5 |
|  2 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 1, 1, 1)$ |                     22 |           5 |
|  3 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 1, 2, 2)$ |                     21 |           5 |
|  4 |           3 |      48 | $(6, 15, 18, 9)$ | $(1, 2, 2, 1)$ |                     21 |           5 |

# Transpositions
In this data we got 917 by transposing consecutive pairs. The distribution of the transposition types is given in the table:

| transposition_type   |   no switch (nested) |   no switch (not nested) |   switch backward |   switch forward |
|:---------------------|---------------------:|-------------------------:|------------------:|-----------------:|
| birth-birth          |                  125 |                      149 |                71 |               66 |
| birth-death          |                    0 |                       61 |                 0 |               37 |
| death-death          |                  131 |                      139 |                80 |               58 |


# Equations
We checked 48 equations. And we can see the distribution of transposiions, satysfying these equations in the tables:

## Birth-Birth Transpositions
| Name      | Transposition Type   | Switch Type            | Order           | Formula                                                                                                                                   | Correct (WC)   | Correct (WOC)   | Correct, % (WC)   | Correct, % (WOC)   |
|:----------|:---------------------|:-----------------------|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------|:---------------|:----------------|:------------------|:-------------------|
| Eq08      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Succ}_1^\text{at}(a, y) = \text{Succ}_1^\text{bt}(x, y) \oplus \{(x, b)\} \oplus \text{Succ}_1^\text{bt}(a, b)$                    | 66/66          | 66/66           | 100.00%           | 100.00%            |
| Eq09      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Succ}_1^\text{at}(x, b) = \text{Succ}_1^\text{bt}(a, b)$                                                                           | 66/66          | 66/66           | 100.00%           | 100.00%            |
| Eq10      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Pred}_1^\text{at}(a, y) = \text{Pred}_1^\text{bt}(x, y)$                                                                           | 66/66          | 66/66           | 100.00%           | 100.00%            |
| Eq11      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Pred}_1^\text{at}(x, b) = \text{Pred}_1^\text{bt}(a, b) \oplus \{(a, y)\}$                                                         | 66/66          | 66/66           | 100.00%           | 100.00%            |
| Eq12      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Succ}_2^\text{at}(a, y) = \text{Succ}_2^\text{bt}(x, y) \oplus \{(x, b), (a, b)\}$                                                 | 66/66          | 66/66           | 100.00%           | 100.00%            |
| Eq13      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Succ}_2^\text{at}(x, b) = \text{Succ}_2^\text{bt}(a, b)$                                                                           | 30/66          | 30/66           | 45.45%            | 45.45%             |
| Eq14      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Pred}_2^\text{at}(a, y) = [\text{Pred}_2^\text{bt}(a, b) \cap \mathcal{L}]$                                                        | 66/66          | 66/66           | 100.00%           | 100.00%            |
| Eq15      | birth-birth          | switch forward         | $a < x < y < b$ | $\text{Pred}_2^\text{at}(x, b) = \text{Pred}_2^\text{bt}(x, y) \oplus \{(a, y)\} \oplus [\text{Pred}_2^\text{bt}(a, b) \cap \mathcal{M}]$ | 66/66          | 66/66           | 100.00%           | 100.00%            |
| Eq16      | birth-birth          | no switch (nested)     | $a < x < b < y$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y) \oplus \{(a, b)\} \oplus \text{Succ}_1^\text{bt}(a, b)$                    | 0/65           | 0/65            | 0.00%             | 0.00%              |
| Eq16      | birth-birth          | no switch (nested)     | $a < x < y < b$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y) \oplus \{(a, b)\} \oplus \text{Succ}_1^\text{bt}(a, b)$                    | 60/60          | 60/60           | 100.00%           | 100.00%            |
| Eq16s     | birth-birth          | no switch (nested)     | $a < x < b < y$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b) \oplus \{(x, y)\} \oplus \text{Succ}_1^\text{bt}(x, y)$                    | 65/65          | 65/65           | 100.00%           | 100.00%            |
| Eq16s     | birth-birth          | no switch (nested)     | $a < x < y < b$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b) \oplus \{(x, y)\} \oplus \text{Succ}_1^\text{bt}(x, y)$                    | 0/60           | 0/60            | 0.00%             | 0.00%              |
| Eq17      | birth-birth          | no switch (nested)     | $a < x < b < y$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b) \oplus \{(x, y)\}$                                                         | 0/65           | 0/65            | 0.00%             | 0.00%              |
| Eq17      | birth-birth          | no switch (nested)     | $a < x < y < b$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b) \oplus \{(x, y)\}$                                                         | 60/60          | 60/60           | 100.00%           | 100.00%            |
| Eq17s     | birth-birth          | no switch (nested)     | $a < x < b < y$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y) \oplus \{(a, b)\}$                                                         | 65/65          | 65/65           | 100.00%           | 100.00%            |
| Eq17s     | birth-birth          | no switch (nested)     | $a < x < y < b$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y) \oplus \{(a, b)\}$                                                         | 0/60           | 0/60            | 0.00%             | 0.00%              |
| Eq26      | birth-birth          | switch backward        | $a < x < y < b$ | $\text{Pred}_2^\text{bt}(a, b) = \text{Pred}_2^\text{at}(a, y) \oplus \{(x, y)\} \oplus [\text{Pred}_2^\text{at}(x, b) \cap \mathcal{M}]$ | 71/71          | 71/71           | 100.00%           | 100.00%            |
| Eq27      | birth-birth          | switch backward        | $a < x < y < b$ | $\text{Pred}_2^\text{bt}(x, y) = [\text{Pred}_2^\text{at}(x, b) \cap \mathcal{L}]$                                                        | 71/71          | 71/71           | 100.00%           | 100.00%            |
| EqPred1ab | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b)$                                                                           | 74/74          | 74/74           | 100.00%           | 100.00%            |
| EqPred1ab | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b)$                                                                           | 75/75          | 75/75           | 100.00%           | 100.00%            |
| EqPred1xy | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y)$                                                                           | 74/74          | 74/74           | 100.00%           | 100.00%            |
| EqPred1xy | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y)$                                                                           | 75/75          | 75/75           | 100.00%           | 100.00%            |
| EqPred2ab | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b)$                                                                           | 74/74          | 74/74           | 100.00%           | 100.00%            |
| EqPred2ab | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b)$                                                                           | 75/75          | 75/75           | 100.00%           | 100.00%            |
| EqPred2xy | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y)$                                                                           | 74/74          | 74/74           | 100.00%           | 100.00%            |
| EqPred2xy | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y)$                                                                           | 75/75          | 75/75           | 100.00%           | 100.00%            |
| EqSucc1ab | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b)$                                                                           | 74/74          | 74/74           | 100.00%           | 100.00%            |
| EqSucc1ab | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b)$                                                                           | 75/75          | 75/75           | 100.00%           | 100.00%            |
| EqSucc1xy | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y)$                                                                           | 74/74          | 74/74           | 100.00%           | 100.00%            |
| EqSucc1xy | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y)$                                                                           | 75/75          | 75/75           | 100.00%           | 100.00%            |
| EqSucc2ab | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b)$                                                                           | 74/74          | 74/74           | 100.00%           | 100.00%            |
| EqSucc2ab | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b)$                                                                           | 75/75          | 75/75           | 100.00%           | 100.00%            |
| EqSucc2xy | birth-birth          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y)$                                                                           | 74/74          | 74/74           | 100.00%           | 100.00%            |
| EqSucc2xy | birth-birth          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y)$                                                                           | 75/75          | 75/75           | 100.00%           | 100.00%            |

## Death-Death Transpositions
| Name      | Transposition Type   | Switch Type            | Order           | Formula                                                                                                                                   | Correct (WC)   | Correct (WOC)   | Correct, % (WC)   | Correct, % (WOC)   |
|:----------|:---------------------|:-----------------------|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------|:---------------|:----------------|:------------------|:-------------------|
| Eq28      | death-death          | switch forward         | $a < x < y < b$ | $\text{Succ}_1^\text{at}(x, b) = \text{Succ}_1^\text{bt}(x, y) \oplus \{(a, y), (a, b)\}$                                                 | 58/58          | 58/58           | 100.00%           | 100.00%            |
| Eq29      | death-death          | switch forward         | $a < x < y < b$ | $\text{Succ}_1^\text{at}(a, y) = \text{Succ}_1^\text{bt}(a, b)$                                                                           | 23/58          | 23/58           | 39.66%            | 39.66%             |
| Eq30      | death-death          | switch forward         | $a < x < y < b$ | $\text{Pred}_1^\text{at}(x, b) = [\text{Pred}_1^\text{bt}(a, b) \cap \mathcal{B}]$                                                        | 58/58          | 58/58           | 100.00%           | 100.00%            |
| Eq31      | death-death          | switch forward         | $a < x < y < b$ | $\text{Pred}_1^\text{at}(a, y) = \text{Pred}_1^\text{bt}(x, y) \oplus \{(x, b)\} \oplus[\text{Pred}_1^\text{bt}(a, b) \cap \mathcal{N}]$  | 58/58          | 58/58           | 100.00%           | 100.00%            |
| Eq32      | death-death          | switch forward         | $a < x < y < b$ | $\text{Succ}_2^\text{at}(x, b) = \text{Succ}_2^\text{bt}(x, y) \oplus \{(a, y)\} \oplus \text{Succ}_2^\text{bt}(a, b)$                    | 58/58          | 58/58           | 100.00%           | 100.00%            |
| Eq33      | death-death          | switch forward         | $a < x < y < b$ | $\text{Succ}_2^\text{at}(a, y) = \text{Succ}_2^\text{bt}(a, b)$                                                                           | 58/58          | 58/58           | 100.00%           | 100.00%            |
| Eq34      | death-death          | switch forward         | $a < x < y < b$ | $\text{Pred}_2^\text{at}(x, b) = \text{Pred}_2^\text{bt}(x, y)$                                                                           | 58/58          | 58/58           | 100.00%           | 100.00%            |
| Eq35      | death-death          | switch forward         | $a < x < y < b$ | $\text{Pred}_2^\text{at}(a, y) = \text{Pred}_2^\text{bt}(a, b) \oplus \{(x, b)\}$                                                         | 58/58          | 58/58           | 100.00%           | 100.00%            |
| Eq36      | death-death          | no switch (nested)     | $a < x < y < b$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y) \oplus \{(a, b)\} \oplus \text{Succ}_2^\text{bt}(a, b)$                    | 65/65          | 65/65           | 100.00%           | 100.00%            |
| Eq36      | death-death          | no switch (nested)     | $x < a < y < b$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y) \oplus \{(a, b)\} \oplus \text{Succ}_2^\text{bt}(a, b)$                    | 0/66           | 0/66            | 0.00%             | 0.00%              |
| Eq36s     | death-death          | no switch (nested)     | $a < x < y < b$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b) \oplus \{(x, y)\} \oplus \text{Succ}_2^\text{bt}(x, y)$                    | 0/65           | 0/65            | 0.00%             | 0.00%              |
| Eq36s     | death-death          | no switch (nested)     | $x < a < y < b$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b) \oplus \{(x, y)\} \oplus \text{Succ}_2^\text{bt}(x, y)$                    | 66/66          | 66/66           | 100.00%           | 100.00%            |
| Eq37      | death-death          | no switch (nested)     | $a < x < y < b$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b) \oplus \{(x, y)\}$                                                         | 65/65          | 65/65           | 100.00%           | 100.00%            |
| Eq37      | death-death          | no switch (nested)     | $x < a < y < b$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b) \oplus \{(x, y)\}$                                                         | 0/66           | 0/66            | 0.00%             | 0.00%              |
| Eq37s     | death-death          | no switch (nested)     | $a < x < y < b$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y) \oplus \{(a, b)\}$                                                         | 0/65           | 0/65            | 0.00%             | 0.00%              |
| Eq37s     | death-death          | no switch (nested)     | $x < a < y < b$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y) \oplus \{(a, b)\}$                                                         | 66/66          | 66/66           | 100.00%           | 100.00%            |
| Eq46      | death-death          | switch backward        | $a < x < y < b$ | $\text{Pred}_1^\text{bt}(a, b) = \text{Pred}_1^\text{at}(x, b) \oplus \{(x, y)\} \oplus [\text{Pred}_1^\text{at}(a, y) \cap \mathcal{N}]$ | 80/80          | 80/80           | 100.00%           | 100.00%            |
| Eq47      | death-death          | switch backward        | $a < x < y < b$ | $\text{Pred}_1^\text{bt}(x, y) = [\text{Pred}_1^\text{at}(a, y) \cap \mathcal{B}]$                                                        | 80/80          | 80/80           | 100.00%           | 100.00%            |
| EqPred1ab | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b)$                                                                           | 67/67          | 67/67           | 100.00%           | 100.00%            |
| EqPred1ab | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b)$                                                                           | 72/72          | 72/72           | 100.00%           | 100.00%            |
| EqPred1xy | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y)$                                                                           | 67/67          | 67/67           | 100.00%           | 100.00%            |
| EqPred1xy | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y)$                                                                           | 72/72          | 72/72           | 100.00%           | 100.00%            |
| EqPred2ab | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b)$                                                                           | 67/67          | 67/67           | 100.00%           | 100.00%            |
| EqPred2ab | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b)$                                                                           | 72/72          | 72/72           | 100.00%           | 100.00%            |
| EqPred2xy | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y)$                                                                           | 67/67          | 67/67           | 100.00%           | 100.00%            |
| EqPred2xy | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y)$                                                                           | 72/72          | 72/72           | 100.00%           | 100.00%            |
| EqSucc1ab | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b)$                                                                           | 67/67          | 67/67           | 100.00%           | 100.00%            |
| EqSucc1ab | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b)$                                                                           | 72/72          | 72/72           | 100.00%           | 100.00%            |
| EqSucc1xy | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y)$                                                                           | 67/67          | 67/67           | 100.00%           | 100.00%            |
| EqSucc1xy | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y)$                                                                           | 72/72          | 72/72           | 100.00%           | 100.00%            |
| EqSucc2ab | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b)$                                                                           | 67/67          | 67/67           | 100.00%           | 100.00%            |
| EqSucc2ab | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b)$                                                                           | 72/72          | 72/72           | 100.00%           | 100.00%            |
| EqSucc2xy | death-death          | no switch (not nested) | $a < x < y < b$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y)$                                                                           | 67/67          | 67/67           | 100.00%           | 100.00%            |
| EqSucc2xy | death-death          | no switch (not nested) | $x < a < y < b$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y)$                                                                           | 72/72          | 72/72           | 100.00%           | 100.00%            |

## Birth-Death Transpositions
| Name      | Transposition Type   | Switch Type            | Order           | Formula                                                                              | Correct (WC)   | Correct (WOC)   | Correct, % (WC)   | Correct, % (WOC)   |
|:----------|:---------------------|:-----------------------|:----------------|:-------------------------------------------------------------------------------------|:---------------|:----------------|:------------------|:-------------------|
| Eq48      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Succ}_1^\text{at}(a, x) = \text{Succ}_1^\text{bt}(a, b)$                      | 37/37          | 37/37           | 100.00%           | 100.00%            |
| Eq49      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Succ}_1^\text{at}(b, y) = \text{Succ}_1^\text{bt}(x, y)$                      | 37/37          | 37/37           | 100.00%           | 100.00%            |
| Eq50      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_1^\text{at}(a, x) = \{(s, t)\in \text{BD}:\; U_1^\text{bt}[t, x] = 1\}$ | 7/37           | 0/37            | 18.92%            | 0.00%              |
| Eq50a     | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_1^\text{at}(a, x) = A_1^\text{bt} \cap B_1^\text{bt}$                   | 7/37           | 37/37           | 18.92%            | 100.00%            |
| Eq50b     | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_1^\text{at}(a, x) = A_1^\text{bt} \cap B_1^\text{at}$                   | 37/37          | 37/37           | 100.00%           | 100.00%            |
| Eq51      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_1^\text{at}(b, y) = \text{Pred}_1^\text{bt}(x, y)$                      | 37/37          | 37/37           | 100.00%           | 100.00%            |
| Eq52      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Succ}_2^\text{at}(a, x) = \text{Succ}_2^\text{bt}(a, b)$                      | 37/37          | 37/37           | 100.00%           | 100.00%            |
| Eq53      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Succ}_2^\text{at}(b, y) = \text{Succ}_2^\text{bt}(x, y)$                      | 37/37          | 37/37           | 100.00%           | 100.00%            |
| Eq54      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_2^\text{at}(a, x) = \text{Pred}_2^\text{bt}(a, b)$                      | 37/37          | 37/37           | 100.00%           | 100.00%            |
| Eq55      | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_2^\text{at}(b, y) = \{(s, t)\in \text{BD}:\; U_2^\text{bt}[b, s] = 1\}$ | 1/37           | 1/37            | 2.70%             | 2.70%              |
| Eq55a     | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_2^\text{at}(b, y) = A_2^\text{bt} \cap B_2^\text{bt}$                   | 1/37           | 37/37           | 2.70%             | 100.00%            |
| Eq55b     | birth-death          | switch forward         | $a < b < x < y$ | $\text{Pred}_2^\text{at}(b, y) = A_2^\text{bt} \cap B_2^\text{at}$                   | 37/37          | 37/37           | 100.00%           | 100.00%            |
| EqPred1ab | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b)$                      | 28/28          | 28/28           | 100.00%           | 100.00%            |
| EqPred1ab | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_1^\text{at}(a, b) = \text{Pred}_1^\text{bt}(a, b)$                      | 33/33          | 33/33           | 100.00%           | 100.00%            |
| EqPred1xy | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y)$                      | 28/28          | 28/28           | 100.00%           | 100.00%            |
| EqPred1xy | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_1^\text{at}(x, y) = \text{Pred}_1^\text{bt}(x, y)$                      | 33/33          | 33/33           | 100.00%           | 100.00%            |
| EqPred2ab | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b)$                      | 28/28          | 28/28           | 100.00%           | 100.00%            |
| EqPred2ab | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_2^\text{at}(a, b) = \text{Pred}_2^\text{bt}(a, b)$                      | 33/33          | 33/33           | 100.00%           | 100.00%            |
| EqPred2xy | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y)$                      | 28/28          | 28/28           | 100.00%           | 100.00%            |
| EqPred2xy | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Pred}_2^\text{at}(x, y) = \text{Pred}_2^\text{bt}(x, y)$                      | 33/33          | 33/33           | 100.00%           | 100.00%            |
| EqSucc1ab | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b)$                      | 28/28          | 28/28           | 100.00%           | 100.00%            |
| EqSucc1ab | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_1^\text{at}(a, b) = \text{Succ}_1^\text{bt}(a, b)$                      | 33/33          | 33/33           | 100.00%           | 100.00%            |
| EqSucc1xy | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y)$                      | 28/28          | 28/28           | 100.00%           | 100.00%            |
| EqSucc1xy | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_1^\text{at}(x, y) = \text{Succ}_1^\text{bt}(x, y)$                      | 33/33          | 33/33           | 100.00%           | 100.00%            |
| EqSucc2ab | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b)$                      | 28/28          | 28/28           | 100.00%           | 100.00%            |
| EqSucc2ab | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_2^\text{at}(a, b) = \text{Succ}_2^\text{bt}(a, b)$                      | 33/33          | 33/33           | 100.00%           | 100.00%            |
| EqSucc2xy | birth-death          | no switch (not nested) | $a < b < x < y$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y)$                      | 28/28          | 28/28           | 100.00%           | 100.00%            |
| EqSucc2xy | birth-death          | no switch (not nested) | $a < x < b < y$ | $\text{Succ}_2^\text{at}(x, y) = \text{Succ}_2^\text{bt}(x, y)$                      | 33/33          | 33/33           | 100.00%           | 100.00%            |

# Incorrect Equations
We have 364 transpositions, such that some equations are incorrect.
We will list some examples.

## Transposition <2, 3> in complex 0
Here is a __birth-birth__ __no switch__ transposition <2, 3>.

$$
a = 2, \; b = 8, \; x = 3, \; y = 14
$$

$$
a < x < b < y
$$

<img src="transpositions_stats-abstract.assets/image-9949aafc4fdd.png" width="1200">

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

<img src="transpositions_stats-abstract.assets/image-29c1a5dc9300.png" width="1200">

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

<img src="transpositions_stats-abstract.assets/image-78ede20437ce.png" width="1200">

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

<img src="transpositions_stats-abstract.assets/image-182884de3864.png" width="1200">

### Birth Death Pairs
|                          |                                                                                                                                                   |
|:-------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition | $\{(17, 23), (3, 14), (2, 8), (10, 27), (4, 6), (11, 32), (0, 9), (13, 29), (15, 26), (5, 7), (16, 25), (19, 22), (12, 31), (18, 24), (20, 21)\}$ |
| After the transposition  | $\{(17, 23), (2, 8), (10, 27), (4, 6), (14, 26), (11, 32), (13, 29), (5, 7), (0, 9), (16, 25), (19, 22), (12, 31), (3, 15), (18, 24), (20, 21)\}$ |

### Relations
|                                | Algorithm 1                                                                                                                                                                                                                                                     | Algorithm 2                                                                                                                                                                                                                                                                                              |
|:-------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition (WC)  | $\{(24, 27), (25, 32), (26, 27), (31, 32), (29, 32), (22, 32), (6, 8), (6, 14), (24, 26), (24, 32), (26, 29), (29, 31), (23, 27), (22, 31), (7, 9), (25, 27), (24, 31), (21, 23), (26, 31), (22, 27), (23, 26), (22, 24), (23, 29), (23, 32), (6, 9), (7, 8)\}$ | $\{(4, 3), (20, 17), (17, 12), (17, 15), (19, 18), (19, 15), (15, 11), (16, 10), (18, 10), (20, 13), (18, 13), (20, 10), (5, 0), (20, 19), (15, 13), (16, 15), (18, 15), (12, 11), (20, 12), (20, 18), (20, 15), (5, 2), (17, 10), (17, 16), (17, 13), (2, 0), (19, 16), (13, 12), (15, 12), (16, 11)\}$ |
| Before the transposition (WOC) | $\{(24, 27), (25, 32), (26, 27), (31, 32), (29, 32), (22, 32), (6, 8), (6, 14), (24, 26), (24, 32), (26, 29), (29, 31), (23, 27), (22, 31), (7, 9), (25, 27), (24, 31), (21, 23), (26, 31), (22, 27), (23, 26), (22, 24), (23, 29), (23, 32), (6, 9), (7, 8)\}$ | $\{(4, 3), (20, 17), (17, 12), (17, 15), (19, 18), (19, 15), (15, 11), (16, 10), (18, 10), (20, 13), (18, 13), (20, 10), (5, 0), (20, 19), (15, 13), (16, 15), (18, 15), (12, 11), (20, 12), (20, 18), (20, 15), (5, 2), (17, 10), (17, 16), (17, 13), (2, 0), (19, 16), (13, 12), (15, 12), (16, 11)\}$ |
| After the transposition (WC)   | $\{(24, 27), (25, 32), (26, 27), (31, 32), (29, 32), (22, 32), (24, 26), (6, 8), (24, 32), (26, 29), (29, 31), (23, 27), (22, 31), (7, 9), (25, 27), (7, 15), (24, 31), (21, 23), (26, 31), (22, 27), (23, 26), (22, 24), (23, 29), (23, 32), (6, 9), (7, 8)\}$ | $\{(4, 3), (20, 17), (14, 13), (17, 12), (19, 18), (16, 10), (18, 10), (20, 13), (18, 13), (20, 10), (5, 0), (20, 19), (14, 12), (19, 14), (12, 11), (20, 12), (20, 18), (14, 11), (5, 2), (17, 10), (17, 16), (17, 13), (2, 0), (19, 16), (13, 12), (16, 11), (18, 14)\}$                               |
| After the transposition (WOC)  | $\{(24, 27), (25, 32), (26, 27), (31, 32), (29, 32), (22, 32), (24, 26), (6, 8), (24, 32), (26, 29), (29, 31), (23, 27), (22, 31), (7, 9), (25, 27), (7, 15), (24, 31), (21, 23), (26, 31), (22, 27), (23, 26), (22, 24), (23, 29), (23, 32), (6, 9), (7, 8)\}$ | $\{(4, 3), (20, 17), (14, 13), (17, 12), (19, 18), (16, 10), (18, 10), (20, 13), (18, 13), (20, 10), (5, 0), (20, 19), (14, 12), (19, 14), (12, 11), (20, 12), (20, 18), (14, 11), (5, 2), (17, 10), (17, 16), (17, 13), (2, 0), (19, 16), (13, 12), (16, 11), (18, 14)\}$                               |

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
| $B_1 = \{(s, t)\in \text{BD}:\; U_1[t, x] = 1\}$ (WOC)             | $\emptyset$                                            | $\{(5, 7)\}$                                           |
| $A_2 = \{(s, t)\in \text{BD}:\; f(b) < f(s) < f(t) < f(y)\}$ (WC)  | $\{(17, 23), (16, 25), (19, 22), (18, 24), (20, 21)\}$ | $\{(17, 23), (16, 25), (19, 22), (18, 24), (20, 21)\}$ |
| $A_2 = \{(s, t)\in \text{BD}:\; f(b) < f(s) < f(t) < f(y)\}$ (WOC) | $\{(17, 23), (16, 25), (19, 22), (18, 24), (20, 21)\}$ | $\{(17, 23), (16, 25), (19, 22), (18, 24), (20, 21)\}$ |
| $B_2 = \{(s, t)\in \text{BD}:\; U_2[b, s] = 1\}$ (WC)              | $\emptyset$                                            | $\{(19, 22), (18, 24)\}$                               |
| $B_2 = \{(s, t)\in \text{BD}:\; U_2[b, s] = 1\}$ (WOC)             | $\emptyset$                                            | $\{(19, 22), (18, 24)\}$                               |

### Equations
There are 4 equations are wrong.

| eq    | left (formula)                  | left (paramatrized)               | left (value) (WC)                  | left (value) (WOC)                 | right (formula)                                      | right (paramatrized)                                  | right (value) (WC)                 | right (value) (WOC)                | correct (WC)   | correct (WOC)   |
|:------|:--------------------------------|:----------------------------------|:-----------------------------------|:-----------------------------------|:-----------------------------------------------------|:------------------------------------------------------|:-----------------------------------|:-----------------------------------|:---------------|:----------------|
| Eq48  | $\text{Succ}_1^\text{at}(a, x)$ | $\text{Succ}_1^\text{at}(3, 15)$  | $\emptyset$                        | $\emptyset$                        | $\text{Succ}_1^\text{bt}(a, b)$                      | $\text{Succ}_1^\text{bt}(3, 14)$                      | $\emptyset$                        | $\emptyset$                        | ✔              | ✔               |
| Eq49  | $\text{Succ}_1^\text{at}(b, y)$ | $\text{Succ}_1^\text{at}(14, 26)$ | $\{(10, 27), (13, 29), (12, 31)\}$ | $\{(10, 27), (13, 29), (12, 31)\}$ | $\text{Succ}_1^\text{bt}(x, y)$                      | $\text{Succ}_1^\text{bt}(15, 26)$                     | $\{(10, 27), (13, 29), (12, 31)\}$ | $\{(10, 27), (13, 29), (12, 31)\}$ | ✔              | ✔               |
| Eq50  | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(3, 15)$  | $\{(5, 7)\}$                       | $\{(5, 7)\}$                       | $\{(s, t)\in \text{BD}:\; U_1^\text{bt}[t, x] = 1\}$ | $\{(s, t)\in \text{BD}:\; U_1^\text{bt}[t, 15] = 1\}$ | $\emptyset$                        | $\emptyset$                        | ✘              | ✘               |
| Eq50a | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(3, 15)$  | $\{(5, 7)\}$                       | $\{(5, 7)\}$                       | $A_1^\text{bt} \cap B_1^\text{bt}$                   | $A_1^\text{bt} \cap B_1^\text{bt}$                    | $\emptyset$                        | $\emptyset$                        | ✘              | ✘               |
| Eq50b | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(3, 15)$  | $\{(5, 7)\}$                       | $\{(5, 7)\}$                       | $A_1^\text{bt} \cap B_1^\text{at}$                   | $A_1^\text{bt} \cap B_1^\text{at}$                    | $\{(5, 7)\}$                       | $\{(5, 7)\}$                       | ✔              | ✔               |
| Eq51  | $\text{Pred}_1^\text{at}(b, y)$ | $\text{Pred}_1^\text{at}(14, 26)$ | $\{(17, 23), (18, 24)\}$           | $\{(17, 23), (18, 24)\}$           | $\text{Pred}_1^\text{bt}(x, y)$                      | $\text{Pred}_1^\text{bt}(15, 26)$                     | $\{(17, 23), (18, 24)\}$           | $\{(17, 23), (18, 24)\}$           | ✔              | ✔               |
| Eq52  | $\text{Succ}_2^\text{at}(a, x)$ | $\text{Succ}_2^\text{at}(3, 15)$  | $\emptyset$                        | $\emptyset$                        | $\text{Succ}_2^\text{bt}(a, b)$                      | $\text{Succ}_2^\text{bt}(3, 14)$                      | $\emptyset$                        | $\emptyset$                        | ✔              | ✔               |
| Eq53  | $\text{Succ}_2^\text{at}(b, y)$ | $\text{Succ}_2^\text{at}(14, 26)$ | $\{(13, 29), (11, 32), (12, 31)\}$ | $\{(13, 29), (11, 32), (12, 31)\}$ | $\text{Succ}_2^\text{bt}(x, y)$                      | $\text{Succ}_2^\text{bt}(15, 26)$                     | $\{(13, 29), (11, 32), (12, 31)\}$ | $\{(13, 29), (11, 32), (12, 31)\}$ | ✔              | ✔               |
| Eq54  | $\text{Pred}_2^\text{at}(a, x)$ | $\text{Pred}_2^\text{at}(3, 15)$  | $\{(4, 6)\}$                       | $\{(4, 6)\}$                       | $\text{Pred}_2^\text{bt}(a, b)$                      | $\text{Pred}_2^\text{bt}(3, 14)$                      | $\{(4, 6)\}$                       | $\{(4, 6)\}$                       | ✔              | ✔               |
| Eq55  | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(14, 26)$ | $\{(19, 22), (18, 24)\}$           | $\{(19, 22), (18, 24)\}$           | $\{(s, t)\in \text{BD}:\; U_2^\text{bt}[b, s] = 1\}$ | $\{(s, t)\in \text{BD}:\; U_2^\text{bt}[14, s] = 1\}$ | $\emptyset$                        | $\emptyset$                        | ✘              | ✘               |
| Eq55a | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(14, 26)$ | $\{(19, 22), (18, 24)\}$           | $\{(19, 22), (18, 24)\}$           | $A_2^\text{bt} \cap B_2^\text{bt}$                   | $A_2^\text{bt} \cap B_2^\text{bt}$                    | $\emptyset$                        | $\emptyset$                        | ✘              | ✘               |
| Eq55b | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(14, 26)$ | $\{(19, 22), (18, 24)\}$           | $\{(19, 22), (18, 24)\}$           | $A_2^\text{bt} \cap B_2^\text{at}$                   | $A_2^\text{bt} \cap B_2^\text{at}$                    | $\{(18, 24), (19, 22)\}$           | $\{(18, 24), (19, 22)\}$           | ✔              | ✔               |

## Transposition <15, 16> in complex 0
Here is a __birth-birth__ __switch forward__ transposition <15, 16>.

$$
a = 15, \; b = 26, \; x = 16, \; y = 25
$$

$$
a < x < y < b
$$

<img src="transpositions_stats-abstract.assets/image-38b2e7024662.png" width="1200">

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

<img src="transpositions_stats-abstract.assets/image-84086b87933b.png" width="1200">

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

<img src="transpositions_stats-abstract.assets/image-2678dbf4ac46.png" width="1200">

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

<img src="transpositions_stats-abstract.assets/image-0ade523799b5.png" width="1200">

### Birth Death Pairs
|                          |                                                                                                                                                   |
|:-------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition | $\{(17, 27), (10, 24), (12, 26), (2, 11), (1, 14), (8, 35), (3, 9), (13, 28), (18, 25), (5, 6), (20, 22), (19, 21), (4, 7), (15, 31), (16, 23)\}$ |
| After the transposition  | $\{(17, 27), (10, 24), (12, 26), (1, 14), (7, 35), (3, 9), (13, 28), (18, 25), (5, 6), (20, 22), (4, 8), (19, 21), (2, 11), (15, 31), (16, 23)\}$ |

### Relations
|                                | Algorithm 1                                                                                                                                                                                                                                 | Algorithm 2                                                                                                                                                                                                                                     |
|:-------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition (WC)  | $\{(25, 35), (21, 25), (21, 31), (31, 35), (23, 31), (9, 14), (11, 14), (27, 28), (6, 14), (25, 31), (24, 35), (26, 35), (21, 24), (21, 27), (22, 28), (22, 31), (7, 9), (6, 7), (21, 23), (22, 27), (23, 35), (7, 11), (6, 9), (25, 26)\}$ | $\{(20, 8), (4, 3), (3, 1), (20, 17), (17, 15), (19, 15), (13, 8), (16, 10), (16, 13), (18, 13), (20, 16), (17, 8), (19, 8), (19, 17), (10, 8), (16, 12), (18, 12), (3, 2), (20, 12), (20, 18), (12, 8), (20, 15), (5, 2), (19, 16), (18, 8)\}$ |
| Before the transposition (WOC) | $\{(25, 35), (21, 25), (21, 31), (31, 35), (23, 31), (9, 14), (11, 14), (27, 28), (6, 14), (25, 31), (24, 35), (26, 35), (21, 24), (21, 27), (22, 28), (22, 31), (7, 9), (6, 7), (21, 23), (22, 27), (23, 35), (7, 11), (6, 9), (25, 26)\}$ | $\{(20, 8), (4, 3), (3, 1), (20, 17), (17, 15), (19, 15), (13, 8), (16, 10), (16, 13), (18, 13), (20, 16), (17, 8), (19, 8), (19, 17), (10, 8), (16, 12), (18, 12), (3, 2), (20, 12), (20, 18), (12, 8), (20, 15), (5, 2), (19, 16), (18, 8)\}$ |
| After the transposition (WC)   | $\{(25, 35), (21, 25), (21, 31), (8, 9), (31, 35), (23, 31), (9, 14), (11, 14), (27, 28), (6, 14), (25, 31), (24, 35), (26, 35), (21, 24), (21, 27), (8, 11), (22, 28), (22, 31), (21, 23), (22, 27), (23, 35), (6, 9), (25, 26)\}$         | $\{(12, 7), (4, 3), (3, 1), (20, 17), (17, 15), (19, 15), (16, 7), (16, 10), (18, 7), (16, 13), (18, 13), (20, 16), (19, 17), (13, 7), (16, 12), (18, 12), (3, 2), (20, 12), (20, 18), (20, 15), (5, 2), (17, 7), (19, 16)\}$                   |
| After the transposition (WOC)  | $\{(25, 35), (21, 25), (21, 31), (8, 9), (31, 35), (23, 31), (9, 14), (11, 14), (27, 28), (6, 14), (25, 31), (24, 35), (26, 35), (21, 24), (21, 27), (8, 11), (22, 28), (22, 31), (21, 23), (22, 27), (23, 35), (6, 9), (25, 26)\}$         | $\{(12, 7), (4, 3), (3, 1), (20, 17), (17, 15), (19, 15), (16, 7), (16, 10), (18, 7), (16, 13), (18, 13), (20, 16), (19, 17), (13, 7), (16, 12), (18, 12), (3, 2), (20, 12), (20, 18), (20, 15), (5, 2), (17, 7), (19, 16)\}$                   |

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
| $B_1 = \{(s, t)\in \text{BD}:\; U_1[t, x] = 1\}$ (WOC)             | $\emptyset$                                                                                    | $\emptyset$                                                                                    |
| $A_2 = \{(s, t)\in \text{BD}:\; f(b) < f(s) < f(t) < f(y)\}$ (WC)  | $\{(10, 24), (12, 26), (16, 23), (13, 28), (18, 25), (20, 22), (19, 21), (15, 31), (17, 27)\}$ | $\{(10, 24), (12, 26), (16, 23), (13, 28), (18, 25), (20, 22), (19, 21), (15, 31), (17, 27)\}$ |
| $A_2 = \{(s, t)\in \text{BD}:\; f(b) < f(s) < f(t) < f(y)\}$ (WOC) | $\{(10, 24), (12, 26), (16, 23), (13, 28), (18, 25), (20, 22), (19, 21), (15, 31), (17, 27)\}$ | $\{(10, 24), (12, 26), (16, 23), (13, 28), (18, 25), (20, 22), (19, 21), (15, 31), (17, 27)\}$ |
| $B_2 = \{(s, t)\in \text{BD}:\; U_2[b, s] = 1\}$ (WC)              | $\emptyset$                                                                                    | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$                                         |
| $B_2 = \{(s, t)\in \text{BD}:\; U_2[b, s] = 1\}$ (WOC)             | $\emptyset$                                                                                    | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$                                         |

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
| Eq55a | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(7, 35)$ | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | $A_2^\text{bt} \cap B_2^\text{bt}$                   | $A_2^\text{bt} \cap B_2^\text{bt}$                   | $\emptyset$                                            | $\emptyset$                                            | ✘              | ✘               |
| Eq55b | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(7, 35)$ | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | $A_2^\text{bt} \cap B_2^\text{at}$                   | $A_2^\text{bt} \cap B_2^\text{at}$                   | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | $\{(17, 27), (12, 26), (13, 28), (18, 25), (16, 23)\}$ | ✔              | ✔               |

## Transposition <35, 36> in complex 3
Here is a __birth-death__ __switch forward__ transposition <35, 36>.

$$
a = 8, \; b = 35, \; x = 36, \; y = 42
$$

$$
a < b < x < y
$$

<img src="transpositions_stats-abstract.assets/image-f08b4923abd6.png" width="1200">

### Birth Death Pairs
|                          |                                                                                                                                                                                         |
|:-------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition | $\{(10, 24), (38, 39), (32, 46), (12, 26), (19, 21), (36, 42), (37, 41), (30, 45), (8, 35), (16, 23), (13, 28), (18, 25), (29, 47), (20, 22), (34, 40), (33, 43), (15, 31), (17, 27)\}$ |
| After the transposition  | $\{(10, 24), (38, 39), (32, 46), (12, 26), (35, 42), (19, 21), (8, 36), (37, 41), (30, 45), (16, 23), (13, 28), (18, 25), (29, 47), (20, 22), (34, 40), (33, 43), (15, 31), (17, 27)\}$ |

### Relations
|                                | Algorithm 1                                                                                                                                                                                                                                                                        | Algorithm 2                                                                                                                                                                                                                                                                                                                             |
|:-------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Before the transposition (WC)  | $\{(25, 35), (41, 43), (42, 45), (21, 25), (21, 31), (31, 35), (23, 31), (27, 28), (25, 31), (41, 45), (24, 35), (26, 35), (43, 45), (21, 24), (46, 47), (21, 27), (22, 28), (22, 31), (39, 47), (42, 43), (41, 47), (21, 23), (22, 27), (23, 35), (40, 45), (25, 26)\}$           | $\{(20, 8), (20, 17), (17, 15), (19, 15), (37, 36), (13, 8), (16, 10), (16, 13), (18, 13), (33, 32), (20, 16), (38, 34), (17, 8), (34, 30), (19, 8), (34, 33), (36, 30), (37, 29), (19, 17), (10, 8), (36, 33), (30, 29), (16, 12), (18, 12), (32, 29), (20, 12), (20, 18), (12, 8), (20, 15), (19, 16), (34, 32), (33, 30), (18, 8)\}$ |
| Before the transposition (WOC) | $\{(25, 35), (41, 43), (42, 45), (21, 25), (21, 31), (31, 35), (23, 31), (27, 28), (25, 31), (41, 45), (24, 35), (26, 35), (43, 45), (21, 24), (46, 47), (21, 27), (22, 28), (22, 31), (39, 47), (42, 43), (41, 47), (21, 23), (22, 27), (23, 35), (40, 45), (25, 26)\}$           | $\{(20, 8), (20, 17), (17, 15), (19, 15), (37, 36), (13, 8), (16, 10), (16, 13), (18, 13), (33, 32), (20, 16), (38, 34), (17, 8), (34, 30), (19, 8), (34, 33), (36, 30), (37, 29), (19, 17), (10, 8), (36, 33), (30, 29), (16, 12), (18, 12), (32, 29), (20, 12), (20, 18), (12, 8), (20, 15), (19, 16), (34, 32), (33, 30), (18, 8)\}$ |
| After the transposition (WC)   | $\{(42, 45), (41, 43), (26, 36), (21, 25), (21, 31), (23, 31), (27, 28), (28, 36), (25, 31), (41, 45), (43, 45), (21, 24), (46, 47), (21, 27), (22, 28), (21, 36), (22, 31), (39, 47), (27, 36), (42, 43), (41, 47), (25, 36), (21, 23), (22, 27), (40, 45), (31, 36), (25, 26)\}$ | $\{(35, 30), (35, 33), (20, 8), (20, 17), (17, 15), (19, 15), (13, 8), (16, 10), (16, 13), (18, 13), (33, 32), (20, 16), (38, 34), (17, 8), (34, 30), (19, 8), (34, 33), (37, 29), (19, 17), (10, 8), (30, 29), (16, 12), (18, 12), (32, 29), (20, 12), (20, 18), (12, 8), (20, 15), (19, 16), (34, 32), (33, 30), (18, 8)\}$           |
| After the transposition (WOC)  | $\{(42, 45), (41, 43), (26, 36), (21, 25), (21, 31), (23, 31), (27, 28), (28, 36), (25, 31), (41, 45), (43, 45), (21, 24), (46, 47), (21, 27), (22, 28), (21, 36), (22, 31), (39, 47), (27, 36), (42, 43), (41, 47), (25, 36), (21, 23), (22, 27), (40, 45), (31, 36), (25, 26)\}$ | $\{(35, 30), (35, 33), (20, 8), (20, 17), (17, 15), (19, 15), (13, 8), (16, 10), (16, 13), (18, 13), (33, 32), (20, 16), (38, 34), (17, 8), (34, 30), (19, 8), (34, 33), (37, 29), (19, 17), (10, 8), (30, 29), (16, 12), (18, 12), (32, 29), (20, 12), (20, 18), (12, 8), (20, 15), (19, 16), (34, 32), (33, 30), (18, 8)\}$           |

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
| $B_1 = \{(s, t)\in \text{BD}:\; U_1[t, x] = 1\}$ (WOC)             | $\emptyset$                                                                                    | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$                               |
| $A_2 = \{(s, t)\in \text{BD}:\; f(b) < f(s) < f(t) < f(y)\}$ (WC)  | $\{(38, 39), (37, 41)\}$                                                                       | $\{(38, 39), (37, 41)\}$                                                                       |
| $A_2 = \{(s, t)\in \text{BD}:\; f(b) < f(s) < f(t) < f(y)\}$ (WOC) | $\{(38, 39), (37, 41)\}$                                                                       | $\{(38, 39), (37, 41)\}$                                                                       |
| $B_2 = \{(s, t)\in \text{BD}:\; U_2[b, s] = 1\}$ (WC)              | $\emptyset$                                                                                    | $\emptyset$                                                                                    |
| $B_2 = \{(s, t)\in \text{BD}:\; U_2[b, s] = 1\}$ (WOC)             | $\emptyset$                                                                                    | $\emptyset$                                                                                    |

### Equations
There are 3 equations are wrong.

| eq    | left (formula)                  | left (paramatrized)               | left (value) (WC)                                                          | left (value) (WOC)                                                         | right (formula)                                      | right (paramatrized)                                  | right (value) (WC)                                                         | right (value) (WOC)                                                        | correct (WC)   | correct (WOC)   |
|:------|:--------------------------------|:----------------------------------|:---------------------------------------------------------------------------|:---------------------------------------------------------------------------|:-----------------------------------------------------|:------------------------------------------------------|:---------------------------------------------------------------------------|:---------------------------------------------------------------------------|:---------------|:----------------|
| Eq48  | $\text{Succ}_1^\text{at}(a, x)$ | $\text{Succ}_1^\text{at}(8, 36)$  | $\emptyset$                                                                | $\emptyset$                                                                | $\text{Succ}_1^\text{bt}(a, b)$                      | $\text{Succ}_1^\text{bt}(8, 35)$                      | $\emptyset$                                                                | $\emptyset$                                                                | ✔              | ✔               |
| Eq49  | $\text{Succ}_1^\text{at}(b, y)$ | $\text{Succ}_1^\text{at}(35, 42)$ | $\{(33, 43), (30, 45)\}$                                                   | $\{(33, 43), (30, 45)\}$                                                   | $\text{Succ}_1^\text{bt}(x, y)$                      | $\text{Succ}_1^\text{bt}(36, 42)$                     | $\{(33, 43), (30, 45)\}$                                                   | $\{(33, 43), (30, 45)\}$                                                   | ✔              | ✔               |
| Eq50  | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(8, 36)$  | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $\{(s, t)\in \text{BD}:\; U_1^\text{bt}[t, x] = 1\}$ | $\{(s, t)\in \text{BD}:\; U_1^\text{bt}[t, 36] = 1\}$ | $\emptyset$                                                                | $\emptyset$                                                                | ✘              | ✘               |
| Eq50a | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(8, 36)$  | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $A_1^\text{bt} \cap B_1^\text{bt}$                   | $A_1^\text{bt} \cap B_1^\text{bt}$                    | $\emptyset$                                                                | $\emptyset$                                                                | ✘              | ✘               |
| Eq50b | $\text{Pred}_1^\text{at}(a, x)$ | $\text{Pred}_1^\text{at}(8, 36)$  | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $A_1^\text{bt} \cap B_1^\text{at}$                   | $A_1^\text{bt} \cap B_1^\text{at}$                    | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | $\{(12, 26), (13, 28), (18, 25), (19, 21), (15, 31), (17, 27)\}$           | ✔              | ✔               |
| Eq51  | $\text{Pred}_1^\text{at}(b, y)$ | $\text{Pred}_1^\text{at}(35, 42)$ | $\emptyset$                                                                | $\emptyset$                                                                | $\text{Pred}_1^\text{bt}(x, y)$                      | $\text{Pred}_1^\text{bt}(36, 42)$                     | $\emptyset$                                                                | $\emptyset$                                                                | ✔              | ✔               |
| Eq52  | $\text{Succ}_2^\text{at}(a, x)$ | $\text{Succ}_2^\text{at}(8, 36)$  | $\emptyset$                                                                | $\emptyset$                                                                | $\text{Succ}_2^\text{bt}(a, b)$                      | $\text{Succ}_2^\text{bt}(8, 35)$                      | $\emptyset$                                                                | $\emptyset$                                                                | ✔              | ✔               |
| Eq53  | $\text{Succ}_2^\text{at}(b, y)$ | $\text{Succ}_2^\text{at}(35, 42)$ | $\{(33, 43), (30, 45)\}$                                                   | $\{(33, 43), (30, 45)\}$                                                   | $\text{Succ}_2^\text{bt}(x, y)$                      | $\text{Succ}_2^\text{bt}(36, 42)$                     | $\{(33, 43), (30, 45)\}$                                                   | $\{(33, 43), (30, 45)\}$                                                   | ✔              | ✔               |
| Eq54  | $\text{Pred}_2^\text{at}(a, x)$ | $\text{Pred}_2^\text{at}(8, 36)$  | $\{(10, 24), (12, 26), (13, 28), (18, 25), (20, 22), (19, 21), (17, 27)\}$ | $\{(10, 24), (12, 26), (13, 28), (18, 25), (20, 22), (19, 21), (17, 27)\}$ | $\text{Pred}_2^\text{bt}(a, b)$                      | $\text{Pred}_2^\text{bt}(8, 35)$                      | $\{(10, 24), (12, 26), (13, 28), (18, 25), (20, 22), (19, 21), (17, 27)\}$ | $\{(10, 24), (12, 26), (13, 28), (18, 25), (20, 22), (19, 21), (17, 27)\}$ | ✔              | ✔               |
| Eq55  | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(35, 42)$ | $\emptyset$                                                                | $\emptyset$                                                                | $\{(s, t)\in \text{BD}:\; U_2^\text{bt}[b, s] = 1\}$ | $\{(s, t)\in \text{BD}:\; U_2^\text{bt}[35, s] = 1\}$ | $\{(34, 40), (30, 45)\}$                                                   | $\{(34, 40), (30, 45)\}$                                                   | ✘              | ✘               |
| Eq55a | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(35, 42)$ | $\emptyset$                                                                | $\emptyset$                                                                | $A_2^\text{bt} \cap B_2^\text{bt}$                   | $A_2^\text{bt} \cap B_2^\text{bt}$                    | $\emptyset$                                                                | $\emptyset$                                                                | ✔              | ✔               |
| Eq55b | $\text{Pred}_2^\text{at}(b, y)$ | $\text{Pred}_2^\text{at}(35, 42)$ | $\emptyset$                                                                | $\emptyset$                                                                | $A_2^\text{bt} \cap B_2^\text{at}$                   | $A_2^\text{bt} \cap B_2^\text{at}$                    | $\emptyset$                                                                | $\emptyset$                                                                | ✔              | ✔               |