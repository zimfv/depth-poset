
# The Problem
There is a boundary matrix given

|    |   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |   9 |   10 |   11 |   12 |
|---:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|-----:|-----:|-----:|
|  0 |   0 |   1 |   1 |   1 |   1 |   1 |   0 |   0 |   0 |   0 |    0 |    0 |    0 |
|  1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |    1 |    1 |    0 |
|  2 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   0 |    1 |    0 |    1 |
|  3 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   0 |   1 |    0 |    0 |    0 |
|  4 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   1 |   1 |    0 |    0 |    0 |
|  5 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   1 |   0 |    0 |    1 |    1 |

and we are making __birth-birth switch forward__ transposition $(2, 3)$.

| a | b | x | y |
| --- | --- | --- | --- |
| 2 | 10 | 3 | 8 |


# Computed sets
## Successors and Predecessers
|                               | $\text{Pred}_1$                                      | $\text{Pred}_2$                               | $\text{Succ}_1$                               | $\text{Succ}_2$                                       |
|:------------------------------|:-----------------------------------------------------|:----------------------------------------------|:----------------------------------------------|:------------------------------------------------------|
| $\text{Set}^\text{bt}(2, 10)$ | $\text{Pred}_1^\text{bt}(2, 10) = \emptyset$         | $\text{Pred}_2^\text{bt}(2, 10) = \emptyset$  | $\text{Succ}_1^\text{bt}(2, 10) = \emptyset$  | $\text{Succ}_2^\text{bt}(2, 10) = \{(3, 8), (4, 7)\}$ |
| $\text{Set}^\text{bt}(3, 8)$  | $\text{Pred}_1^\text{bt}(3, 8) = \{(5, 6), (4, 7)\}$ | $\text{Pred}_2^\text{bt}(3, 8) = \{(2, 10)\}$ | $\text{Succ}_1^\text{bt}(3, 8) = \emptyset$   | $\text{Succ}_2^\text{bt}(3, 8) = \{(5, 6)\}$          |
| $\text{Set}^\text{at}(2, 8)$  | $\text{Pred}_1^\text{at}(2, 8) = \{(5, 6), (4, 7)\}$ | $\text{Pred}_2^\text{at}(2, 8) = \{(3, 10)\}$ | $\text{Succ}_1^\text{at}(2, 8) = \{(3, 10)\}$ | $\text{Succ}_2^\text{at}(2, 8) = \{(4, 7)\}$          |
| $\text{Set}^\text{at}(3, 10)$ | $\text{Pred}_1^\text{at}(3, 10) = \{(2, 8)\}$        | $\text{Pred}_2^\text{at}(3, 10) = \emptyset$  | $\text{Succ}_1^\text{at}(3, 10) = \emptyset$  | $\text{Succ}_2^\text{at}(3, 10) = \{(5, 6), (2, 8)\}$ |

## $\mathcal{L}$ and $\mathcal{M}$ sets
|                                                               | Before the Transposition     | After the Transposition      |
|:--------------------------------------------------------------|:-----------------------------|:-----------------------------|
| $\mathcal{L} = \{(s, t)\in \text{BD}:\; f(t) < f(y)\}$        | $\{(0, 1), (5, 6), (4, 7)\}$ | $\{(0, 1), (5, 6), (4, 7)\}$ |
| $\mathcal{M} = \{(s, t)\in \text{BD}:\; f(y) < f(t) < f(b)\}$ | $\emptyset$                  | $\emptyset$                  |

# Equations
|   eq | left (formula)                  | left (paramatrized)              | left (value)         | right (formula)                                                                                           | right (paramatrized)                                                                                       | right (value)                  | correct   |
|-----:|:--------------------------------|:---------------------------------|:---------------------|:----------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------|:-------------------------------|:----------|
|    8 | $\text{Succ}_1^\text{at}(a, y)$ | $\text{Succ}_1^\text{at}(2, 8)$  | $\{(3, 10)\}$        | $\text{Succ}_1^\text{bt}(x, y) \oplus \{(x, b)\} \oplus \text{Succ}_1^\text{bt}(a, b)$                    | $\text{Succ}_1^\text{bt}(3, 8) \oplus \{(3, 10)\} \oplus \text{Succ}_1^\text{bt}(2, 10)$                   | $\{(3, 10)\}$                  | ✔         |
|    9 | $\text{Succ}_1^\text{at}(x, b)$ | $\text{Succ}_1^\text{at}(3, 10)$ | $\emptyset$          | $\text{Succ}_1^\text{bt}(a, b)$                                                                           | $\text{Succ}_1^\text{bt}(2, 10)$                                                                           | $\emptyset$                    | ✔         |
|   10 | $\text{Pred}_1^\text{at}(a, y)$ | $\text{Pred}_1^\text{at}(2, 8)$  | $\{(5, 6), (4, 7)\}$ | $\text{Pred}_1^\text{bt}(x, y)$                                                                           | $\text{Pred}_1^\text{bt}(3, 8)$                                                                            | $\{(5, 6), (4, 7)\}$           | ✔         |
|   11 | $\text{Pred}_1^\text{at}(x, b)$ | $\text{Pred}_1^\text{at}(3, 10)$ | $\{(2, 8)\}$         | $\text{Pred}_1^\text{bt}(a, b) \oplus \{(a, y)\}$                                                         | $\text{Pred}_1^\text{bt}(2, 10) \oplus \{(2, 8)\}$                                                         | $\{(2, 8)\}$                   | ✔         |
|   12 | $\text{Succ}_2^\text{at}(a, y)$ | $\text{Succ}_2^\text{at}(2, 8)$  | $\{(4, 7)\}$         | $\text{Succ}_2^\text{bt}(x, y) \oplus \{(x, b), (a, b)\}$                                                 | $\text{Succ}_2^\text{bt}(3, 8) \oplus \{(3, 10), (2, 10)\}$                                                | $\{(2, 10), (5, 6), (3, 10)\}$ | ✘         |
|   13 | $\text{Succ}_2^\text{at}(x, b)$ | $\text{Succ}_2^\text{at}(3, 10)$ | $\{(5, 6), (2, 8)\}$ | $\text{Succ}_2^\text{bt}(a, b)$                                                                           | $\text{Succ}_2^\text{bt}(2, 10)$                                                                           | $\{(3, 8), (4, 7)\}$           | ✘         |
|   14 | $\text{Pred}_2^\text{at}(a, y)$ | $\text{Pred}_2^\text{at}(2, 8)$  | $\{(3, 10)\}$        | $[\text{Pred}_2^\text{bt}(a, b) \cap \mathcal{L}]$                                                        | $[\text{Pred}_2^\text{bt}(2, 10) \cap \mathcal{L}]$                                                        | $\emptyset$                    | ✘         |
|   15 | $\text{Pred}_2^\text{at}(x, b)$ | $\text{Pred}_2^\text{at}(3, 10)$ | $\emptyset$          | $\text{Pred}_2^\text{bt}(x, y) \oplus \{(a, y)\} \oplus [\text{Pred}_2^\text{bt}(a, b) \cap \mathcal{M}]$ | $\text{Pred}_2^\text{bt}(3, 8) \oplus \{(2, 8)\} \oplus [\text{Pred}_2^\text{bt}(2, 10) \cap \mathcal{M}]$ | $\{(2, 10), (2, 8)\}$          | ✘         |
