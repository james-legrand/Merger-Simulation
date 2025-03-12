This first note covers the most simple of demand structures: linear demand. Credit for this section goes to [Davis(2006)](https://www.biicl.org/files/2757_peter_davis_-_coordinated_effects_merger_simulation_with_linear_demands.pdf).
### Model Setup
The demand for good $k \in \{1, \dots, J\} \equiv \mathcal{J} $ demand is written as a linear function of the prices of all the goods in the market: 
```math
D_k(p_1, \dots, p_N) = 
\begin{cases}
    a_k + \sum_{j=1}^{J}b_{kj}p_j & \text{if } a_k + \sum_{j=1}^{J}b_{kj}p_j \geq 0 \\
    0 & \text{otherwise}
\end{cases}
```
with demand intercept parameter $a_k$ and slope parameter $b_{ij}$ describing the change in demand for product $k$ when good $j$'s price increases by 1 unit. 
### Analysis

```python
import numpy as np
```
We now consider the pricing game, wherein each firm $i$ produces a subset of the available products, $\mathcal{J}_i \subseteq \mathcal{J}$, and choses the prices of these products to maximise its profits. Note at this point that the set $\mathcal{J}_i$ is pre-specified and considered fixed; it is not part of the optimisation problem. Consider the maximisation:
```math
\max_{p_j} \sum_{j \in \mathcal{J_i}} (p_j-c_j)D_j(\mathbf{p}) \quad \text{s.t.} \quad p_j \geq 0 \quad \forall j \in \mathcal{J_i}
```
where $\mathbf{p} = (p_1, \dots, p_N)$ and $c_j$ is the marginal cost of product $j$, assumed constant. In the empirical literature it is common for us to assume that all goods are sold in positive quantities and with positive prices (as such the constraints do not bind in equilibrium). For this reason, consider the first order conditions to the unconstrained problem: 
```math
D_k(\mathbf{p}) + \sum_{j \in \mathcal{J_i}} \frac{\partial D_j(\mathbf{p})}{\partial p_k} (p_j - c_j) = 0, \quad \forall k \in \mathcal{J_i} \\
\implies  a_k + \sum_{j=1}^J b_{kj}p_j + \sum_{j \in \mathcal{J_i}}b_{jk}(p_j - c_j) = 0, \quad \forall k \in \mathcal{J_i}
```
At this point, it is useful to introduce the 'ownership matrix' to standardise the first order conditions. Specifically, the $J \times J$ matrix $\Delta$, with $j,k^{th}$ element:
```math
\Delta_{kj} = 
\begin{cases}
    1 & \text{if $j, k$ produced by same firm} \\
    0 & \text{otherwise}
\end{cases}
```
where, by construction $\Delta_{kj} = \Delta_{jk}$ for all $j,k \in \mathcal{J}$. Changing the ownership structure in unilateral effects merger simulations reduces to solely changing this indicator matrix. Using this new matrix we can reconstruct the first order condition by summing over all products:
```math
  a_k + \sum_{j=1}^J b_{kj}p_j + \sum_{j=1}^J \Delta_{jk} b_{jk}(p_j - c_j) = 0 \quad \forall k \in \mathcal{J_i}
```
Note that there is one of these first order conditions for each $k \in \mathcal{J_i}$. Since every product is owned by some firm, we obtain a total of $J$ first order conditions - one for each product. We can then stack up these conditions as follows. We first define the matrices $\mathbf{a}$ and $\mathbf{B}$ as vectors of demand coefficients as follows:

```math
\underbrace{\mathbf{a}}_{J \times 1} = 
\begin{pmatrix}
a_1 \\
\vdots \\
a_J
\end{pmatrix}

\quad

\underbrace{\mathbf{B}}_{J \times J} = 
\begin{pmatrix}
b_{11} & \cdots & b_{j1}& \cdots & b_{J1} \\
\vdots & & \vdots & & \vdots \\
b_{1k} & \cdots& b_{jk} & \cdots & b_{Jk} \\
\vdots & & \vdots & & \vdots \\
b_{1J} & \cdots& b_{jJ} & \cdots & b_{JJ}
\end{pmatrix}
```

It is also useful to define the Hadamard (or element-by-element) product of $\Delta$ and $\mathbf{B}$:
```math
\underbrace{ \Delta \odot \mathbf{B}}_{J \times J} = 
\begin{pmatrix}
\Delta_{11}b_{11} & \cdots & \Delta_{j1}b_{j1}& \cdots & \Delta_{J1}b_{J1} \\
\vdots & & \vdots & & \vdots \\
\Delta_{1k}b_{1k} & \cdots& \Delta_{jk}b_{jk} & \cdots & \Delta_{Jk}b_{Jk} \\
\vdots & & \vdots & & \vdots \\
\Delta_{1J}b_{1J} & \cdots& \Delta_{jJ}b_{jJ} & \cdots & \Delta_{JJ}b_{JJ}
\end{pmatrix}
```
This convenient notation allows us to rewrite the demand system for all goods as $D(\mathbf{p}) = \mathbf{a} + \mathbf{B^Tp}$, and the FOCs can be written as $\mathbf{0}  = \mathbf{a}+ \mathbf{B^Tp} + (\Delta \odot \mathbf{B})(\mathbf{p-c})$.

```python
D_p = a + B.T @ price
```

The solution to this set of equations is the Nash equilibrium vector of prices, $\mathbf{p^*} = (p_1^*,\dots, p_J^*)$ since each firm is choosing the prices of its products to maximise its profits given the prices charged by other firms. Rearranging this: 
```math
\begin{align*}
\mathbf{0}  &= \mathbf{a}+ \mathbf{B^Tp} + (\Delta \odot \mathbf{B})(\mathbf{p-c}) \\
\implies 
(\mathbf{B^T} + (\Delta \odot \mathbf{B}))\mathbf{p} &= -\mathbf{a} +(\Delta \odot \mathbf{B})\mathbf{c} \\
\implies 
\mathbf{p^*} &= (\mathbf{B^T} + (\Delta \odot \mathbf{B}))^{-1}(-\mathbf{a} +(\Delta \odot \mathbf{B})\mathbf{c})
\end{align*}
```

```python
Delta_hadamard_B = delta * B
inverse_term = np.linalg.inv(B.T + Delta_hadamard_B)
price = inverse_term @ (-a + (Delta_hadamard_B @ c))
```
The convenience of linear demand, is that all objects of interest can be computed easily for a given ownership structure. For example, once the nash prices are computed, demand is simply given by: $D(\mathbf{p}) = \mathbf{a} + \mathbf{B^Tp^*}$ and profits derived from each product are $(\mathbf{p^*-c}) \odot D(\mathbf{p})$. The profits of each firm can then be calculated by summing across the owned products (the corresponding row of the ownership matrix): $\Pi_i(\mathbf{p^*}) = \Delta^T_j ((\mathbf{p^*-c}) \odot D(\mathbf{p}))$

```python
profits = delta.T @ ((price - c) * D_p)
```

Now it's time to simulate! 
Consider a firm $i$ with a set of products $\mathcal{J_i^{pre}}$ and marginal cost $c_j^{pre}$ $\forall j \in \mathcal{J_i^{pre}}$. Post merger the corresponding values are given by $\mathcal{J_i^{post}}$ and marginal cost $c_j^{post}$. These ownership structures are captured in the matrix $\Delta$, so a unilateral effects simulation with linear demand amounts to calculating the NE prices in each case: 
```math
\mathbf{p^{pre}} = (\mathbf{B^T} + (\Delta^{pre} \odot \mathbf{B}))^{-1}(-\mathbf{a} +(\Delta^{pre} \odot \mathbf{B})\mathbf{c}^{pre}) \\
\mathbf{p^{post}} = (\mathbf{B^T} + (\Delta^{post} \odot \mathbf{B}))^{-1}(-\mathbf{a} +(\Delta^{post} \odot \mathbf{B})\mathbf{c}^{post})
```
Oftentimes data on marginal costs will not be available. To circumvent this issue, we can make the assumption of profit maxing pre-merger and extract the original cost. We recreate the analysis using this approach in appendix [A1: Cost Estimation](#a1-cost-estimation).

Consider a market with 3 firms $j = 1,2,3$, with each firm owning one product pre-merger. This means the ownership matrix is the $3\times3$ identity matrix. For now we also assume that no efficiencies result from the merger, such that $c^{pre}=c^{post}=1$. 
```python
delta = np.identity(3)
c = np.full((3, 1), 1)
```
Suppose further that the linear demand structure for the first product is given by:
```math
q_1 = 10 - 2p_1 + 0.3p_2 + 0.3p_3
```
and the others for $j=2,3$ are symmetrically defined with the coefficient -2 on own price, and the coefficients 0.3 on rival products' prices. In practice we don't know the values of these parameters and must calibrate these as well, this is a problem considered in later sections.

```python
a = np.full((3, 1), 10)
B = np.full((3, 3), 0.3)
np.fill_diagonal(B, -2)
```
Running our code from earlier now allows us to compute both the Nash equilibrium prices and profits in this environment. The table below describes prices (with profits in brackets) under several ownership structures: firstly where each firm owns one good, secondly when firms 1 and 3 merge (into firm 1), and thirdly a pure monopolist.

| Firm    | (1,1,1)         | (2,1)         | (3)           |
| --------| -------         | -----         | -------       |
| Firm 1  | $3.53 (12.80)$  | $3.76 (25.82)$| $4.07 (39.62)$|
| Firm 2  | $3.53 (12.80)$  | $3.56 (13.14)$|               |
| Firm 3  | $3.53 (12.80)$  |               |               |

#### Comparative Statics: Cost Savings
We might care about what level of cost saving is needed for a merger to result in a price increase. To do this, it is useful to define the cost change paramter as follows: $\gamma_i = \frac{c_i^{post}-c_i^{pre}}{c_i^{pre}}$. This parameter captures the degree of efficiency gain, with a lower value of $\gamma$ reflecting greater cost savings. We vary the parameter $\gamma_i$ between $-0.75$ and $0.75$ and simulate a merger between firms 1 and 3 as above, i.e. the (2,1) structure in the table above. The resulting prices and profits are plotted below. 

![Linear Cost Statics](../images/linear_cost_statics.png)

Note that the newly merged firm sells both products 1 and 3 at the same price, since the problem is symmetric in all goods. 

As the cost savings increase (and the efficiency term becomes negative) the price change decreases until the merger eventually results in a decrease in price. This is to say, that even in the case of efficiencies, price decreases do not necessarily follow. Absent cost savings the merger increases prices, thus a large cost saving is required for the post-merger price to fall.

### Appendix

#### A1: Cost estimation
Given the observed pre-merger prices $\mathbf{p}^{pre}$ and the ownership matrix $\Delta^{pre}$, we may solve the first order conditions for pre-merger marginal costs:

```math
\mathbf{c^{pre}} = (\Delta^{pre} \odot \mathbf{B})^{-1} (\mathbf{a}+(\mathbf{B^T} + (\Delta \odot \mathbf{B}))\mathbf{p^{pre}})
```
For example, if we have the simplest case of independent goods each owned by a seperate firm ($\mathbf{B}$ and $\Delta$ are diagonal), and for simplicity $\mathbf{a = 1}$:

```math
\begin{align*}
\mathbf{c^{pre}} &= (\Delta^{pre} \odot \mathbf{B})^{-1} (\mathbf{a}+(\mathbf{B^T} + (\Delta \odot \mathbf{B}))\mathbf{p^{pre}}) \\
&= \mathbf{I^{-1}(1}+2\mathbf{Ip^{pre}}) \\
&= \mathbf{1+2p^{pre}}
\end{align*}
```
For example, if prices pre-merger are 1 for each firm, the implied cost would be 3 for each firm. We can then plug this estimate into our formulas above, and proceed with simulation as normal.