# Soft Dynamic Time Warping

The choice of the distance metric for time series of variable sizes is an important task. Simple distance metrics that align i-th point of one time series with the i-th point of the other give low similarity score. Non-linear alignments match similar pieces even if they are shifted in time.

The Dynamic Time Warping approach seeks a minimum-cost alignment in the distance matrix under the constraints between two time series.  It can be computed by dynamic programming in quadratic time.  However, DTW is not differentiable, is sensitive to noise, and is known to lead to bad local optima when used as a loss.

$$\text{DTW(x, y)} := \smash{\displaystyle\min_{A \in A (n, m)}} [A, \Delta \text(x, y)]$$

Soft-DTW, proposed by M. Cuturi and M. Blondel in 2018, is a smoothed formulation of DTW. It addresses these issues of DTW. It replaces the minimum over alignments with a soft minimum, which has the effect of inducing a probability distribution over all alignments. Soft-DTW can still be computed by dynamic programming in the same complexity.

\begin{equation}
\text{dtw}^\gamma(\text{x, y}) := \text{min}^\gamma [(A, \Delta \text(x, y)), A \in A (n,m)],
\end{equation}

\begin{equation}
\text{min}^{\gamma} \{a_1, ..., a_n\} := \begin{cases} \min_{i \leq n} a_i, \gamma=0 \\ -\gamma \log \sum_{i=1}^n \exp^{-a_i/\gamma}, \gamma > 0 \end{cases}
\end{equation}

To ensure numerical stability, we computed operator $\text{min}^\gamma$ using the usual log-sum-exp stabilization trick, namely that

\begin{equation}
\log \sum_{i} \exp^{z_i} = (\max_j z_j) + \log \sum_{i} \exp^{z_i−max_j z_j}
\end{equation}

Our implementation repeats forward and backward recursion from the original paper. The autograd function computes the smoothed DTW for a distance matrix. Calling backward on the result gives a derivative of the DTW with respect to the distance matrix.

![picture](https://drive.google.com/uc?export=view&id=1Xmfdpc8JlYsn_erpDiCQs778YNqAEpvY)
