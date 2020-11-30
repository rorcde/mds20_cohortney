Soft Dynamic Time Warping

The choice of the distance metric for time series of variable sizes is an important task. Simple distance metrics that align i-th point of one time series with the i-th point of the other give low similarity score. Non-linear alignments match similar pieces even if they are shifted in time.

The Dynamic Time Warping approach seeks a minimum-cost alignment in the distance matrix under the constraints between two time series.  It can be computed by dynamic programming in quadratic time.  However, DTW is not differentiable, is sensitive to noise, and is known to lead to bad local optima when used as a loss.

Soft-DTW, proposed by M. Cuturi and M. Blondel in 2017, is a smoothed formulation of DTW. It addresses these issues of DTW. It replaces the minimum over alignments with a soft minimum, which has the effect of inducing a probability distribution over all alignments. Soft-DTW can still be computed by dynamic programming in the same complexity.

To ensure numerical stability, we computed operator min^gamma using the usual log-sum-exp stabilization trick, namely that

Our implementation repeats forward and backward recursion from the original paper. The autograd function computes the smoothed DTW for a distance matrix. Calling backward on the result gives a derivative of the DTW with respect to the distance matrix.