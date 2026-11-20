The determinant is nonzero, so the linear system has a unique solution.

$$A=\begin{bmatrix}2&1\\1&2\end{bmatrix},\qquad \mathrm{det}(A)=4-1=3$$

Use the two by two inverse formula and multiply by the given vector.

$$A^{-1}=\frac{1}{3}\begin{bmatrix}2&-1\\-1&2\end{bmatrix}$$

$$x=A^{-1}b=\begin{bmatrix}\frac{5}{3}\\-\frac{1}{3}\end{bmatrix},\qquad Ax=\begin{bmatrix}3\\1\end{bmatrix}$$

The characteristic equation gives two distinct positive eigenvalues.

$$\mathrm{det}(A-\lambda I)=(2-\lambda)^2-1=(\lambda-3)(\lambda-1)=0$$

Normalize an eigenvector for each eigenvalue. The resulting columns are orthogonal.

$$q_1=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix},\qquad q_2=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}$$

$$Q=\frac{1}{\sqrt{2}}\begin{bmatrix}1&1\\1&-1\end{bmatrix},\qquad D=\begin{bmatrix}3&0\\0&1\end{bmatrix}$$

Direct multiplication verifies both orthonormality and the decomposition.

$$Q^TQ=I,\qquad QDQ^T=\begin{bmatrix}2&1\\1&2\end{bmatrix}=A$$
