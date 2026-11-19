# ruff: noqa: E501
"""Authored difficult render-only references; never part of personal training."""


def cases() -> dict[str, str]:
    return {
        "Spectral decomposition": r"""Consider the symmetric tridiagonal matrix below. Find an orthonormal eigenbasis and use it to describe the solution of the differential system. Symmetry is useful because eigenvectors belonging to different eigenvalues are orthogonal.

$$A=\begin{pmatrix}2&1&0\\1&2&1\\0&1&2\end{pmatrix},\qquad \frac{d\vec{x}}{dt}=-A\vec{x}$$

Expanding the determinant along the first row gives the characteristic equation. The three distinct roots are positive, so every component of the solution decays. This conclusion follows from the spectrum rather than from inspection of individual matrix entries.

$$\mathrm{det}(A-\lambda I)=(2-\lambda)\left((2-\lambda)^2-2\right)=0$$

$$\lambda_1=2-\sqrt{2},\quad\lambda_2=2,\quad\lambda_3=2+\sqrt{2}$$

$$u_1=\frac{1}{2}\begin{pmatrix}1\\-\sqrt{2}\\1\end{pmatrix},\quad u_2=\frac{1}{\sqrt{2}}\begin{pmatrix}1\\0\\-1\end{pmatrix},\quad u_3=\frac{1}{2}\begin{pmatrix}1\\\sqrt{2}\\1\end{pmatrix}$$

Let $Q=(u_1,u_2,u_3)$ and let $c_k=u_k^{\mathrm{T}}\vec{x}(0)$. Orthogonality separates the differential equations into three scalar equations. The different exponential factors determine the relative decay of their coefficients.

$$Q^{\mathrm{T}}Q=I,\qquad A=Q\,\mathrm{diag}(\lambda_1,\lambda_2,\lambda_3)Q^{\mathrm{T}}$$

$$\vec{x}(t)=\sum_{k=1}^{3}c_k e^{-\lambda_k t}u_k,\qquad \lim_{t\to\infty}\vec{x}(t)=0$$
""",
        "Parameter differentiation": r"""For positive a and real b, evaluate the following improper integrals. The exponential factor provides sufficient decay for convergence. We first calculate a simpler integral and then differentiate with respect to the parameter.

$$I(a,b)=\int_0^\infty e^{-ax}\sin(bx)\,dx,\qquad J(a,b)=\int_0^\infty x e^{-ax}\sin(bx)\,dx$$

Integration by parts introduces a companion cosine integral. The boundary term vanishes at infinity. At zero, the sine term vanishes while the cosine term contributes one. Keeping these different boundary contributions separate avoids a sign error.

$$K(a,b)=\int_0^\infty e^{-ax}\cos(bx)\,dx$$

$$I=\frac{b}{a}K,\qquad K=\frac{1}{a}-\frac{b}{a}I$$

$$I\left(1+\frac{b^2}{a^2}\right)=\frac{b}{a^2},\qquad I(a,b)=\frac{b}{a^2+b^2}$$

On every compact interval of positive a, an integrable exponential bound permits differentiation under the integral sign. The derivative brings down a factor of negative x. The minus sign must therefore be retained when identifying the second integral.

$$J(a,b)=-\frac{\partial I}{\partial a}=\frac{2ab}{(a^2+b^2)^2}$$

$$I(2,3)=\frac{3}{13},\qquad J(2,3)=\frac{12}{169}$$

Both expressions are odd functions of b, as required by the sine factor. At b equal to zero both integrands vanish. These checks test the algebra independently of the integration procedure.
""",
        "Regression and uncertainty": r"""Fit an intercept and a slope to five observations. Use the residual degrees of freedom to estimate the error variance, and express a two-sided confidence interval for the slope under independent normal errors with constant variance.

$$x=(-2,-1,0,1,2),\qquad y=(-3,-3,4,3,9)$$

The centered explanatory values simplify the normal equations. The sum of the squared x values is ten, and the cross product with y is thirty. The intercept is the sample mean of y because the sample mean of x is zero.

$$X^{\mathrm{T}}X=\begin{pmatrix}5&0\\0&10\end{pmatrix},\qquad X^{\mathrm{T}}y=\begin{pmatrix}10\\30\end{pmatrix}$$

$$\hat{\beta}=(X^{\mathrm{T}}X)^{-1}X^{\mathrm{T}}y=\begin{pmatrix}2\\3\end{pmatrix},\qquad \hat y_i=2+3x_i$$

The residuals are not all small, even though their sum and their cross product with x are zero. These orthogonality conditions verify the least squares calculation. They do not establish the normality or independence assumptions.

$$e=(1,-2,2,-2,1),\qquad \sum_{i=1}^{5}e_i=0,\quad\sum_{i=1}^{5}x_i e_i=0$$

$$SSE=14,\qquad s^2=\frac{14}{5-2}=\frac{14}{3},\qquad \mathrm{SE}(\hat\beta_1)=\sqrt{\frac{7}{15}}$$

The interval uses the Student quantile with three degrees of freedom. The corresponding statistic for a zero slope divides the estimated slope by its standard error. A numerical decision would compare this statistic with the chosen critical value.

$$\beta_1:\quad 3\pm t_{0.975,3}\sqrt{\frac{7}{15}},\qquad T=\frac{3}{\sqrt{7/15}},\qquad T^2=\frac{135}{7}$$
""",
        "Underdamped RLC response": r"""A series RLC circuit has inductance one henry, resistance one ohm, and capacitance one farad. At time zero a constant source of three volts is applied. Initially the capacitor is uncharged and the current is zero. Find the charge and the current for nonnegative time.

$$q''+q'+q=3,\qquad q(0)=0,\qquad q'(0)=0,\qquad i=q'$$

The characteristic roots form a complex conjugate pair. A constant particular solution is three. The negative real part of each root gives exponential damping, while the imaginary part determines the oscillation frequency.

$$r^2+r+1=0,\qquad r=-\frac{1}{2}\pm\frac{\sqrt{3}}{2}\,\mathrm{i},\qquad\omega=\frac{\sqrt{3}}{2}$$

$$q(t)=3+e^{-t/2}\left(C_1\cos(\omega t)+C_2\sin(\omega t)\right)$$

Substitution of the initial charge gives the first coefficient. Differentiation and substitution of the initial current give the second. The two terms involving the cosine cancel when the final charge expression is differentiated.

$$C_1=-3,\qquad -\frac{C_1}{2}+\omega C_2=0,\qquad C_2=-\sqrt{3}$$

$$q(t)=3\left[1-e^{-t/2}\left(\cos(\omega t)+\frac{\sin(\omega t)}{\sqrt{3}}\right)\right]$$

$$i(t)=2\sqrt{3}\,e^{-t/2}\sin(\omega t)$$

Finally, the charge approaches three coulombs and the current approaches zero. The damped oscillation can temporarily reverse the current direction. This behavior is consistent with energy exchange between the inductor and the capacitor while resistance dissipates energy.

$$\lim_{t\to\infty}q(t)=3,\qquad\lim_{t\to\infty}i(t)=0$$
""",
    }
