"""Full worked explanations for four existing benchmark questions; no private homework."""


def complete_homework() -> dict[str, str]:
    return {
        "core-elimination": r"""Solve the system by Gaussian elimination.

$$x+y+z=6$$

$$2x-y+z=3$$

$$x+2y-z=2$$

Write the augmented matrix. Subtract twice row one from row two, and row one from row three.

$$\begin{pmatrix}1&1&1&6\\2&-1&1&3\\1&2&-1&2\end{pmatrix}$$

$$\begin{pmatrix}1&1&1&6\\0&-3&-1&-9\\0&1&-2&-4\end{pmatrix}$$

The last two rows give the following equations.

$$-3y-z=-9,\quad y-2z=-4$$

Add three times the second equation to the first.

$$-7z=-21,\quad z=3$$

Back substitution gives the unique solution.

$$y=2z-4=2,\quad x=6-y-z=1$$

Check the solution in all three original equations.

$$1+2+3=6,\quad 2-2+3=3,\quad 1+4-3=2$$
""",
        "core-regression": r"""Fit a least squares line to the observations (1,2), (2,3), and (3,5).

The model is $y=a+bx$. First calculate the sample means.

$$\bar{x}=\frac{1+2+3}{3}=2,\quad\bar{y}=\frac{2+3+5}{3}=\frac{10}{3}$$

Compute the centered sums for the slope.

$$S_{xx}=(-1)^2+0^2+1^2=2$$

$$S_{xy}=(-1)\left(-\frac{4}{3}\right)+0\left(-\frac{1}{3}\right)+1\left(\frac{5}{3}\right)=3$$

The slope and intercept are therefore

$$b=\frac{S_{xy}}{S_{xx}}=\frac{3}{2},\quad a=\bar{y}-b\bar{x}=\frac{1}{3}$$

Evaluate the fitted line at each observed value of x.

$$\hat{y}_1=\frac{11}{6},\quad\hat{y}_2=\frac{10}{3},\quad\hat{y}_3=\frac{29}{6}$$

The residuals are $e_i=y_i-\hat{y}_i$.

$$e_1=\frac{1}{6},\quad e_2=-\frac{1}{3},\quad e_3=\frac{1}{6}$$

They sum to zero. The residual sum of squares is

$$SSE=\frac{1}{36}+\frac{1}{9}+\frac{1}{36}=\frac{1}{6}$$
""",
        "core-improper-integral": r"""
Evaluate the improper integral and determine whether it converges.

$$I=\int_1^{\infty}x^{-3}\,dx$$

The upper endpoint is infinite. Replace it by a finite value b before applying the power rule.

$$I=\lim_{b\to\infty}\int_1^b x^{-3}\,dx$$

Check an antiderivative by differentiation.

$$F(x)=-\frac{1}{2x^2},\quad F'(x)=x^{-3}$$

Evaluate the finite integral using its two endpoints.

$$\int_1^b x^{-3}\,dx=-\frac{1}{2b^2}+\frac{1}{2}$$

Since $b^{-2}$ tends to zero, the limit exists.

$$I=\frac{1}{2}$$

The integrand is positive, and this finite positive result is consistent with convergence.

For a finite cutoff B, the remaining tail is

$$R(B)=\int_B^{\infty}x^{-3}\,dx=\frac{1}{2B^2}$$

For example, $B=10$ leaves a tail of $R(10)=1/200=0.005$.

The integral from one to infinity of $x^{-p}$ converges for $p>1$; here $p=3$.
""",
        "core-rl-transient": r"""
A 12 V source drives a series circuit with resistance 6 ohm and inductance 3 H.

The initial current is zero. Find the current and the time constant.

Kirchhoff's voltage law gives the differential equation

$$L\frac{di}{dt}+Ri=V$$

Substitute the circuit values and divide by the inductance.

$$3\frac{di}{dt}+6i=12,\quad\frac{di}{dt}+2i=4$$

The complementary solution decays exponentially. A constant particular solution is two amperes.

$$i(t)=2+Ce^{-2t}$$

Use the initial condition to find the remaining constant.

$$i(0)=0=2+C,\quad C=-2$$

Hence the current and the time constant are

$$i(t)=2(1-e^{-2t})\ \mathrm{A},\quad t\ge0$$

$$\tau=\frac{L}{R}=\frac{3}{6}=0.5\ \mathrm{s}$$

Check both the differential equation and the limiting current.

$$3(4e^{-2t})+6(2-2e^{-2t})=12$$

$$\lim_{t\to\infty}i(t)=2\ \mathrm{A}$$
""",
    }
