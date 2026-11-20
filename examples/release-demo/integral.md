Polynomial division separates the integrand into terms with simple antiderivatives.

$$\frac{x^2}{1+x}=x-1+\frac{1}{1+x}$$

Integrate each term and check the result by differentiation.

$$F(x)=\frac{x^2}{2}-x+\log(1+x)$$

$$F'(x)=x-1+\frac{1}{1+x}=\frac{x^2}{1+x}$$

Now evaluate at the limits.

$$I=\int_0^1\frac{x^2}{1+x}\,dx=F(1)-F(0)=\log 2-\frac{1}{2}$$

On this interval, the denominator lies between one and two. Multiplying the
reciprocal bounds by a nonnegative numerator preserves their order.

$$\frac{x^2}{2}\leq\frac{x^2}{1+x}\leq x^2,\qquad 0\leq x\leq1$$

Integrating the bounds gives a useful independent check.

$$\frac{1}{6}\leq I\leq\frac{1}{3}$$

The exact answer is positive and its decimal value lies inside this interval.

$$I\approx0.193147,\qquad \frac{1}{6}<\log2-\frac{1}{2}<\frac{1}{3}$$
