"""Exact checks of the six frozen solutions used by the math glyph comparison."""

from sympy import I, Matrix, cos, diag, diff, exp, eye, limit, oo, simplify, sin, sqrt, symbols


def check_reference_math() -> dict:
    """Verify the reference calculations independently of the glyph renderer."""
    z = symbols("z", real=True)
    spectral = Matrix([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
    vectors = Matrix.hstack(
        Matrix([1, -sqrt(2), 1]) / 2,
        Matrix([1, 0, -1]) / sqrt(2),
        Matrix([1, sqrt(2), 1]) / 2,
    )
    eigenvalues = diag(2 - sqrt(2), 2, 2 + sqrt(2))
    orthogonality = (vectors.T * vectors - eye(3)).applyfunc(simplify)
    eigenvectors = (spectral * vectors - vectors * eigenvalues).applyfunc(simplify)

    a = symbols("a", positive=True)
    b = symbols("b", real=True)
    integral = (1 / (a - I * b)).as_real_imag()[1]
    derivative_integral = 2 * a * b / (a**2 + b**2) ** 2

    design = Matrix([[1, x] for x in range(-2, 3)])
    observations = Matrix([-3, -3, 4, 3, 9])
    residual = observations - design * Matrix([2, 3])
    sse = residual.dot(residual)
    slope_variance = (sse / 3) * (design.T * design).inv()[1, 1]

    t = symbols("t", nonnegative=True)
    omega = sqrt(3) / 2
    charge = 3 * (1 - exp(-t / 2) * (cos(omega * t) + sin(omega * t) / sqrt(3)))
    current = 2 * sqrt(3) * exp(-t / 2) * sin(omega * t)

    matrix = Matrix([[2, 1], [1, 2]])
    alpha, beta, rho, w, phi = symbols("alpha beta rho omega phi", real=True)
    quadratic = alpha**2 + rho * alpha * beta + beta**2
    checks = {
        "Spectral decomposition": {
            "characteristic_polynomial": simplify(
                (spectral - z * eye(3)).det() - (2 - z) * ((2 - z) ** 2 - 2)
            )
            == 0,
            "orthonormal_eigenbasis": orthogonality == Matrix.zeros(3) == eigenvectors,
        },
        "Parameter differentiation": {
            "imaginary_part_of_exponential_integral": simplify(integral - b / (a * a + b * b)) == 0,
            "minus_parameter_derivative": simplify(-diff(integral, a) - derivative_integral) == 0,
        },
        "Regression and uncertainty": {
            "normal_equations": (
                design.T * design == diag(5, 10) and design.T * observations == Matrix([10, 30])
            ),
            "residuals_and_standard_error": (
                residual == Matrix([1, -2, 2, -2, 1])
                and design.T * residual == Matrix.zeros(2, 1)
                and sse == 14
                and slope_variance * 15 == 7
                and 9 / slope_variance * 7 == 135
            ),
        },
        "Underdamped RLC response": {
            "ode_and_initial_conditions": (
                simplify(diff(charge, t, 2) + diff(charge, t) + charge) == 3
                and charge.subs(t, 0) == 0
                and current.subs(t, 0) == 0
            ),
            "charge_derivative_and_limits": (
                simplify(diff(charge, t) - current) == 0
                and limit(charge, t, oo) == 3
                and limit(current, t, oo) == 0
            ),
        },
        "Matrix and weighted mean": {
            "characteristic_polynomial": simplify((matrix - z * eye(2)).det() - ((2 - z) ** 2 - 1))
            == 0,
            "roots_and_weighted_mean": (
                set(matrix.eigenvals()) == {1, 3} and (1 * 2 + 3 * 4) * 2 == (1 + 3) * 7
            ),
        },
        "Derivatives and Greek parameters": {
            "partial_and_time_derivatives": (
                diff(quadratic, alpha) == 2 * alpha + rho * beta
                and diff(sin(w * t + phi), t) == w * cos(w * t + phi)
            ),
            "limits": limit(exp(-t / a), t, oo) == 0 and limit(sin(t) / t, t, 0) == 1,
        },
    }
    if not all(passed for case in checks.values() for passed in case.values()):
        raise ValueError(f"Frozen reference calculation failed: {checks}")
    return checks
