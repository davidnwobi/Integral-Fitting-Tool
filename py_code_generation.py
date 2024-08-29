from itertools import combinations_with_replacement
from collections import Counter
from typing import List

def generate_polynomial_function_py(n: int, k: int) -> str:
    """
    Generate a Python function for evaluating a polynomial of degree `k` with `n` variables.

    Parameters
    ----------
    n : int
        The number of variables in the polynomial.

    k : int
        The degree of the polynomial.

    Returns
    -------
    str
        A string representing a Python function that evaluates the polynomial.
    """

    # I couldn't tell you how this works, but it does. An eaiser way might be to use the sympy library.

    terms = ""
    variables = [f'vars[{i}]' for i in range(n)]
    coeff_counter = 0
    for degree in range(k + 1):
        for exponents in combinations_with_replacement(range(n), degree):
            exponent_count = Counter(exponents)
            term = ' * '.join(f'{variables[i]} ** {exponent_count[i]}' if exponent_count[i] > 1 else variables[i]
                           for i in range(n) if exponent_count[i] > 0)
            if not term:
                term = '1'
                term += f' * coeffs[{coeff_counter}]'
                terms += f'{term}'
                coeff_counter += 1
            else:
                term += f' * coeffs[{coeff_counter}]'
                coeff_counter += 1
                terms += f' + {term}'

    func = f'''
        
def eval_poly(vars: Sequence[float]) -> float:
    return {terms} + intercept
    '''

    return func

def generate_integration_py(param_names: List[str]) -> str:
    """
    Generate a Python function for evaluating the integral of a function using Gauss quadrature.

    Parameters
    ----------
    param_names : List[str]
        A list of parameter names used in the model.

    Returns
    -------
    str
        A string representing a Python function that performs the integration.
    """
    


    func = f'''

lb: int = 1
ub: int = 15

def integrate(f: tp.Callable[[float, tp.Tuple[float, ...]], float], a: float, b: float, params: tp.Tuple[float, ...] = ()) -> float:

    expo = int(eval_poly([np.log2(max(limit[0], min(limit[1],param))) for limit, param in zip(limits,params)]) + 1)   
    n = int(pow(2, max(lb, min(ub, expo))))

    xg, wg = get_gauss_points(n)

    y = (b-a)*(xg+1)/2 + a
    return (b-a)/2 * np.sum(wg*f(y, *params))
    '''

    return func
