from itertools import combinations_with_replacement
from collections import Counter
from typing import List

def generate_polynomial_function_c(n: int, k: int) -> str:

    """
    Generate a C function for evaluating a polynomial of degree `k` with `n` variables.

    Parameters
    ----------
    n : int
        The number of variables in the polynomial.

    k : int
        The degree of the polynomial.

    Returns
    -------
    str
        A string representing a C function that evaluates the polynomial.
    """

        # I couldn't tell you how this works, but it does. An eaiser way might be to use the sympy library.

    terms = ""
    variables = [f'var{i}' for i in range(n)]
    coeff_counter = 0
    for degree in range(k + 1):
        for exponents in combinations_with_replacement(range(n), degree):
            exponent_count = Counter(exponents)

            explict_power = lambda var, exp: ' * '.join(f'{var}' for _ in range(exp)) if exp > 1 else var

            term = ' * '.join(f'({explict_power(variables[i],exponent_count[i])})' if exponent_count[i] > 1 else variables[i]
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

    v = [f'double var{i}' for i in range(n)]
    vars = ', '.join(v)
    func = f'''
        
double eval_poly({vars}){{
    return {terms} + intercept;
}}
    '''
    
    return func

def generate_log2_func() -> str:
    """
    Generate a C function for a reasonably fast and accurate approximation of log2 for float values.

    The approximation is accurate for numbers in the range [2, 2^24], with the following errors:
    - Max absolute error: 0.009708743539436071
    - Max relative error: 0.007738984406645373

    Returns
    -------
    str
        A string representing a C function that approximates log2 for float values.
    """


    func = f'''
inline float log2m (float val)
{{
   int * const    exp_ptr = (int *) (&val);
   int            x = *exp_ptr;
   const int      log_2 = ((x >> 23) & 255) - 128;
   x &= ~(255 << 23);
   x += 127 << 23;
   *exp_ptr = x;

   val = ((-1.0f/3) * val + 2) * val - 2.0f/3;   // (1)

   return (val + log_2);
}}

'''
    return func

def generate_integration_c(param_names: List[str]) -> str:
    """
    Generate a C function for evaluating the integral of a function using Gauss quadrature.

    Parameters
    ----------
    param_names : List[str]
        A list of parameter names used in the model.

    Returns
    -------
    str
        A string representing a C function that performs the integration.
    """

    func = f'''
    
typedef void (*Integrand)(double x, {', '.join(f'double ' for _ in param_names)}, double* out, int n, int i);
''' 

    func += f'''
void integrate(Integrand f, double a, double b, {', '.join(f'double {name}' for name in param_names)}, double* res){{\n\n'''
    
    func += f'''
    // Determine the number of points for the Gauss quadrature
    int expo = (int)(eval_poly({', '.join(f'log2m(max(limits[{i}][0],min(limits[{i}][1], {name})))' for i,name in enumerate(param_names))}) + 1);
    int n = (int)(pow(2, max(1, min(15, expo))));
    
    double *xg, *wg;
    get_gauss_points(n, &xg, &wg);
    
    // Perform the integration
    *res = 0;
    for (int i = 0; i < n; i++){{
        double temp;
        f(a + (b - a) * 0.5 * (xg[i] + 1), {', '.join(name for name in param_names)}, &temp, n, i);
        *res += temp * wg[i];
    }}
    *res *= (b - a) * 0.5;
  
    '''
    
    func += f'''
}}'''

    return func

def generate_integration2_c(param_names: List[str]) -> str:
    """
    Generate a C function for evaluating the integral of a function using Gauss quadrature, with two output results.

    Parameters
    ----------
    param_names : List[str]
        A list of parameter names used in the model.

    Returns
    -------
    str
        A string representing a C function that performs the integration and outputs two results.
    """

    func = f'''
    
typedef double (*Integrand2)(double x, {', '.join(f'double {name}' for name in param_names)}, double*, double*, int, int);

''' 

    func += f'''

void integrate2(Integrand2 f, double a, double b, {', '.join(f'double {name}' for name in param_names)}, double* res1, double* res2){{\n\n'''
    
    func += f'''
    // Determine the number of points for the Gauss quadrature
    int expo = (int)(eval_poly({', '.join(f'log2m(max(limits[{i}][0],min(limits[{i}][1], {name})))' for i,name in enumerate(param_names))}) + 1);
    int n = (int)(pow(2, max(1, min(15, expo))));
    
    int n = (int)(pow(2, max(1, min(15, expo))));

    double *xg, *wg;
    get_gauss_points(n, &xg, &wg);

    // Perform the integration
    *res1 = 0;
    *res2 = 0;

    for (int i = 0; i < n; i++){{
        double t1, t2;
        f(a + (b - a) * 0.5 * (xg[i] + 1), {', '.join(name for name in param_names)}, &temp, n, i);
        *res1 += t1 * wg[i];
        *res2 += t2 * wg[i];
    }}

    *res1 *= (b - a) * 0.5;
    *res2 *= (b - a) * 0.5;
  
    '''

    func += f'''
}}'''
    
    return func
    