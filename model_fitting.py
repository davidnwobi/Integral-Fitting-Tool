import numpy as np
import numpy.typing as npt
from itertools import product
from kron.kron import get_gauss_kronrod_points
import typing as tp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from typing import Callable, List, Tuple, Dict, Union, Optional
import numpy as np
import os
import plotly.graph_objects as go
from c_code_generation import generate_polynomial_function_c, generate_integration_c, generate_log2_func, generate_integration2_c
from py_code_generation import generate_polynomial_function_py, generate_integration_py
from joblib import Parallel, delayed


n_kronrod = np.power(2, np.arange(1, 16)) # This are the points available for the Gauss-Kronrod quadrature rule


def rel_error_kronrod(
    n: int,
    func: tp.Callable[[np.float64, *tp.Tuple[np.float64, ...]], np.float64],
    a: np.float64,
    b: np.float64,
    params: tp.Tuple[np.float64, ...])->np.float64:

    """
    Compute the relative error for a Gauss-Kronrod quadrature rule with n points for a given function and parameter space.

    Parameters
    ----------
    n : `int`
        The number of points in the quadrature rule.

    func : `Callable`
        The function to be integrated.

    a : `float64`
        The lower bound of the integral.

    b : `float64`
        The upper bound of the integral.

    params : `Tuple(float64, ...)`
        The parameters for the function.

    Returns
    -------
    `float64`
    """

    xk, wk, _, wg = get_gauss_kronrod_points(n)

    def integrate(points, weights, weights_gauss)->np.float64:
        """Integrate the function using the Gauss-Kronrod quadrature rule and return the higher and lower order estimates."""
        y = (b-a)*(points+1)/2.0 + a
        feval = func(y, *params)
        higher_order = (b-a)/2.0 * np.sum(weights*feval, axis=-1)
        lower_order = (b-a)/2.0 * np.sum(weights_gauss*feval[1:-1:2], axis=-1)
        return higher_order, lower_order

    sol_h, sol_l = integrate(xk, wk, wg)

    if sol_h == 0 and sol_l == 0:
        return np.finfo(np.float64).eps
    return np.abs((sol_h-sol_l)/sol_h) if sol_h != 0 else sol_l

def estimate_error(
        param_prod: Tuple[float, ...],
        func: Callable[[float, *Tuple[float, ...]], float], 
        a: float,
        b: float,
        rtol: float,
        full_output=False) -> Tuple[int, float]:
    """
    Estimate the number of points needed for the Gauss-Kronrod quadrature rule to achieve the desired relative error.

    Parameters
    ----------
    param_prod : `Tuple[float, ...]`
        The parameters for the function.

    func : `Callable[[float, Tuple[float, ...]], float]`
        The function to be integrated.

    a : `float`
        The lower bound of the integral.

    b : `float`
        The upper bound of the integral.

    rtol : `float`
        The desired relative error.

    Returns
    -------
    Tuple[int, float]
        A tuple where the first element is the number of points needed to achieve the desired relative error, 
        and the second element is the computed relative error.
    """
    out = 0
    rel_err = 0.0
    # Find the minimum n such that the relative error is less than the tolerance
    for i, n in enumerate(n_kronrod):
        rel_err = rel_error_kronrod(n, func, a,b, param_prod)
        
        if rel_err < rtol:
            n = i
            out = n
            print("Relative error for Param", param_prod, " is ", rel_err, "with ", 2**(n+1) , " points")
            break
    else:
        print("Kronrod failed for Param", param_prod, " with error ", rel_err, ". Setting n to the maximum value, ", 2**(len(n_kronrod)), " points")
        n = len(n_kronrod)-1
        out = n
    
    return out, rel_err

def compute_tuned_quad_dict(
    func: Callable[[float, *Tuple[float, ...]], float],
    a: float,
    b: float, 
    param_prod: List[Tuple[float, ...]],
    rtol: float,
    full_output=False,
    n_jobs: int = -2) -> Tuple[Dict[Tuple[float, ...], int], Dict[Tuple[float, ...], float]]:

    """
    Compute the tuned quadrature points for the given function and parameter space.

    Parameters  
    ----------
    func : `Callable[[float, Tuple[float, ...]], float]`
        The function to be integrated.

    a : `float`
        The lower bound of the integral.

    b : `float`
        The upper bound of the integral.

    param_prod : `List[Tuple[float, ...]]`
        A list of parameter tuples to be passed to the function.

    rtol : `float`
        The desired relative error.

    full_output : `bool`, optional
        If `True`, the function will return the computed relative errors along with the quadrature points.

    n_jobs : `int`, optional
        The number of jobs to run in parallel. Default is `-2`, which uses all but one core.

    Returns
    -------
    Tuple[Dict[Tuple[float, ...], int], Dict[Tuple[float, ...], float]]
        A tuple containing two dictionaries:
        - The first dictionary maps parameter tuples to the number of quadrature points needed.
        - The second dictionary maps parameter tuples to the computed relative errors (returns a filled dictionary if `full_output` is `True` else empty).
    """


    tuned_quad = dict()
    tuned_quad_rel_err = dict()

    out = np.array(Parallel(n_jobs=n_jobs)(delayed(estimate_error)(param, func, a,b, rtol, full_output) for param in param_prod))

    for k, param in enumerate(param_prod):
        tuned_quad[param] = tuned_quad[param] = n_kronrod[int(out[k][0])]
        if full_output:
            tuned_quad_rel_err[param] = out[k][1]

    return tuned_quad, tuned_quad_rel_err

def tune_quadrature(
        func: Callable[[float, Dict[str, float]], float],
        a: float,
        b: float,
        reg_params_list: List[npt.NDArray],
        rtol: float,
        full_output=False,
        n_jobs: int = -2) -> Tuple[Dict[Tuple[float, ...], int], Dict[Tuple[float, ...], float]]:
    
    """
    Tune the quadrature points for the given function and parameter space.

    Parameters
    ----------
    func : `Callable[[float, Dict[str, float]], float]`
        The function to be integrated.

    a : `float`
        The lower bound of the integral.

    b : `float`
        The upper bound of the integral.

    reg_params_list : `List[NDArray]`
        A list of arrays representing the regularization parameters.

    rtol : `float`
        The desired relative error.

    full_output : `bool`, optional
        If `True`, the function will return the computed relative errors along with the quadrature points.

    n_jobs : `int`, optional
        The number of jobs to run in parallel. Default is `-2`, which uses all but one core.

    Returns
    -------
    Tuple[Dict[Tuple[float, ...], int], Dict[Tuple[float, ...], float]]
        A tuple containing two dictionaries:
        - The first dictionary maps parameter tuples to the number of quadrature points needed.
        - The second dictionary maps parameter tuples to the computed relative errors (only returned if `full_output` is `True`).
    """
    
    param_prod = list(product(*reg_params_list))
    return compute_tuned_quad_dict(
        func=func, 
        a=a, 
        b=b, 
        param_prod=param_prod, 
        rtol=rtol,
        full_output=full_output,
        n_jobs=n_jobs)



def fit_dict(
        tuned_quad_dict: Dict[Tuple[float, ...], int], 
        degree: int,
        plot_param_space: bool = False) -> Tuple[int, int, np.ndarray, float, Pipeline]:
    '''
    Fit a polynomial to the tuned quadrature dictionary.
    
    Parameters
    ----------
    tuned_quad_dict : Dict[Tuple[float, ...], int]
        The tuned quadrature dictionary mapping parameter tuples to quadrature points.
    
    degree : int
        The degree of the polynomial to fit.
        
    plot_param_space : bool, default=False
        If True, plot the parameter space for 2 parameter models.
    
    Returns
    -------
    Tuple[int, int, np.ndarray, float, Pipeline]
        A tuple containing:
        - `n_params` (int): Number of parameters in the keys of the dictionary.
        - `degree` (int): The degree of the polynomial fitted.
        - `coeff` (np.ndarray): Coefficients of the fitted polynomial.
        - `intercept` (float): Intercept of the fitted polynomial.
        - `model` (Pipeline): The fitted polynomial model pipeline.
    '''
    n_params = len(list(tuned_quad_dict.keys())[0])

    
    # Fit an n-dimennsional polynomial of degree `degree` to the tuned quadrature dictionary
    features = np.vstack(np.array([np.log2(np.array(k)) for k in tuned_quad_dict.keys()])) # log2 of the parameters
    values = np.log2(np.array(list(tuned_quad_dict.values())))
    
    

    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(features, values)
    prediction = model.predict(features)

    
    coeff = model.named_steps['linearregression'].coef_
    intercept = model.named_steps['linearregression'].intercept_
    
    # Look at the parameter space for 2 parameter models
    if n_params == 2 and plot_param_space:
        x = features[:, 0]
        y = features[:, 1]
        z = values.reshape((int(np.sqrt(len(values))), int(np.sqrt(len(values)))))
        fig = go.Figure()
        fig = go.Figure(data=[go.Surface(z=z, colorscale='Viridis')])
        

        prediction = np.array([(max(1,min(15, int(p)+1))) for p in prediction])
        prediction = prediction.reshape((int(np.sqrt(len(values))), int(np.sqrt(len(values)))))
        
        fig.add_trace(go.Surface(z=prediction, colorscale='Turbo', opacity=0.6))
        fig.update_geos(projection_type="orthographic")
        fig.show()

    return n_params, degree, coeff, intercept, model


def save_model_py_file(
        model_name: str, 
        param_names: List[str], 
        limits: List[Tuple[float, float]],
        poly_degree: int,
        coeffs: npt.NDArray[np.float64],
        intercept: float):
    """
    Save a Python file that defines a polynomial model and an integration function.

    Parameters
    ----------
    model_name : str
        The base name for the Python file to be created.

    param_names : List[str]
        A list of parameter names used in the model.

    limits : List[Tuple[float, float]]
        A list of tuples defining the lower and upper limits for each parameter.

    poly_degree : int
        The degree of the polynomial to be used in the model.

    coeffs : np.ndarray
        The coefficients of the polynomial.

    intercept : float
        The intercept of the polynomial.
    """

    py_file_name = model_name + "_integration.py"

    py_script = f"""

import numpy as np
import typing as tp
from collections.abc import Sequence
import sys
sys.path.insert(0, "../kron")
from kron import get_gauss_points

__all__ = ["integrate"]
coeffs = {np.array(np.array2string(coeffs, separator=",", precision=10))}
intercept  = {intercept}
limits = [{', '.join(map(str, limits))}]

"""
    
    # Generate the polynomial function and the integration function
    py_script += generate_polynomial_function_py(len(param_names), poly_degree)
    py_script += generate_integration_py(param_names)

    # Write the script to a Python file
    with open(py_file_name, "w") as f:
        f.write(py_script)
        
def save_model_c_file(
        model_name: str, 
        param_names: List[str], 
        limits: List[Tuple[float, float]],
        poly_degree: int,
        coeffs: npt.NDArray[np.float64],
        intercept: float):
    """
    Save a C file that defines a polynomial model and an integration function.

    Parameters
    ----------
    model_name : str
        The base name for the C file to be created.

    param_names : List[str]
        A list of parameter names used in the model.

    limits : List[Tuple[float, float]]
        A list of tuples defining the lower and upper limits for each parameter.

    poly_degree : int
        The degree of the polynomial to be used in the model.

    coeffs : npt.NDArray[np.float64]
        The coefficients of the polynomial.

    intercept : float
        The intercept of the polynomial.
    """


    c_file_name = model_name + "_integration.c"
    c_script = f'''

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

const double coeffs[] = {{{", ".join(map(str, coeffs))}}};
const double intercept = {intercept};
const double limits[][2] = {{{", ".join(f'{{{l[0]}, {l[1]}}}' for l in limits)}}};

'''
    c_script += generate_log2_func()
    c_script += generate_polynomial_function_c(len(param_names), poly_degree)
    c_script += generate_integration_c(param_names)
    c_script += generate_integration2_c(param_names)
    
    with open(c_file_name, "w") as f:
        f.write(c_script)





def fit_model(
        model_name: str, 
        integrand: tp.Callable[[np.float64, *tp.Tuple[np.float64, ...]], np.float64],
        a: float,
        b: float,
        param_names: List[str],
        reg_param_list: List[npt.NDArray[np.float64]], 
        rtol: np.float64,
        degree: Optional[int] = 2,
        update: bool = False,
        full_output: bool = False,
        n_jobs: int = -2,
        plot_param_space: bool = False
    ) -> Union[None, Tuple[Dict[Tuple[float, ...], int], Dict[Tuple[float, ...], float], object]]:
    """
    Fit a model to the given integrand and save the generated model to Python and C files.

    Parameters
    ----------
    model_name : str
        The base name for the model files to be created.

    integrand : callable
        The function to be integrated.

    a : float
        The lower bound of the integration.

    b : float
        The upper bound of the integration.

    param_names : List[str]
        A list of parameter names used in the model.

    reg_param_list : List[npt.NDArray[np.float64]]
        A list of numpy arrays defining the parameter ranges for each parameter.

    rtol : np.float64
        The relative tolerance for the quadrature.

    degree : Optional[int], default=2
        The degree of the polynomial to be fitted.

    update : bool, default=False
        If True, overwrite existing model files.

    full_output : bool, default=False
        If True, return additional output such as the fitted model and relative errors.
        
    n_jobs : int, default=-2
        The number of parallel jobs to run.
        
    plot_param_space : bool, default=False
        If True, plot the parameter space for 2 parameter models.

    Returns
    -------
    Union[None, Tuple[Dict[Tuple[float, ...], int], Dict[Tuple[float, ...], float], object]]
        If full_output is True, returns a tuple containing:
        - tuned_quad_dict : Dict[Tuple[float, ...], int]
            The tuned quadrature dictionary.
        - tuned_quad_rel_err : Dict[Tuple[float, ...], float]
            The relative errors associated with the tuned quadrature points.
        - model : Pipeline
            The fitted polynomial model.
        Otherwise, returns None.
    """

    file_name = f'{model_name}_integration.py'
    if not os.path.exists(file_name) or update:   
        limits = [(rg[0], rg[-1]) for rg in reg_param_list]

        # Compute the required number of quadrature points for each parameter tuple then fit a polynomial to the results
        tuned_quad_dict, tuned_quad_rel_err = tune_quadrature(integrand, a, b, reg_param_list, rtol=rtol, full_output=full_output, n_jobs=n_jobs)
        n_params, degree, coeffs, intercept, model = fit_dict(tuned_quad_dict, degree, plot_param_space)

        
        save_model_py_file(model_name, param_names, limits, degree, coeffs, intercept)
        save_model_c_file(model_name, param_names, limits, degree, coeffs, intercept)
        
        if full_output:
            return tuned_quad_dict, tuned_quad_rel_err, model

