# Integral Fitting Tool

Integral Fitting is a tool designed to determine the number of sampling points required to compute a **1D** integral using a Gauss–Legendre quadrature. This tool is particularly useful in scenarios where a specific integral needs to be computed multiple times with varying parameters, such as in SasView.

## Overview

To determine the number of points, the Gauss-Kronrod rule is used to estimate the accuracy for a provided parameter. The number of points is then incremented from 2 to 32,768 in powers of 2. The estimated points are subsequently fitted to a polynomial.

Note: Due to the nature of the problem, the fitting is done using the base-2 logarithm of the parameter and the sampling points. Once this fitting is complete, determining the number of points requires evaluating the polynomial and raising the result to the power of 2, converting it into an integer.

## Example Usage

The main function of interest is `fit_model`:

```python
def fit_model(model_name: str, integrand, a, b, param_names, reg_param_list, rtol, degree=2, update=False, full_output=False, n_jobs: int = -2, plot_param_space: bool = False)
```

This function calculates the required points and fits a polynomial model to them.

## Function Parameters

- **model_name**: The name to append to the output files.
- **integrand**: A function that behaves similarly to the target integrand. The integrand should ignore scaling factors when being created. Typically, its form is:

```python
tp.Callable[[np.float64, *tp.Tuple[np.float64, ...]], np.float64]
```

It's recommended to create the integrand in a generic fashion. For example, for a cylinder model, the integrand would be:

```python
def integrand(x, q, L, R):
    return (np.sinc(q * L / 2 * np.cos(x) / np.pi) * J1x_x(q * R * np.sin(x))) ** 2 * np.sin(x)
```

However, a simpler and more efficient approach would be:

```python
def integrand(x, A, B):
    return (np.sinc(A * np.cos(x) / np.pi) * J1x_x(B * np.sin(x))) ** 2 * np.sin(x)
```

Here, you only need to worry about two parameters instead of three and a sensible parameter space is easier to define.

- **a, b**: The integration range. For the cylinder model example, this would be 0 to np.pi.
- **param_names**: The names of the parameters.
- **reg_param_list**: The parameter space to evaluate.
- **rtol**: Relative tolerance. Note that the Gauss-Kronrod rule only provides an estimate of the error, so the specified tolerance does not guarantee that tolerance will be met.
- **degree**: The degree of the polynomial to fit. Default is 2.
- **update**: If set to True, the function will overwrite existing output files. Default is False.
- **full_output**: If set to True, the function returns a dictionary with each parameter tested and the corresponding number of points, a dictionary with each parameter tested and the relative error estimated by the Gauss-Kronrod rule, and the model itself. Default is False.
- **n_jobs**: Number of jobs for parallel processing. Default is -2.
- **plot_param_space**: If set to True, the parameter space is plotted. Default is False.

## Example Setup for Cylinder Model

```python

import numpy as np
from scipy.special import j1
from model_fitting import fit_model

a = 0
b = np.pi / 2

def J1x_x(x):
    return np.where(x != 0, j1(x) / x, 0.5)

def integrand(x, A, B):
    return (np.sinc(A * np.cos(x) / np.pi) * J1x_x(B * np.sin(x))) ** 2 * np.sin(x)

n = 40
rtol = 1e-3
reg_params2 = [np.geomspace(1, 1000000, n), np.geomspace(1, 1000000, n)]
params_names = ["A", "B"]

fit_model("cylinder_high_res", integrand, a, b, params_names, reg_params2, rtol)

```

This setup will generate two files: `cylinder_high_res_integration.c` and `cylinder_high_res_integration.py`.

## Key Components of the Output Files

1. `eval_poly` Function: This function uses `coeffs` and `intercept` to evaluate the polynomial for the **base-2 logarithim** of the required points. it expects the parameters to be **base-2 logarithms** of the actual parameters.

   ### Python:


   ```python
   coeffs = [0.0, 0.4447375199, 0.519781895, 0.0090581797, -0.0370485252, 0.0066942664]
   intercept = 2.2240044191383124

   def eval_poly(vars: Sequence[float]) -> float:
       return (vars[0] * coeffs[1] + vars[1] * coeffs[2] + vars[0] ** 2 * coeffs[3] + 
               vars[0] * vars[1] * coeffs[4] + vars[1] ** 2 * coeffs[5] + intercept)
   ```

   ### C:

   ```c
   const double coeffs[] = {0.0, 0.44473751986882, 0.5197818950131304, 0.00905817971635138, -0.0370485251727869, 0.006694266375182011};
   const double intercept = 2.2240044191383124;

   double eval_poly(double var0, double var1) {
       return (var0 * coeffs[1] + var1 * coeffs[2] + (var0 * var0) * coeffs[3] +
               var0 * var1 * coeffs[4] + (var1 * var1) * coeffs[5] + intercept);
   }
   ```
2. Integration Function: Example of how integration would work in practice.

   ```python
   lb = 1
   ub = 15

   def integrate(f, a: float, b: float, params) -> float:
       # Convert to log2 first
       expo = int(eval_poly([np.log2(max(limit[0], min(limit[1], param))) 
                           for limit, param in zip(limits, params)]) + 1)
       # Clamp the exponent to the available range, then raise 2 to it.
       n = int(pow(2, max(lb, min(ub, expo))))

       xg, wg = get_gauss_points(n)
       y = (b - a) * (xg + 1) / 2 + a
       return (b - a) / 2 * np.sum(wg * f(y, *params))

   ```

## Optional Features

- **Overwriting Files**: Use `update=True` in fit_model to overwrite existing output files.
- **Polynomial Degree**: The model is fitted to an n-dimensional quadratic polynomial by default. Use the `degree` parameter to change this.
- **Plotting Parameter Space**: For a 2-parameter model, set `plot_param_space=True` to visualize the parameter space.
- **Parallel Processing**: By default, the parameter space is evaluated in parallel using joblib.
- **Full Output Access**: Setting `full_output=True` in fit_model provides access to the data generated by the model, including tested parameters, relative errors, and the fitted model.

Example usage of `full_output`:

```python
tuned_quad_dict, tuned_quad_rel_err, model = fit_model("cylinder_high_res", integrand, a, b, params_names, reg_params2, rtol, update=True, degree=2, full_output=True, n_jobs=-2, plot_param_space=True)

features = np.vstack(np.array([np.log2(np.array(k)) for k in tuned_quad_dict.keys()]))
val = np.array(list(tuned_quad_dict.values()))

predictions = model.predict(features)
predictions = np.power(2, np.array([(max(1, min(15, int(p) + 1))) for p in predictions]))
print(np.where(predictions >= val, 1, 0).mean())

```

Note: The `+1` in the predictions ensures that the number of points for evaluation is at least equal to or greater than the number of points actually required. While not the most efficient method, it ensures accurate results most of the time.

## Hacks

These cases typically arise with nested and higher-order integrals. The program isn't specifically designed for these scenarios and involves quite a bit of setup (this is left as an exercise to the user). However, these are cases I have encountered when trying to set it up.

### Integration Range as a Parameter

Take the case of the capped cylinder kernel, [capped_cylinder](https://www.sasview.org/docs/user/models/capped_cylinder.html). The lower bound of the inner integral is a varying parameter.

I implement it as an unused parameter A:

```python
def integrand(t, A, B, C, D):
    return np.cos(B * t + C) * (1 - t**2) * J1x_nb(D * (1 - t**2)**0.5)

n = 20
rtol = 1e-3
reg_params2 = [np.geomspace(0.01, 0.9999, n), np.geomspace(1, 1000000, n), np.geomspace(1, 1000000, n), np.geomspace(1, 1000000, n)]
params_names = ["A", "B", "C", "D"]
fit_model("capped_cylinder_kernel", integrand, a, b, params_names, reg_params2, rtol, update=True)
```

Within the source code in `estimate_error`, I manually set the lower limit:

```python
def estimate_error(param_prod, func, a, b, rtol, full_output=False):
    out = 0
    rel_err = 0.0
    for i, n in enumerate(n_kronrod):
        # Change
        a = param_prod[0]
        #
        rel_err = rel_error_kronrod(n, func, a, b, param_prod)
        if rel_err < rtol:
            out = i
            break
    else:
        out = len(n_kronrod) - 1
    return out, rel_err
```

### Some Combination of the Parameter Space is Invalid

Once again, consider the capped cylinder example. Look at the outer integral.

```python
def integrand(alpha, q, radius_cap, radius, half_length):
    sin_theta = np.sin(alpha)
    cos_theta = np.cos(alpha)
    qab = q * sin_theta
    qc = q * cos_theta
    h = -np.sqrt(radius_cap**2 - radius**2)

    form = _fq(q * sin_theta, q * cos_theta, h, radius_cap, radius, half_length)
    return form * form * sin_theta

n = 8
rtol = 1e-3
reg_params2 = [np.geomspace(0.0005, 0.5, n),  np.geomspace(10, 200000, n), np.geomspace(10, 200000, n), np.geomspace(10, 200000, n)]

params_names = ["q", "radius_cap", "radius", "half_length"]
fit_model("capped_cylinder", integrand_2param, a, b, params_names, reg_params2, rtol, update=True)

```

Ignore `_fq`. It encapsulates all the nasty bits. However, `radius_cap` clearly needs to be greater than or equal to `radius` at all times. You can manually enforce this by modifying the source code, specifically, `tune_quadrature` as follows:

```python
def tune_quadrature(func, a, b, reg_params_list, rtol):
    param_prod = list(product(*reg_params_list))
    # Change
    param_prod = np.array(param_prod)
    mask = np.where(param_prod[:, 1] >= param_prod[:, 2])
    param_prod = param_prod[mask]
    #
    return compute_tuned_quad_dict(
        func=func,
        a=a,
        b=b,
        param_prod=param_prod,
        rtol=rtol
    )
```

Python's duck typing should prevent any errors from occurring, even though it's a 2D array instead of a list of tuples but you can always convert it back. 

If you are going down this route and want to deal with higher order integrals, you are encouraged look at the source code as well as some examples I played around with.
