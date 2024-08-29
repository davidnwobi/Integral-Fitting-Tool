"""
Typical setup case for tuning and integrand
"""


import numpy as np
from scipy.special import j1
from model_fitting import fit_model

a = 0
b = np.pi/2


def J1x_nb(x):
    
    return np.where(x != 0, j1(x)/x, 0.5)

def integrand_2param(
    x, A, B):

    return (np.sinc(A * np.cos(x)/np.pi) * J1x_nb(B * np.sin(x)))**2*np.sin(x)

n = 100
rtol = 1e-3
reg_params2 = [np.geomspace(1, 1000000, n), np.geomspace(1, 1000000, n)]
params_names = ["A", "B"]
tuned_quad_dict, tuned_quad_rel_err, model = fit_model("cylinder_high_res", integrand_2param, a, b, params_names, reg_params2, rtol, update=True, degree=2, full_output=True, n_jobs=-2, plot_param_space=True)

features = np.vstack(np.array([np.log2(np.array(k)) for k in tuned_quad_dict.keys()]))
val = np.array(list(tuned_quad_dict.values()))

predictions = model.predict(features)
predictions = np.power(2,np.array([(max(1,min(15, int(p)+1))) for p in predictions]))
print(np.where(predictions >= val, 1, 0).mean())

