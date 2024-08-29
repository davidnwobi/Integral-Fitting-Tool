import numpy as np
from model_fitting import fit_model
from scipy.special import spherical_jn

a = 0
b = 1


def J1x_nb(x):
    
    return np.where(x != 0, spherical_jn(1,x)/x*3, 1)


def integrand_2param(x, q, A, B):
    v_square_minus_one =(B/A)**2-1
    r = A*(1+x**2*v_square_minus_one)**0.5
    f = J1x_nb(q*r)
    return f*f
    
    
n = 100
rtol = 1e-3
reg_params2 = [np.geomspace(0.00005, 0.5, n), np.geomspace(1, 200000, n), np.geomspace(1, 200000, n)]
params_names = ["q", "A", "B"]

tuned_quad_dict, tuned_quad_rel_err, model = fit_model("ellipsoid2", integrand_2param, a, b, params_names, reg_params2, rtol, update=True, degree=4, full_output=True, n_jobs=-2, plot_param_space=True)

features = np.vstack(np.array([np.log2(np.array(k)) for k in tuned_quad_dict.keys()]))
val = np.array(list(tuned_quad_dict.values()))

predictions = model.predict(features)
predictions = np.power(2,np.array([(max(1,min(15, int(p)+1))) for p in predictions]))
print(np.where(predictions >= val, 1, 0).mean())
