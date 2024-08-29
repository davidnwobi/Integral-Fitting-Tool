import numpy as np
from scipy.special import j1
from model_fitting import fit_model

a = 0
b = np.pi/2


# @vectorize([float64(float64)])
def J1x_nb(x):
    
    return np.where(x != 0, 2*j1(x)/x, 1)


def integrand(x, a_q, b_q):
    sn = np.sin(x)
    cn = np.cos(x)
    arg = np.sqrt((a_q*sn)**2 + (b_q*cn)**2) # When a_q == b_q, this is just a_q. Very easy to integrate.
    yy  = J1x_nb(arg)
    return yy*yy

n = 100
rtol = 1e-3
reg_params2 = [np.geomspace(1, 1000000, n), np.geomspace(1, 1000000, n)]
params_names = ["a_q", "b_q"]

tuned_quad_dict, tuned_quad_rel_err, model = fit_model("elliptical_crosssection_high_res", integrand, a, b, params_names, reg_params2, rtol, update=True, degree=3, full_output=True, n_jobs=-2, plot_param_space=True)

features = np.vstack(np.array([np.log2(np.array(k)) for k in tuned_quad_dict.keys()]))
val = np.array(list(tuned_quad_dict.values()))

predictions = model.predict(features)
predictions = np.power(2,np.array([(max(1,min(15, int(p)+1))) for p in predictions]))
print(np.where(predictions >= val, 1, 0).mean())
