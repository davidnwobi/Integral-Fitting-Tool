import numpy as np
from model_fitting import fit_model
from scipy.special import spherical_jn

a = 0
b = np.pi/2


def J1x_nb(x):
    
    return np.where(x != 0, 3*spherical_jn(1,x)/x, 1)


def integrand(x, q, A, B, C, D):
    r1 = ((A*np.sin(x))**2 + (B*np.cos(x))**2)**0.5
    r2 = ((C*np.sin(x))**2 + (D*np.cos(x))**2)**0.5
    return J1x_nb(q*r1) + J1x_nb(q*r2)


    
n = 15
rtol = 1e-3
reg_params2 = [np.geomspace(0.00005, 1, n), np.geomspace(1, 1000000, n), np.geomspace(1, 1000000, n), np.geomspace(1, 1000000, n), np.geomspace(1, 1000000, n)]
params_names = ["q", "A", "B", "C", "D"]

tuned_quad_dict, tuned_quad_rel_err, model = fit_model("core_shell_ellipsoid", integrand, a, b, params_names, reg_params2, rtol, update=True, degree=3, full_output=True, n_jobs=-2, plot_param_space=True)

features = np.vstack(np.array([np.log2(np.array(k)) for k in tuned_quad_dict.keys()]))
val = np.array(list(tuned_quad_dict.values()))

predictions = model.predict(features)
predictions = np.power(2,np.array([(max(1,min(15, int(p)+1))) for p in predictions]))
print(np.where(predictions >= val, 1, 0).mean())
