import numpy as np
from scipy.special import j1
from model_fitting import fit_model

a = 0
b = np.pi/2


# @vectorize([float64(float64)])
def J1x_nb(x):
    
    return np.where(x != 0, 2*j1(x)/x, 1)



def _fq(qab, qc, radius, thickness, length):
    lam1 = J1x_nb((radius+thickness)*qab)
    lam2 = J1x_nb(radius*qab)
    gamma_sq = (radius/(radius+thickness))**2
    psi = (lam1 - gamma_sq*lam2)/(1.0 - gamma_sq)
    t2 = np.sin(0.5*length*qc)/(0.5*length*qc)
    return psi*t2

def integrand(x, q, radius, thickness, length):
    sin_theta = np.sin(x)
    cos_theta = np.cos(x)
    form = _fq(q*sin_theta, q*cos_theta, radius, thickness, length)
    return form*form

n = 15
rtol = 1e-3
reg_params2 = [np.geomspace(0.0005, .5, n), np.geomspace(1, 1000000, n), np.geomspace(1, 1000000, n), np.geomspace(1, 1000000, n)]
params_names = ["q", "radius", "thickness", "length"]

tuned_quad_dict, tuned_quad_rel_err, model = fit_model("hollow_cylinder_high_res", integrand, a, b,  params_names, reg_params2, rtol, update=True, degree=3, full_output=True, n_jobs=-2, plot_param_space=True)

features = np.vstack(np.array([np.log2(np.array(k)) for k in tuned_quad_dict.keys()]))
val = np.array(list(tuned_quad_dict.values()))

predictions = model.predict(features)
predictions = np.power(2,np.array([(max(1,min(15, int(p)+1))) for p in predictions]))
print(np.where(predictions >= val, 1, 0).mean())