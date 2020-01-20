import numpy as nmp
import theano.tensor as tns
import pymc3 as pmc

from GPyclone import aux

##
def get_model(x, r, R, vaf0, K=10):
    nsamples = r.shape[1]
    r, R, vaf0 = r[:, :, None], R[:, :, None], vaf0[:, :, None]
    idxs = aux.corr_vector_to_matrix_indices(K)
    with pmc.Model() as model:
        w = pmc.Dirichlet('w', nmp.ones(K))
        lw = tns.log(w)

        # alpha = pmc.Gamma('alpha', 1.0, 1.0)
        # u = pmc.Beta('u', 1.0, alpha, shape=K-1)
        # lw = aux.stick_breaking_log(u)

        rho = pmc.Gamma('rho', 1.0, 1.0)
        Cc = tns.fill_diagonal(pmc.LKJCorr('C', eta=2.0, n=K)[idxs], 1.0)
        Cr = aux.cov_quad_exp(x, 1.0, rho)
        mu_psi = pmc.MatrixNormal('mu_psi', mu=nmp.zeros((nsamples, K)), rowcov=Cr, colcov=Cc, shape=(nsamples, K))
        psi = pmc.Normal('psi', mu=mu_psi, sd=0.1, shape=(nsamples, K))
        phi = pmc.Deterministic('phi', pmc.invlogit(psi))

        # psi = pmc.MvNormal('psi', mu=nmp.zeros(K), tau=nmp.eye(K), shape=(nsamples, K))    
        # phi = pmc.Deterministic('phi', pmc.invlogit(psi))

        theta = pmc.Deterministic('theta', vaf0 * phi[None, :, :])
        pmc.DensityDist('r', aux.binmixND_logp_fcn(R, theta, lw), observed=r)
    return model
