import numpy as nmp
import numpy.random as rnd
import pandas as pnd

import clonosGP.aux as aux
import clonosGP.stats as sts


##
def get_1sample(sampleid = 'S0', weights=(0.65, 0.25, 0.10), z=None, phi=(1.0, 0.5, 0.25), nmuts=100, rho=0.9, mean_depth=1000):
    CNm = nmp.ones(nmuts, dtype='int')
    CNt = nmp.repeat(2, nmuts)
    CNn = nmp.repeat(2, nmuts)
    R = rnd.poisson(mean_depth, size=nmuts)
    z = rnd.choice(len(weights), size=nmuts, replace=True, p=weights) if z is None else z
    phi = nmp.asarray(phi)[z]
    weights = nmp.asarray(weights)[z]
    VAF0 = aux.calculate_vaf0(rho, CNm, CNt, CNn)
    theta = VAF0 * phi
    r = rnd.binomial(R, theta)
    mutid = [f'M{i+1}' for i in range(nmuts)]

    return pnd.DataFrame({
        'SAMPLEID': sampleid,
        'PURITY': rho,
        'MUTID': mutid,
        'r': r,
        'R': R, 
        'CNn': CNn,
        'CNt': CNt,
        'CNm': CNm,
        'theta': theta,
        'VAF0': VAF0, 
        'phi': phi,
        'w': weights
    })


#
def get_Nsamples(nclusters=3, nmuts=100, nsamples=10, h2=None, tau=None, weights=None, rho=(0.8, 0.9), mean_depth=(40, 40), depths=None):
    weights = nmp.ones(nclusters) / nclusters if weights is None else weights
    tau = rnd.gamma(1, 1) if tau is None else tau
    h2 = rnd.gamma(1, 1) if h2 is None else h2
    times = nmp.linspace(0, 1, nsamples) + rnd.uniform(-1/nsamples, 1/nsamples, nsamples); times[0] = 0; times[-1] = 1
    sids = [f'S{_+1}' for _ in range(nsamples)]
    mids = [f'M{_+1}' for _ in range(nmuts)]

    cov = h2 * sts.cov_mat32(times[:, None], times, tau).eval()
    y = rnd.multivariate_normal(nmp.zeros(nsamples), cov, size=nclusters)
    y = 1 / (1 + nmp.exp(-y))

    z = rnd.choice(nclusters, size=nmuts, p=weights)
    phi = y[z, :]

    lam = rnd.uniform(mean_depth[0], mean_depth[1], size=nsamples)
    rho = rnd.uniform(rho[0], rho[1], size=nsamples)

    R = rnd.poisson(lam, size=(nmuts, nsamples)) if depths is None else rnd.choice(depths, size=(nmuts, nsamples))
    r = rnd.binomial(R, 0.5 * rho * phi)

    r = pnd.DataFrame(r, index=pnd.Index(mids, name='MUTID'), columns=pnd.MultiIndex.from_arrays([sids, times], names=['SAMPLEID', 'TIME']))
    R = pnd.DataFrame(R, index=pnd.Index(mids, name='MUTID'), columns=pnd.MultiIndex.from_arrays([sids, times], names=['SAMPLEID', 'TIME']))
    phi = pnd.DataFrame(phi, index=pnd.Index(mids, name='MUTID'), columns=pnd.MultiIndex.from_arrays([sids, times], names=['SAMPLEID', 'TIME']))
    cid = pnd.DataFrame({'CLUSTERID': z+1}, index=pnd.Index(mids, name='MUTID')).reset_index()

    return pnd.merge(
        pnd.concat({'r': r, 'R': R, 'PHI': phi}, axis=1).stack(level=(-2,-1)).reset_index(), 
        cid).sort_values(['TIME', 'MUTID']).assign(TAU = tau, H2 = h2)

