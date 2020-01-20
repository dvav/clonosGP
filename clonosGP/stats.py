import numpy as nmp
import pandas as pnd
import theano.tensor as tns
import pymc3 as pmc


##
def bin_lpdf(r, R, theta):
    return tns.gammaln(R + 1.0) - tns.gammaln(r + 1.0) - tns.gammaln(R - r + 1.0)\
        + r*tns.log(theta) + (R - r)*tns.log1p(-theta)


##
def binmix_logp_fcn(R, theta, lw):
    def logp(r):
        lp = lw + bin_lpdf(r, R, theta).sum(1)
        return pmc.logsumexp(lp, 0)#.sum()
    return logp


##
def betabin_lpdf(r, R, a, b):
    return tns.gammaln(R + 1.0) - tns.gammaln(r + 1.0) - tns.gammaln(R - r + 1.0)\
        + tns.gammaln(r + a) + tns.gammaln(R - r + b) - tns.gammaln(R + a + b)\
        + tns.gammaln(a + b) - tns.gammaln(a) - tns.gammaln(b)


##
def betabinmix_logp_fcn(R, u, theta, lw):
    a = u * theta
    b = u * (1.0 - theta)
    def logp(r):
        lp = lw + betabin_lpdf(r, R, a, b).sum(1)
        return pmc.logsumexp(lp, 0)#.sum()
    return logp


##
def cov_expquad(x1, x2, tau):
    return tns.exp(-0.5 * tau * (x1 - x2)**2)


##
def cov_exp(x1, x2, tau):
    return tns.exp(-tns.sqrt(tau) * tns.abs_(x1 - x2))


##
def cov_mat32(x1, x2, tau):
    r = tns.abs_(x1 - x2)
    c = tns.sqrt(3.0) * r * tns.sqrt(tau)
    return (1.0 + c) * tns.exp(-c)


##
def cov_mat52(x1, x2, tau):
    r = tns.abs_(x1 - x2)
    c = tns.sqrt(5.0) * r * tns.sqrt(tau)
    return (1.0 + c + 5.0/3.0 * r**2 * tau) * tns.exp(-c)


##
def stick_breaking_log(u):
    """Return log of weights from stick-breaking process."""
    lu = tns.concatenate((tns.log(u), [0.0]))
    cs = tns.concatenate(([0.0], tns.cumsum(tns.log1p(-u))))
    lw = lu + cs

    return lw


##
COV_FCNS = {
        'ExpQ': cov_expquad,
        'Exp': cov_exp,
        'Mat32': cov_mat32,
        'Mat52': cov_mat52    
    }


##
def calculate_cluster_weights(trace, threshold, alpha):
    w_samples = nmp.exp(trace['lw'])
    
    # re-weight cluster weights
    w = nmp.median(w_samples, 0)
    wids = w < threshold
    w_samples[:, wids] = 0
    w_samples = w_samples / nmp.sum(w_samples, 1, keepdims=True)

    # median, credible interval
    w_lo, w, w_hi = nmp.quantile(w_samples, [0.5*alpha, 0.5, 1 - 0.5*alpha], axis=0) 
    
    #
    return pnd.DataFrame({
        'CLUSTERID': nmp.arange(w.size) + 1, 
        'W': w, 
        'W_LO': w_lo, 
        'W_HI': w_hi
    })


##
def calculate_cluster_centres(data, trace, alpha):
    phi_samples = trace['phi']
    
    phi_lo, phi, phi_hi = nmp.quantile(phi_samples, [0.5*alpha, 0.5, 1 - 0.5*alpha], axis=0)
    
    sid = data['samples'].SAMPLEID
    cid = nmp.arange(phi_samples.shape[1]) + 1
    centres = pnd.concat({
        'PHI': pnd.DataFrame(phi, index=cid, columns=sid), 
        'PHI_LO': pnd.DataFrame(phi_lo, index=cid, columns=sid), 
        'PHI_HI': pnd.DataFrame(phi_hi, index=cid, columns=sid)
    }, axis=1).stack().reset_index().rename(columns={'level_0': 'CLUSTERID'})
    
    if 'TIME2' in data['samples']:
        centres = pnd.merge(centres, data['samples'][['SAMPLEID', 'TIME2']], how='left', on = 'SAMPLEID')
        centres = centres[['CLUSTERID', 'SAMPLEID', 'TIME2', 'PHI', 'PHI_LO', 'PHI_HI']]

    #
    return centres


##
def calculate_ccf_and_hard_clusters(data, trace, threshold, alpha):
    r, R, VAF0 = data['r'].values.T, data['R'].values.T, data['VAF0'].values.T

    r, R, VAF0 = r[None, None, :, :], R[None, None, :, :], VAF0[None, None, :, :]
    phi, lw = trace.phi, trace.lw
    theta = VAF0 * phi[:, :, :, None]

    # re-weight cluster weights
    w_samples = nmp.exp(lw)
    w = nmp.median(w_samples, 0)
    wids = w < threshold
    w_samples[:, wids] = 0
    w_samples = w_samples / nmp.sum(w_samples, 1, keepdims=True)
    lw = nmp.log(w_samples)

    # calculate logliks
    if 'u' in trace.varnames:   # implies BetaBinomial model
        u = trace.u[:, None, :, None]
        a = u * theta
        b = u * (1.0 - theta)
        lp = betabin_lpdf(r, R, a, b).eval()
    else:   # implies Binomial model
        lp = bin_lpdf(r, R, theta).eval()        
    
    # ppd
    w = nmp.exp(lp + lw[:, :, None, None])      
    ppd_ = nmp.sum(w * R, axis=1)
    ppd_lo, ppd, ppd_hi = nmp.quantile(ppd_, [alpha * 0.5, 0.5, 1 - alpha * 0.5], axis=0)
    
    ppd = pnd.concat({
        'PPD': pnd.DataFrame(ppd.T, index=data['r'].index, columns=data['r'].columns), 
        'PPD_LO': pnd.DataFrame(ppd_lo.T, index=data['r'].index, columns=data['r'].columns), 
        'PPD_HI': pnd.DataFrame(ppd_hi.T, index=data['r'].index, columns=data['r'].columns)
    }, axis=1)

    lppd = nmp.ma.masked_invalid(nmp.log(ppd_)).sum(axis=(1,2))
    lppd_lo, lppd, lppd_hi = nmp.quantile(lppd, [alpha * 0.5, 0.5, 1 - alpha * 0.5])

    lppd = pnd.DataFrame({'PPD': lppd, 'PPD_LO': lppd_lo, 'PPD_HI': lppd_hi}, index=[0])

    # ccf
    w = nmp.exp(lp.sum(2) + lw[:, :, None])
    w = w / nmp.sum(w, axis=1, keepdims=True)
    ws = nmp.cumsum(w, 1)
    niters, _, nmuts = ws.shape
    r = nmp.random.rand(niters, 1, nmuts)
    ids = nmp.sum(ws <= r, axis=1)
    ccf_samples = phi[nmp.arange(niters)[:, None], ids, :]
    ccf_lo, ccf, ccf_hi = nmp.quantile(ccf_samples, [alpha * 0.5, 0.5, 1 - alpha * 0.5], axis=0)
    
    ccf = pnd.concat({
        'CCF': pnd.DataFrame(ccf, index=data['r'].index, columns=data['r'].columns), 
        'CCF_LO': pnd.DataFrame(ccf_lo, index=data['r'].index, columns=data['r'].columns), 
        'CCF_HI': pnd.DataFrame(ccf_hi, index=data['r'].index, columns=data['r'].columns)
    }, axis=1)

    # hard clusters
    cids = nmp.nonzero(wids)[0] + 1
    clusters = nmp.argmax(nmp.median(w, axis=0), axis=0)
    clusters = pnd.DataFrame({'CLUSTERID': clusters+1}, index=data['r'].index).reset_index()
    clusters = clusters.assign(CLUSTERID = nmp.where(clusters.CLUSTERID.isin(cids), 'uncertain', clusters.CLUSTERID))
    
    #
    return pnd.merge(pnd.concat([ccf, ppd], axis=1).stack().reset_index(), clusters), lppd


##
def calculate_ppd(data, trace, threshold, alpha, npoints):
    VAF, R, VAF0 = data['VAF'].values.T, data['R'].values.T, data['VAF0'].values.T

    R, VAF0 = nmp.mean(R, axis=1)[None, None, :, None], nmp.mean(VAF0, axis=1)[None, None, :, None]
    phi, lw = trace.phi, trace.lw
    theta = VAF0 * phi[:, :, :, None]

    # re-weight cluster weights
    w_samples = nmp.exp(lw)
    w = nmp.median(w_samples, 0)
    wids = w < threshold
    w_samples[:, wids] = 0
    w_samples = w_samples / nmp.sum(w_samples, 1, keepdims=True)
    lw = nmp.log(w_samples)

    # calculate logliks
    vaf = nmp.asarray([nmp.linspace(nmp.min(_), nmp.max(_), num=npoints) for _ in VAF])
    if 'u' in trace.varnames:   # implies BetaBinomial model
        u = trace.u[:, None, :, None]
        a = u * theta
        b = u * (1.0 - theta)
        lp = betabin_lpdf(vaf[None, None, :, :] * R, R, a, b).eval()
    else:   # implies Binomial model
        lp = bin_lpdf(vaf[None, None, :, :] * R, R, theta).eval()        
    
    # ppd
    w = nmp.exp(lp + lw[:, :, None, None])  
    ppd = nmp.sum(w * R, axis=1)
    ppd_lo, ppd, ppd_hi = nmp.quantile(ppd, [alpha * 0.5, 0.5, 1 - alpha * 0.5], axis=0)
    
    mid = [f'M{_+1}' for _ in range(npoints)]
    ppd = pnd.concat({
        'VAF': pnd.DataFrame(vaf.T, index=mid, columns=data['r'].columns), 
        'PPD': pnd.DataFrame(ppd.T, index=mid, columns=data['r'].columns), 
        'PPD_LO': pnd.DataFrame(ppd_lo.T, index=mid, columns=data['r'].columns), 
        'PPD_HI': pnd.DataFrame(ppd_hi.T, index=mid, columns=data['r'].columns)
    }, axis=1)

    #
    return ppd.stack().reset_index().rename(columns={'level_0': 'MUTID'})


##
def calculate_cluster_centres_gp(data, trace, prior, cov, npoints, alpha, *args, **kargs):
    phi_samples = trace['phi']
    tau_samples = trace['tau']
    h2_samples = trace['h2']    
    y_samples = nmp.log(phi_samples) - nmp.log1p(-phi_samples)
    
    t = data['samples'].TIME2.values.ravel()
    t1 = nmp.linspace(0, 1, npoints)

    if prior == 'GP0':
        tau_samples = tau_samples[:, None, None, None]
        h2_samples = h2_samples[:, None, None, None]
    elif prior in ['GP1', 'GP2']:
        tau_samples = tau_samples[:, None, None, None]
        h2_samples = h2_samples[:, :, None, None]
    else:  # implies prior is GP3
        tau_samples = tau_samples[:, :, None, None]
        h2_samples = h2_samples[:, :, None, None]        
    y_samples = sample_post_gp(phi_samples.shape[1], t1, t, y_samples, tau_samples, h2_samples, COV_FCNS[cov])
    phi_samples = 1.0 / (1.0 + nmp.exp(-y_samples))

    phi_lo, phi, phi_hi = nmp.quantile(phi_samples, [0.5*alpha, 0.5, 1 - 0.5*alpha], axis=0)

    sid = [f'S{_}' for _ in range(1, t1.size+1)]
    cid = nmp.arange(phi_samples.shape[1]) + 1
    centres = pnd.concat({
        'PHI': pnd.DataFrame(phi, index=cid, columns=pnd.MultiIndex.from_arrays([sid, t1])), 
        'PHI_LO': pnd.DataFrame(phi_lo, index=cid, columns=pnd.MultiIndex.from_arrays([sid, t1])), 
        'PHI_HI': pnd.DataFrame(phi_hi, index=cid, columns=pnd.MultiIndex.from_arrays([sid, t1]))
    }, axis=1).stack(level=(1, 2)).reset_index().rename(columns={'level_0': 'CLUSTERID', 'level_1': 'SAMPLEID', 'level_2': 'TIME'}).sort_values(['TIME', 'CLUSTERID'])

    #
    return centres


##
def sample_post_gp(K, t1, t, y, tau, h2, cov_fcn, *args, **kargs):
    N, Mnew, Mold = h2.shape[0], t1.size, t.size    # number of MC samples, number of clusters, number of new and old samples
    K11 = h2 * cov_fcn(t1[None, None, None, :], t1[None, None, :, None], 1).eval()**tau + nmp.eye(Mnew)*1e-6
    K22 = h2 * cov_fcn(t[None, None, None, :], t[None, None, :, None], 1).eval()**tau + nmp.eye(Mold)*1e-6
    K12 = h2 * cov_fcn(t[None, None, None, :], t1[None, None, :, None], 1).eval()**tau
    K21 = nmp.moveaxis(K12, -1, 2)

    K22_inv = nmp.linalg.inv(K22)

    mu = nmp.matmul(nmp.matmul(K12, K22_inv), y[:, :, :, None])
    S = K11 - nmp.matmul(nmp.matmul(K12, K22_inv), K21)
    L = nmp.linalg.cholesky(S)
    z = nmp.random.randn(N, K, Mnew, 1)

    return nmp.squeeze(mu + nmp.matmul(L, z), axis=-1)    # remove last dimension, as always singular


##
def calculate_dispersions(data, trace, alpha):
    s_samples = 1/trace['u']
    s_lo, s, s_hi = nmp.quantile(s_samples, [0.5*alpha, 0.5, 1 - 0.5*alpha], axis=0)
        
    disps = pnd.DataFrame({
        'SAMPLEID': data['samples'].SAMPLEID,
        'S': s, 'S_LO': s_lo, 'S_HI': s_hi
    })
    
    if 'TIME2' in data['samples']:
        disps = pnd.merge(data['samples'][['SAMPLEID', 'TIME2']], disps, how='left', on='SAMPLEID')

    #
    return disps


##
def calculate_scales(trace, alpha):
    len_samples, h2_samples, lw_samples = 1/trace['tau'], trace['h2'], trace['lw']
    
    l_lo, l, l_hi = nmp.quantile(len_samples, [0.5*alpha, 0.5, 1 - 0.5*alpha], axis=0)
    h2_lo, h2, h2_hi = nmp.quantile(h2_samples, [0.5*alpha, 0.5, 1 - 0.5*alpha], axis=0)

    h2 = pnd.DataFrame({
        'CLUSTERID': nmp.arange(lw_samples.shape[1]) + 1,
        'L': h2, 'L_LO': h2_lo, 'L_HI': h2_hi
    })

    l = pnd.DataFrame({
        'CLUSTERID': nmp.arange(lw_samples.shape[1]) + 1,
        'L': l, 'L_LO': l_lo, 'L_HI': l_hi
    })

    #
    return l, h2
