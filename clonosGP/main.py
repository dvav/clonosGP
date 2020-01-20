import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('clonosGP')

import numpy as nmp
import pandas as pnd
import pymc3 as pmc

import clonosGP.aux as aux
import clonosGP.model_factory as modelfct
import clonosGP.stats as sts


MODEL_ARGS = {
    'K': 20,
    'alpha': 1.0,
    'prior': 'Flat',
    'eta': 2.0,
    'cov': 'Mat32',
    'lik': 'BBin',
    'threshold': 0.00,
    'ci_alpha': 0.05,
    'npoints': 100
}

PYMC3_ARGS = {
    'draws': 1000,
    'tune': 1000,
    'target_accept': 0.95,
    'niters': int(10e3),
    'method': 'advi',
    'learning_rate': 1e-2,
    'random_seed': 42
}


##
def infer(data, model_args={}, pymc3_args={}):
    model_args = {**MODEL_ARGS, **model_args}     # model arguments
    pymc3_args = {**PYMC3_ARGS, **pymc3_args}     # sampler arguments
    data = aux.prepare_data(data)                 # validate and pre-process data

    # choose and run model
    model = modelfct.get_model(data, **model_args)
    if pymc3_args['method'] == 'nuts':
        fit = None
        trace = pmc.sample(model=model, 
                           draws=pymc3_args['draws'],
                           tune=pymc3_args['tune'],
                           chains=1,
                           compute_convergence_checks=False,
                           target_accept=pymc3_args['target_accept'],
                           random_seed=pymc3_args['random_seed'])
    elif pymc3_args['method'] == 'nfvi':
        fit = pmc.NFVI(model=model,
                       flow=pymc3_args['flow'],
                       random_seed=pymc3_args['random_seed']
        ).fit(n=pymc3_args['niters'], 
              obj_optimizer=pmc.adam(learning_rate=pymc3_args['learning_rate']))
        trace = fit.sample(pymc3_args['draws'])      
    else:
        fit = pmc.fit(model=model,
                      n=pymc3_args['niters'], 
                      method=pymc3_args['method'], 
                      obj_optimizer=pmc.adam(learning_rate=pymc3_args['learning_rate']),
                      random_seed=pymc3_args['random_seed'])
        trace = fit.sample(pymc3_args['draws'])

    # post-processing
    logger.info('Calculating posterior cluster weights and centres.')
    weights = sts.calculate_cluster_weights(trace, model_args['threshold'], model_args['ci_alpha'])
    centres = sts.calculate_cluster_centres(data, trace, model_args['ci_alpha'])

    logger.info('Calculating posterior CCF values.')
    posts, lppd = sts.calculate_ccf_and_hard_clusters(data, trace, model_args['threshold'], model_args['ci_alpha'])
    
    logger.info('Calculating posterior predictive distribution.')
    ppd = sts.calculate_ppd(data, trace, model_args['threshold'], model_args['ci_alpha'], model_args['npoints'])

    if model_args['prior'] in ['GP0', 'GP1', 'GP2', 'GP3']:
        logger.info('Calculating GP-related quantities.')
        try:
            centres_gp = sts.calculate_cluster_centres_gp(data, trace, prior=model_args['prior'], cov=model_args['cov'], 
                                                          npoints=model_args['npoints'], alpha=model_args['ci_alpha'])
        except:    # ExpQ sometimes throws a singular matrix error
            logger.error('Exception occured while calculating GP-related quantities.')
            centres_gp = None

        l, h2 = sts.calculate_scales(trace, model_args['ci_alpha'])
    else:
        centres_gp, l, h2 = None, None, None
        

    if model_args['lik'] == 'BBin':
        logger.info('Calculating dispersion(s).')
        disps = sts.calculate_dispersions(data, trace, model_args['ci_alpha'])
    else:
        disps = None
    
    # return tidy data
    data = aux.pivot_longer(data)
    data = pnd.merge(data, posts)
    
    #
    logger.info('Finished.')
    return {
        'model': model,
        'fit': fit,
        'trace': trace,
        'data': data, 
        'weights': weights,
        'centres': centres,
        'centres_gp': centres_gp,
        'PPD': ppd,
        'LPPD': lppd,
        'disps': disps,
        'lengths': l,
        'amplitudes': h2,
        'model_args': model_args,
        'pymc3_args': pymc3_args
    }
    