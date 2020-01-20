import numpy as nmp
import pandas as pnd
import matplotlib.pyplot as plt


##
def plot_1sample(df, ppd, ax, bins=100, title=None, hist=True):
    if hist is True:
        ax.hist(df.VAF, bins=bins, density=True, label='Data', color='gray', linewidth=1)
        ax.plot(ppd.VAF, ppd.PPD, color='red', label='Model')
        ax.set_xlabel('variant allele fraction')
        ax.set_ylabel('density')
        ax.set_title(title)
        ax.legend(loc='best')
    else:
        ax.plot(df.VAF, df.CCF * df.VAF0, 'ok', alpha=0.5)
        ax.plot(ax.get_xlim(), ax.get_xlim(), '--k', linewidth=1)
        ax.set_xlabel('observed variant allele fraction')
        ax.set_ylabel('predicted variant allele fraction')
        ax.set_title(title)

    #
    return ax


##
def plot_1sample_ccf(df, ax):
    ax.errorbar(df.CLUSTERID, df.PHI, yerr=[df.PHI - df.PHI_LO, df.PHI_HI - df.PHI], fmt='ok', linewidth=1)
    ax.set_xticks(df.CLUSTERID)
    plt.xlabel('cluster ID')
    plt.ylabel('cancer cell fraction')


##
def plot_weights(df, thr, ax):
    # ax.axhline(y=thr, linestyle='dashed', color='k', linewidth=1)    
    ax.errorbar(df.CLUSTERID, df.W, yerr=[df.W-df.W_LO, df.W_HI-df.W], fmt='ok', linewidth=1)
    ax.set_xticks(df.CLUSTERID)
    ax.set_xlabel('cluster ID')
    ax.set_ylabel('cluster weights')

    #
    return ax


##
def plot_1trace(trace, ax, ylabel=None):
    ax.plot(trace, linewidth=1, color='black')
    ax.set_xlabel('iteration')
    ax.set_ylabel(ylabel)

    #
    return ax


##
def plot_Ntraces(trace, cids, ax, ylabel=None, cmap='Set2'):
    colors = plt.cm.get_cmap(cmap, len(cids)).colors
    colors = dict(zip(cids, colors))
    for cid in cids:
        ax.plot(trace[:, cid-1], color=colors[cid], label=cid, linewidth=1)
    ax.set_xlabel('iteration')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', title='cluster ID')

    #
    return ax


##
def plot_phi(data, centres, centres_gp, cids, ax, cmap='Set2', show_ci=True, show_legend=True):
    centres = centres[centres.CLUSTERID.isin(cids)]
    samples = centres.SAMPLEID.unique()
    colors = plt.cm.get_cmap(cmap, len(cids)).colors
    colors = dict(zip(cids, colors))

    x = centres.TIME2.unique() if 'TIME2' in centres else samples  
    for cid, df in centres.groupby('CLUSTERID'):
        if centres_gp is not None:
            t = centres_gp.TIME.unique()
            ds = centres_gp[centres_gp.CLUSTERID == cid]
            if show_ci is True:
                ax.fill_between(t, y1=ds.PHI_LO, y2=ds.PHI_HI, alpha=0.5, color=colors[cid])            
                ax.plot(t, ds.PHI, color=colors[cid], label=cid, linewidth=1)
                ax.plot(x, df.PHI, 'o', color=colors[cid])
            else:
                ax.plot(t, ds.PHI, color=colors[cid], label=cid, linewidth=1)
                ax.errorbar(x, df.PHI, yerr=[df.PHI-df.PHI_LO, df.PHI_HI-df.PHI], fmt='o', color=colors[cid], linewidth=1)
        else:
            ax.errorbar(x, df.PHI, yerr=[df.PHI-df.PHI_LO, df.PHI_HI-df.PHI], fmt='o', color=colors[cid], linewidth=1)
            ax.plot(x, df.PHI, color=colors[cid], label=cid, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=90)
    ax.set_xlabel('samples')
    ax.set_ylabel('cancer cell fraction')
    if show_legend is True:
        ax.legend(loc='best', title='cluster ID')
    
    #
    return ax


##
def plot_vaf(data, ax, cmap='Set2', show_legend=True):
    data1 = data[data.CLUSTERID == 'uncertain']
    data2 = data[data.CLUSTERID != 'uncertain'].assign(CLUSTERID = lambda df: df.CLUSTERID.astype('int'))
    cids = data2.CLUSTERID.unique()
    colors = plt.cm.get_cmap(cmap, len(cids)).colors
    colors = dict(zip(cids, colors))
    samples = data.SAMPLEID.unique()
    x = data.TIME2.unique() if 'TIME2' in data else samples

    for cid, _ in data2.groupby('CLUSTERID'):
        for _, df in _.groupby('MUTID'):
            ax.plot(x, df.VAF, color=colors[cid], linewidth=1, label=cid)

    if not data1.empty:
        for _, df in data1.groupby('MUTID'):
            ax.plot(x, df.VAF, color='gray', linestyle='--', linewidth=1, label = 'uncertain')

    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=90)
    ax.set_xlabel('samples')
    ax.set_ylabel('variant allele fraction')
    if show_legend is True:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', title='cluster ID')

    #
    return ax


##
def plot_vaf_mono(data, ax):
    samples = data.SAMPLEID.unique()
    x = data.TIME2.unique() if 'TIME2' in data else samples
    for _, df in data.groupby('MUTID'):
        ax.plot(x, df.VAF, color='gray', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=90)
    ax.set_xlabel('samples')
    ax.set_ylabel('variant allele fraction')

    #
    return ax


##
def plot_disps(df, ax):
    samples = df.SAMPLEID.values
    x = df.TIME2.unique() if 'TIME2' in df.columns else samples
    ax.errorbar(x, df.S, yerr=[df.S-df.S_LO, df.S_HI-df.S], fmt='o-', color = 'black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=90)
    ax.set_xlabel('samples')
    ax.set_ylabel('dispersion')

    #
    return ax


##
def plot_length(df, ax, ylabel):
    ax.errorbar(df.CLUSTERID, y = df.L, yerr=[df.L - df.L_LO, df.L_HI - df.L], fmt='ok', linewidth=1)
    ax.set_xticks(df.CLUSTERID)    
    ax.set_xlabel('cluster ID')
    ax.set_ylabel(ylabel)

    #
    return ax


##
def plot_metrics(df, ax, label=None):
    ax.axhline(y=df.MEDIAN.values[0], linestyle='--', color='k', linewidth=1)
    ax.errorbar(df.LABEL, df.MEDIAN, yerr=[df.MEDIAN-df.LOW, df.HIGH-df.MEDIAN], fmt='ko', linewidth=1)
    ax.set_xticklabels(df.LABEL, rotation=90)
    ax.set_ylabel(label)

    #
    return ax


##
def plot_samples(res, figsize=(10, 10), samples=None, ncols=2, bins=100, hist=True):
    data, ppd = res['data'], res['PPD']
    data = data if samples is None else data[data.SAMPLEID.isin(samples)]
    ppd = ppd if samples is None else ppd[ppd.SAMPLEID.isin(samples)]
    samples = data.SAMPLEID.unique()
    nsamples = len(samples)

    fig = plt.figure(figsize=figsize) 
    for i in range(nsamples):
        ax = plt.subplot(int(nmp.ceil(nsamples/ncols)), ncols, i+1)
        df1 = data[data.SAMPLEID == samples[i]]
        df2 = ppd[ppd.SAMPLEID == samples[i]]  
        plot_1sample(df1, df2, ax, bins, f'Sample {samples[i]}', hist)

    plt.tight_layout()

    #
    return fig


##
def plot_vi_bin_1sample_flat(res, figsize=(10, 10)):
    data, ppd, fit, weights = res['data'], res['PPD'], res['fit'], res['weights']
    thr, ccf = res['model_args']['threshold'], res['centres']
    cids = [int(_) for _ in data.CLUSTERID.unique() if _ != 'uncertain']

    fig = plt.figure(figsize=figsize)
    grd = fig.add_gridspec(2, 2)
    
    if fit is None:
        pass
    else:
        plot_1trace(fit.hist, fig.add_subplot(grd[0, 0]), ylabel='LOSS')        
    plot_weights(weights, thr, fig.add_subplot(grd[0, 1]))
    
    plot_1sample(data, ppd, fig.add_subplot(grd[1, 0]), title=None)
    plot_1sample_ccf(ccf[ccf.CLUSTERID.isin(cids)], fig.add_subplot(grd[1, 1]))

    plt.tight_layout()

    # 
    return fig


##
def plot_vi_bbin_1sample_flat(res, figsize=(10, 10)):
    data, ppd, fit, weights = res['data'], res['PPD'], res['fit'], res['weights']
    thr, disps, ccf = res['model_args']['threshold'], res['disps'], res['centres']
    cids = [int(_) for _ in data.CLUSTERID.unique() if _ != 'uncertain']

    fig = plt.figure(figsize=figsize)
    grd = fig.add_gridspec(3, 2)
    
    if fit is None:
        pass
    else:
        plot_1trace(fit.hist, fig.add_subplot(grd[0, 0]), ylabel='LOSS')        
    plot_weights(weights, thr, fig.add_subplot(grd[0, 1]))
    
    plot_1sample(data, ppd, fig.add_subplot(grd[1, :]), title=None)
    
    plot_1sample_ccf(ccf[ccf.CLUSTERID.isin(cids)], fig.add_subplot(grd[2, 0]))
    plot_disps(disps, fig.add_subplot(grd[2, 1]))

    plt.tight_layout()

    # 
    return fig


##
def plot_vi_bin_Nsamples_flat(res, figsize=(10, 10)):
    data, fit, weights, centres = res['data'], res['fit'], res['weights'], res['centres']
    thr = res['model_args']['threshold']
    cids = [int(_) for _ in data.CLUSTERID.unique() if _ != 'uncertain']

    fig = plt.figure(figsize=figsize)
    grd = fig.add_gridspec(3, 2)
    
    if fit is None:
        pass
    else:
        plot_1trace(fit.hist, fig.add_subplot(grd[0, 0]), ylabel='LOSS')        
    plot_weights(weights, thr, fig.add_subplot(grd[0, 1]))
    plot_vaf(data, fig.add_subplot(grd[1, :]))
    plot_phi(data, centres, None, cids, fig.add_subplot(grd[2, :]))

    plt.tight_layout()

    # 
    return fig


##
def plot_vi_bin_Nsamples_gp(res, figsize=(10, 10)):
    data, fit, weights, centres, centres_gp = res['data'], res['fit'], res['weights'], res['centres'], res['centres_gp']
    thr, l, h2 = res['model_args']['threshold'], res['lengths'], res['amplitudes']
    cids = [int(_) for _ in data.CLUSTERID.unique() if _ != 'uncertain']

    fig = plt.figure(figsize=figsize)
    grd = fig.add_gridspec(4, 2)
    
    if fit is None:
        pass
    else:
        plot_1trace(fit.hist, fig.add_subplot(grd[0, 0]), ylabel='LOSS')        
    plot_weights(weights, thr, fig.add_subplot(grd[0, 1]))
    plot_vaf(data, fig.add_subplot(grd[1, :]))
    plot_phi(data, centres, centres_gp, cids, fig.add_subplot(grd[2, :]))
    plot_length(l[l.CLUSTERID.isin(cids)], fig.add_subplot(grd[3, 0]), ylabel='lengths')
    plot_length(h2[h2.CLUSTERID.isin(cids)], fig.add_subplot(grd[3, 1]), ylabel='amplitudes')    

    plt.tight_layout()

    # 
    return fig


##
def plot_vi_bbin_Nsamples_flat(res, figsize=(10, 10)):
    data, fit, weights, centres = res['data'], res['fit'], res['weights'], res['centres']
    thr, disps = res['model_args']['threshold'], res['disps']
    cids = [int(_) for _ in data.CLUSTERID.unique() if _ != 'uncertain']

    fig = plt.figure(figsize=figsize)
    grd = fig.add_gridspec(4, 2)
    
    if fit is None:
        pass
    else:
        plot_1trace(fit.hist, fig.add_subplot(grd[0, 0]), ylabel='LOSS')        
    plot_weights(weights, thr, fig.add_subplot(grd[0, 1]))
    plot_vaf(data, fig.add_subplot(grd[1, :]))
    plot_phi(data, centres, None, cids, fig.add_subplot(grd[2, :]))
    plot_disps(disps, fig.add_subplot(grd[3, :]))

    plt.tight_layout()

    # 
    return fig


##
def plot_vi_bbin_Nsamples_gp(res, figsize=(10, 10)):
    data, fit, weights, centres, centres_gp = res['data'], res['fit'], res['weights'], res['centres'], res['centres_gp']
    thr, disps, l, h2 = res['model_args']['threshold'], res['disps'], res['lengths'], res['amplitudes']
    cids = [int(_) for _ in data.CLUSTERID.unique() if _ != 'uncertain']

    fig = plt.figure(figsize=figsize)
    grd = fig.add_gridspec(5, 2)
    
    if fit is None:
        pass
    else:
        plot_1trace(fit.hist, fig.add_subplot(grd[0, 0]), ylabel='LOSS')        
    plot_weights(weights, thr, fig.add_subplot(grd[0, 1]))
    plot_vaf(data, fig.add_subplot(grd[1, :]))
    plot_phi(data, centres, centres_gp, cids, fig.add_subplot(grd[2, :]))
    plot_disps(disps, fig.add_subplot(grd[3, :]))
    plot_length(l[l.CLUSTERID.isin(cids)], fig.add_subplot(grd[4, 0]), ylabel='lengths')
    plot_length(h2[h2.CLUSTERID.isin(cids)], fig.add_subplot(grd[4, 1]), ylabel='amplitudes')    

    plt.tight_layout()

    # 
    return fig


##
def plot_summary(res, figsize=(10, 10)):
    if res['data'].SAMPLEID.nunique() == 1:
        if res['model_args']['lik'] == 'Bin':
            fig = plot_vi_bin_1sample_flat(res, figsize)
        else:  # implies lik is BBin
            fig = plot_vi_bbin_1sample_flat(res, figsize)
    else:   # implies more than one samples present
        if res['model_args']['lik'] == 'Bin':
            if ('tau' in res['trace'].varnames) and ('h2' in res['trace'].varnames):
                fig = plot_vi_bin_Nsamples_gp(res, figsize)
            else:  # implies prior is Flat 
                fig = plot_vi_bin_Nsamples_flat(res, figsize)
        else:  # implies lik is BetaBin
            if ('tau' in res['trace'].varnames) and ('h2' in res['trace'].varnames):
                fig = plot_vi_bbin_Nsamples_gp(res, figsize)
            else:  # implies prior is Flat 
                fig = plot_vi_bbin_Nsamples_flat(res, figsize)

    #
    return fig
