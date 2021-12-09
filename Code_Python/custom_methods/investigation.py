import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def point_biserial_corr(continuous: pd.DataFrame, binary: pd.DataFrame) -> pd.DataFrame:
    '''
    05/11/21
    'continuous' must be all continuous valued, float type DataFrame.
    'binary' must be binary, convertable to int DataFrame.
    Note: Can convert Series to DataFrame using .to_frame()
    '''
    from scipy.stats import pointbiserialr
    corr = pd.DataFrame(index=continuous.columns, columns=binary.columns, dtype='float')
    for bin in corr.columns:
        for cont in corr.index:
            corr.loc[cont, bin] = pointbiserialr(x=binary[bin].astype('int'), y=continuous[cont])[0]
    return corr

def correlation_ratios(categorical: pd.DataFrame, continuous: pd.DataFrame) -> pd.DataFrame:
    '''
    05/13/21
    inputs must be data frames
    Note: This may take a long time...
    '''
    from dython.nominal import correlation_ratio
    result_df = pd.DataFrame(index=categorical.columns, columns=continuous.columns, dtype='float')
    for cat in categorical.columns:
        for cont in continuous.columns:
            result_df.loc[cat, cont] = correlation_ratio(categories=categorical[cat], measurements=continuous[cont])
    return result_df

def plot_heatmap(corr:pd.DataFrame, visual_specs:dict, fmt:str='.3f', annot:bool=True, fname:str=None, image_dest:str=None):
    '''
    05/19/21
    plots heatmap of correlation matrix
    '''
    fontsize = visual_specs['fontsize']
    sns.heatmap(corr, cmap=visual_specs['palette'], fmt=fmt, annot=annot)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=90)
    if not (image_dest is None):
         plt.savefig(image_dest + fname + '.png', dpi=300, bbox_inches='tight')
    return

def plot_pn_mo_away_on(pos:pd.DataFrame, neg:pd.DataFrame, on:str, visual_specs:dict, image_dest:str=None):
    fontsize = visual_specs['fontsize']
    palette = visual_specs['palette']
    mean = neg[on].describe()['mean']
    std = neg[on].describe()['std']

    sns.lineplot(data=pos, x="MO_AWAY", y=on, palette=palette, label='P: mean +/- std')
    xmin = pos.MO_AWAY.min()
    xmax = pos.MO_AWAY.max()

    plt.hlines(mean-std, xmin=xmin, xmax=xmax, color='black', label='N: mean +/- std', linestyle='dashed')
    plt.hlines(mean, xmin=xmin, xmax=xmax, color='black', label='N: mean', linestyle='dashdot')
    plt.hlines(mean+std, xmin=xmin, xmax=xmax, color='black', linestyle='dashed')
    plt.legend(fontsize=fontsize)
    
    plt.ylabel(on, fontsize=fontsize)
    plt.yticks(fontsize=fontsize-2)
    plt.xlabel('MO_AWAY', fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.legend(fontsize=fontsize)
    
    if not (image_dest is None):
        plt.savefig(image_dest + 'PN_mo_away_on_'+ on + '.png', dpi=300, bbox_inches='tight')
    return


def pop_proportion(pos:pd.DataFrame, neg:pd.DataFrame, attribute:str, visual_specs:dict, image_dest:str=None, xlim:tuple=None, bins:int=30):
    '''
    05/20/21
    prints description of P, N cases on "attribute", then plots split probability histogram of P, N subpopulations on "attribute"
    saves plot if "image_dest" is defined
    NOTE: xlim is in terms of bin indexes, not bin edge values
    '''
    fontsize = visual_specs['fontsize']
    palette = visual_specs['palette']
    saturation= visual_specs['saturation']
    
    upper = max(pos[attribute].max(), neg[attribute].max())
    lower = min(pos[attribute].min(), neg[attribute].min())
    bins = pd.Series(np.linspace(lower, upper, bins), name='bin_bounds')

    pos_prob = (pd.cut(pos[attribute], bins=bins, duplicates='drop', ordered=False, include_lowest=True, labels=bins.iloc[:-1].round(1)).\
        value_counts().rename('prob') / len(pos)).to_frame().reset_index()
    pos_prob['Class'] = 'Positive'

    neg_prob = (pd.cut(neg[attribute], bins=bins, duplicates='drop', ordered=False, include_lowest=True, labels=bins.iloc[:-1].round(1)).\
        value_counts().rename('prob') / len(neg)).to_frame().reset_index()
    neg_prob['Class'] = 'Negative'

    all_prob = pos_prob
    all_prob = all_prob.append(neg_prob).rename({'index':'bin'}, axis=1)

    sns.barplot(
        data=all_prob, 
        x='bin', 
        y='prob',
        hue='Class',
        palette=palette,
        saturation=saturation
    )

    plt.ylabel('Proportion of Subpopulation', fontsize=fontsize)
    plt.yticks(fontsize=fontsize-2)
    plt.xlabel(attribute, fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2, rotation=90)
    plt.legend(fontsize=fontsize)
    plt.xlim(xlim)
    
    if not (image_dest is None):
         plt.savefig(image_dest + 'PN_on'+ attribute + '.png', dpi=300, bbox_inches='tight')
    return