import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
import scipy.stats as stats


def bar_graphs(axes, experiment, names, query, save_name=None, colors=None):
    taus = []
    taus_err = []
    if 'slow' in query:
        speed_error = '% slow'
    elif 'fast' in query:
        speed_error = '% fast'
    else:
        speed_error = '% slow'
    # x = []
    if colors == None:
        colors = []
    ages = []
    proteins = []
    for name in names:
        xxp = experiment.dict_experiments[name][2]
        protein = experiment.dict_experiments[name][0]
        tau = xxp.fit_parameters[query]
        # print(tau, tqu)
        hpf = experiment.dict_experiments[name][1]
        err = xxp.fit_parameters[query] * xxp.fit_parameters_sigma[speed_error]
        color = experiment.get_protein_color(protein, 'early' if hpf < 15 else 'late')
        if not colors:
            colors.append(color)
        taus.append(tau)
        proteins.append(protein)
        taus_err.append(err)
        # ages.append(f'{hpf} hpf \n n={int(xxp.sample_count)}')
        ages.append(f'{protein} \n {hpf} hpf')
    xval = [a for a in range(len(names))]
    axes.bar(xval, taus, yerr=taus_err, color=colors, tick_label=ages, width=0.75, capsize=3,
             ecolor='#231F20')
    axes.set_ylabel(r'Recovery time constant $\tau$ (in sec)')
    # ax.legend()
    if save_name is not None:
        plt.savefig('/Volumes/HELHEIM/analyzed_data/diffusivity/save_path' + os.path.sep + save_name + '.pdf')


def swarmy_boxes(axes, data_dict, colors=None):
    df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in data_dict.items()]))
    if colors is None:
        colors = ["#ffb3b3", "#ffc28a", "#f3dd6d", "#92c57d", "#9bac8b", "#698c86", "#366b81", "#405373", "#4a3a64"]

    sns.boxplot(data=df, ax=axes, fliersize=0, palette=colors)
    sns.swarmplot(data=df, ax=axes, color='#22262a', s=4)


def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


def perform_statistical_test(data_dict, independant: bool, homogeneity_of_variance, name1=None, value1=None, name2=None,
                             value2=None, z_cuttoff=3.5, remove_outliers=True):
    if data_dict == {}:
        data_dict = {
            name1: value1,
            name2: value2
        }

    data_names = list(data_dict.keys())
    if len(data_names) > 2:
        raise Exception('Tried to compare more than two data sets')

    normalities = []
    outliers = None

    # Removing outliers and testing for normality
    for baka in data_dict.keys():
        values = np.asarray(data_dict.get(baka), dtype='float')
        values = values[np.logical_not(np.isnan(values))]
        std = np.std(values)
        med = np.median(values)
        mad = stats.median_abs_deviation(values)
        z_scores = [0.6745 * ((x - med) / mad) for x in values]

        cleaned_values = []
        for ids, val in enumerate(z_scores):
            if val < z_cuttoff and remove_outliers:
                cleaned_values.append(values[ids])

        data_dict.update({baka: cleaned_values})

        outliers = False if remove_outliers is True else True
        try:
            _, p_v = stats.normaltest(values)
        except ValueError:
            p_v = 1
        normalities.append(p_v)

    normality = all(p > 0.05 for p in normalities)

    # Decision if dataset is parametric
    if homogeneity_of_variance and normality and outliers is False:
        parametric = True
    else:
        parametric = False

    # Statistic decision tree
    if independant == True and parametric == False:
        _, p = stats.mannwhitneyu(data_dict.get(data_names[0]), data_dict.get(data_names[1]))
        stat_test = 'Mann-Whitney-U'
    elif independant == True and parametric == True:
        _, p = stats.ttest_ind(data_dict.get(data_names[0]), data_dict.get(data_names[1]))
        stat_test = 'Independant t-test'
    elif independant == False and parametric == False:
        _, p, = stats.wilcoxon(data_dict.get(data_names[0]), data_dict.get(data_names[1]))
        stat_test = 'Wilcoxon'
    elif independant == False and parametric == True:
        _, p, = stats.ttest_rel(data_dict.get(data_names[0]), data_dict.get(data_names[1]))
        stat_test = 'paired t-test'

    documentation = {
        'Homogeneity of Variance': homogeneity_of_variance,
        'Normal Distribution': [p > 0.05 for p in normalities],
        'Parametric': parametric,
        'Independant Sets': independant,
        'Outliers': 'Not yet implemented',
        # TODO implement that outliers are shown here or atleast if outliers were removed
        'Performed test': stat_test,
        'P-value': p,
        'Star significance': convert_pvalue_to_asterisks(p)
    }

    return p, convert_pvalue_to_asterisks(p), documentation
