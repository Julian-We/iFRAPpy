from ifrappy.main_iFRAP import ExperimentGroup
import numpy as np
from tqdm import tqdm


def generate_frap_data(protein_dict, experiment_group_path, pckl_name='FRAP_group', generate_freshly=True):
    """

    :param pckl_name:
    :param generate_freshly:
    :param experiment_group_path:
    :param protein_dict: Dictionary containing {'Name of protein' : [hours post fertilization: int, path to folder: str,
     molecular_weight: float -> kwarg and disorder: float -> kwarg]}
    :return: Experiment object
    """
    frap = ExperimentGroup(experiment_group_path, pickle_name=pckl_name)
    try:
        # Try to recover the frap experiment from a pickle file
        frap.recover_experiment()

        # Compare the number of experiments from the last execution to this execution. If there are not the same, then
        # re-execute the experiment
        pathlist = []
        for _, path in protein_dict.items():
            pathlist.append(path)
        if len(frap.filelist) != len(pathlist):
            len_ident = False
        else:
            len_ident = True

    except Exception as e:
        print(e)
        len_ident = True

    if not generate_freshly and not len_ident:
        inpt = input(
            f'The number of folders in this experiment has changed since last time. Do you want to caluclate the data freshly? (y/n)')
        if inpt.lower() == 'y' or inpt.lower() == 'yes':
            generate_freshly = True

    if generate_freshly:
        frap = ExperimentGroup(experiment_group_path, pickle_name=pckl_name)
        for k, v in tqdm(protein_dict.items()):
            j = k[:k.rfind('#')] if '#' in k else k
            if len(v) <= 2:
                exp = frap.add_experiment(j, v[0], v[1])
            elif len(v) > 2:
                exp = frap.add_experiment(j, v[0], v[1], molecular_weight=v[2], disorder=v[3])

    return frap


def generate_granule_data(**kwargs):
    the_dict = {
        f'Vasa(AA1-164)#{1}': [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_hypergerm/10hpf'],
        f'Vasa(AA1-164)#{2}': [24, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_hypergerm/24hpf'],
        f'Full length Vasa#{1}': [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_full_vasa/10hpf'],
        f'Full length Vasa#{2}': [24, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_full_vasa/24hpf'],
        f'Dead end#{1}': [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_dnd/10hpf'],
        f'Dead end#{2}': [24, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_dnd/24hpf'],
        f'Dazl#{1}': [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_dazl/10hpf'],
        f'Dazl#{2}': [24, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_dazl/24hpf'],
        f'Dazl F91A#{1}': [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_mutdazl/10hpf'],
        f'Granulito#{1}': [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_gra/10hpf'],
        f'Granulito#{2}': [24, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_gra/24hpf'],
        f'Tdrd7#{1}': [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_tdrd7/10hpf'],
        f'Nanos#{1}': [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_nanos/10hpf'],
        f'Nanos#{2}': [24, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_nanos/24hpf']
    }

    furapu = generate_frap_data(the_dict, '/Volumes/HELHEIM/analyzed_data/diffusivity/group_data',
                                pckl_name='all_granule_components', **kwargs)

    return furapu


def generate_glycogenase_data(**kwargs):
    the_dict = {
        'Control': [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/glycogen_overexpression/10hpf/ctrl',
                    43.90804, 0.2979],
        'Glycogenase OEx': [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/glycogen_overexpression/10hpf/oex',
                            43.90804, 0.2979],
        'Granulito wo RNA': [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_gra/10hpf',
                            43.90804, 0.2979]

    }

    furapu = generate_frap_data(the_dict, '/Volumes/HELHEIM/analyzed_data/diffusivity/group_data',
                                pckl_name='Glycogen_depletion', **kwargs)

    return furapu

# hyper10 = frap.add_experiment('Vasa(AA1-164)', 10,
#                               '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_hypergerm/10hpf',
#                               molecular_weight=44.25506, disorder=1)
# hyper24 = frap.add_experiment('Vasa(AA1-164)', 24, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_hypergerm/24hpf',
#                               molecular_weight=44.25506, disorder=1)
#
# vasa10 = frap.add_experiment('Full length Vasa', 10,
#                              '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_full_vasa/10hpf',
#                              molecular_weight=104.06211, disorder=0.4056)
# vasa24 = frap.add_experiment('Full length Vasa', 24,
#                              '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_full_vasa/24hpf',
#                              molecular_weight=104.06211, disorder=0.4056)
#
# dnd10 = frap.add_experiment('Dead end', 10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_dnd/10hpf',
#                             molecular_weight=73.17418, disorder=0.2871)
# dnd24 = frap.add_experiment('Dead end', 24, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_dnd/24hpf',
#                             molecular_weight=73.17418, disorder=0.2871)
#
# dazl10 = frap.add_experiment('Dazl', 10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_dazl/10hpf',
#                              molecular_weight=52.93738, disorder=0.4760)
# dazl24 = frap.add_experiment('Dazl', 24, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_dazl/24hpf',
#                              molecular_weight=52.93738, disorder=0.4760)
#
# mutdazl = frap.add_experiment('Dazl F91A', 10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_mutdazl/10hpf',
#                               molecular_weight=52.95139, disorder=0.4760)
#
# gra10 = frap.add_experiment('Granulito', 10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_gra/10hpf',
#                             molecular_weight=43.90804, disorder=0.2979)
# gra24 = frap.add_experiment('Granulito', 24, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_gra/24hpf',
#                             molecular_weight=43.90804, disorder=0.2979)
#
# tdrd7 = frap.add_experiment('Tdrd7', 10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_tdrd7/10hpf',
#                             molecular_weight=148, disorder=0)
#
# nos10 = frap.add_experiment('Nanos', 10, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_nanos/10hpf',
#                             molecular_weight=44.98995, disorder=0)
# nos24 = frap.add_experiment('Nanos', 24, '/Volumes/HELHEIM/analyzed_data/diffusivity/FRAP_nanos/24hpf',
#                             molecular_weight=44.98995, disorder=0)


def generate_rna_data(**kwargs):
    the_dict = {
        "Nanos3'UTR 400pg" : [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/202306_RNAinfluence/400pg/nos'],
        "Globin3'UTR 400pg": [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/202306_RNAinfluence/400pg/glob'],
        "Tdrd7a3'UTR 400pg": [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/202306_RNAinfluence/400pg/tdrd7'],
        "Nanos3'UTR 600pg": [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/202306_RNAinfluence/600pg/nos'],
        "Globin3'UTR 600pg": [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/202306_RNAinfluence/600pg/globin'],
        "Tdrd7a3'UTR 600pg": [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/202306_RNAinfluence/600pg/tdrd7a'],
    }

    furapu = generate_frap_data(the_dict, '/Volumes/HELHEIM/analyzed_data/diffusivity/group_data',
                                pckl_name='RNA_influence', **kwargs)

    return furapu


def generate_tdrd7mo_data(**kwargs):
    the_dict = {
        "Control MO" : [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/tdrd7MO_gra/controlMO'],
        "Tdrd7a MO": [10, '/Volumes/HELHEIM/analyzed_data/diffusivity/tdrd7MO_gra/tdrd7MO']
    }

    furapu = generate_frap_data(the_dict, '/Volumes/HELHEIM/analyzed_data/diffusivity/group_data',
                                pckl_name='Tdrd7a_KD_granulito ', **kwargs)

    return furapu
