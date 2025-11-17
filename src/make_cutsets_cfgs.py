'''
Script to create yaml files corresponding to a set of cutset to perform
the cut variation analysis in the combined and correlated cases
python3 make_cutsets_cfgs.py config_flow.yml -o path/to/output [--correlated]
Without --correlated, the script will create yaml files for the combined case
'''
import yaml
import argparse
import os
import numpy as np
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))  # Append parent directory

def pad_to_length(list, target_len):
    '''
        Function to pad a list to a target length
        Args:
            lst (list): list to be padded
            target_len (int): target length of the list
        Returns:
            list: padded list
    '''
    list_length = len(list)
    len_offset = target_len - list_length
    return list + [list[-1]] * len_offset if list_length < target_len else list

def make_yaml_cutvar(flow_config, outputdir, correlated):
    '''
        Function to create a yaml file with a set of cuts for ML
        Args:
            flow_config (str): path to the flow config file
            outputdir (str): path to the output directory
    '''
    with open(flow_config, 'r') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(f'{outputdir}/cutsets', exist_ok=True)
    cut_var = cfg['ry_setup']['cut_variation']
    fd_cuts = np.arange(cut_var['fd_min'], cut_var['fd_max'], cut_var['fd_step'])
    for i_cut, fd_cut in enumerate(fd_cuts):
        combinations = {
            'icutset': f"{i_cut:02}",
            'pt_min': cfg['ry_setup']['pt_range'][0], 
            'pt_max': cfg['ry_setup']['pt_range'][1],
            'score_bkg_min': 0.0,
            'score_bkg_max': cut_var['bkg_max'],
            'score_fd_min': float(fd_cut),
            'score_fd_max': 1.0
        }
        print(f'{outputdir}/cutsets/cutset_{i_cut:02}.yml')
        with open(f'{outputdir}/cutsets/cutset_{i_cut:02}.yml', 'w') as file:
            yaml.dump(combinations, file, default_flow_style=False, sort_keys=False)

    print(f'Cutsets saved in {outputdir}/cutsets')

def make_yaml_ry(flow_config, outputdir, correlated):
    '''
        Function to create a yaml file with a set of cuts for ML
        Args:
            flow_config (str): path to the flow config file
            outputdir (str): path to the output directory
    '''
    with open(flow_config, 'r') as f:
        cfg = yaml.safe_load(f)

    pt_bin_cfg = {
        'icutset': 'central',
        'pt_min': cfg['ry_setup']['pt_range'][0], 
        'pt_max': cfg['ry_setup']['pt_range'][1],
        'score_bkg_min': 0.0,
        'score_bkg_max': cfg['ry_setup']['bkg_cut_central'],
        'score_fd_min': cfg['ry_setup']['fd_min_cut_central'],
        'score_fd_max': cfg['ry_setup']['fd_max_cut_central']
    }

    outdir = os.path.join(cfg["outdir"], cfg["outfolder"])
    os.makedirs(outdir, exist_ok=True)
    with open(f'{outdir}/cutset.yml', 'w') as file:
        yaml.dump(pt_bin_cfg, file, default_flow_style=False, sort_keys=False)

    print(f'Cutsets saved in {outdir}/cutset.yml')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('flow_config', metavar='text', default='config_flow.yml')
    parser.add_argument("--outputdir", "-o", metavar="text", default=".", help="output directory")
    parser.add_argument("--config_type", "-cfg_type", metavar="text", default="ry", help="Produce yml files for ry (raw yield) or cut variation")
    args = parser.parse_args()

    if args.config_type == 'cutvar':
        make_yaml_cutvar(args.flow_config, args.outputdir, args.config_type)
    elif args.config_type == 'ry':
        make_yaml_ry(args.flow_config, args.outputdir, args.config_type)
    else:
        raise ValueError(f"Unknown type {args.config_type}, please use 'ry' or 'cutvar'")
