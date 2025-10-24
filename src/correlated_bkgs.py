'''
    Normalizations are given for the fit function to use a signal PDF and
    template from hMassTotalCorrBkgs multiplied by a common normalization 
    constant.
'''

import pandas as pd
import matplotlib.pyplot as plt
import uproot
import numpy as np
import ROOT
from array import array
import os
import sys
import argparse
import yaml
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import logger, get_centrality_bins, make_dir_root_file
from corr_bkgs_brs import final_states

def produce_corr_bkgs_templs_trees(config, cutset_config):
    print("Producing correlated backgrounds templates - trees output")
    with open(cutset_config, 'r') as f:
        cfg_cutset = yaml.safe_load(f)

    full_dfs = []
    cfg_corrbkgs = config["corr_bkgs"]
    tables = [[] for table in cfg_corrbkgs["table_names"]]
    with uproot.open(cfg_corrbkgs["input_file"]) as f:
        for table_name, table_list in zip(cfg_corrbkgs["table_names"], tables):
            for iKey, key in enumerate(f.keys()):
                if table_name in key:
                    dfData = f[key].arrays(library='pd')
                    table_list.append(dfData)

            full_table_df = pd.concat([df for df in table_list], ignore_index=True)
            full_dfs.append(full_table_df)
    full_df = pd.concat(full_dfs, axis=1)

    ### Centrality selection
    _, (centMin, centMax) = get_centrality_bins(config["centrality"])
    full_df = full_df.query(f"fCentrality >= {centMin} and fCentrality < {centMax}")
    
    # pt-differential mass shifts
    shift = np.zeros(len(full_df))
    if cfg_corrbkgs.get('shift_mass_pt_diff'):
        if isinstance(cfg_corrbkgs["shift_mass"], float):
            shift_values = [cfg_corrbkgs["shift_mass"]] * len(cfg_cutset["Pt"]["min"])
        else:
            shift_values = cfg_corrbkgs["shift_mass"]
        for ptmin, ptmax, pt_diff_shift in zip(cfg_cutset["Pt"]["min"], cfg_cutset["Pt"]["max"], shift_values):
            mask = (full_df["pt"] >= ptmin) & (full_df["pt"] < ptmax)
            shift[mask] = pt_diff_shift

    full_df.loc[:, "fM"] = full_df["fM"] + shift

    # Process corr bkgs channels
    final_states_to_include = cfg_corrbkgs["include_final_states"]
    sgn_fin_state = cfg_corrbkgs['sgn_fin_state']
    fit_data_file = ROOT.TFile(cutset_config.replace("cutset", "proj").replace(".yml", ".root"), "READ")
    out_file_name = cutset_config.replace("cutset", "corrbkg").replace(".yml", ".root")
    write_file_mode = "recreate"

    corr_bkgs_info_dict = {}
    for ipt_bin, (pt_min, pt_max, score_bkg_max, score_fd_min, score_fd_max) in enumerate(zip(cfg_cutset["Pt"]["min"],
                                                                                              cfg_cutset["Pt"]["max"],
                                                                                              cfg_cutset["score_bkg"]["max"],
                                                                                              cfg_cutset["score_FD"]["min"],
                                                                                              cfg_cutset["score_FD"]["max"])):
        pt_key = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        print(f"pt_key: {pt_key}")
        corr_bkgs_info_dict[pt_key] = {}
        histo_weights_dict = {}
        print(f"Processing pt bin: {pt_min} - {pt_max}")
        mass_min = pt_bin_fit_cfg['fit_range'][0]
        mass_max = pt_bin_fit_cfg['fit_range'][1]
        query_str = f"fPt >= {pt_min} and fPt < {pt_max} and fM >= {mass_min} and fM < {mass_max}"
        # query_str = f"fPt >= {pt_min} and fPt < {pt_max} and {config['bkg_score_column']} < {score_bkg_max} and {config['fd_score_column']} >= {score_fd_min} and {config['fd_score_column']} < {score_fd_max} and fM >= {mass_min} and fM < {mass_max}"
        cutset_sel_df = full_df.query(query_str)

        for fin_state, fin_state_info in final_states.items():
            print(f"Processing final state: {fin_state}")
            if not any(fin_state in name for name in final_states_to_include):
                continue

            selected_df = cutset_sel_df.query(fin_state_info['query'])
            print(f"Number of selected candidates for final state {fin_state}: {len(selected_df)}")
            if len(selected_df) > 0:
                with getattr(uproot, write_file_mode)(out_file_name) as outfile:
                    print(f"Writing tree for {pt_key}/{fin_state}/treeMass")
                    outfile[f"{pt_key}/{fin_state}/treeMass"] = selected_df['fM'].to_frame()
                    write_file_mode = "update"

                hBRs = ROOT.TH1F("hBRs", "hBRs;Branching Ratio", 4, 0, 4)
                hBRs.GetXaxis().SetBinLabel(1, "MC")
                br_mc = fin_state_info.get(f"abundance_to_{config['Dmeson']}", 1) * (fin_state_info[f'br_sim_{cfg_corrbkgs["coll_system"]}'])
                hBRs.SetBinContent(1, br_mc)
                hBRs.GetXaxis().SetBinLabel(2, "PDG")
                br_pdg = fin_state_info['br_pdg']
                hBRs.SetBinContent(2, br_pdg)
                hBRs.GetXaxis().SetBinLabel(3, "Raw yield")
                raw_yield = len(selected_df)
                hBRs.SetBinContent(3, raw_yield)
                hBRs.GetXaxis().SetBinLabel(4, "RY * (PDG/MC)")
                hBRs.SetBinContent(4, raw_yield * (br_pdg/br_mc))
                histo_weights_dict[fin_state] = [raw_yield * (br_pdg/br_mc), len(selected_df)]

                if fin_state == sgn_fin_state:
                    signal_df = selected_df
                    total_signal_weight = raw_yield * (br_pdg/br_mc)

                corr_bkgs_info_dict[pt_key][fin_state] = {
                    "br_mc": br_mc,
                    "br_pdg": br_pdg,
                    "raw_yield": raw_yield,
                    "weight_to_sgn": raw_yield * (br_pdg/br_mc),
                }

        n_final_states = len(corr_bkgs_info_dict[pt_key])

        hWeightsAnchorSignal = ROOT.TH1F("hWeightsAnchorSignal", "hWeightsAnchorSignal", n_final_states, 0, n_final_states)
        for i_fin_state, (name, (weight, histo)) in enumerate(histo_weights_dict.items()):
            print(f"name: {name}, weight: {weight}")
            hWeightsAnchorSignal.GetXaxis().SetBinLabel(i_fin_state+1, name)
            hWeightsAnchorSignal.SetBinContent(i_fin_state+1, weight)

        # Normalize weights histogram to the total signal weight
        hWeightsAnchorSignal.Scale(1 / total_signal_weight)
        corr_bkgs_info_dict[pt_key]['hWeightsSummary'] = hWeightsAnchorSignal

    # Write output histograms
    outfile = ROOT.TFile(out_file_name, "UPDATE")
    for pt_key, pt_corr_bkgs in corr_bkgs_info_dict.items():
        for fin_state, fin_state_info in pt_corr_bkgs.items():
            if fin_state == 'hWeightsSummary':
                continue
            hBRs = ROOT.TH1F("hBRs", "hBRs;Branching Ratio", 4, 0, 4)
            hBRs.GetXaxis().SetBinLabel(1, "MC")
            hBRs.SetBinContent(1, fin_state_info['br_mc'])
            hBRs.GetXaxis().SetBinLabel(2, "PDG")
            hBRs.SetBinContent(2, fin_state_info['br_pdg'])
            hBRs.GetXaxis().SetBinLabel(3, "Raw yield")
            hBRs.SetBinContent(3, fin_state_info['raw_yield'])
            hBRs.GetXaxis().SetBinLabel(4, "RY * (PDG/MC)")
            hBRs.SetBinContent(4, fin_state_info['weight_to_sgn'])

            outfile.cd(f"{pt_key}/{fin_state}")
            print(f"Writing hBRs for {pt_key}/{fin_state}/hBRs")
            hBRs.Write()

        hWeightsAnchorSignal = pt_corr_bkgs['hWeightsSummary']
        outfile.cd(pt_key)
        print(f"Writing total histograms for {pt_key}")
        hWeightsAnchorSignal.Write()
    outfile.Close()

def produce_corr_bkgs_templs_histos(config, cutset_config):

    with open(cutset_config, 'r') as f:
        cfg_cutset = yaml.safe_load(f)

    full_dfs = []
    cfg_corrbkgs = config["corr_bkgs"]
    tables = [[] for table in cfg_corrbkgs["table_names"]]
    with uproot.open(cfg_corrbkgs["input_file"]) as f:
        for table_name, table_list in zip(cfg_corrbkgs["table_names"], tables):
            for iKey, key in enumerate(f.keys()):
                if table_name in key:
                    dfData = f[key].arrays(library='pd')
                    table_list.append(dfData)

            full_table_df = pd.concat([df for df in table_list], ignore_index=True)
            full_dfs.append(full_table_df)
    full_df = pd.concat(full_dfs, axis=1)

    ### Centrality selection
    _, (centMin, centMax) = get_centrality_bins(config["centrality"])
    print(f"Applying centrality selection: {centMin} - {centMax}")
    print(f"Initial number of candidates: {len(full_df)}")
    full_df = full_df.query(f"fCentrality >= {centMin} and fCentrality < {centMax}")
    print(f"Number of candidates after centrality selection: {len(full_df)}")
    # pt-differential mass shifts
    shift = np.zeros(len(full_df))
    if cfg_corrbkgs.get('shift_mass_pt_diff'):
        if isinstance(cfg_corrbkgs["shift_mass"], float):
            shift_values = [cfg_corrbkgs["shift_mass"]] * len(cfg_cutset["Pt"]["min"])
        else:
            shift_values = cfg_corrbkgs["shift_mass"]
        for ptmin, ptmax, pt_diff_shift in zip(cfg_cutset["Pt"]["min"], cfg_cutset["Pt"]["max"], shift_values):
            mask = (full_df["pt"] >= ptmin) & (full_df["pt"] < ptmax)
            shift[mask] = pt_diff_shift

    full_df.loc[:, "fM"] = full_df["fM"] + shift

    # Process corr bkgs channels
    final_states_to_include = cfg_corrbkgs["include_final_states"]
    sgn_fin_state = cfg_corrbkgs['sgn_fin_state']
    print(f"cutset_config.replace('cutset', 'proj').replace('.yml', '.root'): {cutset_config.replace('cutset', 'proj').replace('.yml', '.root')}")
    fit_data_file = ROOT.TFile(cutset_config.replace("cutset", "proj").replace(".yml", ".root"), "READ")
    out_file_name = cutset_config.replace("cutset", "corrbkg").replace(".yml", ".root")
    outfile = ROOT.TFile(out_file_name, "RECREATE")

    for ipt_bin, (pt_min, pt_max, score_bkg_max, score_fd_min, score_fd_max) in enumerate(zip(cfg_cutset["Pt"]["min"],
                                                                                              cfg_cutset["Pt"]["max"],
                                                                                              cfg_cutset["score_bkg"]["max"],
                                                                                              cfg_cutset["score_FD"]["min"],
                                                                                              cfg_cutset["score_FD"]["max"])):
        pt_key = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        print(f"pt_key: {pt_key}")
        fit_data = fit_data_file.Get(f"{pt_key}/hMassData")
        histo_weights_dict = {}
        print(f"Processing pt bin: {pt_min} - {pt_max}")
        pt_bin_fit_cfg = config["ry_extraction"]["pt_bins"][ipt_bin]
        mass_min = pt_bin_fit_cfg['fit_range'][0]
        mass_max = pt_bin_fit_cfg['fit_range'][1]
        try:
            query_str = f"fPt >= {pt_min} and fPt < {pt_max} and fMlScore0 < {score_bkg_max} and fMlScore1 >= {score_fd_min} and fMlScore1 < {score_fd_max} and fM >= {mass_min} and fM < {mass_max}"
            cutset_sel_df = full_df.query(query_str)
        except Exception as e:
            print(f"Exception in applying query with ML selections --> only mass and pt!")
            query_str = f"fPt >= {pt_min} and fPt < {pt_max} and fM >= {mass_min} and fM < {mass_max}"
            cutset_sel_df = full_df.query(query_str)
        print(f"Number of candidates after cutset selection: {len(cutset_sel_df)}")

        for fin_state, fin_state_info in final_states.items():
            print(f"Processing final state: {fin_state}")

            if not any(fin_state == name for name in final_states_to_include) and fin_state != sgn_fin_state:
                continue

            selected_df = cutset_sel_df.query(fin_state_info['query'])
            print(f"Number of selected candidates for final state {fin_state}: {len(selected_df)}")
            if len(selected_df) > 0:
                make_dir_root_file(f"{pt_key}/{fin_state}", outfile)
                outfile.cd(f"{pt_key}/{fin_state}")

                # Fill tree from DataFrame
                histo_channel = fit_data.Clone()
                histo_channel.Reset('ICESM')
                histo_channel.SetName("hMass")
                for cand_mass in selected_df['fM']:
                    histo_channel.Fill(cand_mass)

                outfile.cd(f"{pt_key}/{fin_state}")
                histo_channel.Smooth(10)
                histo_channel.Write()
                hBRs = ROOT.TH1F("hBRs", "hBRs;Branching Ratio", 4, 0, 4)
                hBRs.GetXaxis().SetBinLabel(1, "MC")
                br_mc = fin_state_info.get(f"abundance_to_{config['Dmeson']}", 1) * (fin_state_info[f'br_sim_{cfg_corrbkgs["coll_system"]}'])
                hBRs.SetBinContent(1, br_mc)
                hBRs.GetXaxis().SetBinLabel(2, "PDG")
                br_pdg = fin_state_info['br_pdg']
                hBRs.SetBinContent(2, br_pdg)
                hBRs.GetXaxis().SetBinLabel(3, "Raw yield")
                raw_yield = len(selected_df)
                hBRs.SetBinContent(3, raw_yield)
                hBRs.GetXaxis().SetBinLabel(4, "RY * (PDG/MC)")
                hBRs.SetBinContent(4, raw_yield * (br_pdg/br_mc))
                print(f"Storing weight for final state {fin_state}: {raw_yield * (br_pdg/br_mc)}")
                histo_weights_dict[fin_state] = [raw_yield * (br_pdg/br_mc), histo_channel]
                hBRs.Write()

        n_final_states = len(histo_weights_dict)

        hMassSignal = fit_data.Clone()
        hMassSignal.Reset('ICESM')
        hMassSignal.SetName("hMassSignal")
        hMassTotalCorrBkgs = fit_data.Clone()
        hMassTotalCorrBkgs.Reset('ICESM')
        hMassTotalCorrBkgs.SetName("hMassTotalCorrBkgs")
        hWeightsAnchorSignal = ROOT.TH1F("hWeightsAnchorSignal", "hWeightsAnchorSignal", n_final_states, 0, n_final_states)
        total_signal_weight = 0
        for i_fin_state, (name, (weight, histo)) in enumerate(histo_weights_dict.items()):
            print(f"Adding signal final state {name} with weight {weight}")
            if name == sgn_fin_state:
                hMassSignal.Add(histo, weight)
                total_signal_weight = weight
            else:
                hMassTotalCorrBkgs.Add(histo, weight)
            hWeightsAnchorSignal.GetXaxis().SetBinLabel(i_fin_state+1, name)
            hWeightsAnchorSignal.SetBinContent(i_fin_state+1, weight)

        # Normalize weights histogram to the total signal weight
        hWeightsAnchorSignal.Scale(1 / total_signal_weight)

        for i_fin_state, (name, (weight, histo)) in enumerate(histo_weights_dict.items()):
            histo.Scale(weight / total_signal_weight)
            outfile.cd(f"{pt_key}/{name}")
            histo.Write("hMass_scaled_br")

        outfile.cd(pt_key)
        hMassSignal.Write()
        hMassTotalCorrBkgs.Write()
        hWeightsAnchorSignal.Write()

    outfile.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("config", metavar="text",
                        default="config.yaml", help="flow configuration file")
    parser.add_argument("cutset_config", metavar="text",
                        default="cfg_cutset.yaml", help="flow configuration file")
    parser.add_argument('--final_states_only', '-fso', action='store_true',
                        help="separate only final states")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    produce_trees = config['input_type'] == 'Tree'

    if args.final_states_only:
        if produce_trees:
            produce_corr_bkgs_templs_trees(config, args.cutset_config)
        else:
            print("PRODUCING CORRELATED BACKGROUNDS TEMPLATES - HISTOGRAMS OUTPUT")
            produce_corr_bkgs_templs_histos(config, args.cutset_config)
    else:
        produce_corr_bkgs_templs(args.config, args.cutset_config)
