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
from ROOT import RooRealVar, RooDataSet, RooArgSet, RooKeysPdf

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
        corr_bkgs_info_dict[pt_key] = {}
        histo_weights_dict = {}
        print(f"Processing pt bin: {pt_min} - {pt_max}")
        mass_min = pt_bin_fit_cfg['fit_range'][0]
        mass_max = pt_bin_fit_cfg['fit_range'][1]
        query_str = f"fPt >= {pt_min} and fPt < {pt_max} and fM >= {mass_min} and fM < {mass_max}"
        # query_str = f"fPt >= {pt_min} and fPt < {pt_max} and {config['bkg_score_column']} < {score_bkg_max} and {config['fd_score_column']} >= {score_fd_min} and {config['fd_score_column']} < {score_fd_max} and fM >= {mass_min} and fM < {mass_max}"
        cutset_sel_df = full_df.query(query_str)

        for fin_state, fin_state_info in final_states.items():
            if not any(fin_state in name for name in final_states_to_include):
                continue

            selected_df = cutset_sel_df.query(fin_state_info['query'])
            if len(selected_df) > 0:
                with getattr(uproot, write_file_mode)(out_file_name) as outfile:
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
            hBRs.Write()

        hWeightsAnchorSignal = pt_corr_bkgs['hWeightsSummary']
        outfile.cd(pt_key)
        hWeightsAnchorSignal.Write()
    outfile.Close()

def fill_smooth_histo(df, histo, n_points_for_sample, n_points_for_kde):

    # Define the RooDataset corresponding to histogram range
    x_min = histo.GetXaxis().GetXmin()
    x_max = histo.GetXaxis().GetXmax()
    x = RooRealVar("x", "x", x_min, x_max)
    data = RooDataSet("data", "data", RooArgSet(x))

    # Fill it from DataFrame
    for i_val, val in enumerate(df['fM']):
        if i_val > n_points_for_kde:
            break
        x.setVal(val)
        data.add(RooArgSet(x))

    # Build a RooKeysPdf (kernel smoothing)
    keys_pdf = RooKeysPdf("keys", "keys", x, data, RooKeysPdf.NoMirror)
    generated = keys_pdf.generate(RooArgSet(x), n_points_for_sample)

    histo_smooth = histo.Clone(f"{histo.GetName()}")
    histo_smooth.Reset("ICESM")
    for i in range(int(generated.numEntries())):
        val = generated.get(i).getRealValue("x")
        histo_smooth.Fill(val)

    histo_smooth.Scale(len(df) / histo_smooth.Integral())
    return histo_smooth

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
        fit_data = fit_data_file.Get(f"{pt_key}/hMassData")
        histo_weights_dict = {}
        print(f"Processing pt bin: {pt_min} - {pt_max}")
        pt_bin_fit_cfg = config["ry_extraction"]["pt_bins"][ipt_bin]
        mass_min = pt_bin_fit_cfg['fit_range'][0]
        mass_max = pt_bin_fit_cfg['fit_range'][1]
        if cfg_corrbkgs.get('reweight_prompt_non_prompt'):
            yield_prompt = len(full_df.query("fOriginMcRec == 1"))
            yield_non_prompt = len(full_df.query("fOriginMcRec == 2"))
            f_prompt_mc = yield_prompt / (yield_prompt + yield_non_prompt) if (yield_prompt + yield_non_prompt) > 0 else 0
            f_non_prompt_mc = 1 - f_prompt_mc

        query_str = f"fPt >= {pt_min} and fPt < {pt_max}"
        cutset_sel_df = full_df.query(query_str)
        print(f"Number of candidates after pt selection: {len(cutset_sel_df)}")

        if cfg_corrbkgs.get('smear_sigma'):
            mean = 0.0
            print(f"Applying mass smearing with sigma = {cfg_corrbkgs['smear_sigma'][ipt_bin]} GeV/c^2")
            cutset_sel_df.loc[:, "fM"] = cutset_sel_df["fM"] + np.random.normal(mean, 
                                                                    cfg_corrbkgs['smear_sigma'][ipt_bin],
                                                                    size=len(cutset_sel_df))


        for fin_state, fin_state_info in final_states.items():
            print(f"Processing final state: {fin_state}")

            if not any(fin_state == name for name in final_states_to_include) and fin_state != sgn_fin_state:
                continue

            channel_df = cutset_sel_df.query(fin_state_info['query'])
            if len(channel_df) == 0:
                continue

            histo_channel = fit_data.Clone()
            histo_channel.Reset('ICESM')
            histo_channel.SetName("hMass")
            for cand_mass in channel_df['fM']:
                histo_channel.Fill(cand_mass)
            
            # Fill tree from DataFrame
            histo_channel_smooth = None
            if cfg_corrbkgs.get('smooth') and fin_state != sgn_fin_state:
                histo_channel_smooth = fit_data.Clone()
                histo_channel_smooth.Reset('ICESM')
                histo_channel_smooth.SetName("hMass_smooth")
                histo_channel_smooth = fill_smooth_histo(channel_df, histo_channel_smooth, cfg_corrbkgs['n_smooth_points'], cfg_corrbkgs['n_points_for_kde'])
                print(f"Smoothed histogram for final state {fin_state} created, entries: {histo_channel_smooth.GetEntries()}.")

            try:
                query_ml_mass = f"fMlScore0 < {score_bkg_max} and fMlScore1 >= {score_fd_min} and fMlScore1 < {score_fd_max} and fM >= {mass_min} and fM < {mass_max}"
                cutset_sel_df_mass = channel_df.query(query_ml_mass)
            except Exception as e:
                query_mass = f"fM >= {mass_min} and fM < {mass_max}"
                cutset_sel_df_mass = channel_df.query(query_mass)

            # print(f"Number of selected candidates for final state {fin_state}: {len(selected_df)}")
            if len(cutset_sel_df_mass) > 0:
                make_dir_root_file(f"{pt_key}/{fin_state}", outfile)
                outfile.cd(f"{pt_key}/{fin_state}")
                outfile.cd(f"{pt_key}/{fin_state}")
                if histo_channel_smooth is not None:
                    histo_channel_smooth.Write()
                histo_channel.Write()
                hBRs = ROOT.TH1F("hBRs", "hBRs;Branching Ratio", 4, 0, 4)
                hBRs.GetXaxis().SetBinLabel(1, "MC")
                br_mc = fin_state_info[f'br_sim_{cfg_corrbkgs["coll_system"]}']
                hBRs.SetBinContent(1, br_mc)
                hBRs.GetXaxis().SetBinLabel(2, "PDG")
                br_pdg = fin_state_info['br_pdg']
                hBRs.SetBinContent(2, br_pdg)
                hBRs.GetXaxis().SetBinLabel(3, "Raw yield")
                raw_yield = len(cutset_sel_df_mass)
                hBRs.SetBinContent(3, raw_yield)
                hBRs.GetXaxis().SetBinLabel(4, "RY * (PDG/MC)")
                hBRs.SetBinContent(4, raw_yield * (br_pdg/br_mc))
                correct_abundance_wrt_sgn = fin_state_info.get(f"abundance_to_{config['Dmeson']}", 1)
                if cfg_corrbkgs.get('reweight_prompt_non_prompt'):
                    f_prompt_real = config['prompt_fraction']
                    f_non_prompt_real = 1 - f_prompt_real
                    sgn_yield = (yield_prompt * (f_prompt_real / f_prompt_mc) +
                                 yield_non_prompt * (f_non_prompt_real / f_non_prompt_mc))
                else:
                    sgn_yield = raw_yield
                histo_weights_dict[fin_state] = [sgn_yield * correct_abundance_wrt_sgn * (br_pdg/br_mc), histo_channel]
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
            if name == sgn_fin_state:
                hMassSignal.Add(histo, weight)
                total_signal_weight = weight
            else:
                hMassTotalCorrBkgs.Add(histo, weight)
            hWeightsAnchorSignal.GetXaxis().SetBinLabel(i_fin_state+1, name)
            hWeightsAnchorSignal.SetBinContent(i_fin_state+1, weight)


        for i_fin_state, (name, (weight, histo)) in enumerate(histo_weights_dict.items()):
            histo.Scale(weight / total_signal_weight)
            outfile.cd(f"{pt_key}/{name}")
            histo.Write("hMass_scaled_br")

        outfile.cd(pt_key)
        hMassSignal.Write()
        hMassTotalCorrBkgs.Write()
        hWeightsAnchorSignal.Write("hWeights")
        # Normalize weights histogram to the total signal weight
        hWeightsAnchorSignal.Scale(1 / total_signal_weight)
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
