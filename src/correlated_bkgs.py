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
from ROOT import RooRealVar, RooDataSet, RooArgSet, RooKeysPdf, TFile, TH3F, TH1F

def get_corr_bkg(cfg_cutset, corr_bkg_file, corr_bkg_chn, fit_range, pt_label, templ_type, output_type, ):
    '''
    Get correlated background template and normalization factor
    '''
    input_folder = f"{pt_label}/{corr_bkg_chn}"
    try:
        hist_pdg_mc_brs = corr_bkg_file.Get(f"{input_folder}/hBRs")
    except:
        logger(f"Could not retrieve hBRs histogram from {input_folder} in file {corr_bkg_file}", "ERROR")
        sys.exit(1)
    br_pdg = hist_pdg_mc_brs.GetBinContent(2)
    br_mc = hist_pdg_mc_brs.GetBinContent(1)
    print(f"Branching ratios: PDG = {br_pdg}, MC = {br_mc}")
    templ_tree_mass = corr_bkg_file.Get(f"{input_folder}/{templ_type}/treeMass")
    templ_tree_mass.SetDirectory(0)
    templ_histo_mass = corr_bkg_file.Get(f"{input_folder}/{templ_type}/hMassSmooth")
    templ_histo_mass.SetDirectory(0)
    full_tree = corr_bkg_file.Get(f"{input_folder}/{templ_type}/treeFracMassScoresBkgFD")
    templ_rdataframe_full = ROOT.RDataFrame(full_tree)
    n_entries = (
        templ_rdataframe_full.Filter(
            f"fMlScore0 < {cfg_cutset['score_bkg_max']} && "
            f"fMlScore1 >= {cfg_cutset['score_fd_min']} && "
            f"fMlScore1 < {cfg_cutset['score_fd_max']} && "
            f"fM >= {fit_range[0]} && fM < {fit_range[1]}"
        ).Count().GetValue()
    )

    frac = (br_pdg / br_mc) * n_entries # TODO: add correction of MC abundances
    if output_type == "hist":
        print(f"Returning frac {frac} for correlated bkg source {corr_bkg_chn}")
        return templ_histo_mass, frac
    elif output_type == "tree":
        print(f"Returning frac {frac} for correlated bkg source {corr_bkg_chn}")
        return templ_tree_mass, frac
    else:
        logger(f"Output type {output_type} not recognized. Choose between 'hist' or 'tree'.", "ERROR")
        sys.exit(1)

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

def shift_templs(cfg_corrbkgs, cutset_sel_df, pt_min, pt_max):
    # pt-differential mass shifts
    
    # Copy the dataframe to avoid modifying the original one
    df = cutset_sel_df.copy(deep=True)
    
    mass_shift = 0.
    print(f"Applying mass shift correction")
    if isinstance(cfg_corrbkgs["shift_mass"], float):
        print(f"Applying constant mass shift: {cfg_corrbkgs['shift_mass']}")
        mass_shift = cfg_corrbkgs["shift_mass"]
    else:
        print(f"Taking mass shifts from file: {cfg_corrbkgs['shift_mass']}")
        logger(f"Taking mass shifts from {cfg_corrbkgs['shift_mass']}", "INFO")
        shifts_file = ROOT.TFile(cfg_corrbkgs['shift_mass'], "READ")
        shifts_histo = shifts_file.Get("delta_mean_data_mc")
        for i_bin in range(1, shifts_histo.GetNbinsX()+1):
            bin_center = shifts_histo.GetBinCenter(i_bin)
            if (bin_center > pt_min and bin_center < pt_max):
                print(f"----> [bin_center: {bin_center}, pt_min: {pt_min}, pt_max: {pt_max}] Applying mass shift: {shifts_histo.GetBinContent(i_bin)} GeV/c^2")
                mass_shift = shifts_histo.GetBinContent(i_bin)
                break
        shifts_histo.SetDirectory(0)
        shifts_file.Close()

    print(f"Shifting mass by {mass_shift} GeV/c^2")
    df.loc[:, "fM"] = df["fM"] + mass_shift
    return df

def smear_templs(cfg_corrbkgs, cutset_sel_df, pt_min, pt_max):
    # pt-differential mass smearing
    mass_smear = 0.
    # Copy the dataframe to avoid modifying the original one
    df = cutset_sel_df.copy(deep=True)
    
    if isinstance(cfg_corrbkgs["smear_mass"], float):
        sigma_smear = cfg_corrbkgs["smear_mass"]
    else:
        logger(f"Taking mass smears from {cfg_corrbkgs['smear_mass']}", "INFO")
        smear_file = ROOT.TFile(cfg_corrbkgs['smear_mass'], "READ")
        smear_histo = smear_file.Get("delta_sigma_data_mc")
        for i_bin in range(1, smear_histo.GetNbinsX()+1):
            bin_center = smear_histo.GetBinCenter(i_bin)
            if (bin_center > pt_min and bin_center < pt_max):
                sigma_smear = smear_histo.GetBinContent(i_bin)
                print(f"----> [bin_center: {bin_center}, pt_min: {pt_min}, pt_max: {pt_max}] Applying mass smearing: {sigma_smear} GeV/c^2")
                break
        smear_histo.SetDirectory(0)
        smear_file.Close()
        if sigma_smear > 0:
            mass_smear = np.random.normal(0.0, sigma_smear, size=len(df)).astype("float32")
            print(f"Smearing mass by sigma = {mass_smear[:10]} GeV/c^2")
            df.loc[:, "fM"] = df["fM"] + mass_smear
        else:
            logger(f"Mass smearing value is {sigma_smear}, no smearing applied.", "WARNING")
    return df

def produce_chn_corrbkg(cfg_corrbkgs, df, outfile, chn_dir, templ_type='raw'):

    outfile.mkdir(f'{chn_dir}/{templ_type}')
    outfile.cd(f'{chn_dir}/{templ_type}')

    histo_mass = TH1F(f"hMass", f"hMass", 700, 1.6, 2.3)
    histo_channel = TH3F(f"hMassScoresBkgFD", f"hMassScoresBkgFD", 700, 1.6, 2.3, 100, 0, 1, 100, 0, 1)
    histo_channel.Reset('ICESM')
    histo_channel.SetName("hMassRaw")

    treeFrac = ROOT.TTree("treeFrac", "treeFrac")
    treeMass = ROOT.TTree("treeMass", "treeMass")

    # Create branches and buffers
    fM_mass = np.zeros(1, dtype=np.float32)
    treeMass.Branch("fM", fM_mass, "fM/F")
    fM_frac = np.zeros(1, dtype=np.float32)
    treeFrac.Branch("fM", fM_frac, "fM/F")
    fPt = np.zeros(1, dtype=np.float32)
    treeFrac.Branch("fPt", fPt, "fPt/F")
    fCentrality = np.zeros(1, dtype=np.float32)
    treeFrac.Branch("fCentrality", fCentrality, "fCentrality/F")
    fMlScore0 = np.zeros(1, dtype=np.float32)
    treeFrac.Branch("fMlScore0", fMlScore0, "fMlScore0/F")
    fMlScore1 = np.zeros(1, dtype=np.float32)
    treeFrac.Branch("fMlScore1", fMlScore1, "fMlScore1/F")

    for mass, score_bkg, score_fd, pt, centrality in zip(df['fM'], df['fMlScore0'], df['fMlScore1'], df['fPt'], df['fCentrality']):
        histo_channel.Fill(mass, score_bkg, score_fd)
        histo_mass.Fill(mass)
        fM_mass[0] = mass
        fM_frac[0] = mass
        fPt[0] = pt
        fCentrality[0] = centrality
        fMlScore0[0] = score_bkg
        fMlScore1[0] = score_fd
        treeFrac.Fill()
        treeMass.Fill()
    print(f"Filled histogram with entries: {histo_channel.GetEntries()}.")

    histo_mass_smooth = histo_mass.Clone()
    histo_mass_smooth.Reset('ICESM')
    histo_mass_smooth.SetName("hMassSmooth")
    histo_mass_smooth = fill_smooth_histo(df, histo_mass_smooth, cfg_corrbkgs['n_smooth_points'], cfg_corrbkgs['n_points_for_kde'])
    histo_mass_smooth.Smooth(100)
    print(f"Smoothed histogram created with entries: {histo_mass_smooth.GetEntries()}.")

    histo_mass.Write('hMassRaw')
    histo_mass_smooth.Write('hMassSmooth')
    histo_channel.Write('hMassBkgFD')
    treeFrac.Write('treeFracMassScoresBkgFD')
    treeMass.Write('treeMass')

def produce_corr_bkgs_templs(config):

    cfg_corrbkgs = config["corr_bkgs"]

    full_dfs = []
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

    cent_sel_df = full_df.query(f"fCentrality >= {centMin} and fCentrality < {centMax}")
    print(f"Initial candidates: {len(full_df)} ----> after cent and pt selection: {len(cent_sel_df)}")

    # Process corr bkgs channels
    sgn_fin_state = cfg_corrbkgs['sgn_fin_state']

    for i_pt, pt_bin_cfg in enumerate(config["pt_bins"]):
        pt_min, pt_max = pt_bin_cfg['pt_range'][0], pt_bin_cfg['pt_range'][1]
        pt_key = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        outfile = TFile(f"{cfg_corrbkgs['outfile']}_{pt_key}.root", "RECREATE")
        print(f"\nProcessing pt bin: {pt_min} - {pt_max}")

        cent_pt_sel_df = cent_sel_df.query(f"fPt >= {pt_min} and fPt < {pt_max}")

        for fin_state, fin_state_info in final_states.items():
            print(f"Processing final state: {fin_state}")

            channel_df = cent_pt_sel_df.query(fin_state_info['query'])
            if len(channel_df) <= cfg_corrbkgs.get("min_entries", 0):
                print(f"----> No candidates for final state: {fin_state}, skipping.")
                continue

            chn_dir = f"{pt_key}/{fin_state}"
            make_dir_root_file(chn_dir, outfile)
            outfile.cd(chn_dir)
            hBRs = ROOT.TH1F("hBRs", "hBRs;Branching Ratio", 2, 0, 2)
            hBRs.GetXaxis().SetBinLabel(1, "MC")
            br_mc = fin_state_info[f'br_sim_{cfg_corrbkgs["coll_system"]}']
            hBRs.SetBinContent(1, br_mc)
            hBRs.GetXaxis().SetBinLabel(2, "PDG")
            br_pdg = fin_state_info['br_pdg']
            hBRs.SetBinContent(2, br_pdg)
            hBRs.Write()

            produce_chn_corrbkg(cfg_corrbkgs, channel_df, outfile, chn_dir, templ_type='raw')

            if cfg_corrbkgs.get('smear_mass'):
                channel_df_smear = smear_templs(cfg_corrbkgs, channel_df, pt_min, pt_max)
                produce_chn_corrbkg(cfg_corrbkgs, channel_df_smear, outfile, chn_dir, templ_type='smear')
            if cfg_corrbkgs.get('shift_mass'):
                channel_df_shift = shift_templs(cfg_corrbkgs, channel_df, pt_min, pt_max)
                produce_chn_corrbkg(cfg_corrbkgs, channel_df_shift, outfile, chn_dir, templ_type='shift')
            if cfg_corrbkgs.get('smear_mass') and cfg_corrbkgs.get('shift_mass'):
                channel_df_smear = smear_templs(cfg_corrbkgs, channel_df, pt_min, pt_max)
                channel_df_shift_smear = shift_templs(cfg_corrbkgs, channel_df_smear, pt_min, pt_max)
                produce_chn_corrbkg(cfg_corrbkgs, channel_df_shift_smear, outfile, chn_dir, templ_type='shift_smear')

        outfile.Close()
        print(f"\nOutput file with correlated backgrounds templates: {cfg_corrbkgs['outfile']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("config", metavar="text",
                        default="config.yaml", help="flow configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger("Producing correlated backgrounds templates", "INFO")
    produce_corr_bkgs_templs(config)




















# def produce_corr_bkgs_templs_trees(config, cutset_config):
#     print("Producing correlated backgrounds templates - trees output")
#     with open(cutset_config, 'r') as f:
#         cfg_cutset = yaml.safe_load(f)

#     full_dfs = []
#     cfg_corrbkgs = config["corr_bkgs"]
#     tables = [[] for table in cfg_corrbkgs["table_names"]]
#     with uproot.open(cfg_corrbkgs["input_file"]) as f:
#         for table_name, table_list in zip(cfg_corrbkgs["table_names"], tables):
#             for iKey, key in enumerate(f.keys()):
#                 if table_name in key:
#                     dfData = f[key].arrays(library='pd')
#                     table_list.append(dfData)

#             full_table_df = pd.concat([df for df in table_list], ignore_index=True)
#             full_dfs.append(full_table_df)
#     full_df = pd.concat(full_dfs, axis=1)

#     ### Centrality selection
#     _, (centMin, centMax) = get_centrality_bins(config["centrality"])
#     full_df = full_df.query(f"fCentrality >= {centMin} and fCentrality < {centMax}")
    
#     # pt-differential mass shifts
#     shifts = np.zeros(len(full_df))
#     if cfg_corrbkgs.get('shift_mass'):
#         if isinstance(cfg_corrbkgs["shift_mass"], float):
#             shift_values = [cfg_corrbkgs["shift_mass"]] * len(cfg_cutset["pt_min"])
#         elif isinstance(cfg_corrbkgs["shift_mass"], list):
#             shift_values = cfg_corrbkgs["shift_mass"]
#         else:
#             logger(f"Taking mass shifts from {cfg_corrbkgs['shift_mass']}", "INFO")
#             shifts_file = ROOT.TFile(cfg_corrbkgs['shift_mass'], "READ")
#             shifts_histo = shifts_file.Get("delta_mean_data_mc")
#             for i_bin in range(1, shifts_histo.GetNbinsX()+1):
#                 shift_values.append(shifts_histo.GetBinContent(i_bin))
#         for ptmin, ptmax, pt_diff_shift in zip(cfg_cutset["pt_min"], cfg_cutset["pt_max"], shift_values):
#             mask = (full_df["fPt"] >= ptmin) & (full_df["fPt"] < ptmax)
#             shifts[mask] = pt_diff_shift

#     full_df.loc[:, "fM"] = full_df["fM"] + shifts

#     # pt-differential mass shifts
#     smears = np.zeros(len(full_df))
#     if cfg_corrbkgs.get('smear_mass'):
#         if isinstance(cfg_corrbkgs["smear_mass"], float):
#             smear_values = [cfg_corrbkgs["smear_mass"]] * len(cfg_cutset["pt_min"])
#         elif isinstance(cfg_corrbkgs["smear_mass"], list):
#             smear_values = cfg_corrbkgs["smear_mass"]
#         else:
#             logger(f"Taking mass shifts from {cfg_corrbkgs['smear_mass']}", "INFO")
#             smear_file = ROOT.TFile(cfg_corrbkgs['smear_mass'], "READ")
#             smear_histo = smear_file.Get("delta_mean_data_mc")
#             for i_bin in range(1, smear_histo.GetNbinsX()+1):
#                 smear_values.append(smear_histo.GetBinContent(i_bin))
#         for ptmin, ptmax, pt_diff_smear in zip(cfg_cutset["pt_min"], cfg_cutset["pt_max"], smear_values):
#             mask = (full_df["fPt"] >= ptmin) & (full_df["fPt"] < ptmax)
#             smears[mask] = pt_diff_smear

#     full_df.loc[:, "fM"] = full_df["fM"] + smears

#     # Process corr bkgs channels
#     final_states_to_include = cfg_corrbkgs["include_final_states"]
#     sgn_fin_state = cfg_corrbkgs['sgn_fin_state']
#     fit_data_file = ROOT.TFile(cutset_config.replace("cutset", "proj").replace(".yml", ".root"), "READ")
#     out_file_name = cutset_config.replace("cutset", "corrbkg").replace(".yml", ".root")
#     write_file_mode = "recreate"

#     corr_bkgs_info_dict = {}
#     for ipt_bin, (pt_min, pt_max, score_bkg_max, score_fd_min, score_fd_max) in enumerate(zip(cfg_cutset["pt_min"],
#                                                                                               cfg_cutset["pt_max"],
#                                                                                               cfg_cutset["score_bkg"]["max"],
#                                                                                               cfg_cutset["score_fd"]["min"],
#                                                                                               cfg_cutset["score_fd"]["max"])):
#         pt_key = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
#         corr_bkgs_info_dict[pt_key] = {}
#         histo_weights_dict = {}
#         print(f"Processing pt bin: {pt_min} - {pt_max}")
#         mass_min = pt_bin_fit_cfg['fit_range'][0]
#         mass_max = pt_bin_fit_cfg['fit_range'][1]
#         query_str = f"fPt >= {pt_min} and fPt < {pt_max} and fM >= {mass_min} and fM < {mass_max}"
#         # query_str = f"fPt >= {pt_min} and fPt < {pt_max} and {config['bkg_score_column']} < {score_bkg_max} and {config['fd_score_column']} >= {score_fd_min} and {config['fd_score_column']} < {score_fd_max} and fM >= {mass_min} and fM < {mass_max}"
#         cutset_sel_df = full_df.query(query_str)

#         for fin_state, fin_state_info in final_states.items():
#             if not any(fin_state in name for name in final_states_to_include):
#                 continue

#             selected_df = cutset_sel_df.query(fin_state_info['query'])
#             if len(selected_df) > 0:
#                 with getattr(uproot, write_file_mode)(out_file_name) as outfile:
#                     outfile[f"{pt_key}/{fin_state}/treeMass"] = selected_df['fM'].to_frame()
#                     write_file_mode = "update"

#                 hBRs = ROOT.TH1F("hBRs", "hBRs;Branching Ratio", 4, 0, 4)
#                 hBRs.GetXaxis().SetBinLabel(1, "MC")
#                 br_mc = fin_state_info.get(f"abundance_to_{config['Dmeson']}", 1) * (fin_state_info[f'br_sim_{cfg_corrbkgs["coll_system"]}'])
#                 hBRs.SetBinContent(1, br_mc)
#                 hBRs.GetXaxis().SetBinLabel(2, "PDG")
#                 br_pdg = fin_state_info['br_pdg']
#                 hBRs.SetBinContent(2, br_pdg)
#                 hBRs.GetXaxis().SetBinLabel(3, "Raw yield")
#                 raw_yield = len(selected_df)
#                 hBRs.SetBinContent(3, raw_yield)
#                 hBRs.GetXaxis().SetBinLabel(4, "RY * (PDG/MC)")
#                 hBRs.SetBinContent(4, raw_yield * (br_pdg/br_mc))
#                 histo_weights_dict[fin_state] = [raw_yield * (br_pdg/br_mc), len(selected_df)]

#                 if fin_state == sgn_fin_state:
#                     signal_df = selected_df
#                     total_signal_weight = raw_yield * (br_pdg/br_mc)

#                 corr_bkgs_info_dict[pt_key][fin_state] = {
#                     "br_mc": br_mc,
#                     "br_pdg": br_pdg,
#                     "raw_yield": raw_yield,
#                     "weight_to_sgn": raw_yield * (br_pdg/br_mc),
#                 }

#         n_final_states = len(corr_bkgs_info_dict[pt_key])

#         hWeightsAnchorSignal = ROOT.TH1F("hWeightsAnchorSignal", "hWeightsAnchorSignal", n_final_states, 0, n_final_states)
#         for i_fin_state, (name, (weight, histo)) in enumerate(histo_weights_dict.items()):
#             hWeightsAnchorSignal.GetXaxis().SetBinLabel(i_fin_state+1, name)
#             hWeightsAnchorSignal.SetBinContent(i_fin_state+1, weight)

#         # Normalize weights histogram to the total signal weight
#         hWeightsAnchorSignal.Scale(1 / total_signal_weight)
#         corr_bkgs_info_dict[pt_key]['hWeightsSummary'] = hWeightsAnchorSignal

#     # Write output histograms
#     outfile = ROOT.TFile(out_file_name, "UPDATE")
#     for pt_key, pt_corr_bkgs in corr_bkgs_info_dict.items():
#         for fin_state, fin_state_info in pt_corr_bkgs.items():
#             if fin_state == 'hWeightsSummary':
#                 continue
#             hBRs = ROOT.TH1F("hBRs", "hBRs;Branching Ratio", 4, 0, 4)
#             hBRs.GetXaxis().SetBinLabel(1, "MC")
#             hBRs.SetBinContent(1, fin_state_info['br_mc'])
#             hBRs.GetXaxis().SetBinLabel(2, "PDG")
#             hBRs.SetBinContent(2, fin_state_info['br_pdg'])
#             hBRs.GetXaxis().SetBinLabel(3, "Raw yield")
#             hBRs.SetBinContent(3, fin_state_info['raw_yield'])
#             hBRs.GetXaxis().SetBinLabel(4, "RY * (PDG/MC)")
#             hBRs.SetBinContent(4, fin_state_info['weight_to_sgn'])

#             outfile.mkdir(fin_state)
#             outfile.cd(fin_state)
#             hBRs.Write()

#         hWeightsAnchorSignal = pt_corr_bkgs['hWeightsSummary']
#         outfile.cd()
#         hWeightsAnchorSignal.Write()
#     outfile.Close()