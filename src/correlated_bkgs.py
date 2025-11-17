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

def get_corr_bkg(cfg_cutset, corr_bkg_file, corr_bkg_chn, fit_range, pt_label, templ_type, output_type, sgn_d_meson='Dplus', corr_abundances=False):
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

    corr_abundance = 1 if not corr_abundances else final_states[corr_bkg_chn].get(f"abundance_to_{sgn_d_meson}", 1)
    if corr_abundance != 1:
        logger(f"Applying abundance correction factor of {corr_abundance} for correlated bkg source {corr_bkg_chn}", "WARNING")
    frac = (br_pdg / br_mc) * n_entries * corr_abundance
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
        histo_mass.Fill(mass)
        fM_mass[0] = mass
        fM_frac[0] = mass
        fPt[0] = pt
        fCentrality[0] = centrality
        fMlScore0[0] = score_bkg
        fMlScore1[0] = score_fd
        treeFrac.Fill()
        treeMass.Fill()
    print(f"Filled histogram with entries: {histo_mass.GetEntries()}.")

    histo_mass_smooth = histo_mass.Clone()
    histo_mass_smooth.Reset('ICESM')
    histo_mass_smooth.SetName("hMassSmooth")
    histo_mass_smooth = fill_smooth_histo(df, histo_mass_smooth, cfg_corrbkgs['n_smooth_points'], cfg_corrbkgs['n_points_for_kde'])
    histo_mass_smooth.Smooth(100)
    print(f"Smoothed histogram created with entries: {histo_mass_smooth.GetEntries()}.")

    histo_mass.Write('hMassRaw')
    histo_mass_smooth.Write('hMassSmooth')
    treeFrac.Write('treeFracMassScoresBkgFD')
    treeMass.Write('treeMass')

def produce_corr_bkgs_templs(cfg):

    full_dfs = []
    tables = [[] for table in cfg["table_names"]]
    with uproot.open(cfg["input_file"]) as f:
        for table_name, table_list in zip(cfg["table_names"], tables):
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

    for i_pt, (pt_min, pt_max) in enumerate(config["pt_bins"]):
        pt_key = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        os.makedirs(os.path.dirname(f"{cfg['outfile']}_{pt_key}.root"), exist_ok=True)
        outfile = TFile(f"{cfg['outfile']}_{pt_key}.root", "RECREATE")
        print(f"\nProcessing pt bin: {pt_min} - {pt_max}")

        cent_pt_sel_df = cent_sel_df.query(f"fPt >= {pt_min} and fPt < {pt_max}")

        for fin_state, fin_state_info in final_states.items():
            print(f"Processing final state: {fin_state}")

            channel_df = cent_pt_sel_df.query(fin_state_info['query'])
            if len(channel_df) <= cfg.get("min_entries", 0):
                print(f"----> No candidates for final state: {fin_state}, skipping.")
                continue

            chn_dir = f"{pt_key}/{fin_state}"
            make_dir_root_file(chn_dir, outfile)
            outfile.cd(chn_dir)
            hBRs = ROOT.TH1F("hBRs", "hBRs;Branching Ratio", 2, 0, 2)
            hBRs.GetXaxis().SetBinLabel(1, "MC")
            br_mc = fin_state_info[f'br_sim_{cfg["coll_system"]}']
            hBRs.SetBinContent(1, br_mc)
            hBRs.GetXaxis().SetBinLabel(2, "PDG")
            br_pdg = fin_state_info['br_pdg']
            hBRs.SetBinContent(2, br_pdg)
            hBRs.Write()

            produce_chn_corrbkg(cfg, channel_df, outfile, chn_dir, templ_type='raw')

            if cfg.get('smear_mass'):
                channel_df_smear = smear_templs(cfg, channel_df, pt_min, pt_max)
                produce_chn_corrbkg(cfg, channel_df_smear, outfile, chn_dir, templ_type='smear')
            if cfg.get('shift_mass'):
                channel_df_shift = shift_templs(cfg, channel_df, pt_min, pt_max)
                produce_chn_corrbkg(cfg, channel_df_shift, outfile, chn_dir, templ_type='shift')
            if cfg.get('smear_mass') and cfg.get('shift_mass'):
                channel_df_smear = smear_templs(cfg, channel_df, pt_min, pt_max)
                channel_df_shift_smear = shift_templs(cfg, channel_df_smear, pt_min, pt_max)
                produce_chn_corrbkg(cfg, channel_df_shift_smear, outfile, chn_dir, templ_type='shift_smear')

        outfile.Close()
        print(f"\nOutput file with correlated backgrounds templates: {cfg['outfile']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("config", metavar="text",
                        default="config.yaml", help="flow configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger("Producing correlated backgrounds templates", "INFO")
    produce_corr_bkgs_templs(config)
