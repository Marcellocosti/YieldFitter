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
    full_df.loc[:, "fM"] = full_df["fM"] + cfg_corrbkgs.get('shift_mass', 0.0)

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
        mass_min = config["ry_extraction"]["MassFitRanges"][ipt_bin][0]
        mass_max = config["ry_extraction"]["MassFitRanges"][ipt_bin][1]
        query_str = f"fPt >= {pt_min} and fPt < {pt_max} and fM >= {mass_min} and fM < {mass_max}"
        # query_str = f"fPt >= {pt_min} and fPt < {pt_max} and {config['bkg_score_column']} < {score_bkg_max} and {config['fd_score_column']} >= {score_fd_min} and {config['fd_score_column']} < {score_fd_max} and fM >= {mass_min} and fM < {mass_max}"
        cutset_sel_df = full_df.query(query_str)

        for fin_state, fin_state_info in final_states.items():
            print(f"Processing final state: {fin_state}")
            if not any(fin_state in name for name in final_states_to_include):
                continue

            selected_df = cutset_sel_df.query(f"abs(fFlagMcMatchRec) == {fin_state_info['flag_mc_match_rec']}")
            print(f"Number of selected candidates for final state {fin_state}: {len(selected_df)}")
            if len(selected_df) > 0:
                with getattr(uproot, write_file_mode)(out_file_name) as outfile:
                    print(f"Writing tree for {pt_key}/{fin_state}/treeMass")
                    outfile[f"{pt_key}/{fin_state}/treeMass"] = selected_df['fM'].to_frame()
                    write_file_mode = "update"

                hBRs = ROOT.TH1F("hBRs", "hBRs;Branching Ratio", 4, 0, 4)
                hBRs.GetXaxis().SetBinLabel(1, "MC")
                br_mc = fin_state_info.get(f"abundance_to_{config['Dmeson']}", 1) * (fin_state_info['br_sim'])
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
    full_df = full_df.query(f"fCentrality >= {centMin} and fCentrality < {centMax}")
    full_df.loc[:, "fM"] = full_df["fM"] + cfg_corrbkgs.get('shift_mass', 0.0)

    # Process corr bkgs channels
    final_states_to_include = cfg_corrbkgs["include_final_states"]
    sgn_fin_state = cfg_corrbkgs['sgn_fin_state']
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
        mass_min = config["ry_extraction"]["MassFitRanges"][ipt_bin][0]
        mass_max = config["ry_extraction"]["MassFitRanges"][ipt_bin][1]
        query_str = f"fPt >= {pt_min} and fPt < {pt_max} and fM >= {mass_min} and fM < {mass_max}"
        # query_str = f"fPt >= {pt_min} and fPt < {pt_max} and {config['bkg_score_column']} < {score_bkg_max} and {config['fd_score_column']} >= {score_fd_min} and {config['fd_score_column']} < {score_fd_max} and fM >= {mass_min} and fM < {mass_max}"
        cutset_sel_df = full_df.query(query_str)

        for fin_state, fin_state_info in final_states.items():
            print(f"Processing final state: {fin_state}")
            if not any(fin_state in name for name in final_states_to_include):
                continue

            selected_df = cutset_sel_df.query(f"abs(fFlagMcMatchRec) == {fin_state_info['flag_mc_match_rec']}")
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
                br_mc = fin_state_info.get(f"abundance_to_{config['Dmeson']}", 1) * (fin_state_info['br_sim'])
                hBRs.SetBinContent(1, br_mc)
                hBRs.GetXaxis().SetBinLabel(2, "PDG")
                br_pdg = fin_state_info['br_pdg']
                hBRs.SetBinContent(2, br_pdg)
                hBRs.GetXaxis().SetBinLabel(3, "Raw yield")
                raw_yield = len(selected_df)
                hBRs.SetBinContent(3, raw_yield)
                hBRs.GetXaxis().SetBinLabel(4, "RY * (PDG/MC)")
                hBRs.SetBinContent(4, raw_yield * (br_pdg/br_mc))
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
            print(f"name: {name}, weight: {weight}")
            if name == sgn_fin_state:
                print(f"Adding signal final state {name} with weight {weight}")
                hMassSignal.Add(histo, weight)
                total_signal_weight = weight
            else:
                hMassTotalCorrBkgs.Add(histo, weight)
                hWeightsAnchorSignal.GetXaxis().SetBinLabel(i_fin_state+1, name)
                hWeightsAnchorSignal.SetBinContent(i_fin_state+1, weight)

        # Normalize weights histogram to the total signal weight
        hWeightsAnchorSignal.Scale(1 / total_signal_weight)

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
            produce_corr_bkgs_templs_histos(config, args.cutset_config)
    else:
        produce_corr_bkgs_templs(args.config, args.cutset_config)





# def produce_corr_bkgs_templs(config, cutset_config):

#     with open(config, 'r') as f:
#         config = yaml.safe_load(f)
#     cfg_corrbkgs = config["corr_bkgs"]

#     with open(cutset_config, 'r') as f:
#         cfg_cutset = yaml.safe_load(f)

#     full_dfs = []
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

#     decays_info = {
#         "Dplus": {
#             "decay_table": final_states_dplus,
#             "mc_abundance": cfg_corrbkgs.get('correct_dplus_abundance', 1)
#         },
#         "Ds": {
#             "decay_table": final_states_ds,
#             "mc_abundance": cfg_corrbkgs.get('correct_ds_abundance', 1)
#         },
#         "DstarD0": {
#             "decay_table": final_states_dstar_to_d0_piplus,
#             "mc_abundance": cfg_corrbkgs.get('correct_dstar_abundance', 1)
#         },
#         "DstarDplus": {
#             "decay_table": final_states_dstar_to_dplus_pi0,
#             "mc_abundance": cfg_corrbkgs.get('correct_dstar_abundance', 1)
#         },
#         "Lc": {
#             "decay_table": final_states_lc,
#             "mc_abundance": cfg_corrbkgs.get('correct_Lc_abundance', 1)
#         },
#         "Xic": {
#             "decay_table": final_states_xic,
#             "mc_abundance": cfg_corrbkgs.get('correct_Xic_abundance', 1)
#         }
#     }

#     ### Extract the total MC branching ratio for all species
#     total_br_mc = {}
#     for particle, info_dict in decays_info.items():
#         total_br_mc_part = 0
#         for fin_state, fin_state_info in info_dict["decay_table"].items():
#             resonant_states = fin_state_info["ResoStates"]
#             for reso_state in resonant_states:
#                 total_br_mc_part += reso_state['br_mc']
#         total_br_mc[particle] = total_br_mc_part

#     # Process corr bkgs channels
#     final_states_to_include = cfg_corrbkgs["include_final_states"]
#     sgn_fin_state = cfg_corrbkgs['sgn_fin_state']
#     outfile = ROOT.TFile(cutset_config.replace("cutset", "corrbkg").replace(".yml", ".root"), "RECREATE")
#     for ipt_bin, (pt_min, pt_max, score_bkg_max, score_fd_min, score_fd_max) in enumerate(zip(cfg_cutset["Pt"]["min"],
#                                                                                               cfg_cutset["Pt"]["max"],
#                                                                                               cfg_cutset["score_bkg"]["max"],
#                                                                                               cfg_cutset["score_FD"]["min"],
#                                                                                               cfg_cutset["score_FD"]["max"])):
#         pt_key = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
#         histo_weights_dict = {}
#         print(f"Processing pt bin: {pt_min} - {pt_max}")
#         mass_min = config["simfit"]["MassFitRanges"][ipt_bin][0]
#         mass_max = config["simfit"]["MassFitRanges"][ipt_bin][1]
#         query_str = f"fPt >= {pt_min} and fPt < {pt_max} and fM >= {mass_min} and fM < {mass_max}"
#         # query_str = f"fPt >= {pt_min} and fPt < {pt_max} and {config['bkg_score_column']} < {score_bkg_max} and {config['fd_score_column']} >= {score_fd_min} and {config['fd_score_column']} < {score_fd_max} and fM >= {mass_min} and fM < {mass_max}"
#         cutset_sel_df = full_df.query(query_str)

#         for particle, info_dict in decays_info.items():
#             for fin_state, fin_state_info in info_dict["decay_table"].items():

#                 if not fin_state.startswith(f"{sgn_fin_state}_") and not any(fin_state in name for name in final_states_to_include):
#                     continue

#                 for reso_state in fin_state_info["ResoStates"]:
#                     selected_df = cutset_sel_df.query(f"abs(fFlagMcMatchRec) == {fin_state_info['FlagFinal']} and fFlagMcDecayChanRec == {reso_state['FlagReso']}")
                    
#                     if len(selected_df) > 0:
#                         make_dir_root_file(f"{pt_key}/{fin_state}/{reso_state['Channel']}", outfile)
#                         outfile.cd(f"{pt_key}/{fin_state}/{reso_state['Channel']}")

#                         # Fill tree from DataFrame
#                         hMass = ROOT.TH1F("hMass", "hMass", 600, 1.6, 2.2)
#                         tree = ROOT.TTree("DecayTree", f"DecayTree {particle} {reso_state['Channel']}")
#                         mass = array("f", [0.])
#                         tree.Branch("fM", mass, "fM/F")

#                         mass_values = selected_df["fM"].to_numpy(dtype="float32")
#                         for val in mass_values:
#                             mass[0] = val
#                             tree.Fill()

#                         tree.Draw("fM >> hMass", "", "goff")
#                         hMass.Smooth(100)
#                         hMass.Write()
#                         hBRs = ROOT.TH1F("hBRs", "hBRs;Branching Ratio", 4, 0, 4)
#                         hBRs.GetXaxis().SetBinLabel(1, "MC")
#                         br_mc = info_dict["mc_abundance"] * (reso_state['br_mc'] / total_br_mc[particle])
#                         hBRs.SetBinContent(1, br_mc)
#                         hBRs.GetXaxis().SetBinLabel(2, "PDG")
#                         br_pdg = reso_state['br_pdg']
#                         hBRs.SetBinContent(2, br_pdg)
#                         hBRs.GetXaxis().SetBinLabel(3, "Raw yield")
#                         raw_yield = tree.GetEntries()
#                         hBRs.SetBinContent(3, raw_yield)
#                         hBRs.GetXaxis().SetBinLabel(4, "RY * (PDG/MC)")
#                         hBRs.SetBinContent(4, raw_yield * (br_pdg/br_mc))
#                         hBRs.Write()
#                         histo_weights_dict[f"{fin_state}_{reso_state['Channel']}"] = [raw_yield * (br_pdg/br_mc), hMass]

#         n_final_states = len(histo_weights_dict)

#         hMassTotalSignal = ROOT.TH1F("hMassTotalSignal", "hMassTotalSignal", 600, 1.6, 2.2)
#         hMassTotalCorrBkgs = ROOT.TH1F("hMassTotalCorrBkgs", "hMassTotalCorrBkgs", 600, 1.6, 2.2)
#         hWeightsAnchorSignal = ROOT.TH1F("hWeightsAnchorSignal", "hWeightsAnchorSignal", n_final_states+1, 0, n_final_states+1)
#         hWeightsAnchorToFirst = ROOT.TH1F("hWeightsAnchorToFirst", "hWeightsAnchorToFirst", n_final_states+1, 0, n_final_states+1)
#         total_signal_weight = 0
#         i_final_state = 1
#         for name, (weight, histo) in histo_weights_dict.items():
#             if name.startswith(f"{sgn_fin_state}_"):
#                 hMassTotalSignal.Add(histo, weight)
#                 total_signal_weight += weight
#             else:
#                 if i_final_state == 1:
#                     weight_first_template = weight
#                 hMassTotalCorrBkgs.Add(histo, weight)
#                 hWeightsAnchorSignal.GetXaxis().SetBinLabel(i_final_state, name)
#                 hWeightsAnchorSignal.SetBinContent(i_final_state, weight)
#                 hWeightsAnchorToFirst.GetXaxis().SetBinLabel(i_final_state, name)
#                 hWeightsAnchorToFirst.SetBinContent(i_final_state, weight)
#                 i_final_state += 1

#         # Normalize weights histogram to the total signal weight
#         hWeightsAnchorSignal.Scale(1 / total_signal_weight)
#         hWeightsAnchorToFirst.Scale(1 / weight_first_template)

#         outfile.cd(pt_key)
#         hMassTotalSignal.Write()
#         hMassTotalCorrBkgs.Write()
#         hWeightsAnchorSignal.Write()
#         hWeightsAnchorToFirst.Write()

#     outfile.Close()
