"""
Script to perform signal extraction as a function of pT for several BDT output scores for cut-variation method
"""

import argparse
import numpy as np
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
import yaml
from ROOT import TFile, TH1F, TH1
from matplotlib.offsetbox import AnchoredText
import sys
import os
import uproot
import pandas as pd
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../utils")
from data_model import get_data_model_dicts, get_histo, get_tree
from utils import add_info_on_canvas
from fit_utils import get_signal_pars_dict
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position

def add_corr_bkgs_to_fitter(fitter, fit_data, tree_corr_bkgs, config, config_fit, mass_min, mass_max):
    """
    Helper function to add correlated backgrounds to the fitter
    """
    # Build the general selection string
    sel_string = f"fM < {mass_max} and fM > {mass_min} and fPt > {config_fit['pt_range'][0]} and fPt < {config_fit['pt_range'][1]}"
    if config.get('correlated_bkgs') and config_fit.get('score_bkg_max'):
        if config['correlated_bkgs'].get('apply_ml_score_sel'):
            sel_string += f" and fMlScore0 < {config_fit['score_bkg_max']}"

    tree_corr_bkgs_selected = tree_corr_bkgs.query(sel_string)

    # extract distribution of signal
    tree_signal = tree_corr_bkgs_selected.query(f"abs(fFlagMcMatchRec) == {config['correlated_bkgs']['signal']['flag_mc_match_rec']}")
    sgn_denom = len(tree_signal) * config['correlated_bkgs']['signal']['br_pdg'] / config['correlated_bkgs']['signal']['br_sim']

    shift_templ_mass = config_fit.get('shift_template_mass', 0.0)
    for i_source, source in enumerate(config["correlated_bkgs"]["sources"]):
        tree_channel = tree_corr_bkgs_selected.query(f"abs(fFlagMcMatchRec) == {source['flag_mc_match_rec']}")
        if isinstance(fit_data, TH1):
            histo_channel = fit_data.Clone()
            histo_channel.Reset('ICESM')
            histo_channel.SetName(f"h{source['name']}")
            for cand_mass in tree_channel['fM']:
                histo_channel.Fill(cand_mass + shift_templ_mass)
            print(f"Correlated background histo {source['name']} has {histo_channel.Integral()} entries")
            fitter.set_background_template(i_source, DataHandler(histo_channel, limits=[mass_min, mass_max]))
        else:
            tree_channel['fM'] = tree_channel['fM'] + shift_templ_mass
            print(f"Correlated background tree {source['name']} has {len(tree_channel)} entries")
            fitter.set_background_kde(i_source, DataHandler(tree_channel, var_name='fM', limits=[mass_min, mass_max], name=f"ciao{i_source}", nbins=100))
        print(f"len(tree_channel): {len(tree_channel)}")
        print(f"sgn_denom: {sgn_denom}")
        fraction = len(tree_channel) * source['br_pdg'] * source.get('correct_abundance', 1.0) / source['br_sim'] / sgn_denom
        print(f"Fraction of {source['name']} to signal: {fraction}")
        fitter.fix_bkg_frac_to_signal_pdf(i_source, 0, fraction)  # fix to 10% of the signal yield

def perform_fit(config, fit_config, data_model_dict, fit_data, tree_corr_bkgs, fitter_name, sgn_funcs, bkg_funcs, mass_min, mass_max):
    """
    Helper function to perform fit
    """

    if config.get('is_tree'):
        fit_data = fit_data.rename(columns={f'{data_model_dict["Data"]["Mass"]}': 'fM'})
        data_hdl = DataHandler(fit_data, var_name='fM', limits=[mass_min, mass_max], nbins=100)
    else:
        data_hdl = DataHandler(fit_data, limits=[mass_min, mass_max])

    label_bkg_pdf = ["Comb. bkg"]
    if tree_corr_bkgs is not None:
        for i_source, source in enumerate(config["correlated_bkgs"]["sources"]):
            bkg_funcs.insert(i_source, "hist") if data_hdl.get_is_binned() else bkg_funcs.insert(i_source, "kde_grid")
            label_bkg_pdf.insert(i_source, source["name"])

    print(f"Performing fit with signal functions {sgn_funcs} and background functions {bkg_funcs}")
    fitter = F2MassFitter(data_hdl, name_signal_pdf=sgn_funcs, tol=0.01, rebin=4,
                          name_background_pdf=bkg_funcs, name=fitter_name, label_bkg_pdf=label_bkg_pdf)

    fitter.set_signal_initpar(0, "frac", 0.2, limits=[0., 1.])

    if fit_config.get("init_pars_sgn"):
        for i_func, func_init_pars in enumerate(fit_config["init_pars_sgn"]):
            for par, val in func_init_pars.items():
                try:
                    if isinstance(val, list) and len(val) == 3:
                        print(f"setting sgn par {par} to {val[0]} with limits {val[1]}, {val[2]}")
                        fitter.set_signal_initpar(i_func, par, val[0], limits=[val[1], val[2]])
                        continue

                    if isinstance(val, (int, float)):
                        print(f"fixing sgn par {par} to {val}")
                        fitter.set_signal_initpar(i_func, par, val, fix=True)
                        continue

                    if isinstance(val, str):
                        par_file = TFile.Open(val, "READ")
                        histo_par = par_file.Get(f"hist_{par}")
                        for i_bin in range(histo_par.GetNbinsX()+1):
                            bin_center = histo_par.GetBinCenter(i_bin)
                            if bin_center > (fit_config['pt_range'][0]) and bin_center < (fit_config['pt_range'][1]):
                                par_val = histo_par.GetBinContent(i_bin)
                                fitter.set_signal_initpar(i_func, par, par_val, fix=True)
                                print(f"fixing sgn par {par} to value {par_val} taken from file {val}")
                                break
                        par_file.Close()
                        continue

                except Exception as e:
                    print(f"Parameter {par} not present in {sgn_funcs[i_func]}!")

    if fit_config.get("init_pars_bkg"):
        for par, val in fit_config["init_pars_bkg"].items():
            try:
                if isinstance(val, list) and len(val) == 3:
                    print(f"setting sgn par {par} to {val[0]} with limits {val[1]}, {val[2]}")
                    fitter.set_background_initpar(len(bkg_funcs)-1, par, val[0], limits=[val[1], val[2]])
                    continue

                if isinstance(val, (int, float)):
                    print(f"fixing sgn par {par} to {val}")
                    fitter.set_background_initpar(len(bkg_funcs)-1, par, val, fix=True)
                    continue

                if isinstance(val, str):
                    par_file = TFile.Open(val, "READ")
                    histo_par = par_file.Get(f"hist_{par}")
                    for i_bin in range(histo_par.GetNbinsX()+1):
                        bin_center = histo_par.GetBinCenter(i_bin)
                        if bin_center > (fit_config['pt_range'][0]) and bin_center < (fit_config['pt_range'][1]):
                            par_val = histo_par.GetBinContent(i_bin)
                            fitter.set_background_initpar(len(bkg_funcs)-1, par, par_val, fix=True)
                            print(f"fixing sgn par {par} to value {par_val} taken from file {val}")
                            break
                    par_file.Close()
                    continue

            except Exception as e:
                print(f"Parameter {par} not present in {bkg_funcs[0]}!")

    if tree_corr_bkgs is not None:
        add_corr_bkgs_to_fitter(fitter, fit_data, tree_corr_bkgs, config, fit_config, mass_min, mass_max)

    fitter.mass_zfit()

    return fitter

def fit(input_config):
    """
    Main function for fitting and saving results
    """

    with open(input_config, "r") as f:
        cfg = yaml.safe_load(f)

    data_model_dict = get_data_model_dicts(cfg, cfg["data_type"])
    tree_corr_bkgs = None
    if cfg.get('correlated_bkgs'):
        tree_corr_bkgs = get_tree(cfg['correlated_bkgs']['input'], cfg['correlated_bkgs']['tree_tables'])
    if cfg.get('is_tree', False):
        print("Fitting from tree")
        tree_sgn = get_tree(cfg['input'], cfg['tree_tables'], cfg.get('tree_query_for_signal', None))

    pt_limits = np.array(cfg["pt_limits"], np.float64)
    input_file = TFile.Open(cfg['input'], "READ")
    for sgn_func in cfg['sgn_funcs']:
        signal_pars_dict = get_signal_pars_dict(sgn_func, pt_limits)
        sgn_func_string = sgn_func if isinstance(sgn_func, str) else "_".join(sgn_func)

        outfile_name = f"{cfg['output']}/{sgn_func_string}/results.root"
        os.makedirs(os.path.dirname(outfile_name), exist_ok=True)
        outfile = TFile(outfile_name, "recreate")
        outfile.Close()

        for ipt, pt_bin in enumerate(cfg['pt_bins']):
            pt_min, pt_max = pt_bin['pt_range']
            mass_min, mass_max = pt_bin['mass_range']
            bincounting = pt_bin.get('bincounting', False)

            pt_label = f"{int(pt_min*10)}_{int(pt_max*10)}"

            sgn_func_pars = []
            if cfg.get('is_tree'):
                if pt_bin.get('score_bkg_max') and cfg.get('correlated_bkgs'):
                    if cfg['correlated_bkgs'].get('apply_ml_score_sel'):
                        tree_sgn_pt = tree_sgn.query(f"{data_model_dict['Data']['Pt']} > {pt_min} and {data_model_dict['Data']['Pt']} < {pt_max} and {data_model_dict['Data']['BkgScoreCol']} < {pt_bin['score_bkg_max']}")
                    else:
                        tree_sgn_pt = tree_sgn
                else:
                    print(f"data_model_dict: {data_model_dict}\n\n")
                    tree_sgn_pt = tree_sgn.query(f"{data_model_dict['Data']['Pt']} > {pt_min} and {data_model_dict['Data']['Pt']} < {pt_max}")
                fitter = perform_fit(cfg, pt_bin, data_model_dict, tree_sgn_pt, tree_corr_bkgs, pt_label, sgn_func, pt_bin['bkg_funcs'], mass_min, mass_max)
            else:
                fit_histo = get_histo(input_file, data_model_dict, cfg, pt_bin)
                fit_histo.Rebin(pt_bin.get("Rebin", 1))
                fitter = perform_fit(cfg, pt_bin, data_model_dict, fit_histo, tree_corr_bkgs, pt_label, sgn_func, pt_bin['bkg_funcs'], mass_min, mass_max)

            print("Fit done!")
            if fitter.get_fit_result.converged:

                rawyield = fitter.get_raw_yield_bincounting(0) if bincounting else fitter.get_raw_yield(0)
                signal_pars_dict["rawyield"].SetBinContent(ipt+1, rawyield[0])
                signal_pars_dict["rawyield"].SetBinError(ipt+1, rawyield[1])

                signal_pars = fitter.get_signal_pars()[0]
                for par_name, val in signal_pars.items():
                    if par_name in signal_pars_dict.keys():
                        sgn_func_pars.append(sgn_func)
                        val = fitter.get_signal_parameter(0, par_name)
                        signal_pars_dict[par_name].SetBinContent(ipt+1, val[0])
                        signal_pars_dict[par_name].SetBinError(ipt+1, val[1])

                fitter.dump_to_root(outfile_name, option="update", suffix=f"_{pt_label}")
                fig, axs = fitter.plot_mass_fit(style="ATLAS", figsize=(8, 8), axis_title=r"$M_\mathrm{K\pi\pi}$ (GeV/$c^2$)",
                                                show_extra_info=True, extra_info_loc=["lower right", "lower left"])
                add_info_on_canvas(axs, "upper left", "pp", pt_min, pt_max)
                fig_res = fitter.plot_raw_residuals(figsize=(8, 8), style="ATLAS", axis_title=r"$M_\mathrm{K\pi\pi}$ (GeV/$c^2$)")

                outdir_figs = f"{os.path.dirname(outfile_name)}/{pt_label}"
                os.makedirs(outdir_figs, exist_ok=True)
                fig.savefig(f"{outdir_figs}/massfit.pdf")
                fig_res.savefig(f"{outdir_figs}/massfitres.pdf")
                print(f"Saved fit figures in {outdir_figs}/massfit.pdf and {outdir_figs}/massfitres.pdf")

        outfile = TFile(outfile_name, "update")
        for histo in signal_pars_dict.values():
            histo.Write()
        outfile.Close()

    input_file.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--cfg_file", "-c", metavar="text",
                        default="config.yml", help="config file")
    args = parser.parse_args()

    fit(args.cfg_file)
