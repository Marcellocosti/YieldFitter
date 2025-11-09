'''
Script for extracting raw yields for D mesons
run: python get_raw_yields.py fitConfigFileName.yml infile.root
'''

import argparse
import yaml
import os
from ROOT import TFile, TH1, TH1D, TH1F # pylint: disable=import-error,no-name-in-module
script_dir = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.join(script_dir, '..', 'utils'))
from fit_utils import rebin_histo, get_data_to_fit, create_hist, add_info_on_canvas
from utils import logger, get_centrality_bins, get_particle_info
import zfit
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
import uproot
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position

def get_corr_bkg_template(path, input_type, file_path):
    
    if input_type == 'Tree':
        print(f"Getting corr bkg template from tree {path}treeMass in file {file_path}")
        with uproot.open(corr_bkg_file_path) as f:
            tree = f[f"{path}treeMass"]
        return tree
    else:
        file = TFile.Open(file_path, "READ")
        print(f"Getting corr bkg template from histogram {path}hMass_smooth in file {file_path}")
        hist = file.Get(f"{path}hMass_smooth")
        if not isinstance(hist, TH1):
            logger(f"---> No smoothed hist found, getting only hMass", "WARNING")
            hist = file.Get(f"{path}hMass")
        hist.SetDirectory(0)
        return hist

def set_fitter_init_pars(full_cfg, fitter, cfg, pt_min, pt_max, n_bkg_functs, n_sgn_functs):
    print(f"\n\nSetting fitter initial parameters for pt range {pt_min} - {pt_max} GeV/c")

    # First init, then eventually override with fix
    if cfg.get("init_pars_sgn"):
        for setting in cfg["init_pars_sgn"]:
            sgn_func_idx = setting[0]
            par_name = setting[1]
            par_val = setting[2]
            par_lims = setting[3]
            fitter.set_signal_initpar(sgn_func_idx, par_name, par_val, limits=par_lims)
            print(f"---> setting sgn par {par_name} to value {par_val}, limits {par_lims}")

    if cfg.get("init_pars_bkg"):
        for setting in cfg["init_pars_bkg"]:
            par_name = setting[0]
            par_val = setting[1]
            par_lims = setting[2]
            fitter.set_background_initpar(n_bkg_functs-1, par_name, par_val, limits=par_lims)
            print(f"---> setting bkg par {par_name} to value {par_val}, limits {par_lims}")

    if cfg.get("fix_pars_sgn"):
        for setting in cfg["fix_pars_sgn"]:
            sgn_func_idx = setting[0]
            par_name = setting[1]
            par_val = setting[2]
            fitter.set_signal_initpar(sgn_func_idx, par_name, par_val, fix=True)
            print(f"---> fixing sgn par {par_name} to value {par_val}")

    if cfg.get("fix_pars_bkg"):
        for setting in cfg["fix_pars_bkg"]:
            par_name = setting[0]
            par_val = setting[1]
            fitter.set_background_initpar(n_bkg_functs-1, par_name, par_val, fix=True)
            print(f"---> fixing bkg par {par_name} to value {par_val}")

    if cfg.get("fix_sgn_from_file"):
        # Initialization from MC fits from file
        for setting in cfg["fix_sgn_from_file"]:
            sgn_func_idx = setting[0]
            par_names = setting[1]
            file_pars = setting[2]
            print(f"Opening file {file_pars} to fix signal parameters {par_names}")
            par_file = TFile.Open(file_pars, "READ")
            for par_name in par_names:
                try:
                    histo_par = par_file.Get(f"hist_{par_name}")
                    for i_bin in range(histo_par.GetNbinsX()+1):
                        bin_center = histo_par.GetBinCenter(i_bin)
                        if bin_center > pt_min and bin_center < pt_max:
                            par_val = histo_par.GetBinContent(i_bin)
                            break

                    # Shift the mean or add smearing to compensate data-MC discrepancies
                    # no sgn_func_idx == 0 as this is only for corr bkgs
                    shift, smear = 0.0, 0.0
                    if full_cfg.get('corr_bkgs') and sgn_func_idx != 0:
                        cfg_corrbkgs = full_cfg['corr_bkgs']
                        if cfg_corrbkgs.get('shift_mass') and 'mu' in par_name:
                            if isinstance(cfg_corrbkgs["shift_mass"], float):
                                shift = cfg_corrbkgs["shift_mass"]
                            elif isinstance(cfg_corrbkgs["shift_mass"], list):
                                shift = cfg_corrbkgs["shift_mass"]
                            else:
                                logger(f"Taking mass shifts from {cfg_corrbkgs['shift_mass']}", "INFO")
                                shifts_file = TFile.Open(cfg_corrbkgs['shift_mass'], "READ")
                                shifts_histo = shifts_file.Get("delta_mean_data_mc")
                                shifts_histo.SetDirectory(0)
                                for ipt in range(smear_histo.GetNbinsX()+1):
                                    bin_center = smear_histo.GetBinCenter(ipt)
                                    if bin_center > pt_min and bin_center < pt_max:
                                        shift = shifts_histo.GetBinContent(ipt+1)
                                        break
                                shifts_file.Close()
                            par_val += shift
                        if cfg_corrbkgs.get('smear_sigma') and 'sigma' in par_name:
                            if isinstance(cfg_corrbkgs["smear_mass"], float):
                                smear = cfg_corrbkgs["smear_mass"]
                            elif isinstance(cfg_corrbkgs["smear_mass"], list):
                                smear = cfg_corrbkgs["smear_mass"]
                            else:
                                logger(f"Taking mass shifts from {cfg_corrbkgs['smear_mass']}", "INFO")
                                smear_file = TFile.Open(cfg_corrbkgs['smear_mass'], "READ")
                                smear_histo = smear_file.Get("delta_sigma_data_mc")
                                smear_histo.SetDirectory(0)
                                for ipt in range(smear_histo.GetNbinsX()+1):
                                    bin_center = smear_histo.GetBinCenter(ipt)
                                    if bin_center > pt_min and bin_center < pt_max:
                                        smear = smear_histo.GetBinContent(ipt+1)
                                        break
                                smear_file.Close()
                            par_val += smear
                    print(f"---> fixing signal parameter {par_name} to value {par_val}, shift {shift}, smear {smear}")
                    fitter.set_signal_initpar(sgn_func_idx, par_name, par_val, fix=True)
                    
                except Exception as e:
                    print(f"        Parameter {par_name} not present!")

            par_file.Close()

def set_corr_bkgs(fitter, corr_bkgs_templs, sgn_bkgs_templs, cfg):

    sgn_func_idx = len(cfg['sgn_func'])
    bkg_func_idx = 0
    for corr_bkg in cfg["corr_bkgs"]['channels']:
        chn = corr_bkg['name']
        if corr_bkg.get('sgn_func'):
            if corr_bkg.get('fix_to'):
                pdf_name = corr_bkg['fix_to']
                frac = sgn_bkgs_templs[pdf_name]['frac']
                sgn_func_idx = sgn_bkgs_templs[pdf_name]['idx']
                print(f"Setting correlated bkg function {sgn_func_idx} fraction " \
                      f"to {frac} wrt signal pdf no. {pdf_idx}")
                fitter.fix_signal_frac_to_signal_pdf(sgn_func_idx, pdf_idx, frac)
            else:
                logger(f"Signal function for correlated bkg source {chn} will be free", level="WARNING")

            sgn_func_idx += 1
            continue

        if corr_bkg.get('bkg_func'):
            data_hdl = corr_bkgs_templs[chn]['data_hdl']
            if corr_bkg['bkg_func'] != 'kde_grid' and corr_bkg['bkg_func'] != 'hist':
                logger(f"Background function for correlated bkg source {chn} not 'kde_grid' or 'hist'", level="ERROR")

            if corr_bkg['bkg_func'] == 'kde_grid':
                print(f"Setting kde for source {chn}")
                fitter.set_background_kde(bkg_func_idx, data_hdl)
            else:
                print(f"Setting histo for source {chn}")
                fitter.set_background_template(bkg_func_idx, data_hdl)

            if corr_bkg.get('fix_to'):
                anchor_pdf_name = corr_bkg['fix_to']
                print(f"\nanchor_pdf_name: {anchor_pdf_name}\n")
                frac = corr_bkgs_templs[chn]['frac'] / corr_bkgs_templs[anchor_pdf_name]['frac']
                pdf_idx = corr_bkgs_templs[anchor_pdf_name]['idx']
                if anchor_pdf_name == 'signal':
                    print(f"Setting correlated bkg template function {bkg_func_idx} fraction " \
                          f"to {frac} wrt signal pdf no. {pdf_idx}")
                    fitter.fix_bkg_frac_to_signal_pdf(bkg_func_idx, pdf_idx, frac)
                else:
                    print(f"Setting correlated bkg template function {bkg_func_idx} fraction " \
                          f"to {frac} wrt bkg pdf no. {pdf_idx}")
                    fitter.fix_bkg_frac_to_bkg_pdf(bkg_func_idx, pdf_idx, frac)

            elif corr_bkg.get('init_to'):
                pdf_name = corr_bkg['init_to']
                idx = corr_bkgs_templs[pdf_name]['idx']
                frac = corr_bkgs_templs[pdf_name]['frac']
                if pdf_name == 'signal':
                    frac_sgn = corr_bkgs_templs['signal']['frac']
                    print(f"Setting correlated bkg template function {bkg_func_idx} initial fraction " \
                        f"to {frac} wrt signal pdf no. {pdf_idx}")
                    fitter.set_background_initpar(idx, "frac", frac/frac_sgn, limits=[0., 1.])
            else:
                logger(f"Background function for correlated bkg source {corr_bkg} without 'fix_to' or 'init_to' key", level="ERROR")
            
            bkg_func_idx += 1
            continue

def perform_fit(pt_bin_cfg, corr_bkg_file_path, \
                data_hdl, pt_label, \
                config_file, pt_min, pt_max, \
                labels_bkg, labels_sgn):

    # Add correlated bkg templates
    corr_bkgs_templs, sgn_bkgs_templs = {}, {}
    bkg_functs, sgn_functs = [], []
    if pt_bin_cfg.get("corr_bkgs"):

        # Get the fraction
        corr_bkg_file = TFile.Open(corr_bkg_file_path, 'read')
        pt_subdir = corr_bkg_file.Get(pt_label)
        hist_fractions = pt_subdir.Get("hWeights")
        weights = {}
        for i_bin in range(1, hist_fractions.GetNbinsX()+1):
            weights[hist_fractions.GetXaxis().GetBinLabel(i_bin)] = hist_fractions.GetBinContent(i_bin)
        hist_fractions.SetDirectory(0)
        corr_bkg_file.Close()
        i_source_corr_bkg_sgn, i_source_corr_bkg_bkg = 0, 0
        sgn_bkgs_templs['signal'] = {}
        sgn_bkgs_templs['signal']['frac'] = weights[pt_bin_cfg["corr_bkgs"]['sgn_fin_state']]
        sgn_bkgs_templs['signal']['idx'] = 0
        corr_bkgs_templs['signal'] = {}
        corr_bkgs_templs['signal']['frac'] = weights[pt_bin_cfg["corr_bkgs"]['sgn_fin_state']]
        corr_bkgs_templs['signal']['idx'] = 0
        for i_source, corr_bkg_source in enumerate(pt_bin_cfg["corr_bkgs"]["channels"]):
            chn_name = corr_bkg_source['name']

            # Use a signal function instead of a mc template
            if corr_bkg_source.get('sgn_func'):
                sgn_bkgs_templs[chn_name] = {}
                print(f"Using signal function for correlated bkg source {chn_name}")
                sgn_functs.append(corr_bkg_source['sgn_func'])
                sgn_bkgs_templs[chn_name]['frac'] = weights[chn_name]
                sgn_bkgs_templs[chn_name]['idx'] = i_source_corr_bkg_sgn + len(pt_bin_cfg['sgn_func'])
                i_source_corr_bkg_sgn += 1
                continue

            print(f"Using MC template for correlated bkg source {chn_name}")
            corr_bkgs_templs[chn_name] = {}
            # Get the correlated bkg template (TTree or TH1)
            corr_bkg_templ = get_corr_bkg_template(f"{pt_label}/{chn_name}/", config_file['input_type'], corr_bkg_file_path)
            if config_file['input_type'] == 'Tree':
                df = tree.arrays(library="pd")
                corr_bkgs_templs[chn_name]['frac'] = weights[chn_name]
                corr_bkgs_templs[chn_name]['data_hdl'] = DataHandler(df, var_name="fM", limits=pt_bin_cfg["fit_range"], nbins=100)
                bkg_functs.append("kde_grid")
            else:
                corr_bkgs_templs[chn_name]['frac'] = weights[chn_name]
                corr_bkgs_templs[chn_name]['data_hdl'] = DataHandler(corr_bkg_templ, limits=pt_bin_cfg["fit_range"], rebin=pt_bin_cfg.get('rebin', 1))
                bkg_functs.append("hist")

            corr_bkgs_templs[chn_name]['idx'] = i_source_corr_bkg_bkg
            i_source_corr_bkg_bkg += 1

    bkg_functs = bkg_functs + pt_bin_cfg['ry_setup']['bkg_func']
    sgn_functs = sgn_functs + pt_bin_cfg['ry_setup']['sgn_func']
    print(f"Using signal function: {sgn_functs} and background function: {bkg_functs}")
    print(f"Using signal labels: {labels_sgn} and background labels: {labels_bkg}")
    fitter = F2MassFitter(data_hdl, label_signal_pdf=labels_sgn, name_signal_pdf=sgn_functs,
                          name_background_pdf=bkg_functs, label_bkg_pdf=labels_bkg, name=pt_label)

    print(f"pt_bin_cfg: {pt_bin_cfg}\n")

    # Set reflection template
    if pt_bin_cfg['ry_setup'].get("corr_bkgs"):
        set_corr_bkgs(fitter, corr_bkgs_templs, sgn_bkgs_templs, pt_bin_cfg['ry_setup']["corr_bkgs"])
    if pt_bin_cfg['ry_setup'].get("init_pars"):
        set_fitter_init_pars(config_file, fitter, pt_bin_cfg['ry_setup']["init_pars"], pt_min, pt_max, len(bkg_functs), len(sgn_functs))
    # fitter.set_signal_initpar(sgn_func_idx, "frac", 0.2, limits=[0., 1.])
    result = fitter.mass_zfit()

    return fitter, result

def produce_func_labels(cfg, decay):
    bkg_labels = []
    sgn_labels = []
    if cfg.get('corr_bkgs'):
        for chn in cfg['corr_bkgs']['channels']:
            if chn.get('sgn_func'):
                sgn_labels.append(chn['name'])
            else:
                bkg_labels.append(chn['name'])

    sgn_labels.append(decay)
    bkg_labels.append("Comb. bkg")
    
    return sgn_labels, bkg_labels

def get_raw_yields(cfg_ry, cfg, data, outfile_name, corr_bkg_file_path, mass_axis_label, bkg_labels, sgn_labels):

    if isinstance(data, TH1):
        data_hdl = DataHandler(data, limits=cfg_ry["fit_range"], rebin=cfg_ry.get('rebin', 1))
    else:
        data_hdl = DataHandler(data, var_name='fM', limits=cfg_ry["fit_range"], nbins=100)

    fit_info = {}

    fitter, result = perform_fit(cfg, corr_bkg_file_path,
                                 data_hdl, pt_label,
                                 config_file, pt_min, pt_max,
                                 bkg_labels, sgn_labels)

    if result.converged: # i.e. has converged
        print(f"Plotting fit results for pt bin {pt_min} - {pt_max} GeV/c")
        fig, axs = fitter.plot_mass_fit(style="ATLAS",
                                        figsize=(8, 8),
                                        axis_title=mass_axis_label,
                                        show_extra_info=True,
                                        logy=cfg_ry.get('logy_plots', False),
                                        extra_info_loc=["lower right", "lower left"])
        add_info_on_canvas(axs, "upper left", "pp", pt_min, pt_max)

        fig_res = fitter.plot_raw_residuals(style="ATLAS",
                                            figsize=(8, 8),
                                            axis_title=mass_axis_label)

        fig_pulls = fitter.plot_std_residuals(style="ATLAS",
                                            figsize=(8, 8),
                                            axis_title=mass_axis_label)

        file_name = os.path.basename(outfile_name).replace('.root', '')
        outdir = os.path.dirname(outfile_name)
        try:
            ryfile, suffix = file_name.split('_', 2)
        except ValueError:
            ryfile, suffix = file_name, ''
        print(f"Saving figures in {outdir}")
        os.makedirs(f"{outdir}/{suffix}", exist_ok=True)
        fig.savefig(os.path.join(outdir, suffix, f"mass_{pt_label}.pdf"))
        fig_res.savefig(os.path.join(outdir, suffix, f"massres_{pt_label}.pdf"))
        fig_pulls.savefig(os.path.join(outdir, suffix, f"masspulls_{pt_label}.pdf"))

        fit_info['ry'] = fitter.get_raw_yield(0)[0]
        fit_info['ry_unc'] = fitter.get_raw_yield(0)[1]
        fit_info['ry_bin_counting'] = fitter.get_raw_yield_bincounting(0)[0]
        fit_info['ry_bin_counting_unc'] = fitter.get_raw_yield_bincounting(0)[1]
        fit_info['signif'] = fitter.get_significance(0)[0]
        fit_info['signif_unc'] = fitter.get_significance(0)[1]
        fit_info['s_over_b'] = fitter.get_signal_over_background(0)[0]
        fit_info['s_over_b_unc'] = fitter.get_signal_over_background(0)[1]
        fit_info['chi2_fits'] = float(fitter.get_chi2())
        fit_info['chi2_over_ndf_fits'] = float(fitter.get_chi2())/fitter.get_ndf()

        signal_pars = fitter.get_signal_pars()
        signal_pars_uncs = fitter.get_signal_pars_uncs()

        for i_sgn_func, (func_dict, func_dict_unc, label) in enumerate(zip(signal_pars, signal_pars_uncs, sgn_labels)):
            for (par_name, val), (par_name, unc) in zip(func_dict.items(), func_dict_unc.items()):
                # print(f"Initializing fit_info for i_sgn_func {i_sgn_func}, par_name {par_name}\n")
                fit_info[f'{label}_par_sgn_{i_sgn_func}_{par_name}'] = val
                fit_info[f'{label}_par_sgn_{i_sgn_func}_{par_name}_unc'] = unc

    print(f"Getting background parameters for pt bin {pt_min} - {pt_max} GeV/c")
    bkg_pars = fitter.get_bkg_pars()
    bkg_pars_uncs = fitter.get_bkg_pars_uncs()

    for i_bkg_func, (func_dict, func_dict_unc, label) in enumerate(zip(bkg_pars, bkg_pars_uncs, bkg_labels)):
        for (par_name, val), (par_name, unc) in zip(func_dict.items(), func_dict_unc.items()):
            # print(f"Setting fit_info for {par_name} at ipt {ipt}, val: {val}, unc: {unc}\n")
            fit_info[f'{label}_par_bkg_{i_bkg_func}_{par_name}'] = val
            fit_info[f'{label}_par_bkg_{i_bkg_func}_{par_name}_unc'] = unc

    # print(f"Storing fit results in {outfile_name}, folder {pt_label}\n")
    fitter.dump_to_root(outfile_name, option="update", folder=pt_label)

    return fit_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('cfgfile', metavar='text', default='config_Ds_Fit.yml')
    parser.add_argument('infile', metavar='text', default='')
    args = parser.parse_args()

    with open(args.cfgfile, 'r', encoding='utf8') as ymlfitConfigFile:
        config_file = yaml.load(ymlfitConfigFile, yaml.FullLoader)

    _, mass_axis_label, decay, _, _, _ = get_particle_info(config_file["Dmeson"])

    # Retrieve config file for fit
    config_fit = copy.deepcopy(config_file)

    # Create outfile name and store fit config
    outfile = args.infile.replace('proj', 'rawyield')
    ofile = uproot.recreate(outfile)
    ofile.close()
    # with open(f"{os.path.dirname(outfile)}/config.yml", 'w', encoding='utf8') as ymlfitConfigFilePrefit:
    #     yaml.dump(config_fit, ymlfitConfigFilePrefit, default_flow_style=None, sort_keys=False)

    # Eventually prepare config file for postfit (signal parameters free)
    if config_file['ry_setup'].get('postfit_sgn'):
        config_file_postfit = copy.deepcopy(config_fit)
        outfile_postfit = args.infile.replace('proj', 'rawyield_postfit')
        os.makedirs(os.path.dirname(outfile_postfit), exist_ok=True)
        ofile_postfit = uproot.recreate(outfile_postfit)
        ofile_postfit.close()

    # Store fit info
    fit_infos, postfit_infos = {}, {}

    corr_bkg_file_path = outfile.replace('rawyield', 'corrbkg')

    pt_lims = []
    cfg_ry = config_file['ry_setup']
    pt_min, pt_max = cfg_ry['pt_range'][0], cfg_ry['pt_range'][1]
    pt_lims = [cfg_ry['pt_range'][0], cfg_ry['pt_range'][1]]

    print(f"Fitting pt bin: {pt_min} - {pt_max} GeV/c")
    pt_label = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"

    # Get data histogram
    data = get_data_to_fit(args.infile, pt_label, config_file['input_type'])

    # Produce function labels and perform fit
    sgn_labels, bkg_labels = produce_func_labels(config_file, decay)
    pt_fit_info = get_raw_yields(cfg_ry, config_file, data, outfile, corr_bkg_file_path, mass_axis_label, bkg_labels, sgn_labels)

    for key in pt_fit_info.keys():
        if key not in fit_infos:
            fit_infos[key] = 0.
        fit_infos[key] = pt_fit_info[key]

    # Eventually perform postfit (signal parameters all free) if requested
    if cfg_ry.get('postfit_sgn'):
        cfg_postfit = copy.deepcopy(config_file_postfit['ry_setup'])
        cfg_postfit['init_pars']['fix_sgn_from_file'] = []
        cfg_postfit['init_pars']['init_pars_sgn'] = []
        cfg_postfit['init_pars']['fix_pars_sgn'] = []
        cfg_postfit['init_pars']['init_pars_bkg'] = []

        tol_param = cfg_postfit['postfit_param_tol']
        tol_frac = cfg_postfit['postfit_sgn_bkg_frac_tol']
        for par_full_name, par_val in pt_fit_info.items():
                
            # Skip uncertainties and fit results quantities
            if 'unc' in par_full_name or 'par_' not in par_full_name:
                continue

            _, _, func_type, func_idx, par_name = par_full_name.split('_', 4)
            func_idx = int(func_idx)
            val = par_val
            cfg_init_pars_postfit = cfg_postfit['init_pars']

            tol = tol_frac if 'frac' in par_name else tol_param
            min_val = val - tol * val if val >= 0 else val + tol * val
            max_val = val + tol * val if val >= 0 else val - tol * val
            print(f"par_full_name: {par_full_name}, func_type: {func_type}, func_idx: {func_idx}, " \
                  f"par_name: {par_name}, val: {val}, min_val: {min_val}, max_val: {max_val}")

            if f'sgn_{len(sgn_labels)-1}_frac' in par_full_name:
                # tighter constraint on the signal fraction, we know it well from prefit
                print(f"Adding to init_pars_sgn for func_idx {func_idx} with tighter tolerance")
                cfg_init_pars_postfit['init_pars_sgn'].append([func_idx, "frac", val, [min_val, max_val]])
            else:
                # Leave free the last signal function (the main one)
                if f'sgn_{len(sgn_labels)-1}_' in par_full_name:
                    print(f"Adding to init_pars_sgn for func_idx {func_idx}")
                    cfg_init_pars_postfit['init_pars_sgn'].append([func_idx, par_name, val, [min_val, max_val]])
                elif '_c' in par_full_name:
                    print(f"Adding to init_pars_bkg")
                    cfg_init_pars_postfit['init_pars_bkg'].append([par_name, val, [min_val, max_val]])
                else:
                    # cfg_init_pars_postfit['init_pars_sgn'].append([func_idx, par_name, val, [min_val, max_val]])
                    logger(f"Parameter {par_name} not constrained!", "WARNING")

        with open(f"{os.path.dirname(outfile_postfit)}/config.yml", 'w', encoding='utf8') as ymlfitConfigFilePrefit:
            yaml.dump(cfg_postfit, ymlfitConfigFilePrefit, default_flow_style=None)
        pt_fit_info_postfit = get_raw_yields(cfg_ry, cfg_postfit, data, outfile_postfit, corr_bkg_file_path, mass_axis_label, bkg_labels, sgn_labels)

        for key in pt_fit_info_postfit.keys():
            if key not in postfit_infos:
                postfit_infos[key] = 0.
            postfit_infos[key] = pt_fit_info_postfit[key]

    file_std_fit = uproot.update(outfile)
    for par in fit_infos.keys():
        file_std_fit[f"h_{par}"] = create_hist(pt_lims, fit_infos[par], fit_infos[par+'_unc'] if par+'_unc' in fit_infos.keys() else 0.)
    file_std_fit.close()

    if cfg_ry.get('postfit_sgn'):
        file_postfit = uproot.update(outfile_postfit)
        for par in postfit_infos.keys():
            file_postfit[f"h_{par}"] = create_hist(pt_lims, postfit_infos[par], postfit_infos[par+'_unc'] if par+'_unc' in postfit_infos.keys() else 0.)
        file_postfit.close()
