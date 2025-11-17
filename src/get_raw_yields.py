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
from correlated_bkgs import get_corr_bkg
from utils import logger, get_centrality_bins, get_particle_info
import zfit
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
import uproot
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position

def set_fitter_init_pars(fitter, cfg, pt_min, pt_max, n_bkg_functs, n_sgn_functs):
    print(f"\n\nSetting fitter initial parameters for pt range {pt_min} - {pt_max} GeV/c")

    # First init, then eventually override with fix
    if cfg["init_pars"].get("init_pars_sgn"):
        for sett in cfg["init_pars"]["init_pars_sgn"]:
            sgn_func_idx, par_name, par_val, par_lims = sett[0], sett[1], sett[2], sett[3]
            fitter.set_signal_initpar(sgn_func_idx, par_name, par_val, limits=par_lims)
            print(f"---> setting sgn par {par_name} to value {par_val}, limits {par_lims}")

    if cfg["init_pars"].get("init_pars_bkg"):
        for sett in cfg["init_pars"]["init_pars_bkg"]:
            par_name, par_val, par_lims = sett[0], sett[1], sett[2]
            fitter.set_background_initpar(n_bkg_functs-1, par_name, par_val, limits=par_lims)
            print(f"---> setting bkg par {par_name} to value {par_val}, limits {par_lims}")

    if cfg["init_pars"].get("fix_pars_sgn"):
        for sett in cfg["init_pars"]["fix_pars_sgn"]:
            sgn_func_idx, par_name, par_val = sett[0], sett[1], sett[2]
            fitter.set_signal_initpar(sgn_func_idx, par_name, par_val, fix=True)
            print(f"---> fixing sgn par {par_name} to value {par_val}")

    if cfg["init_pars"].get("fix_pars_bkg"):
        for sett in cfg["init_pars"]["fix_pars_bkg"]:
            par_name, par_val = sett[0], sett[1]
            fitter.set_background_initpar(n_bkg_functs-1, par_name, par_val, fix=True)
            print(f"---> fixing bkg par {par_name} to value {par_val}")

    if cfg["init_pars"].get("fix_sgn_from_file"):
        # Initialization from MC fits from file
        for sett in cfg["init_pars"]["fix_sgn_from_file"]:
            sgn_func_idx, par_names, file_pars = sett[0], sett[1], sett[2]
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
                    shift, smear = 0.0, 0.0
                    if cfg.get('corr_bkgs') and sgn_func_idx != (n_sgn_functs-1):
                        cfg_corrbkgs = cfg['corr_bkgs']
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
                                logger(f"Taking mass smears from {cfg_corrbkgs['smear_mass']}", "INFO")
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

def set_corr_bkgs_fracs(fitter, bkg_templs, sgn_templs, cfg):

    for corr_bkg in cfg["corr_bkgs"]['channels']:
        chn = corr_bkg['name']
        print(f"\nSetting correlated bkg source {chn}")
        if corr_bkg.get('sgn_func'):
            if corr_bkg.get('fix_to') or corr_bkg.get('init_to'):
                pdf_idx = sgn_templs[chn]['idx']
                anchor_pdf = corr_bkg.get('fix_to', corr_bkg.get('init_to'))
                anchor_pdf_idx = sgn_templs[anchor_pdf]['idx']
                frac = sgn_templs[chn]['frac'] / sgn_templs[anchor_pdf]['frac']
                print(f"frac of chn {chn} wrt anchor {anchor_pdf}: {sgn_templs[chn]['frac']} / {sgn_templs[anchor_pdf]['frac']} = {frac}")
                
                if corr_bkg.get('fix_to'):
                    print(f"Fixing correlated bkg function {pdf_idx} fraction " \
                        f"to {frac} wrt signal pdf no. {anchor_pdf_idx}")
                    fitter.fix_signal_frac_to_signal_pdf(pdf_idx, anchor_pdf_idx, frac)
                else:
                    print(f"Setting correlated bkg function {pdf_idx} fraction " \
                        f"to {frac} wrt signal pdf no. {anchor_pdf_idx}")
                    fitter.set_signal_initpar(pdf_idx, "frac", frac, limits=[0., 1.])
            else:
                logger(f"Signal function for correlated bkg source {chn} not fixed nor initialized!", level="WARNING")
            continue

        if corr_bkg.get('bkg_func'):
            data_hdl = bkg_templs[chn]['data_hdl']
            pdf_idx = bkg_templs[chn]['idx']
            if corr_bkg['bkg_func'] != 'kde_grid' and corr_bkg['bkg_func'] != 'hist':
                logger(f"Background function for correlated bkg source {chn} not 'kde_grid' or 'hist'", level="ERROR")

            if corr_bkg['bkg_func'] == 'kde_grid':
                print(f"Setting kde for source {chn}")
                fitter.set_background_kde(pdf_idx, data_hdl)
            else:
                print(f"Setting histo for source {chn}")
                fitter.set_background_template(pdf_idx, data_hdl)

            if corr_bkg.get('fix_to'):
                anchor_pdf = corr_bkg['fix_to']
                print(f"frac of chn {chn} wrt anchor {anchor_pdf}: {bkg_templs[chn]['frac']} / {bkg_templs[anchor_pdf]['frac']}")
                frac = bkg_templs[chn]['frac'] / bkg_templs[anchor_pdf]['frac']
                print(f"frac of chn {chn} wrt anchor {anchor_pdf}: {bkg_templs[chn]['frac']} / {bkg_templs[anchor_pdf]['frac']} = {frac}")
                anchor_pdf_idx = bkg_templs[anchor_pdf]['idx']
                if anchor_pdf == 'signal':
                    print(f"Setting correlated bkg template function {pdf_idx} fraction " \
                          f"to {frac} wrt signal pdf no. {anchor_pdf_idx}")
                    fitter.fix_bkg_frac_to_signal_pdf(pdf_idx, anchor_pdf_idx, frac)
                else:
                    print(f"Setting correlated bkg template function {pdf_idx} fraction " \
                          f"to {frac} wrt bkg pdf no. {anchor_pdf_idx}")
                    fitter.fix_bkg_frac_to_bkg_pdf(pdf_idx, anchor_pdf_idx, frac)

            elif corr_bkg.get('init_to'):
                pdf_name = corr_bkg['init_to']
                idx = bkg_templs[pdf_name]['idx']
                frac = bkg_templs[pdf_name]['frac']
                if pdf_name == 'signal':
                    frac_sgn = bkg_templs['signal']['frac']
                    print(f"Setting correlated bkg template function {pdf_idx} initial fraction " \
                        f"to {frac} wrt signal pdf no. {pdf_idx}")
                    fitter.set_background_initpar(idx, "frac", frac/frac_sgn, limits=[0., 1.])
            else:
                logger(f"Background function for correlated bkg source {corr_bkg} without 'fix_to' or 'init_to' key", level="WARNING")

            continue

def perform_fit(pt_cfg, cfg_cutset, corr_bkg_file_path, \
                data_hdl, pt_label, \
                cfg_file, pt_min, pt_max, \
                labels_bkg, labels_sgn):

    if pt_cfg.get("corr_bkgs"):
        corr_bkg_file = TFile.Open(corr_bkg_file_path, 'read')

    # Add correlated bkg templates
    bkg_templs, sgn_templs = {}, {}
    bkg_functs, sgn_functs = [], []
    if pt_cfg.get("corr_bkgs"):

        # Get the fraction
        i_templ_sgn, i_templ_bkg = 0, 0
        for i_source, corr_bkg_source in enumerate(pt_cfg["corr_bkgs"]["channels"]):
            chn = corr_bkg_source['name']

            # Use a signal function instead of a mc template
            if corr_bkg_source.get('sgn_func'):
                sgn_templs[chn] = {}
                print(f"Using signal function for correlated bkg source {chn}")
                sgn_functs.append(corr_bkg_source['sgn_func'])
                _, sgn_templs[chn]['frac'] = get_corr_bkg(cfg_cutset, corr_bkg_file, chn, pt_cfg["fit_range"], pt_label, cfg_file['templ_features'], cfg_file['templ_type'], cfg_file['Dmeson'], cfg_file.get('correct_abundances', False))
                sgn_templs[chn]['idx'] = i_templ_sgn
                i_templ_sgn += 1
                continue

            print(f"Using MC template for correlated bkg source {chn}")
            bkg_templs[chn] = {}
            # Get the correlated bkg template (TTree or TH1)
            if cfg_file['input_type'] == 'Tree':
                bkg_templs[chn]['tree'], bkg_templs[chn]['frac'] = get_corr_bkg(cfg_cutset, corr_bkg_file, chn, pt_cfg["fit_range"], pt_label, cfg_file['templ_features'], cfg_file['templ_type'], cfg_file['Dmeson'], cfg_file.get('correct_abundances', False))
                df = bkg_templs[chn]['tree'].arrays(library="pd")
                bkg_templs[chn]['data_hdl'] = DataHandler(df, var_name="fM", limits=pt_cfg["fit_range"], nbins=100)
                bkg_functs.append("kde_grid")
            else:
                bkg_templs[chn]['hist'], bkg_templs[chn]['frac'] = get_corr_bkg(cfg_cutset, corr_bkg_file, chn, pt_cfg["fit_range"], pt_label, cfg_file['templ_features'], cfg_file['templ_type'], cfg_file['Dmeson'], cfg_file.get('correct_abundances', False))
                bkg_templs[chn]['data_hdl'] = DataHandler(bkg_templs[chn]['hist'], limits=pt_cfg["fit_range"], rebin=pt_cfg.get('rebin', 1))
                bkg_functs.append("hist")

            bkg_templs[chn]['idx'] = i_templ_bkg
            i_templ_bkg += 1

    sgn_templs['signal'] = {}
    sgn_templs['signal']['idx'] = len(sgn_templs)-1
    bkg_templs['signal'] = {}
    bkg_templs['signal']['idx'] = len(sgn_templs)-1

    if pt_cfg.get("corr_bkgs"):
        _, sgn_frac = get_corr_bkg(cfg_cutset, corr_bkg_file, pt_cfg["corr_bkgs"]['sgn_fin_state'], pt_cfg["fit_range"], pt_label, cfg_file['templ_features'], cfg_file['templ_type'])
        print(f"Signal fraction for correlated bkg source {pt_cfg['corr_bkgs']['sgn_fin_state']}: {sgn_frac}")
        sgn_templs['signal']['frac'] = sgn_frac
        bkg_templs['signal']['frac'] = sgn_frac
    bkg_functs = bkg_functs + pt_cfg['bkg_func']
    sgn_functs = sgn_functs + pt_cfg['sgn_func']

    print(f"Using signal function: {sgn_functs} and background function: {bkg_functs}")
    print(f"Using signal labels: {labels_sgn} and background labels: {labels_bkg}")
    fitter = F2MassFitter(data_hdl, label_signal_pdf=labels_sgn, name_signal_pdf=sgn_functs,
                          name_background_pdf=bkg_functs, label_bkg_pdf=labels_bkg, name=pt_label)

    # Set corr bkgs template
    if pt_cfg.get("corr_bkgs"):
        print(f"sgn_templs: {sgn_templs}\n")
        set_corr_bkgs_fracs(fitter, bkg_templs, sgn_templs, pt_cfg)
    if pt_cfg.get("init_pars"):
        set_fitter_init_pars(fitter, pt_cfg, pt_min, pt_max, len(bkg_functs), len(sgn_functs))
    # fitter.set_signal_initpar(sgn_func_idx, "frac", 0.2, limits=[0., 1.])
    if pt_cfg['sgn_func'] == ['hist']:
        sgn_templ, sgn_frac = get_corr_bkg(cfg_cutset, corr_bkg_file, pt_cfg['sgn_template'], pt_cfg["fit_range"], pt_label, cfg_file['templ_features'], cfg_file['templ_type'])
        
        fitter.set_signal_template(len(labels_sgn)-1, DataHandler(sgn_templ, limits=pt_cfg["fit_range"], rebin=pt_cfg.get('rebin', 1)))
    result = fitter.mass_zfit()

    if pt_cfg.get("corr_bkgs"):
        corr_bkg_file.Close()

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

def get_raw_yields(cfg_ry, cfg, cfg_cutset, data, outfile_name, corr_bkg_file_path, mass_axis_label, bkg_labels, sgn_labels):

    if isinstance(data, TH1):
        data_hdl = DataHandler(data, limits=cfg_ry["fit_range"], rebin=cfg_ry.get('rebin', 1))
    else:
        data_hdl = DataHandler(data, var_name='fM', limits=cfg_ry["fit_range"], nbins=100)

    fit_info = {}

    fitter, result = perform_fit(cfg_ry, cfg_cutset, corr_bkg_file_path,
                                 data_hdl, pt_label,
                                 cfg_file, pt_min, pt_max,
                                 bkg_labels, sgn_labels)

    if result.converged: # i.e. has converged
        print(f"Plotting fit results for pt bin {pt_min} - {pt_max} GeV/c")
        fig, axs = fitter.plot_mass_fit(style="ATLAS",
                                        figsize=(8, 8),
                                        axis_title=mass_axis_label,
                                        show_extra_info=True if cfg_ry['sgn_func'] != ['hist'] else False,
                                        logy=cfg_ry.get('logy_plots', False),
                                        extra_info_loc=["lower right", "lower left"]
                                       )
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

        try:
            fit_info['ry'] = fitter.get_raw_yield(0)[0]
            fit_info['ry_unc'] = fitter.get_raw_yield(0)[1]
        except Exception as e:
            logger(f"Could not get raw yield: {e}", "WARNING")
            fit_info['ry'] = -1.
            fit_info['ry_unc'] = -1.
        try:
            fit_info['ry_bin_counting'] = fitter.get_raw_yield_bincounting(0)[0]
            fit_info['ry_bin_counting_unc'] = fitter.get_raw_yield_bincounting(0)[1]
        except Exception as e:
            logger(f"Could not get raw yield from bin counting: {e}", "WARNING")
            fit_info['ry_bin_counting'] = -1.
            fit_info['ry_bin_counting_unc'] = -1.
        try:
            fit_info['signif'] = fitter.get_significance(0)[0]
            fit_info['signif_unc'] = fitter.get_significance(0)[1]
        except Exception as e:
            logger(f"Could not get significance: {e}", "WARNING")
            fit_info['signif'] = -1.
            fit_info['signif_unc'] = -1.
        try:
            fit_info['s_over_b'] = fitter.get_signal_over_background(0)[0]
            fit_info['s_over_b_unc'] = fitter.get_signal_over_background(0)[1]
        except Exception as e:
            logger(f"Could not get signal over background: {e}", "WARNING")
            fit_info['s_over_b'] = -1.
            fit_info['s_over_b_unc'] = -1.
        try:
            fit_info['chi2_fits'] = float(fitter.get_chi2())
            fit_info['chi2_over_ndf_fits'] = float(fitter.get_chi2())/fitter.get_ndf()
        except Exception as e:
            logger(f"Could not get chi2: {e}", "WARNING")
            fit_info['chi2_fits'] = -1.
            fit_info['chi2_over_ndf_fits'] = -1.

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
    parser.add_argument('cfgfile', metavar='text', default='config_fit.yml')
    parser.add_argument('cutsetfile', metavar='text', default='cutset.yml')
    parser.add_argument('infile', metavar='text', default='input_file.root')
    args = parser.parse_args()

    with open(args.cfgfile, 'r', encoding='utf8') as ymlfitConfigFile:
        cfg_file = yaml.load(ymlfitConfigFile, yaml.FullLoader)

    with open(args.cutsetfile, 'r', encoding='utf8') as ymlcutsetFile:
        cutset_file = yaml.load(ymlcutsetFile, yaml.FullLoader)

    _, mass_axis_label, decay, _, _, _ = get_particle_info(cfg_file["Dmeson"])

    # Retrieve config file for fit
    config_fit = copy.deepcopy(cfg_file)

    # Create outfile name and store fit config
    outfile = args.infile.replace('proj', 'rawyield')
    ofile = uproot.recreate(outfile)
    ofile.close()

    # Store fit info
    fit_infos, postfit_infos = {}, {}

    pt_lims = []
    cfg_ry = cfg_file['ry_setup']
    pt_min, pt_max = cfg_ry['pt_range'][0], cfg_ry['pt_range'][1]
    pt_lims = [cfg_ry['pt_range'][0], cfg_ry['pt_range'][1]]

    print(f"Fitting pt bin: {pt_min} - {pt_max} GeV/c")
    pt_label = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"

    # Get data histogram
    data = get_data_to_fit(args.infile, pt_label, cfg_file['input_type'])

    # Produce function labels and perform fit
    sgn_labels, bkg_labels = produce_func_labels(cfg_ry, decay)
    corr_bkg_file_path = f"{cfg_file['corr_bkg_file']}_{pt_label}.root" if cfg_file.get('corr_bkg_file') else None
    pt_fit_info = get_raw_yields(cfg_ry, cfg_file, cutset_file, data, outfile, corr_bkg_file_path, mass_axis_label, bkg_labels, sgn_labels)

    for key in pt_fit_info.keys():
        if key not in fit_infos:
            fit_infos[key] = 0.
        fit_infos[key] = pt_fit_info[key]

    file_std_fit = uproot.update(outfile)
    for par in fit_infos.keys():
        file_std_fit[f"h_{par}"] = create_hist(pt_lims, fit_infos[par], fit_infos[par+'_unc'] if par+'_unc' in fit_infos.keys() else 0.)
    file_std_fit.close()

    # Eventually perform postfit (signal parameters all free) if requested
    if cfg_ry.get('postfit_sgn'):

        cfg_file_postfit = copy.deepcopy(config_fit)
        outfile_postfit = args.infile.replace('proj', 'rawyield_postfit')
        os.makedirs(os.path.dirname(outfile_postfit), exist_ok=True)
        ofile_postfit = uproot.recreate(outfile_postfit)
        ofile_postfit.close()

        cfg_postfit = copy.deepcopy(cfg_file_postfit['ry_setup'])
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

        file_postfit = uproot.update(outfile_postfit)
        for par in postfit_infos.keys():
            file_postfit[f"h_{par}"] = create_hist(pt_lims, postfit_infos[par], postfit_infos[par+'_unc'] if par+'_unc' in postfit_infos.keys() else 0.)
        file_postfit.close()
