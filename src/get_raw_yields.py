'''
Script for extracting raw yields for D mesons
run: python get_raw_yields.py fitConfigFileName.yml inFileName.root
'''

import argparse
import numpy as np
import yaml
import os
from ROOT import TFile, TH1, TH1D, TH1F # pylint: disable=import-error,no-name-in-module
script_dir = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.join(script_dir, '..', 'utils'))
from fit_utils import RebinHisto, get_data_to_fit
from utils import logger, get_centrality_bins, get_particle_info
import zfit
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
from matplotlib.offsetbox import AnchoredText
import uproot
from hist import Hist
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position

def get_corr_bkg_template(path, input_type, file_path):
    
    if input_type == 'Tree':
        print(f"Getting corr bkg template from tree {path}treeMass in file {file_path}")
        with uproot.open(corr_bkg_file_path) as f:
            tree = f[f"{path}treeMass"]
        return tree
    else:
        print(f"Getting corr bkg template from histogram {path}hMass in file {file_path}")
        file = TFile.Open(file_path, "READ")
        try:
            hist = file.Get(f"{path}hMass_smooth")
            print(f"Using smoothed hist")
        except Exception:
            print(f"No smoothed hist found, getting original hist")
            hist = file.Get(f"{path}hMass")
        hist.SetDirectory(0)
        return hist

def set_fitter_init_pars(fitter, cfg, pt_min, pt_max, n_bkg_functs):
    print(f"Setting fitter initial parameters for pt range {pt_min} - {pt_max} GeV/c")

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

                    print(f"---> fixing signal parameter {par_name} to value {par_val}")
                    fitter.set_signal_initpar(sgn_func_idx, par_name, par_val, fix=True)
                    
                except Exception as e:
                    print(f"        Parameter {par_name} not present!")

            par_file.Close()

def create_hist(pt_lims, contents, errors, label_pt=r"$p_\mathrm{T}~(\mathrm{GeV}/c)$"):
    """
    Helper method to create histogram

    Parameters
    ----------

    - pt_lims (list): list of pt limits
    - contents (list): histogram contents
    - errors (list): histogram errors
    - label_pt (str): label for x axis

    Returns
    ----------
    - histogram (hist.Hist)

    """

    pt_cent = [(pt_min+pt_max)/2 for pt_min, pt_max in zip(pt_lims[:-1], pt_lims[1:])]
    histo = Hist.new.Var(pt_lims, name="x", label=label_pt).Weight()
    histo.fill(pt_cent, weight=contents)
    histo.view(flow=False).variance = np.array(errors)**2

    return histo

def add_info_on_canvas(axs, loc, system, pt_min, pt_max, fitter=None):
    """
    Helper method to add text on flarefly mass fit plot

    Parameters
    ----------
    - axs: matplotlib.figure.Axis
        Axis instance of the mass fit figure

    - loc: str
        Location of the info on the figure

    - system: str
        System (pp, MC pp)

    - pt_min: float
        Minimum pT value in the pT range

    - pt_max: float
        Maximum pT value in the pT range

    - fitter: F2MassFitter
        Fitter instance allowing to access chi2 and ndf if wanted
    """
    xspace = " "
    text = xspace
    if fitter is not None:
        chi2 = fitter.get_chi2()
        ndf = fitter.get_ndf()
        text += fr"$\chi^2 / \mathrm{{ndf}} =${chi2:.2f} / {ndf} $\simeq$ {chi2/ndf:.2f}""\n"

    text += "\n\n"
    text += xspace + system + ", " + r"$\sqrt{s} = 13.6$ TeV" + "\n"
    text += xspace + fr"{pt_min:.1f} < $p_{{\mathrm{{T}}}}$ < {pt_max:.1f} GeV/$c$, $|y|$ < 0.5""\n"

    anchored_text = AnchoredText(text, loc=loc, frameon=False)
    axs.add_artist(anchored_text)

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
                print(f"Setting correlated bkg template function {bkg_func_idx} fraction " \
                      f"to {frac} wrt signal pdf no. {pdf_idx}")
                if anchor_pdf_name == 'signal':
                    fitter.fix_bkg_frac_to_signal_pdf(bkg_func_idx, pdf_idx, frac)
                else:
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

def get_raw_yields(fitConfigFileName, inFileName):
    #______________________________________________________
    # Read configuration file
    with open(fitConfigFileName, 'r', encoding='utf8') as ymlfitConfigFile:
        config_file = yaml.load(ymlfitConfigFile, yaml.FullLoader)

    # Set outfile name
    outfile_name = os.path.join(os.path.dirname(os.path.dirname(inFileName)),
                               'rawyields',
                               os.path.basename(inFileName).replace('proj', 'rawyield'))
    file_root = uproot.recreate(outfile_name)
    file_root.close()

    decay_channel = r"\pi^{\plus} K^{-} \pi^{\plus}"
    particle_name = config_file["Dmeson"]
    pdg_id = 411

    # Store fit info
    raw_yields, raw_yields_unc = [], []
    raw_yields_bin_counting, raw_yields_bin_counting_unc = [], []
    signif, signif_unc, s_over_b, s_over_b_unc = [], [], [], []
    means, means_unc, sigmas, sigmas_unc = [], [], [], []
    dict_pars = {}

    # Open file with data projections
    infile = TFile(inFileName, "READ")
    corr_bkg_file_path = outfile_name.replace('rawyield', 'corrbkg')

    pt_lims = []
    for ipt, (pt_bin_cfg) in enumerate(config_file['ry_extraction']['pt_bins']):
        if ipt == 0:
            pt_lims.append(pt_bin_cfg['pt_range'][0])
        pt_lims.append(pt_bin_cfg['pt_range'][1])
        pt_min, pt_max = pt_bin_cfg['pt_range']
        print(f"Fitting pt bin: {pt_min} - {pt_max} GeV/c")
        pt_label = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        # Get data histogram
        data_pt = get_data_to_fit(inFileName, pt_label, config_file['input_type'])

        if isinstance(data_pt, TH1):
            data_hdl = DataHandler(data_pt, limits=pt_bin_cfg["fit_range"],
                                   rebin=pt_bin_cfg.get('rebin', 1))
        else:
            data_hdl = DataHandler(data_pt, var_name='fM', limits=pt_bin_cfg["fit_range"], nbins=100)

        bkg_functs = []
        sgn_functs = pt_bin_cfg['sgn_func'].copy()
        print(f"len(pt_bin_cfg['bkg_func']): {len(pt_bin_cfg['bkg_func'])}")
        label_bkg_pdf = []
        if len(pt_bin_cfg['sgn_func']) > 1:
            label_signal_pdf = pt_bin_cfg['sgn_func_labels']
        else:
            label_signal_pdf = [config_file['ry_extraction']['signal_label']]
            # label_signal_pdf = [rf"$\mathrm{{{particle_name}}}$ signal"]
        print(f"\n\nUsing signal function: {sgn_functs} and background function: {bkg_functs}")

        # Add correlated bkg templates
        corr_bkgs_templs, sgn_bkgs_templs = {}, {}
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
                    label_signal_pdf.append(rf"$\mathrm{{{chn_name}}}$")
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
                label_bkg_pdf.append(chn_name)

        label_bkg_pdf = label_bkg_pdf + ["Comb. bkg"]
        bkg_functs = bkg_functs + pt_bin_cfg['bkg_func']
        print(f"\n\ncorr_bkgs_templs: {corr_bkgs_templs}\n\n")
        print(f"Using signal function: {sgn_functs} and background function: {bkg_functs}")
        
        for label in label_signal_pdf:
            # Account for different parameters in different pt bins
            if label in dict_pars.keys():
                continue
            dict_pars[label] = {}
        for label in label_bkg_pdf:
            if label in dict_pars.keys():
                continue
            dict_pars[label] = {}

        fitter_pt = F2MassFitter(data_hdl,
                                 name_signal_pdf=sgn_functs,
                                 name_background_pdf=bkg_functs,
                                 name=f"{particle_name}_{pt_label}",
                                 label_signal_pdf=label_signal_pdf,
                                 label_bkg_pdf=label_bkg_pdf
                                )

        # Set reflection template
        print(f"len(pt_bin_cfg['bkg_func']): {len(pt_bin_cfg['bkg_func'])}")
        sgn_func_idx = pt_bin_cfg.get("sgn_func_idx", 0)  # Assuming first signal function is the main signal
        if pt_bin_cfg.get("corr_bkgs"):
            set_corr_bkgs(fitter_pt, corr_bkgs_templs, sgn_bkgs_templs, pt_bin_cfg)
        if pt_bin_cfg.get("init_pars"):
            set_fitter_init_pars(fitter_pt, pt_bin_cfg["init_pars"], pt_min, pt_max, len(bkg_functs))
        fitter_pt.set_signal_initpar(sgn_func_idx, "frac", 0.2, limits=[0., 1.])
        result = fitter_pt.mass_zfit()

        if result.converged:
            fig, axs = fitter_pt.plot_mass_fit(style="ATLAS",
                                               figsize=(8, 8),
                                               axis_title=rf"$M(\mathrm{{{decay_channel}}})$ (GeV/$c^2$)",
                                               show_extra_info=True,
                                               logy=config_file['ry_extraction'].get('logy_plots', False),
                                               extra_info_loc=["lower right", "lower left"])
            add_info_on_canvas(axs, "upper left", "pp", pt_min, pt_max)

            fig_res = fitter_pt.plot_raw_residuals(style="ATLAS",
                                                   figsize=(8, 8),
                                                   axis_title=rf"$M(\mathrm{{{decay_channel}}})$ (GeV/$c^2$)")

            fig_pulls = fitter_pt.plot_std_residuals(style="ATLAS",
                                                   figsize=(8, 8),
                                                   axis_title=rf"$M(\mathrm{{{decay_channel}}})$ (GeV/$c^2$)")

            outdir = os.path.join(os.path.dirname(os.path.dirname(inFileName)), 'rawyields')
            print(f"Saving figures in {outdir}")
            fig.savefig(os.path.join(outdir, f"{particle_name}_mass_{pt_label}.pdf"))
            fig_res.savefig(os.path.join(outdir, f"{particle_name}_massres_{pt_label}.pdf"))
            fig_pulls.savefig(os.path.join(outdir, f"{particle_name}_masspulls_{pt_label}.pdf"))

            rawy, rawy_unc = fitter_pt.get_raw_yield(0)
            rawy_bc, rawy_unc_bc = fitter_pt.get_raw_yield_bincounting(0)
            sign, sign_unc = fitter_pt.get_significance(0)
            soverb, soverb_unc = fitter_pt.get_signal_over_background(0)
            mean, mean_unc = fitter_pt.get_signal_parameter(0, "mu")
            sigma, sigma_unc = fitter_pt.get_signal_parameter(0, "sigma")

            raw_yields.append(rawy)
            raw_yields_unc.append(rawy_unc)
            raw_yields_bin_counting.append(rawy_bc)
            raw_yields_bin_counting_unc.append(rawy_unc_bc)
            signif.append(sign)
            signif_unc.append(sign_unc)
            s_over_b.append(soverb)
            s_over_b_unc.append(soverb_unc)
            means.append(mean)
            means_unc.append(mean_unc)
            sigmas.append(sigma)
            sigmas_unc.append(sigma_unc)

            signal_pars = fitter_pt.get_signal_pars()
            signal_pars_uncs = fitter_pt.get_signal_pars_uncs()
            for i_label, label in enumerate(label_signal_pdf):
                for (par_name, val), (par_name, unc) in zip(signal_pars[i_label].items(), signal_pars_uncs[i_label].items()):
                    if par_name not in dict_pars[label].keys():
                        dict_pars[label][f'{par_name}'] = [0.0] * len(config_file['ry_extraction']['pt_bins'])
                        dict_pars[label][f'{par_name}_uncs'] = [0.0] * len(config_file['ry_extraction']['pt_bins'])

                    dict_pars[label][f'{par_name}'][ipt] = val
                    dict_pars[label][f'{par_name}_uncs'][ipt] = unc

            bkg_pars = fitter_pt.get_bkg_pars()
            bkg_pars_uncs = fitter_pt.get_bkg_pars_uncs()
            for i_label, label in enumerate(label_bkg_pdf):
                for (par_name, val), (par_name, unc) in zip(bkg_pars[i_label].items(), bkg_pars_uncs[i_label].items()):
                    if par_name not in dict_pars[label].keys():
                        dict_pars[label][f'{par_name}'] = [0.0] * len(config_file['ry_extraction']['pt_bins'])
                        dict_pars[label][f'{par_name}_uncs'] = [0.0] * len(config_file['ry_extraction']['pt_bins'])

                    dict_pars[label][f'{par_name}'][ipt] = val
                    dict_pars[label][f'{par_name}_uncs'][ipt] = unc

            fitter_pt.dump_to_root(outfile_name, option="update", suffix=f"_{pt_label}")

    file_root = uproot.update(outfile_name)
    file_root["h_rawyields"] = create_hist(pt_lims, raw_yields, raw_yields_unc)
    file_root["h_rawyields_bin_counting"] = create_hist(pt_lims, raw_yields_bin_counting, raw_yields_bin_counting_unc)
    file_root["h_significance"] = create_hist(pt_lims, signif, signif_unc)
    file_root["h_soverb"] = create_hist(pt_lims, s_over_b, s_over_b_unc)
    for func in dict_pars.keys():
        for par_name, par_vals_uncs in dict_pars[func].items():
            if par_name.endswith('_uncs'):
                continue
            file_root[f"h_{func}_{par_name}"] = create_hist(pt_lims, dict_pars[func][par_name], dict_pars[func][f'{par_name}_uncs'])
    file_root.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('configfitFileName', metavar='text', default='config_Ds_Fit.yml')
    parser.add_argument('inFileName', metavar='text', default='')
    args = parser.parse_args()

    get_raw_yields(
        args.configfitFileName,
        args.inFileName,
    )