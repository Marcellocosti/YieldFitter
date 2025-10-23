'''
Script for extracting raw yields for D mesons
run: python get_raw_yields.py fitConfigFileName.yml inFileName.root
'''

import argparse
import numpy as np
import yaml
import os
# import itertools
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

#from utils.kde_producer import kde_producer # TODO: add correlated backgrounds

def get_corr_bkg_template(path, input_type, file_path):
    
    if input_type == 'Tree':
        print(f"Getting corr bkg template from tree {path}treeMass in file {file_path}")
        with uproot.open(corr_bkg_file_path) as f:
            tree = f[f"{path}treeMass"]
        return tree
    else:
        print(f"Getting corr bkg template from histogram {path}hMass in file {file_path}")
        file = TFile.Open(file_path, "READ")
        hist = file.Get(f"{path}hMass")
        hist.SetDirectory(0)
        return hist

def set_fitter_init_pars(fitter, cfg, pt_min, pt_max):
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
            bkg_func_idx = setting[0]
            par_name = setting[1]
            par_val = setting[2]
            fitter.set_background_initpar(bkg_func_idx, par_name, par_val, fix=True)
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
            bkg_func_idx = setting[0]
            par_name = setting[1]
            par_val = setting[2]
            par_lims = setting[3]
            fitter.set_background_initpar(bkg_func_idx, par_name, par_val, limits=par_lims)
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

# pylint: disable=too-many-arguments
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
    particle_name = "Dplus"
    pdg_id = 411

    # Store fit info
    raw_yields, raw_yields_unc = [], []
    signif, signif_unc, s_over_b, s_over_b_unc = [], [], [], []
    means, means_unc, sigmas, sigmas_unc = [], [], [], []

    # Open file with data projections
    infile = TFile(inFileName, "READ")
    corr_bkg_file_path = outfile_name.replace('rawyield', 'corrbkg')

    for ipt, (pt_bin_cfg) in enumerate(config_file['ry_extraction']['pt_bins']):
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
        label_signal_pdf = [rf"$\mathrm{{{particle_name}}}$ signal"]

        # Add correlated bkg templates
        corr_bkgs_templs, sgn_bkgs_templs = [], []
        if pt_bin_cfg.get("incl_corr_bkgs"):

            # Get the fraction
            corr_bkg_file = TFile.Open(corr_bkg_file_path, 'read')
            pt_subdir = corr_bkg_file.Get(pt_label)
            hist_fractions = pt_subdir.Get("hWeightsAnchorSignal")
            hist_fractions.SetDirectory(0)
            corr_bkg_file.Close()

            sgn_fin_state = config_file["corr_bkgs"]["sgn_fin_state"]
            axis = hist_fractions.GetXaxis()
            for corr_bkg_source in pt_bin_cfg["incl_corr_bkgs"]:
                print(f"\n\nFetching template for channel {corr_bkg_source}")

                # find corresponding fraction
                frac_to_sgn = next((hist_fractions.GetBinContent(i) for i in range(1, hist_fractions.GetNbinsX() + 1)
                                    if axis.GetBinLabel(i) == corr_bkg_source), 0)

                # Use a signal function instead of a mc template
                if any(corr_bkg_source == corr_bkg_source_func for corr_bkg_source_func in pt_bin_cfg.get('corr_bkgs_as_sgn_functs', [])):
                    print(f"Using signal function for correlated bkg source {corr_bkg_source}")
                    for i_func, corr_bkg_sgn_func in enumerate(pt_bin_cfg['corr_bkgs_sgn_functs']):
                        sgn_functs.append(corr_bkg_sgn_func)
                        sgn_bkgs_templs.append(frac_to_sgn)
                    label_signal_pdf.append(rf"$\mathrm{{{corr_bkg_source}}}$")
                    continue

                print(f"Using MC template for correlated bkg source {corr_bkg_source}")
                # Get the correlated bkg template (TTree or TH1)
                corr_bkg_templ = get_corr_bkg_template(f"{pt_label}/{corr_bkg_source}/", config_file['input_type'], corr_bkg_file_path)
                if config_file['input_type'] == 'Tree':
                    df = tree.arrays(library="pd")
                    corr_bkgs_templs.append([
                        DataHandler(df, var_name="fM", limits=pt_bin_cfg["fit_range"],
                                    nbins=100),
                        frac_to_sgn
                    ])
                    bkg_functs.append("kde_grid")
                else:
                    corr_bkgs_templs.append([
                        DataHandler(corr_bkg_templ, limits=pt_bin_cfg["fit_range"],
                                    rebin=pt_bin_cfg.get('rebin', 1)),
                        frac_to_sgn
                    ])
                    bkg_functs.append("hist")

                label_bkg_pdf.append(corr_bkg_source)

        # sgn_functs = sgn_functs + pt_bin_cfg['sgn_func']
        label_bkg_pdf = label_bkg_pdf + ["Comb. bkg"]
        bkg_functs = bkg_functs + pt_bin_cfg['bkg_func']
        print(f"\n\ncorr_bkgs_templs: {corr_bkgs_templs}\n\n")
        print(f"Using signal function: {sgn_functs} and background function: {bkg_functs}")
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
        if pt_bin_cfg.get("incl_corr_bkgs"):
            # Fix the fractions for templates modelled with MC (background functions)
            for i_fin_state, (corr_bkg_templ, frac_to_sgn) in enumerate(corr_bkgs_templs):
                print(f"Setting correlated bkg {i_fin_state} fraction " \
                      f"to {frac_to_sgn} wrt signal pdf no. {sgn_func_idx}")
                if isinstance(data_pt, TH1):
                    fitter_pt.set_background_template(i_fin_state, corr_bkg_templ)
                else:
                    fitter_pt.set_background_kde(i_fin_state, corr_bkg_templ)
                fitter_pt.fix_bkg_frac_to_signal_pdf(i_fin_state, sgn_func_idx, frac_to_sgn)

            # Fix the fractions for templates modelled with functions (signal functions)
            for i_func, frac_to_sgn in enumerate(sgn_bkgs_templs):
                print(f"len(pt_bin_cfg['sgn_func']): {len(pt_bin_cfg['sgn_func'])}, fixing frac")
                fitter_pt.fix_signal_frac_to_signal_pdf(i_func+len(pt_bin_cfg['sgn_func']), sgn_func_idx, frac_to_sgn)

        fitter_pt.set_signal_initpar(sgn_func_idx, "frac", 0.2, limits=[0., 1.])
        if pt_bin_cfg.get("init_pars"):
            set_fitter_init_pars(fitter_pt, pt_bin_cfg["init_pars"], pt_min, pt_max)
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

            outdir = os.path.join(os.path.dirname(os.path.dirname(inFileName)), 'rawyields')
            print(f"Saving figures in {outdir}")
            fig.savefig(os.path.join(outdir, f"{particle_name}_mass_{pt_label}.pdf"))
            fig_res.savefig(os.path.join(outdir, f"{particle_name}_massres_{pt_label}.pdf"))

            rawy, rawy_unc = fitter_pt.get_raw_yield(0)
            sign, sign_unc = fitter_pt.get_significance(0)
            soverb, soverb_unc = fitter_pt.get_signal_over_background(0)
            mean, mean_unc = fitter_pt.get_signal_parameter(0, "mu")
            sigma, sigma_unc = fitter_pt.get_signal_parameter(0, "sigma")

            raw_yields.append(rawy)
            raw_yields_unc.append(rawy_unc)
            signif.append(sign)
            signif_unc.append(sign_unc)
            s_over_b.append(soverb)
            s_over_b_unc.append(soverb_unc)
            means.append(mean)
            means_unc.append(mean_unc)
            sigmas.append(sigma)
            sigmas_unc.append(sigma_unc)

            fitter_pt.dump_to_root(outfile_name, option="update", suffix=f"_{pt_label}")

    file_root = uproot.update(outfile_name)
    pt_lims = config_file['ptbins']
    file_root["h_rawyields"] = create_hist(pt_lims, raw_yields, raw_yields_unc)
    file_root["h_significance"] = create_hist(pt_lims, signif, signif_unc)
    file_root["h_soverb"] = create_hist(pt_lims, s_over_b, s_over_b_unc)
    file_root["h_means"] = create_hist(pt_lims, means, means_unc)
    file_root["h_sigmas"] = create_hist(pt_lims, sigmas, sigmas_unc)
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