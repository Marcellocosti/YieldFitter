'''
Script for extracting raw yields for D mesons
run: python get_raw_yields.py fitConfigFileName.yml inFileName.root
'''

import argparse
import numpy as np
import yaml
import os
import itertools
from ROOT import TLatex, TFile, TCanvas, TLegend, TH1, TH1D, TH1F, TGraphAsymmErrors # pylint: disable=import-error,no-name-in-module
from ROOT import gROOT, gPad, gInterpreter, kBlack, kRed, kBlue, kMagenta, kAzure, kOrange, kGreen, kFullCircle, kFullSquare, kOpenCircle # pylint: disable=import-error,no-name-in-module
import zfit
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
from matplotlib.offsetbox import AnchoredText
import uproot
from hist import Hist
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position

#from utils.kde_producer import kde_producer # TODO: add correlated backgrounds


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


def get_raw_yields(fitConfigFileName):
    #______________________________________________________
    # Read configuration file
    with open(fitConfigFileName, 'r', encoding='utf8') as ymlfitConfigFile:
        config_file = yaml.load(ymlfitConfigFile, yaml.FullLoader)
    
    cfg_fit = config_file['ry_extraction']

    # Set outfile name
    file_root = uproot.recreate("rawyields.root")
    file_root.close()

    decay_channel = r"\pi^{\plus} K^{-} \pi^{\plus}"
    particle_name = "Dplus"
    pdg_id = 411

    # Load fit configuration
    pt_mins = config_file['ptbins'][:-1]
    pt_maxs = config_file['ptbins'][1:]
    bkg_functs = cfg_fit["BkgFunc"]
    sgn_functs = cfg_fit["SgnFunc"]

    print(f"Signal functions: {sgn_functs}")
    print(f"Background functions: {bkg_functs}")

    fix_means = cfg_fit["FixMean"]
    fix_sigmas = cfg_fit["FixSigma"]
    if not isinstance(fix_means, list):
        fix_means = [fix_means] * len(pt_mins)
    if not isinstance(fix_sigmas, list):
        fix_sigmas = [fix_sigmas] * len(pt_mins)

    rebin_factors = cfg_fit["Rebin"]
    if not isinstance(rebin_factors, list):
        rebin_factors = [rebin_factors] * len(pt_mins)

    # Store fit info
    raw_yields, raw_yields_unc = [], []
    signif, signif_unc, s_over_b, s_over_b_unc = [], [], [], []
    means, means_unc, sigmas, sigmas_unc = [], [], [], []
    means_mc, means_mc_unc, sigmas_mc, sigmas_mc_unc = [], [], [], []

    # Open file with data projections
    infile = TFile("data.root", "READ")
    if cfg_fit.get("IncludeCorrBkgs"):
        corr_bkg_file = TFile.Open("corrbkg.root", 'read')

    for ipt, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs)):
        print(f"Fitting pt bin: {pt_min} - {pt_max} GeV/c")
        print(f"Background function: {bkg_functs[ipt]}")
        pt_label = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        # Get data histogram
        hist_data_pt = infile.Get(f"{pt_label}/hMassData")

        data_hdl = DataHandler(hist_data_pt, limits=cfg_fit["MassFitRanges"][ipt],
                               rebin=rebin_factors[ipt])

        label_bkg_pdf = ["Comb. bkg"]

        # Add correlated bkg templates
        print("\n\nADDING CORRELATED BKGS\n\n")
        corr_bkgs_templs = []
        if cfg_fit.get("IncludeCorrBkgs"):
            pt_subdir = corr_bkg_file.Get(pt_label)
            hist_fractions = pt_subdir.Get("hWeightsAnchorSignal")

            sgn_fin_state = config_file["corr_bkgs"]["sgn_fin_state"]
            axis = hist_fractions.GetXaxis()
            i_fin_state = 0
            for key in pt_subdir.GetListOfKeys():
                # print(f"Found key: {key.GetName()}")
                name = key.GetName()
                if name == sgn_fin_state or name.startswith("h"):
                    # print(f"Skipping {name}")
                    continue

                # find corresponding fraction
                frac_to_sgn = next((hist_fractions.GetBinContent(i) for i in range(1, hist_fractions.GetNbinsX() + 1)
                                    if axis.GetBinLabel(i) == name), 0)

                print(f"Adding correlated bkg final state: {name}")

                hist = pt_subdir.Get(f"{name}/hMass")
                print(f"Type of hist: {type(hist)}")
                corr_bkgs_templs.append([
                    DataHandler(hist, limits=cfg_fit["MassFitRanges"][ipt],
                                nbins=hist_data_pt.GetNbinsX()),
                    frac_to_sgn
                ])
                bkg_functs[ipt].insert(i_fin_state, "hist")
                label_bkg_pdf.insert(i_fin_state, key.GetName())
                # bkg_functs[ipt].insert(i_fin_state, "hist")
                # label_bkg_pdf.insert(i_fin_state, key.GetName())
                i_fin_state += 1

        # quit()
        print(f"Using signal function: {sgn_functs[ipt]} and background function: {bkg_functs[ipt]}")
        fitter_pt = F2MassFitter(data_hdl,
                                 name_signal_pdf=sgn_functs[ipt],
                                 name_background_pdf=bkg_functs[ipt],
                                 name=f"{particle_name}_{pt_label}",
                                 label_signal_pdf=[rf"$\mathrm{{{particle_name}}}$ signal"],
                                 label_bkg_pdf=label_bkg_pdf
                                )

        # Set reflection template
        print(f"corr_bkgs_templs: {corr_bkgs_templs}")
        if cfg_fit.get("IncludeCorrBkgs"):
            print(f"bkg_functs[ipt]: {bkg_functs[ipt]}")
            for i_fin_state, (histo, frac_to_sgn) in enumerate(corr_bkgs_templs):
                print(f"Setting correlated bkg {i_fin_state} with fraction to signal {frac_to_sgn} and histo {type(histo)}")
                fitter_pt.set_background_template(i_fin_state, histo)
                # fitter_pt.fix_bkg_frac_to_signal_pdf(i_fin_state, 0, 0.)
                fitter_pt.fix_bkg_frac_to_signal_pdf(i_fin_state, 0, frac_to_sgn)

        if not fix_means[ipt]:
            fitter_pt.set_particle_mass(0, pdg_id=pdg_id, limits=[1.8, 1.9])
        else:
            fitter_pt.set_particle_mass(0, pdg_id=pdg_id, fix=True)
        if not fix_sigmas[ipt]:
            fitter_pt.set_signal_initpar(0, "sigma", 0.05, limits=[0.01, 0.1])
        else:
            fitter_pt.set_signal_initpar(0, "sigma", 0.05, fix=True)

        fitter_pt.set_background_initpar(2, "c0", 0.4)
        fitter_pt.set_background_initpar(2, "c1", 0.01)
        fitter_pt.set_background_initpar(2, "c2", -0.02)
        fitter_pt.set_signal_initpar(0, "frac", 0.2, limits=[0., 1.])
        result = fitter_pt.mass_zfit()
        if result.converged:
            fig, axs = fitter_pt.plot_mass_fit(style="ATLAS",
                                               figsize=(8, 8),
                                               axis_title=rf"$M(\mathrm{{{decay_channel}}})$ (GeV/$c^2$)",
                                               show_extra_info=True,
                                               extra_info_loc=["lower right", "lower left"])
            add_info_on_canvas(axs, "upper left", "pp", pt_min, pt_max)

            fig_res = fitter_pt.plot_raw_residuals(style="ATLAS",
                                                   figsize=(8, 8),
                                                   axis_title=rf"$M(\mathrm{{{decay_channel}}})$ (GeV/$c^2$)")

            fig.savefig(f"./{particle_name}_mass_{pt_label}.pdf")
            fig_res.savefig(f"./{particle_name}_massres_{pt_label}.pdf")

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
            # quit()

    file_root = uproot.update(outfile_name)
    pt_lims = config_file['ptbins']
    file_root["h_rawyields"] = create_hist(pt_lims, raw_yields, raw_yields_unc)
    file_root["h_significance"] = create_hist(pt_lims, signif, signif_unc)
    file_root["h_soverb"] = create_hist(pt_lims, s_over_b, s_over_b_unc)
    file_root["h_means"] = create_hist(pt_lims, means, means_unc)
    file_root["h_sigmas"] = create_hist(pt_lims, sigmas, sigmas_unc)
    file_root["h_means_mc"] = create_hist(pt_lims, means_mc, means_mc_unc)
    file_root["h_sigmas_mc"] = create_hist(pt_lims, sigmas_mc, sigmas_mc_unc)
    file_root.close()

    if cfg_fit.get("IncludeCorrBkgs"):
        corr_bkg_file.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('config', metavar='text', default='config_Ds_Fit.yml')
    args = parser.parse_args()

    get_raw_yields(args.config)