'''
Module with function definitions and fit utils
'''

from ROOT import TFile, TMath, TF1, kBlue, kGreen, TDatabasePDG, TH1D, TH1F # pylint: disable=import-error,no-name-in-module
import uproot
from hist import Hist
import numpy as np
from matplotlib.offsetbox import AnchoredText

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
        print(f"chi2: {chi2}, ndf: {ndf}")
        text += fr"$\chi^2 / \mathrm{{ndf}} =${chi2:.2f} / {ndf} $\simeq$ {chi2/ndf:.2f}""\n"

    text += "\n\n"
    text += xspace + system + ", " + r"$\sqrt{s} = 13.6$ TeV" + "\n"
    text += xspace + fr"{pt_min:.1f} < $p_{{\mathrm{{T}}}}$ < {pt_max:.1f} GeV/$c$, $|y|$ < 0.5""\n"

    anchored_text = AnchoredText(text, loc=loc, frameon=False)
    axs.add_artist(anchored_text)

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

def get_data_to_fit(inFileName, pt_dir, input_type):
    '''
    Get data histogram to fit from input file depending on input type

    Parameters
    ----------
    - inFileName: input ROOT file name
    - input_type: type of input ('Sparse' or 'Histo')

    Returns
    ----------
    - hist_data_pt: histogram with data to fit
    '''
    if input_type == "Sparse":
        infile = TFile.Open(inFileName, "READ")
        data_to_fit_pt = infile.Get(f"{pt_dir}/hMassData")
        data_to_fit_pt.SetDirectory(0) 
        infile.Close()
    elif input_type == "Tree":
        with uproot.open(inFileName) as infile:
            data_to_fit_pt_tree = infile[f"{pt_dir}/treeMassData"]
            data_to_fit_pt = data_to_fit_pt_tree.arrays(library="pd")
    else:
        raise ValueError(f"Input type '{input_type}' not recognized. Use 'Sparse' or 'Histo'.")
    return data_to_fit_pt

def rebin_histo(h_orig, reb, first_use = 0):
    '''
    Rebin histogram, from bin firstUse to lastUse
    Use all bins if firstUse=-1
    If ngroup is not an exact divider of the number of bins,
    the bin width is kept as reb*original width
    and the range of rebinned histogram is adapted
    '''
    
    n_bin_orig = h_orig.GetNbinsX()
    first_bin_orig = 1
    last_bin_orig = n_bin_orig
    n_bin_orig_used = n_bin_orig
    n_bin_final = n_bin_orig / reb
    
    if first_use >= 1: 
        first_bin_orig = first_use
        n_bin_final = (n_bin_orig-first_use+1) / reb
        n_bin_orig_used = n_bin_final * reb
        last_bin_orig = first_bin_orig + n_bin_orig_used - 1
    else:
        exc = n_bin_orig_used % reb
        if exc != 0: 
            n_bin_orig_used -= exc
            last_bin_orig = first_bin_orig + n_bin_orig_used - 1

    n_bin_final = round(n_bin_final)
    print(f"Rebin from {n_bin_orig} bins to {n_bin_final} bins -- Used bins = {n_bin_orig_used} in range {first_bin_orig}-{last_bin_orig}\n")
    
    low_lim = h_orig.GetXaxis().GetBinLowEdge(first_bin_orig)
    hi_lim = h_orig.GetXaxis().GetBinUpEdge(last_bin_orig)
    hRebin = TH1D(f"{h_orig.GetName()}-rebin", h_orig.GetTitle(), n_bin_final, low_lim, hi_lim)
    last_summed = first_bin_orig-1
    
    for iBin in range(1, n_bin_final+1):
        sum = 0.
        sume2 = 0.
        for _ in range(reb):
            sum += h_orig.GetBinContent(last_summed+1)
            sume2 += (h_orig.GetBinError(last_summed+1) * h_orig.GetBinError(last_summed+1))
            last_summed += 1
            
        hRebin.SetBinContent(iBin, sum)
        hRebin.SetBinError(iBin, TMath.Sqrt(sume2))
    
    return hRebin

def get_signal_pars_dict(sgn_func, pt_limits):
    """
    Helper function to get signal pars dict
    """
    signal_pars_dict = {}
    if sgn_func == ["gaussian"]:
        signal_pars = ['rawyield', 'mu', 'sigma']
    elif sgn_func == ["doublegaus"]:
        signal_pars = ['rawyield', 'mu', 'sigma1', 'sigma2']
    elif sgn_func == ["doublecbsymm"]:
        signal_pars = ['rawyield', 'mu', 'sigma', 'alpha', 'n']
    elif sgn_func == ["doublecb"]:
        signal_pars = ['rawyield', 'mu', 'sigma', 'alphal', 'nl', 'alphar', 'nr']
    elif sgn_func == ["doublecb", "doublecb"]:
        signal_pars = ['rawyield1', 'mu1', 'sigma1', 'alphal1', 'nl1', 'alphar1', 'nr1',
                       'rawyield2', 'mu2', 'sigma2', 'alphal2', 'nl2', 'alphar2', 'nr2']
    elif sgn_func == ["genergausexptailsymm"]:
        signal_pars = ['rawyield', 'mu', 'sigma', 'alpha']
    elif sgn_func == ["genergausexptail"]:
        signal_pars = ['rawyield', 'mu', 'sigmal', 'alphal', 'sigmar', 'alphar']
    elif sgn_func == ["bifurgaus"]:
        signal_pars = ['rawyield', 'mu', 'sigmal', 'alphal']
    elif sgn_func == ["cauchy"]:
        signal_pars = ['rawyield', 'mu', 'gamma']
    elif sgn_func == ["voigtian"]:
        signal_pars = ['rawyield', 'mu', 'sigma', 'gamma']
    else:
        raise ValueError(f"Signal function '{sgn_func}' not recognized.")

    for par in signal_pars:
        signal_pars_dict[par] = TH1F(f"hist_{par}", f";#it{{p}}_{{T}} (GeV/#it{{c}}); {par}",
                                     len(pt_limits)-1, pt_limits)
    return signal_pars_dict