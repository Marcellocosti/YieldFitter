import ctypes
import ROOT
import yaml
import argparse
import os
import numpy as np
from ROOT import TFile, TCanvas, TH1F, TLegend, gROOT  # pylint: disable=import-error,no-name-in-module
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../utils")
from StyleFormatter import SetGlobalStyle, SetObjectStyle

SetGlobalStyle(titleoffsety=1.1, maxdigits=3, topmargin=0.1,
               bottommargin=0.4, leftmargin=0.3, rightmargin=0.15,
               labelsizey=0.04, setoptstat=0, setopttitle=0,
               setdecimals=True, titleoffsetx=0.91, titlesizex=0.05)

def eval_eff(recoCounts, genCounts, recoCountsError, genCountsError):
    '''
    Method to compute efficiency

    Parameters
    ----------
    - recoCounts: number of reconstructed D
    - genCounts: number of genertated D
    - recoCountsError: error on number of reconstructed D
    - genCountsError: error on number of generated D

    Returns
    ----------
    - efficiency, error on efficiency
    '''
    hTmpNum = TH1F('hTmpNum', '', 1, 0, 1)
    hTmpDen = TH1F('hTmpDen', '', 1, 0, 1)
    hTmpNum.SetBinContent(1, recoCounts)
    hTmpDen.SetBinContent(1, genCounts)
    hTmpNum.SetBinError(1, recoCountsError)
    hTmpDen.SetBinError(1, genCountsError)
    hTmpNum.Divide(hTmpNum, hTmpDen, 1., 1, 'B')
    
    efficiency = hTmpNum.GetBinContent(1)
    error = hTmpNum.GetBinError(1)
    
    hTmpDen.Delete()
    hTmpNum.Delete()

    return efficiency, error

def compute_eff(config, inputFile, batch=False):
    '''
    Method to compute efficiency from input file

    Parameters
    ----------
    - config_file: configuration file
    - inputFile: input file with histograms
    - batch: run in batch mode
    '''

    #_____________________________________________________________________________________
    # Set batch mode
    gROOT.SetBatch(batch)

    #_____________________________________________________________________________________
    # Load input files
    infile = ROOT.TFile.Open(inputFile)
    ptBins = config['ptbins']
    nPtBins = len(ptBins)

    #_____________________________________________________________________________________
    # define histograms
    hEffPrompt       = TH1F('hEffPrompt',       ';#it{p}_{T} (GeV/#it{c}); Efficiency',     nPtBins-1, np.asarray(ptBins, 'd'))
    hEffFD           = TH1F('hEffFD',           ';#it{p}_{T} (GeV/#it{c}); Efficiency',     nPtBins-1, np.asarray(ptBins, 'd'))
    hYieldPromptGen  = TH1F('hYieldPromptGen',  ';#it{p}_{T} (GeV/#it{c}); # Generated MC', nPtBins-1, np.asarray(ptBins, 'd'))
    hYieldFDGen      = TH1F('hYieldFDGen',      ';#it{p}_{T} (GeV/#it{c}); # Generated MC', nPtBins-1, np.asarray(ptBins, 'd'))
    hYieldPromptReco = TH1F('hYieldPromptReco', ';#it{p}_{T} (GeV/#it{c}); # Reco MC',      nPtBins-1, np.asarray(ptBins, 'd'))
    hYieldFDReco     = TH1F('hYieldFDReco',     ';#it{p}_{T} (GeV/#it{c}); # Reco MC',      nPtBins-1, np.asarray(ptBins, 'd'))
    SetObjectStyle(hEffPrompt, markerstyle=20,
                   markercolor=ROOT.kOrange+1,
                   markersize=1., linecolor=ROOT.kOrange+1)
    SetObjectStyle(hEffFD, markerstyle=21,
                   linestyle=2,
                   markercolor=ROOT.kAzure+2, markersize=1.,
                   linecolor=ROOT.kAzure+2)
    SetObjectStyle(hYieldPromptGen, color=ROOT.kRed+1, markerstyle=20)
    SetObjectStyle(hYieldFDGen, color=ROOT.kAzure+4, markerstyle=21, markersize=1.5, linewidh=2, linestyle=7)
    SetObjectStyle(hYieldPromptReco, color=ROOT.kRed+1, markerstyle=20)
    SetObjectStyle(hYieldFDReco, color=ROOT.kAzure+4, markerstyle=21, markersize=1.5, linewidh=2, linestyle=7)

    #_____________________________________________________________________________________
    # Compute efficiency
    for iPt, (ptMin, ptMax) in enumerate(zip(ptBins[:-1], ptBins[1:])):
        ## get input histograms, adjustments needed for reflections?
        pt_label = f'pt_{int(ptMin*10)}_{int(ptMax*10)}'
        hRecoPrompt = infile.Get(f'{pt_label}/hPromptPt')
        hRecoFD = infile.Get(f'{pt_label}/hFDPt')
        hGenPrompt = infile.Get(f'{pt_label}/hPromptGenPt')
        hGenFD = infile.Get(f'{pt_label}/hFDGenPt')

        ## load the values
        nRecoPromptUnc, nGenPromptUnc, nRecoFDUnc, nGenFDUnc = (ctypes.c_double() for _ in range(4))
        nRecoPrompt = hRecoPrompt.IntegralAndError(0, hRecoPrompt.GetNbinsX()+1, nRecoPromptUnc)
        nGenPrompt  = hGenPrompt.IntegralAndError(0, hGenPrompt.GetNbinsX()+1, nGenPromptUnc)
        nRecoFD     = hRecoFD.IntegralAndError(0, hRecoFD.GetNbinsX()+1, nRecoFDUnc)
        nGenFD      = hGenFD.IntegralAndError(0, hGenFD.GetNbinsX()+1, nGenFDUnc)

        hYieldPromptGen.SetBinContent(iPt+1, nGenPrompt)
        hYieldPromptGen.SetBinError(iPt+1, nGenPromptUnc.value)
        hYieldFDGen.SetBinContent(iPt+1, nGenFD)
        hYieldFDGen.SetBinError(iPt+1, nGenFDUnc.value)
        hYieldPromptReco.SetBinContent(iPt+1, nRecoPrompt)
        hYieldPromptReco.SetBinError(iPt+1, nRecoPromptUnc.value)
        hYieldFDReco.SetBinContent(iPt+1, nRecoFD)
        hYieldFDReco.SetBinError(iPt+1, nRecoFDUnc.value)

        ## calculate efficiency
        effPrompt, effPromptUnc = eval_eff(nRecoPrompt, nGenPrompt, nRecoPromptUnc.value, nGenPromptUnc.value)
        hEffPrompt.SetBinContent(iPt+1, effPrompt)
        hEffPrompt.SetBinError(iPt+1, effPromptUnc)
        effFD, effFDUnc = eval_eff(nRecoFD, nGenFD, nRecoFDUnc.value, nGenFDUnc.value)
        hEffFD.SetBinContent(iPt+1, effFD)
        hEffFD.SetBinError(iPt+1, effFDUnc)

    #_____________________________________________________________________________________
    # Draw histograms
    leg = TLegend(0.6, 0.2, 0.8, 0.4)
    leg.SetTextSize(0.045)
    leg.SetFillStyle(0)
    leg.AddEntry(hEffPrompt, "Prompt", "p")
    leg.AddEntry(hEffFD, "Feed-down", "p")

    cEff = TCanvas('cEff', '', 800, 800)
    cEff.DrawFrame(ptBins[0], 1.e-5, ptBins[-1], 1.,
                ';#it{p}_{T} (GeV/#it{c});Efficiency;')
    cEff.SetLogy()
    hEffPrompt.Draw('same')
    hEffFD.Draw('same')
    leg.Draw()

    #_____________________________________________________________________________________
    # Save output
    outFileName = os.path.join(os.path.dirname(os.path.dirname(inputFile)),
                               'effs',
                               os.path.basename(inputFile).replace('proj', 'eff'))
    os.makedirs(os.path.dirname(outFileName), exist_ok=True)
    outFile = TFile(outFileName, 'recreate')
    hEffPrompt.Write()
    hEffFD.Write()
    hYieldPromptGen.Write()
    hYieldFDGen.Write()
    hYieldPromptReco.Write()
    hYieldFDReco.Write()
    outFile.Close()

    outFileNamePDF = outFileName.replace('.root', '.pdf')
    cEff.SaveAs(outFileNamePDF)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text",
                        default="config.yaml", help="configuration file")
    parser.add_argument('infileName', metavar='text', help='projection file')
    parser.add_argument("--batch", "-b", action="store_true",
                        help="batch mode")
    args = parser.parse_args()

    #_____________________________________________________________________________________
    # Read configuration file
    with open(args.config, 'r', encoding='utf8') as ymlconfig:
        config = yaml.load(ymlconfig, yaml.FullLoader)

    compute_eff(
            config=config,
            inputFile=args.infileName,
            batch=args.batch
        )