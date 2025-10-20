'''
python script for the computation of the fractions of prompt and feed-down D for all cutset
the output is saved in the input efficiency directory
run: python ComputeDataDrivenFraction.py --inputdir path/to/input --outputdir path/to/output --suffix text
'''

import argparse
import yaml
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import logger
from frac_utils import GetPromptFDFractionCutSet  # pylint: disable=import-error,no-name-in-module # pyignore # type: ignore
from load_utils import load_cutVar_histos, load_eff_histos, load_root_files
from ROOT import TFile, TH1, TCanvas, TLegend, gROOT, kRed, kBlue  # pylint: disable=import-error,no-name-in-module # pyignore # type: ignore
from StyleFormatter import SetGlobalStyle # pylint: disable=import-error,no-name-in-module # pyignore # type: ignore

TH1.AddDirectory(False)

def data_driven_frac(outputDir, iFile, hEffPrompt, hEffFD, \
                     hPromptFrac, hFDFrac, hPromptFracCorr, hFDFracCorr, \
                     hCorrYieldPrompt, hCorrYieldFD, hCovPromptPrompt, hCovPromptFD, hCovFDFD):
    for iPt in range(hEffPrompt.GetNbinsX()):
        ptMin = hEffPrompt.GetBinLowEdge(iPt+1)
        ptMax = ptMin+hEffPrompt.GetBinWidth(iPt+1)
        ptCent = hEffPrompt.GetBinCenter(iPt+1)
        effAccPrompt = hEffPrompt.GetBinContent(iPt+1)
        effAccFD = hEffFD.GetBinContent(iPt+1)
        effAccPromptUnc = hEffPrompt.GetBinError(iPt+1)
        effAccFDUnc = hEffFD.GetBinError(iPt+1)

        corrYieldPrompt = hCorrYieldPrompt.GetBinContent(iPt+1)
        corrYieldFD = hCorrYieldFD.GetBinContent(iPt+1)
        covPromptPrompt = hCovPromptPrompt.GetBinContent(iPt+1)
        covPromptFD = hCovPromptFD.GetBinContent(iPt+1)
        covFDFD = hCovFDFD.GetBinContent(iPt+1)

        fracPromptFD, uncFracPromptFD = GetPromptFDFractionCutSet(effAccPrompt, effAccFD, corrYieldPrompt, corrYieldFD,
                                                                  covPromptPrompt, covFDFD, covPromptFD)

        fracPromptFDcorr, uncFracPromptFDcorr = GetPromptFDFractionCutSet(1., 1., corrYieldPrompt, corrYieldFD,
                                                                          covPromptPrompt, covFDFD, covPromptFD)

        hPromptFrac.SetBinContent(iPt+1, fracPromptFD[0])
        hPromptFrac.SetBinError(iPt+1, uncFracPromptFD[0])
        hFDFrac.SetBinContent(iPt+1, fracPromptFD[1])
        hFDFrac.SetBinError(iPt+1, uncFracPromptFD[1])
        hPromptFracCorr.SetBinContent(iPt+1, fracPromptFDcorr[0])
        hPromptFracCorr.SetBinError(iPt+1, uncFracPromptFDcorr[0])
        hFDFracCorr.SetBinContent(iPt+1, fracPromptFDcorr[1])
        hFDFracCorr.SetBinError(iPt+1, uncFracPromptFDcorr[1])

    SetGlobalStyle(padleftmargin=0.18, padbottommargin=0.14)

    legFrac = TLegend(0.2, 0.84, 0.4, 0.94)
    legFrac.SetBorderSize(0)
    legFrac.SetFillStyle(0)
    legFrac.SetTextSize(0.045)
    legFrac.AddEntry(hPromptFrac, 'Prompt', 'p')
    legFrac.AddEntry(hFDFrac, 'Non-prompt', 'p')

    legEff = legFrac.Clone('legEff')
    legEff.SetY1(0.2)
    legEff.SetY2(0.4)

    ptMin = hPromptFrac.GetBinLowEdge(1)
    cFrac = TCanvas('cFrac', '', 800, 800)
    cFrac.DrawFrame(ptMin, 0., ptMax, 1.2, ';#it{p}_{T} (GeV/#it{c}); fraction')
    hPromptFrac.SetLineColor(kRed)
    hPromptFrac.Draw('same')
    hFDFrac.SetLineColor(kBlue)
    hFDFrac.Draw('same')
    legFrac.Draw()
    cFrac.Update()

    cFracCorrFrac = TCanvas('cFracCorrFrac', '', 800, 800)
    cFracCorrFrac.DrawFrame(ptMin, 0., ptMax, 1.2, ';#it{p}_{T} (GeV/#it{c}); corrected fraction')
    hPromptFracCorr.SetLineColor(kRed)
    hPromptFracCorr.Draw('same')
    hFDFracCorr.SetLineColor(kBlue)
    hFDFracCorr.Draw('same')
    legFrac.Draw()
    cFracCorrFrac.Update()

    cEff = TCanvas('cEff', '', 800, 800)
    cEff.DrawFrame(ptMin, 1.e-4, ptMax, 1., ';#it{p}_{T} (GeV/#it{c}); (Acc#times#font[152]{e})')
    cEff.SetLogy()
    hEffPrompt.SetLineColor(kRed)
    hEffPrompt.Draw('same')
    hEffFD.SetLineColor(kBlue)
    hEffFD.Draw('same')
    legEff.Draw()
    cEff.Update()

    outFile = TFile(os.path.join(outputDir, f'frac_{iFile:02}.root'), 'recreate')
    hEffPrompt.Write()
    hEffFD.Write()
    hPromptFrac.Write()
    hFDFrac.Write()
    hPromptFracCorr.Write()
    hFDFracCorr.Write()
    cFrac.Write()
    cFracCorrFrac.Write()
    cEff.Write()
    outFile.Close()

def main_data_driven_frac(cutVarFile, effPath, batch=False):

    if batch:
        gROOT.SetBatch()
    
    hEffPrompts, hEffFDs, hPromptFracs, hFDFracs, hPromptFracCorrs, hFDFracCorrs = load_eff_histos(load_root_files(effPath, 'eff_'))
    hCorrYieldPrompt, hCorrYieldFD, hCovPromptPrompt, hCovPromptFD, hCovFDFD = load_cutVar_histos(cutVarFile)
    outputDir = os.path.join(os.path.dirname(effPath), 'frac')
    logger(f'Output directory: {outputDir}', level='INFO')
    os.makedirs(outputDir, exist_ok=True)
    for iFile in range(len(hEffPrompts)):
        data_driven_frac(
            outputDir,
            iFile,
            hEffPrompts[iFile],
            hEffFDs[iFile],
            hPromptFracs[iFile],
            hFDFracs[iFile],
            hPromptFracCorrs[iFile],
            hFDFracCorrs[iFile],
            hCorrYieldPrompt,
            hCorrYieldFD,
            hCovPromptPrompt,
            hCovPromptFD,
            hCovFDFD
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("cutVarFile", metavar="text",
                        help="path to the cut variation file")
    parser.add_argument("effPath", metavar="text",
                        help="path to the efficiency file")
    parser.add_argument("--batch", '-b', action='store_true',
                        help="run in batch mode")
    args = parser.parse_args()

    main_data_driven_frac(
        cutVarFile=args.cutVarFile,
        effPath=args.effPath,
        batch=args.batch
    )
