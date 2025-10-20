'''
Script to perform cut variation, outputs are saved in the same directory as the input efficiency files
'''
import argparse
import yaml
import sys
import os
import numpy as np # type: ignore
from itertools import product
from ROOT import TFile, TH1, TH1F, TH2F, TCanvas, TLegend, TLatex, gROOT, gStyle, gRandom  # pylint: disable=import-error,no-name-in-module
from ROOT import kBlack, kRed, kAzure, kGreen, kRainBow # pylint: disable=import-error,no-name-in-module
from ROOT import kFullCircle, kFullSquare, kOpenSquare, kOpenCircle # pylint: disable=import-error,no-name-in-module
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import logger
from frac_utils import GetMinimisation
from load_utils import load_root_files
from StyleFormatter import SetGlobalStyle, SetObjectStyle

TH1.AddDirectory(False)

def minimise_chi2(config, ptmins, ptmaxs, hRawYields, hEffPrompt, hEffFD, inputPath):

    hRawYieldsVsCut, hRawYieldsVsCutReSum, hRawYieldPromptVsCut, hRawYieldFDVsCut = [], [], [], []
    hEffPromptVsCut, hEffFDVsCut = [], []
    hPromptFracVsCut, hFDFracVsCut = [], []
    hCorrMatrixCutSets = []
    cFinalResPt = []

    hCorrYieldPrompt = hRawYields[0].Clone('hCorrYieldPrompt')
    hCorrYieldPrompt.SetTitle(';#it{p}_{T} (GeV/#it{c}); #it{N}_{prompt}')
    SetObjectStyle(hCorrYieldPrompt, color=kRed+1, fillcolor=kRed+1, markerstyle=kFullCircle)

    hCorrYieldFD = hRawYields[0].Clone('hCorrYieldFD')
    hCorrYieldFD.SetTitle(';#it{p}_{T} (GeV/#it{c}); #it{N}_{non-prompt}')
    SetObjectStyle(hCorrYieldFD, color=kAzure+4, fillcolor=kAzure+4, markerstyle=kFullSquare)

    hCovCorrYields = [[hRawYields[0].Clone('hCovPromptPrompt'), hRawYields[0].Clone('hCovPromptFD')],
                      [hRawYields[0].Clone('hCovFDPrompt'), hRawYields[0].Clone('hCovFDFD')]]

    for iRow, row in enumerate(hCovCorrYields):
        for iCol, hCov in enumerate(row):
            SetObjectStyle(hCov, linecolor=kBlack)
            rowName = '#it{N}_{prompt}' if iRow == 0 else '#it{N}_{non-prompt}'
            colName  = '#it{N}_{prompt}' if iCol == 0   else '#it{N}_{non-prompt}'
            hCov.SetTitle(f';#it{{p}}_{{T}} (GeV/#it{{c}}); #sigma({rowName}, {colName})')

    oCuts = []
    for iPt, (ptMin, ptMax) in enumerate(zip(ptmins, ptmaxs)):
        listRawYield, listRawYieldUnc, listEffPrompt,\
        listEffPromptUnc, listEffFD, listEffFDUnc = ([] for i in range(6))

        for iCut, (hRaw, hEffP, hEffF) in enumerate(zip(hRawYields, hEffPrompt, hEffFD)):
            #! due to there is filter on the file before, using skip_cuts to skip the duplicated cut sets
            # if skip_cuts is defined check if the cut number is present for that pt
            if 'skip_cuts' in config['cut_variation']['corr_bdt_cut']['skip_cuts']:
                if iCut in config['cut_variation']['corr_bdt_cut']['skip_cuts'][iPt]:
                    print(f'Skipping cut set {iCut} for pt {ptMin:.1f}-{ptMax:.1f}')
                    continue
            listRawYield.append(hRaw.GetBinContent(iPt+1))
            listRawYieldUnc.append(hRaw.GetBinError(iPt+1))
            listEffPrompt.append(hEffP.GetBinContent(iPt+1))
            listEffPromptUnc.append(hEffP.GetBinError(iPt+1))
            listEffFD.append(hEffF.GetBinContent(iPt+1))
            listEffFDUnc.append(hEffF.GetBinError(iPt+1))
            oCuts.append(iCut)

        nSets = len(listRawYield)
        print(f'Pt: {ptMin:.1f}-{ptMax:.1f}, iPt: {iPt+1}')
        for i in range(len(listEffPrompt)):
            print(f'({oCuts[i]}) Eff Prompt: {listEffPrompt[i]:.6f}    Eff FD: {listEffFD[i]:.6f}    Raw Yield: {listRawYield[i]:.2f}')

        corrYields, covMatrixCorrYields, chiSquare, matrices = \
            GetMinimisation(listEffPrompt, listEffFD, listRawYield, listEffPromptUnc, listEffFDUnc, listRawYieldUnc)

        hCorrYieldPrompt.SetBinContent(iPt+1, corrYields.item(0))
        hCorrYieldPrompt.SetBinError(iPt+1, np.sqrt(covMatrixCorrYields.item(0, 0)))
        hCorrYieldFD.SetBinContent(iPt+1, corrYields.item(1))
        hCorrYieldFD.SetBinError(iPt+1, np.sqrt(covMatrixCorrYields.item(1, 1)))
        for covElem in product(range(2), range(2)):
            hCovCorrYields[covElem[0]][covElem[1]].SetBinContent(iPt+1, covMatrixCorrYields.item(covElem))
            hCovCorrYields[covElem[0]][covElem[1]].SetBinError(iPt+1, 0.)

        ptString = f'pT{ptMin:.1f}_{ptMax:.1f}'
        commonString = f'{ptMin:.1f} < #it{{p}}_{{T}} < {ptMax:.1f}  GeV/#it{{c}};cut set'
        hRawYieldsVsCut.append(TH1F(f'hRawYieldsVsCutPt_{ptString}',         f'{commonString};raw yield',           nSets, 0.5, nSets + 0.5))
        hRawYieldsVsCutReSum.append(TH1F(f'hRawYieldsVsCutReSum_{ptString}', f'{commonString};raw yield',           nSets, 0.5, nSets + 0.5))
        hRawYieldPromptVsCut.append(TH1F(f'hRawYieldPromptVsCut_{ptString}', f'{commonString};raw yield',           nSets, 0.5, nSets + 0.5))
        hRawYieldFDVsCut.append(TH1F(f'hRawYieldFDVsCut_{ptString}',         f'{commonString};raw yield',           nSets, 0.5, nSets + 0.5))
        hEffPromptVsCut.append(TH1F(f'hEffPromptVsCut_{ptString}',           f'{commonString};efficiency',          nSets, 0.5, nSets + 0.5))
        hEffFDVsCut.append(TH1F(f'hEffFDVsCut_{ptString}',                   f'{commonString};efficiency',          nSets, 0.5, nSets + 0.5))
        hPromptFracVsCut.append(TH1F(f'hPromptFracVsCut_{ptString}',         f'{commonString};#it{{f}}_{{prompt}}', nSets, 0.5, nSets + 0.5))
        hFDFracVsCut.append(TH1F(f'hFDFracVsCut_{ptString}',                 f'{commonString};#it{{f}}_{{FD}}',     nSets, 0.5, nSets + 0.5))

        SetObjectStyle(hRawYieldsVsCut[iPt],      linecolor=kBlack,     markercolor=kBlack, markerstyle=kFullCircle)
        SetObjectStyle(hRawYieldsVsCutReSum[iPt], linecolor=kGreen+2)
        SetObjectStyle(hRawYieldPromptVsCut[iPt], color=kRed+1,         fillcolor=kRed+1,   markerstyle=kOpenCircle, fillalpha=0.3, fillstyle=3145)
        SetObjectStyle(hRawYieldFDVsCut[iPt],     color=kAzure+4,       fillcolor=kAzure+4, markerstyle=kOpenSquare, fillalpha=0.3, fillstyle=3154)
        SetObjectStyle(hEffPromptVsCut[iPt],      color=kRed+1,         markerstyle=kFullCircle)
        SetObjectStyle(hEffFDVsCut[iPt],          color=kAzure+4,       markerstyle=kFullSquare)
        SetObjectStyle(hPromptFracVsCut[iPt],     color=kRed+1,         markerstyle=kFullCircle)
        SetObjectStyle(hFDFracVsCut[iPt],         color=kAzure+4,       markerstyle=kFullSquare)

        hCorrMatrixCutSets.append(TH2F(f'hCorrMatrixCutSets_{ptString}', f'{commonString};cut set', 
                                       nSets, 0.5, nSets + 0.5, nSets, 0.5, nSets + 0.5))
        
        for mEl in product(range(nSets), range(nSets)):
            hCorrMatrixCutSets[iPt].SetBinContent(mEl[0]+1, mEl[1]+1, matrices['corrMatrix'].item(mEl[0], mEl[1]))

        for iCutSet, (rawY, effP, effF, rawYunc, effPunc, effFunc) in enumerate(zip(listRawYield, listEffPrompt, listEffFD,
                                                                                listRawYieldUnc, listEffPromptUnc,
                                                                                listEffFDUnc)):
            # efficiency
            hEffPromptVsCut[iPt].SetBinContent(iCutSet+1, effP)
            hEffPromptVsCut[iPt].SetBinError(iCutSet+1, effPunc)
            hEffFDVsCut[iPt].SetBinContent(iCutSet+1, effF)
            hEffFDVsCut[iPt].SetBinError(iCutSet+1, effFunc)

            # raw yields (including prompt and non-prompt raw yields)
            hRawYieldsVsCut[iPt].SetBinContent(iCutSet+1,       rawY)
            hRawYieldsVsCut[iPt].SetBinError(iCutSet+1,         rawYunc)
            hRawYieldPromptVsCut[iPt].SetBinContent(iCutSet+1,  corrYields.item(0) * effP)
            hRawYieldPromptVsCut[iPt].SetBinError(iCutSet+1,    np.sqrt(covMatrixCorrYields.item(0, 0)) * effP)
            hRawYieldFDVsCut[iPt].SetBinContent(iCutSet+1,      corrYields.item(1) * effF)
            hRawYieldFDVsCut[iPt].SetBinError(iCutSet+1,        np.sqrt(covMatrixCorrYields.item(1, 1)) * effF)
            hRawYieldsVsCutReSum[iPt].SetBinContent(iCutSet+1,  hRawYieldPromptVsCut[iPt].GetBinContent(iCutSet+1) +
                                                                hRawYieldFDVsCut[iPt].GetBinContent(iCutSet+1))

            # prompt fraction
            fPrompt    = effP * corrYields.item(0) / (effP * corrYields.item(0) + effF * corrYields.item(1))
            defPdeNP   = (effP * (effP * corrYields.item(0) + effF * corrYields.item(1)) - effP**2 * corrYields.item(0)) \
                       / (effP * corrYields.item(0) + effF * corrYields.item(1))**2
            defPdeNF   = - effF * effP * corrYields.item(0) / (effP * corrYields.item(0) + effF * corrYields.item(1))**2
            fPromptUnc = np.sqrt(defPdeNP**2 * covMatrixCorrYields.item(0, 0) +
                                defPdeNF**2 * covMatrixCorrYields.item(1, 1) +
                                2 * defPdeNP * defPdeNF * covMatrixCorrYields.item(1, 0))

            # feed-down fraction
            fFD      = effF * corrYields.item(1) / (effP * corrYields.item(0) + effF * corrYields.item(1))
            defFdeNF = (effF * (effF * corrYields.item(1) + effP * corrYields.item(0)) - effF**2 * corrYields.item(1)) \
                     / (effP * corrYields.item(0) + effF * corrYields.item(1))**2
            defFdeNP = - effF * effP * corrYields.item(1) / (effP * corrYields.item(0) + effF * corrYields.item(1))**2
            fFDUnc   = np.sqrt(defFdeNF**2 * covMatrixCorrYields.item(1, 1) +
                            defFdeNP**2 * covMatrixCorrYields.item(0, 0) +
                            2 * defFdeNF * defFdeNP * covMatrixCorrYields.item(1, 0))

            hPromptFracVsCut[iPt].SetBinContent(iCutSet+1, fPrompt)
            hPromptFracVsCut[iPt].SetBinError(iCutSet+1, fPromptUnc)
            hFDFracVsCut[iPt].SetBinContent(iCutSet+1, fFD)
            hFDFracVsCut[iPt].SetBinError(iCutSet+1, fFDUnc)

        if iPt == 0:
            legDistr = TLegend(0.45, 0.68, 0.75, 0.89)
            legDistr.SetFillStyle(0)
            legDistr.SetBorderSize(0)
            legDistr.SetTextSize(0.045)

            legEff = TLegend(0.2, 0.2, 0.4, 0.4)
            legEff.SetFillStyle(0)
            legEff.SetBorderSize(0)
            legEff.SetTextSize(0.045)

            legFrac = TLegend(0.2, 0.79, 0.4, 0.89)
            legFrac.SetFillStyle(0)
            legFrac.SetBorderSize(0)
            legFrac.SetTextSize(0.045)

            latInfo = TLatex()
            latInfo.SetNDC()
            latInfo.SetTextSize(0.045)
            latInfo.SetTextFont(42)
            latInfo.SetTextColor(1)

            legDistr.AddEntry(hRawYieldsVsCut[iPt],      'Measured raw yield',  'lpe')
            legDistr.AddEntry(hRawYieldPromptVsCut[iPt], 'Prompt',              'f')
            legDistr.AddEntry(hRawYieldFDVsCut[iPt],     'Non-prompt',          'f')
            legDistr.AddEntry(hRawYieldsVsCutReSum[iPt], 'Prompt + non-prompt', 'l')
            legEff.AddEntry(hEffPromptVsCut[iPt],        'Prompt',              'lpe')
            legEff.AddEntry(hEffFDVsCut[iPt],            'Non-prompt',          'lpe')
            legFrac.AddEntry(hPromptFracVsCut[iPt],      'Prompt',              'lpe')
            legFrac.AddEntry(hFDFracVsCut[iPt],          'Non-prompt',          'lpe')

        cFinalResPt.append(TCanvas(f'cFinalRes_{ptString}', '', 800, 800))
        cFinalResPt[-1].Divide(2, 2)
        # top-left pad: correlation
        cFinalResPt[-1].cd(1).SetRightMargin(0.14)
        hCorrMatrixCutSets[iPt].Draw('colz text')
        # top-right: template
        hFrameDistr = cFinalResPt[-1].cd(2).DrawFrame(0.5, 0., nSets + 0.5, hRawYieldsVsCut[iPt].GetMaximum() * 1.2,
                                    f'{commonString};raw yield')
        hFrameDistr.GetYaxis().SetDecimals()
        hRawYieldsVsCut[iPt].Draw('same')
        hRawYieldPromptVsCut[iPt].DrawCopy('histsame')
        hRawYieldFDVsCut[iPt].DrawCopy('histsame')
        hRawYieldsVsCutReSum[iPt].Draw('same')
        legDistr.Draw()
        #latInfo.DrawLatex(0.47, 0.65, f'#chi^{{2}} / ndf = {chiSquare:.3f}') # DO NOT TRUST CHI2 IN RUN 3
        # bottom-left: efficiency
        cFinalResPt[-1].cd(3).DrawFrame(0.5, hEffPromptVsCut[iPt].GetMinimum()/5, nSets + 0.5, 1., f'{commonString};efficiency')
        cFinalResPt[-1].cd(3).SetLogy()
        hEffPromptVsCut[iPt].DrawCopy('same')
        hEffFDVsCut[iPt].DrawCopy('same')
        legEff.Draw()
        # bottom-right: fraction
        cFinalResPt[-1].cd(4).DrawFrame(0.5, 0., nSets + 0.5, 1.8, f'{commonString};fraction')
        hPromptFracVsCut[iPt].DrawCopy('Esame')
        hFDFracVsCut[iPt].DrawCopy('Esame')
        legFrac.Draw()
        cFinalResPt[-1].Update()

    nPtBins = hCorrYieldPrompt.GetNbinsX()
    cCorrYield = TCanvas('cCorrYield', '', 800, 800)
    cCorrYield.DrawFrame(hCorrYieldPrompt.GetBinLowEdge(1), 1.,
                        hCorrYieldPrompt.GetBinLowEdge(nPtBins) + hCorrYieldPrompt.GetBinWidth(nPtBins),
                        hCorrYieldPrompt.GetMaximum() * 1.2, ';#it{p}_{T} (GeV/#it{c});corrected yield')
    cCorrYield.SetLogy()
    hCorrYieldPrompt.Draw('same')
    hCorrYieldFD.Draw('same')
    legEff.Draw()

    outDir = os.path.join(os.path.dirname(inputPath), 'cutVar')
    os.makedirs(outDir, exist_ok=True)
    outFileName = os.path.join(outDir, 'cutVar.root')
    outFile = TFile(outFileName, 'recreate')
    cCorrYield.Write()
    hCorrYieldPrompt.Write()
    hCorrYieldFD.Write()
    for covElem in product(range(2), range(2)):
        hCovCorrYields[covElem[0]][covElem[1]].Write()
    for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs)):
        logger(f'Saving cut variation results for pt {ptmin:.1f}-{ptmax:.1f} GeV/c')
        outFile.mkdir(f"pt{ptmin:.1f}_{ptmax:.1f}")
        outFile.cd(f"pt{ptmin:.1f}_{ptmax:.1f}")
        cFinalResPt[iPt].Write()
        hRawYieldsVsCut[iPt].Write()
        hRawYieldPromptVsCut[iPt].Write()
        hRawYieldFDVsCut[iPt].Write()
        hRawYieldsVsCutReSum[iPt].Write()
        hEffPromptVsCut[iPt].Write()
        hEffFDVsCut[iPt].Write()
        hPromptFracVsCut[iPt].Write()
        hFDFracVsCut[iPt].Write()
        hCorrMatrixCutSets[iPt].Write()
    outFile.Close()

    for iPt in range(len(ptmins)):
        if iPt == 0:
            cFinalResPt[iPt].SaveAs(f'{outDir}/FinalResPt.pdf[')
        cFinalResPt[iPt].SaveAs(f'{outDir}/FinalResPt.pdf')
        if iPt == hRawYields[0].GetNbinsX() - 1:
            cFinalResPt[iPt].SaveAs(f'{outDir}/FinalResPt.pdf]')
        cFinalResPt[iPt].SaveAs(f'{outDir}/FinalResPt_pt{ptmins[iPt]}_{ptmaxs[iPt]}.png')
    

def compute_frac_cut_var(config_flow, inputPathRy, inputPathEff, batch=False):

    gROOT.SetBatch(batch)
    gStyle.SetPaintTextFormat("4.2f")

    # load configuration
    with open(config_flow, 'r') as ymlCfgFile:
        config = yaml.safe_load(ymlCfgFile)
        ptmins = config['ptbins'][:-1]
        ptmaxs = config['ptbins'][1:]

    rawYieldFiles = load_root_files(inputPathRy, 'raw_yields_')
    effFiles = load_root_files(inputPathEff, 'eff_')
    
    if len(effFiles) != len(rawYieldFiles):
        logger(f'Number of efficiency files ({len(effFiles)}) does not match number of raw yield files ({len(rawYieldFiles)}).', level='ERROR')
        raise ValueError(f'Number of efficiency files ({len(effFiles)}) does not match number of raw yield files ({len(rawYieldFiles)}).')
    
    effFiles.sort()
    rawYieldFiles.sort()

    hRawYields, hEffPrompt, hEffFD = [], [], []

    # load inputs raw yields and efficiencies
    for inFileNameRawYield, inFileNameEff in zip(rawYieldFiles, effFiles):

        inFileRawYield = TFile.Open(inFileNameRawYield)
        # if hRawYieldsSimFit not in file, skip it
        if not inFileRawYield.GetListOfKeys().Contains('hRawYieldsSimFit'):
            logger(f'File {inFileNameRawYield} does not contain hRawYieldsSimFit, skipping.', level='WARNING')
            continue
        hRawYields.append(inFileRawYield.Get('hRawYieldsSimFit'))
        
        inFileEff = TFile.Open(inFileNameEff)
        hEffPrompt.append(inFileEff.Get('hEffPrompt'))
        hEffFD.append(inFileEff.Get('hEffFD'))

    SetGlobalStyle(padleftmargin=0.15, padtopmargin=0.08, titleoffsetx=1.,
                titleoffsety=1.4, opttitle=1, palette=kRainBow, maxdigits=2)

    legDistr = TLegend(0.45, 0.68, 0.75, 0.89)
    legDistr.SetFillStyle(0)
    legDistr.SetBorderSize(0)
    legDistr.SetTextSize(0.045)

    legEff = TLegend(0.2, 0.2, 0.4, 0.4)
    legEff.SetFillStyle(0)
    legEff.SetBorderSize(0)
    legEff.SetTextSize(0.045)

    legFrac = TLegend(0.2, 0.79, 0.4, 0.89)
    legFrac.SetFillStyle(0)
    legFrac.SetBorderSize(0)
    legFrac.SetTextSize(0.045)

    latInfo = TLatex()
    latInfo.SetNDC()
    latInfo.SetTextSize(0.045)
    latInfo.SetTextFont(42)
    latInfo.SetTextColor(1)

    minimise_chi2(config, ptmins, ptmaxs, hRawYields, hEffPrompt, hEffFD, inputPathEff)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("config_flow", metavar="text",
                        default="config_flow.yaml", help="flow configuration file")
    parser.add_argument("infilePathRy", metavar="text",
                        default="input", help="input path to the raw yields")
    parser.add_argument("infilePathEff", metavar="text",
                        default="input", help="input path to the efficiencies")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="run in batch mode")
    args = parser.parse_args()

    compute_frac_cut_var(args.config_flow, args.infilePathRy, args.infilePathEff,
                         batch=args.batch)