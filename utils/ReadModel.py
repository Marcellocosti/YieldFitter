
'''
This file incorporates code from DmesonAnalysis (GPL-3.0).
Original source: https://github.com/DmesonAnalysers/DmesonAnalysis
Modifications made by https://github.com/flowHF/hf-vn.
'''
'''
Script with helper functions to load model predictions from txt files
'''

import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline

def InterpolateModel(ptCent, yCent, yMin=None, yMax=None):
    '''
    Helper function to interpolate model predictions.
    The returned splines will raise an error if applied out of the data boundary.

    Parameters
    -----------
    ptCent: list of pT centres to interpolate
    yCent: list of central values to interpolate
    yMin: list of min values to interpolate
    yMax: list of max values to interpolate

    Returns:
    -----------
    splinesAll: dictionary with splines {yCent, yMin, yMax}
    ptMin: minimum pt for which the interpolation is valid
    ptMax: maximum pt for which the interpolation is valid
    '''

    splinesAll = {}
    splinesAll['yCent'] = InterpolatedUnivariateSpline(ptCent, yCent, ext='raise', check_finite=True)

    if yMin is not None and yMin.any():
        splinesAll['yMin'] = InterpolatedUnivariateSpline(ptCent, yMin, ext='raise', check_finite=True)
    if yMax is not None and yMax.any():
        splinesAll['yMax'] = InterpolatedUnivariateSpline(ptCent, yMax, ext='raise', check_finite=True)

    return splinesAll, min(ptCent), max(ptCent)


def ReadFONLL(fileNameFONLL, isPtDiff=False, Dmeson='Dzero'):
    '''
    Helper function to read FONLL txt or root files
    
    input:
    ----------
        - fileNameFONLL: FONLL file name
        - isPtDiff: flag to tell the function if the model is in pt differential form
        - Dmeson: D meson type (Dzero, Dplus, B)

    Returns:
    -----------
        - splineFONLL: interpolated values of the FONLL
        - dfFONLL: pandas dataframe with original values
        - ptMin: minimum pt for which the model is valid
        - ptMax: maximum pt for which the model is valid
    '''
    if fileNameFONLL.endswith('.txt'):
        # the header number is starting from 0
        dfFONLL = pd.read_csv(fileNameFONLL, sep=" ", header=0).astype('float64')
        if not isPtDiff:
            dfFONLL['ptcent'] = (dfFONLL['ptmin']+dfFONLL['ptmax']) / 2
            dfFONLL['central_ptdiff'] = dfFONLL['central'] / (dfFONLL['ptmax']-dfFONLL['ptmin'])
            dfFONLL['min_ptdiff'] = dfFONLL['min'] / (dfFONLL['ptmax']-dfFONLL['ptmin'])
            dfFONLL['max_ptdiff'] = dfFONLL['max'] / (dfFONLL['ptmax']-dfFONLL['ptmin'])
        else:
            dfFONLL.rename(columns={'pt': 'ptcent'}, inplace=True)
            dfFONLL.rename(columns={'central': 'central_ptdiff'}, inplace=True)
            dfFONLL.rename(columns={'min': 'min_ptdiff'}, inplace=True)
            dfFONLL.rename(columns={'max': 'max_ptdiff'}, inplace=True)
        
    elif fileNameFONLL.endswith('.root'):
        from ROOT import TFile
        fonllpred = {}
        ptcent, central_ptdiff, min_ptdiff, max_ptdiff = [], [], [], []
        fonllfile = TFile.Open(fileNameFONLL)
        for key in ['Central', 'Min', 'Max']:
            # to be corrected
            if Dmeson != 'B':
                DmesonPrompt = Dmeson
                if Dmeson == 'Dplus':
                    DmesonPrompt = 'Dzero'
                elif Dmeson == 'Dzero':
                    DmesonPrompt = 'Dplus'
                fonllpred[key] = fonllfile.Get(f'Prompt/{DmesonPrompt}/hFonllPrompt{DmesonPrompt}{key}')
                fonllpred[key].SetName(f"fonll_prompt_{key}")
                fonllpred[key].SetDirectory(0)
            else:
                fonllpred[key] = fonllfile.Get(f'hFonllBhadron{key}')
                fonllpred[key].SetName(f"fonll_{key}")
                fonllpred[key].SetDirectory(0)
        fonllfile.Close()

        nPtBins = fonllpred['Central'].GetNbinsX()
        for iBin in range(1,nPtBins+1):
            ptcent.append(fonllpred['Central'].GetBinCenter(iBin))
            central_ptdiff.append(fonllpred['Central'].GetBinContent(iBin))
            min_ptdiff.append(fonllpred['Min'].GetBinContent(iBin))
            max_ptdiff.append(fonllpred['Max'].GetBinContent(iBin))

        dfFONLL = pd.DataFrame({'ptcent': ptcent, 'central_ptdiff': central_ptdiff, 'min_ptdiff': min_ptdiff, 'max_ptdiff': max_ptdiff})

    splineFONLL, ptMin, ptMax = InterpolateModel(dfFONLL['ptcent'], dfFONLL['central_ptdiff'],
                                                dfFONLL['min_ptdiff'], dfFONLL['max_ptdiff'])

    return splineFONLL, dfFONLL, ptMin, ptMax

def ReadTAMU(fileNameTAMU):
    '''
    Helper function to read TAMU txt files

    Parameters
    -----------
    fileNameTAMU: TAMU file name

    Returns:
    -----------
    splineTAMU: dictionary with splines {yCent, yMin, yMax}
    dfTAMU: pandas dataframe with original values
    ptMin: minimum pt for which the model is valid
    ptMax: maximum pt for which the model is valid
    '''
    dfTAMU = pd.read_csv(fileNameTAMU, sep=' ', comment='#')
    if 'R_AA_max' in dfTAMU and 'R_AA_min' in dfTAMU:
        dfTAMU['R_AA'] = (dfTAMU['R_AA_min'] + dfTAMU['R_AA_max']) / 2 #central value taken as average of min and max
        splineTAMU, ptMin, ptMax = InterpolateModel(dfTAMU['PtCent'], dfTAMU['R_AA'],
                                                    dfTAMU['R_AA_min'], dfTAMU['R_AA_max'])
    else:
        splineTAMU, ptMin, ptMax = InterpolateModel(dfTAMU['PtCent'], dfTAMU['R_AA'])

    return splineTAMU, dfTAMU, ptMin, ptMax