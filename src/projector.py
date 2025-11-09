'''
Script to project the MC distributions and apply the pt weights from the AnRes.root of Dtask
python3 projector.py config_flow.yml --cutsetConfig config_cutset.yml [-c --correlated]
If the last argument is not provided, the script will project the combined cutsets.
'''
import ROOT
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import yaml
import argparse
import sys
import os
import glob
from pathlib import Path
from functools import partial
from ROOT import TFile, TObject
from alive_progress import alive_bar
from scipy.interpolate import make_interp_spline
import uproot
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../utils")
from data_model import get_pt_preprocessed_inputs, apply_selection, print_entries
from utils import reweight_histo_1D, reweight_histo_2D, reweight_histo_3D, make_dir_root_file, logger

ROOT.TH1.AddDirectory(False)

import yaml
from ROOT import TFile

def proj_multitrial_sparse(config, multitrial_folder):

    pt_bin_label = Path(multitrial_folder).name
    logger(f"Processing multitrial projections for pt bin {pt_bin_label} ...", level='INFO')

    # Load default cutsets
    default_cutsets = [f"{config['outdir']}/cutvar_{config['suffix']}_combined/cutsets/{f}" for f in os.listdir(f"{config['outdir']}/cutvar_{config['suffix']}_combined/cutsets") if f.endswith('.yml')]
    # Load Mass and MassSp histos from the default cases
    default_histos = {}
    for default_cutset in default_cutsets:
        suffix = os.path.basename(default_cutset).replace(".yml", "").replace("cutset_", "")
        default_proj = TFile.Open(default_cutset.replace(".yml", ".root").replace("cutset", "proj"), "READ")
        default_histos[suffix] = {}
        default_histos[suffix]['Mass'] = default_proj.Get(f"{pt_bin_label}/hMassData")
        default_proj.Close()

    def process_cutset(multitrial_dir, default_histos):
        trial_number = Path(multitrial_dir).name.replace("trial_", "")
        try:
            with open(f"{multitrial_dir}/config_trial_{trial_number}.yml", 'r') as ymlCutSetFile:
                config_trial = yaml.safe_load(ymlCutSetFile)
        except Exception as e:
            logger(f"Error opening or reading config file for trial {trial_number}: {e}", level='ERROR')
            return
        
        multitrial_cutsets = glob.glob(f"{multitrial_dir}/cutsets/*.yml")
        for multitrial_cutset in multitrial_cutsets:
            suffix = os.path.basename(multitrial_cutset).replace(".yml", "").replace("cutset_", "")
            output_dir = os.path.dirname(multitrial_cutset).replace('cutset', 'proj')
            os.makedirs(output_dir, exist_ok=True)
            output_path = multitrial_cutset.replace('.yml', '.root').replace('cutset', 'proj')
            output_file = TFile.Open(output_path, "RECREATE")
            output_file.mkdir(pt_bin_label)
            output_file.cd(pt_bin_label)
            default_histos[suffix]['Mass'].Write("hMassData")
            output_file.Close()

    # Parallel execution
    multitrial_dirs = [f for f in glob.glob(f"{multitrial_folder}/trial_*/")]
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(partial(process_cutset, default_histos=default_histos), multitrial_dirs)

def proj_data_sparse(data_dict, axes, proj_scores, writeopt):

    proj_vars = ['Mass', 'score_FD', 'score_bkg'] if proj_scores else ['Mass']
    proj_axes = [axes['Data'][var] for var in proj_vars]

    for var, ax in zip(proj_vars, proj_axes):
        for isparse, (_, sparse) in enumerate(data_dict.items()):
            hist_var_temp = sparse.Projection(ax)
            hist_var_temp.SetName(f'h{var.capitalize()}_{isparse}')
            if isparse == 0:
                hist_var = hist_var_temp.Clone(f'h{var.capitalize()}')
                hist_var.Reset()

            hist_var.Add(hist_var_temp)

        hist_var.Write(f'h{var.capitalize()}Data', writeopt)

def proj_data_tree(out_file_path, cols_dict, data_frames_dict, proj_scores, pt_min, pt_max, writeopt='recreate'):
    # Columns to keep
    branches_to_keep = ['Mass', 'score_FD', 'score_bkg'] if proj_scores else ['Mass']
    print(f"Branches to keep: {branches_to_keep}")
    print(f"Columns dict: {cols_dict}")
    cols_to_keep = [cols_dict['Data'][branch] for branch in branches_to_keep]

    for key, dataset in data_frames_dict.items():
        # Convert uproot TTree → pandas DataFrame with only selected branches
        # df = dataset.arrays(cols_to_keep, library="pd")

        with getattr(uproot, writeopt)(out_file_path) as f:
            f[f"pt_{int(pt_min*10)}_{int(pt_max*10)}/treeMassData"] = dataset

# Recreate must be generalized
def proj_mc_reco_tree(inputsReco, pt_min, pt_max, out_file_path, writeopt='recreate'):

    for i_dataset, (key, dataset) in enumerate(inputsReco.items()):
        # Convert uproot TTree → pandas DataFrame with only selected branches
        # df = dataset.arrays(cols_to_keep, library="pd")

        print(f"dataset.head(): {dataset.head()}")

        dataset_mass = dataset['fM']
        with getattr(uproot, writeopt)(out_file_path) as f:
            f[f"pt_{int(pt_min*10)}_{int(pt_max*10)}/treeMass{key}"] = dataset_mass.to_frame()

        if i_dataset == 0:
            writeopt = 'update'  # after the first dataset, switch to update mode

        dataset_pt = dataset['fPt']
        with getattr(uproot, writeopt)(out_file_path) as f:
            f[f"pt_{int(pt_min*10)}_{int(pt_max*10)}/treePt{key}"] = dataset_pt.to_frame()

    # for key, sparse in sparsesReco.items():
    #     if key != 'RecoPrompt' and key != 'RecoFD':
    #         sparse.Projection(axes[key]['Mass']).Write(f'h{key}Mass')
    #         sparse.Projection(axes[key]['Pt']).Write(f'h{key}Pt')

    # hMassPrompt = sparsesReco['RecoPrompt'].Projection(axes['RecoPrompt']['Mass'])
    # hMassPrompt.SetName(f'hPromptMass_{ptMin}_{ptMax}')
    # hMassFD = sparsesReco['RecoFD'].Projection(axes['RecoFD']['Mass'])
    # hMassFD.SetName(f'hFDMass_{ptMin}_{ptMax}')

    # ### project pt prompt
    # hPtPrompt = sparsesReco['RecoPrompt'].Projection(axes['RecoPrompt']['Pt'])
    # if sPtWeightsD:
    #     hPtPrompt = reweight_histo_1D(hPtPrompt, sPtWeightsD, binned=False)

    # ### project pt FD
    # if sPtWeightsD:
    #     hPtFD = reweight_histo_1D(sparsesReco['RecoFD'].Projection(axes['RecoFD']['Pt']), sPtWeightsD, binned=False)
    # elif sPtWeightsB:
    #     if Bspeciesweights:
    #         hPtFD = reweight_histo_3D(
    #             sparsesReco['RecoFD'].Projection(axes['RecoFD']['Pt'], axes['RecoFD']['pt_bmoth'], axes['RecoFD']['flag_bhad']), 
    #             sPtWeightsB, Bspeciesweights
    #         )
    #     else:
    #         hPtFD = reweight_histo_2D(
    #             sparsesReco['RecoFD'].Projection(axes['RecoFD']['pt_bmoth'], axes['RecoFD']['Pt']),          # 2D projection: Projection(ydim, xdim)
    #             sPtWeightsB, binned=False
    #         )
    # elif Bspeciesweights:
    #     hPtFD = reweight_histo_2D(
    #         sparsesReco['RecoFD'].Projection(axes['RecoFD']['flag_bhad'], axes['RecoFD']['Pt']),             # 2D projection: Projection(ydim, xdim)
    #         Bspeciesweights, binned=True
    #     )
    # else:
    #     hPtFD = sparsesReco['RecoFD'].Projection(axes['RecoFD']['Pt'])

    # ## write the output 
    # hMassPrompt.Write('hPromptMass', writeopt)
    # hMassFD.Write('hFDMass', writeopt)
    # hPtPrompt.Write('hPromptPt', writeopt)
    # hPtFD.Write('hFDPt', writeopt)

def proj_mc_gen_tree(sparsesGen, writeopt):

    print("proj_mc_gen_tree not implemented yet")
    # for key, sparse in sparsesGen.items():
    #     if key != 'GenPrompt' and key != 'GenFD':
    #         sparse.Projection(axes[key]['Pt']).Write(f'h{key}Pt')

    # ### prompt
    # hGenPtPrompt = sparsesGen['GenPrompt'].Projection(axes['GenPrompt']['Pt'])
    # if sPtWeightsD:
    #     hGenPtPrompt = reweight_histo_1D(hGenPtPrompt, sPtWeightsD, binned=False)

    # ### FD
    # if sPtWeightsD:
    #     hGenPtFD = reweight_histo_1D(sparsesGen['GenFD'].Projection(axes['GenFD']['Pt']), sPtWeightsD, binned=False)
    # elif sPtWeightsB:
    #     if Bspeciesweights:
    #         hGenPtFD = reweight_histo_3D(
    #             sparsesGen['GenFD'].Projection(axes['GenFD']['Pt'], axes['GenFD']['pt_bmoth'], axes['GenFD']['flag_bhad']),
    #             sPtWeightsB, Bspeciesweights
    #         )
    #     else:
    #         hGenPtFD = reweight_histo_2D(
    #             sparsesGen['GenFD'].Projection(axes['GenFD']['pt_bmoth'], axes['GenFD']['Pt']),         # 2D projection: Projection(ydim, xdim)
    #             sPtWeightsB, binned=False
    #         )
    # elif Bspeciesweights:
    #     hGenPtFD = reweight_histo_2D(
    #         sparsesGen['GenFD'].Projection(axes['GenFD']['flag_bhad'], axes['GenFD']['Pt']),            # 2D projection: Projection(ydim, xdim)
    #         Bspeciesweights, binned=True
    #     )
    # else:
    #     hGenPtFD = sparsesGen['GenFD'].Projection(axes['GenFD']['Pt'])

    # ## write the output
    # hGenPtPrompt.Write('hPromptGenPt', writeopt)
    # hGenPtFD.Write('hFDGenPt', writeopt)

def proj_mc_reco_sparse(sparsesReco, sPtWeightsD, sPtWeightsB, Bspeciesweights, writeopt):

    for key, sparse in sparsesReco.items():
        if key != 'RecoPrompt' and key != 'RecoFD':
            sparse.Projection(axes[key]['Mass']).Write(f'h{key}Mass')
            sparse.Projection(axes[key]['Pt']).Write(f'h{key}Pt')

    hMassPrompt = sparsesReco['RecoPrompt'].Projection(axes['RecoPrompt']['Mass'])
    hMassPrompt.SetName(f'hPromptMass_{ptMin}_{ptMax}')
    hMassFD = sparsesReco['RecoFD'].Projection(axes['RecoFD']['Mass'])
    hMassFD.SetName(f'hFDMass_{ptMin}_{ptMax}')

    ### project pt prompt
    hPtPrompt = sparsesReco['RecoPrompt'].Projection(axes['RecoPrompt']['Pt'])
    if sPtWeightsD:
        hPtPrompt = reweight_histo_1D(hPtPrompt, sPtWeightsD, binned=False)

    ### project pt FD
    if sPtWeightsD:
        hPtFD = reweight_histo_1D(sparsesReco['RecoFD'].Projection(axes['RecoFD']['Pt']), sPtWeightsD, binned=False)
    elif sPtWeightsB:
        if Bspeciesweights:
            hPtFD = reweight_histo_3D(
                sparsesReco['RecoFD'].Projection(axes['RecoFD']['Pt'], axes['RecoFD']['pt_bmoth'], axes['RecoFD']['flag_bhad']), 
                sPtWeightsB, Bspeciesweights
            )
        else:
            hPtFD = reweight_histo_2D(
                sparsesReco['RecoFD'].Projection(axes['RecoFD']['pt_bmoth'], axes['RecoFD']['Pt']),          # 2D projection: Projection(ydim, xdim)
                sPtWeightsB, binned=False
            )
    elif Bspeciesweights:
        hPtFD = reweight_histo_2D(
            sparsesReco['RecoFD'].Projection(axes['RecoFD']['flag_bhad'], axes['RecoFD']['Pt']),             # 2D projection: Projection(ydim, xdim)
            Bspeciesweights, binned=True
        )
    else:
        hPtFD = sparsesReco['RecoFD'].Projection(axes['RecoFD']['Pt'])

    ## write the output 
    hMassPrompt.Write('hPromptMass', writeopt)
    hMassFD.Write('hFDMass', writeopt)
    hPtPrompt.Write('hPromptPt', writeopt)
    hPtFD.Write('hFDPt', writeopt)

def proj_mc_gen_sparse(sparsesGen, sPtWeightsD, sPtWeightsB, Bspeciesweights, writeopt):

    for key, sparse in sparsesGen.items():
        if key != 'GenPrompt' and key != 'GenFD':
            sparse.Projection(axes[key]['Pt']).Write(f'h{key}Pt')

    ### prompt
    hGenPtPrompt = sparsesGen['GenPrompt'].Projection(axes['GenPrompt']['Pt'])
    if sPtWeightsD:
        hGenPtPrompt = reweight_histo_1D(hGenPtPrompt, sPtWeightsD, binned=False)

    ### FD
    if sPtWeightsD:
        hGenPtFD = reweight_histo_1D(sparsesGen['GenFD'].Projection(axes['GenFD']['Pt']), sPtWeightsD, binned=False)
    elif sPtWeightsB:
        if Bspeciesweights:
            hGenPtFD = reweight_histo_3D(
                sparsesGen['GenFD'].Projection(axes['GenFD']['Pt'], axes['GenFD']['pt_bmoth'], axes['GenFD']['flag_bhad']),
                sPtWeightsB, Bspeciesweights
            )
        else:
            hGenPtFD = reweight_histo_2D(
                sparsesGen['GenFD'].Projection(axes['GenFD']['pt_bmoth'], axes['GenFD']['Pt']),         # 2D projection: Projection(ydim, xdim)
                sPtWeightsB, binned=False
            )
    elif Bspeciesweights:
        hGenPtFD = reweight_histo_2D(
            sparsesGen['GenFD'].Projection(axes['GenFD']['flag_bhad'], axes['GenFD']['Pt']),            # 2D projection: Projection(ydim, xdim)
            Bspeciesweights, binned=True
        )
    else:
        hGenPtFD = sparsesGen['GenFD'].Projection(axes['GenFD']['Pt'])

    ## write the output
    hGenPtPrompt.Write('hPromptGenPt', writeopt)
    hGenPtFD.Write('hFDGenPt', writeopt)

def get_pt_weights(cfgProj):
    """Get pt weights and return weights flags with spline

    Args:
        cfgProj (dict): Configuration dictionary for projections

    Outputs:
        sPtWeights (spline): Spline for ptWeights interpolation
        sPtWeightsB (spline): Spline for ptWeightsB weights interpolation
        Bspeciesweights (str): B species weights # TODO
    """

    # REVIEW: the ptWeights inputed is a list, but the ptWeights outputed is a TH1D object
    # and actually ptweights is used as a flag
        # compute info for pt weights
    if not cfgProj.get('PtWeightsFile'):
        logger('No pt weights for D and B mesons provided in the config file!', level='WARNING')
        return None, None, None
        
    ptWeightsFile = TFile.Open(cfgProj["PtWeightsFile"], 'r')

    if cfgProj.get('ApplyPtWeightsD'):
        hPtWeightsD = ptWeightsFile.Get('hPtWeightsFONLLtimesTAMUDcent')
        ptBinCentersD = [ (hPtWeightsD.GetBinLowEdge(i)+hPtWeightsD.GetBinLowEdge(i+1))/2 for i in range(1, hPtWeightsD.GetNbinsX()+1)]
        ptBinContentsD = [hPtWeightsD.GetBinContent(i) for i in range(1, hPtWeightsD.GetNbinsX()+1)]
        sPtWeights = make_interp_spline(ptBinCentersD, ptBinContentsD)
    else:
        logger('pt weights for D mesons will not be provided!', level='WARNING')
        sPtWeights = None

    if cfgProj.get('ApplyPtWeightsB'):
        hPtWeightsB = ptWeightsFile.Get('hPtWeightsFONLLtimesTAMUBcent')
        ptBinCentersB = [ (hPtWeightsB.GetBinLowEdge(i)+hPtWeightsB.GetBinLowEdge(i+1))/2 for i in range(1, hPtWeightsB.GetNbinsX()+1)]
        ptBinContentsB = [hPtWeightsB.GetBinContent(i) for i in range(1, hPtWeightsB.GetNbinsX()+1)]
        sPtWeightsB = make_interp_spline(ptBinCentersB, ptBinContentsB)
    else:
        logger('pt weights for B mesons will not be provided!', level='WARNING')
        sPtWeightsB = None

    if cfgProj.get('ApplyBSpeciesWeights'):
        Bspeciesweights = config['Bspeciesweights']
    else:
        logger('B species weights will not be provided!', level='WARNING')
        Bspeciesweights = None
    
    return sPtWeights, sPtWeightsB, Bspeciesweights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text",
                        default="config.yaml", help="flow configuration file")
    parser.add_argument('--cutsetConfig', "-cc", metavar='text', type=str, nargs='?',
                        const=None, default='cutsetConfig.yaml',
                        help='Optional cutset configuration file (default: cutsetConfig.yaml)')
    parser.add_argument("--multitrial_folder", "-multfolder", metavar="text",
                        default="", help="Produce projection files for multitrial systematics")
    parser.add_argument("--outputDir", "-o", metavar="text",
                        default="", help="output directory, used only for directly running the script")
    args = parser.parse_args()

    with open(args.config, 'r') as ymlCfgFile:
        config = yaml.load(ymlCfgFile, yaml.FullLoader)
    operations = config["operations"]

    if args.multitrial_folder != "":
        logger(f"Running multitrial projections with config: {args.config}", level='INFO')
        proj_multitrial_sparse(config, args.multitrial_folder)
        sys.exit(0)

    with open(args.cutsetConfig, 'r') as ymlCutSetFile:
        print(f"Opening cutset config file: {args.cutsetConfig}")
        print(f"ymlCutSetFile: {ymlCutSetFile}")
        cutSetCfg = yaml.load(ymlCutSetFile, yaml.FullLoader)
        if cutSetCfg['icutset'] != 'central':
            iCut = f"{int(cutSetCfg['icutset']):02d}"
            outDir = os.path.join(os.path.dirname(os.path.dirname(args.cutsetConfig)), 'projs') if args.outputDir == "" else args.outputDir
            outfilePath = os.path.join(outDir, f"proj_{iCut}.root")
            os.makedirs(outDir, exist_ok=True)
        else:
            outfilePath = args.cutsetConfig.replace('.yml', '.root').replace('cutset', 'proj')
            outDir = os.path.dirname(outfilePath)
            os.makedirs(outDir, exist_ok=True)
    print(f"Output projection file: {outfilePath}")
    is_sparse_data_type = True if config["input_type"] == "Sparse" else False
    if is_sparse_data_type:
        logger(f"Input data is sparse", level='INFO')
        print(f"operations: {operations}\n\n")
        if operations["do_proj_data"] or operations["do_proj_mc"]:
            if os.path.exists(outfilePath):
                logger(f"Found previous projection file {outfilePath}, will update it", level='INFO')
                outfile = TFile.Open(outfilePath, 'UPDATE')
            else:
                logger(f"No previous projection file found, will create a new one at {outfilePath}", level='INFO')
                outfile = TFile(outfilePath, 'RECREATE')
        else:
            sys.exit(0)
    else:
        write_opt_data_tree, write_opt_mc_tree = 'recreate', 'update'  # default write options for tree output
        if not operations["do_proj_data"] and operations["do_proj_mc"]:
            write_opt_mc_tree = 'recreate'
    print(f"Starting projections for cutset {cutSetCfg['icutset']} ...")
    write_opt_data = TObject.kOverwrite if operations.get("do_proj_data") else 0 
    write_opt_mc = TObject.kOverwrite if operations.get("do_proj_mc") else 0 
    print(f"write_opt_data: {write_opt_data}, write_opt_mc: {write_opt_mc}")
    # compute info for pt weights
    if operations.get("do_proj_mc"):
        sPtWeightsD, sPtWeightsB, Bspeciesweights = get_pt_weights(config["projections"]) if config['projections'].get('PtWeightsFile') else (None, None, None)

    ptMin = cutSetCfg['pt_min']
    ptMax = cutSetCfg['pt_max']
    bkg_min = cutSetCfg['score_bkg_min']
    bkg_max = cutSetCfg['score_bkg_max']
    fd_min = cutSetCfg['score_FD_min']
    fd_max = cutSetCfg['score_FD_max']
    print(f"Applying cuts: bkg [{bkg_min}, {bkg_max}], FD [{fd_min}, {fd_max}]")
    # Cut on centrality and pt on data applied in the preprocessing
    print(f'Projecting distributions for {ptMin:.1f} < pT < {ptMax:.1f} GeV/c')
    pt_label = f"pt_{int(ptMin*10)}_{int(ptMax*10)}"
    inputsData, inputsReco, inputsGen, axes = get_pt_preprocessed_inputs(config, pt_label, is_sparse_data_type)
    if is_sparse_data_type:
        make_dir_root_file(pt_label, outfile)
        outfile.cd(pt_label)

    if operations.get("do_proj_data"):
        print(f"inputsData: {inputsData}")
        for key, _ in inputsData.items():
            print_entries(inputsData[key], "Entries before selection")
            print(f"type inputsData[key]: {type(inputsData[key])}")
            inputsData[key] = apply_selection(inputsData[key], axes['Data'], "score_bkg", bkg_min, bkg_max)
            print_entries(inputsData[key], "Entries after selection bkg")
            inputsData[key] = apply_selection(inputsData[key], axes['Data'], "score_FD", fd_min, fd_max)
            print_entries(inputsData[key], "Entries after selection FD and bkg")
        if is_sparse_data_type:
            proj_data_sparse(inputsData, axes, config["projections"].get('storeML'), write_opt_data)
        else:
            proj_data_tree(outfilePath, axes, inputsData, config["projections"].get('storeML'), ptMin, ptMax, write_opt_data_tree)
        print(f"Projected data!")

    if operations.get("do_proj_mc"):
        for key, _ in inputsReco.items():
            inputsReco[key] = apply_selection(inputsReco[key], axes[key], 'score_bkg', bkg_min, bkg_max)
            inputsReco[key] = apply_selection(inputsReco[key], axes[key], 'score_FD', fd_min, fd_max)

        if is_sparse_data_type:
            proj_mc_reco_sparse(inputsReco, sPtWeightsD, sPtWeightsB, Bspeciesweights, write_opt_mc)
        else:
            proj_mc_reco_tree(inputsReco, ptMin, ptMax, outfilePath, write_opt_mc_tree)
        print("Projected mc reco!")
        if is_sparse_data_type:
            proj_mc_gen_sparse(inputsGen, sPtWeightsD, sPtWeightsB, Bspeciesweights, write_opt_mc)
        else:
            # proj_mc_gen_tree(inputsGen, write_opt_mc)
            logger("MC Gen projection for Tree input type not implemented yet", level='WARNING')
        print("Projected mc gen!")

    print('\n\n')

    if is_sparse_data_type:
        outfile.Close()
    print(f"\n\nClosing output file: {outfilePath}")
    print("Done!")