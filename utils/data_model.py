import sys
from alive_progress import alive_bar
import os
import uproot
import pandas as pd
from ROOT import TFile, TTree, THnSparse # pyright: ignore # type: ignore
sys.path.append("./")
from utils import logger
import ROOT

def print_entries(data, stage_desc=""):
    if isinstance(data, ROOT.THnSparse):
        n_entries = data.GetEntries()
        print(f"{stage_desc} - THnSparse entries: {n_entries}\n")
    elif isinstance(data, ROOT.TTree):
        n_entries = data.GetEntries()
        print(f"{stage_desc} - TTree entries: {n_entries}\n")
    elif isinstance(data, pd.core.frame.DataFrame):
        n_entries = len(data)
        print(f"{stage_desc} - DataFrame entries: {n_entries}\n")
    else:
        print(f"{stage_desc} - Unknown data type: {type(data)}\n")

def apply_selection(data, data_model_dict, var, min_val, max_val):
    """
    Apply selection on dataframe column
    """

    print(f"Applying selection on {var} in range {min_val} - {max_val}")

    if isinstance(data, THnSparse):
        logger(f"Applying sparse selection on {var} in range {min_val} - {max_val}", level="INFO")
        try:
            data.GetAxis(data_model_dict[var]).SetRangeUser(min_val, max_val)
            return data.Clone()
        except Exception as e:
            logger(f"Error applying sparse selection on {var}: {e}", level="WARNING")
            return data

    elif isinstance(data, pd.core.frame.DataFrame):
        logger(f"Applying dataframe selection on {data_model_dict[var]} in range {min_val} - {max_val}", level="INFO")
        try:
            print(f"Applying query: {data_model_dict[var]} >= {min_val} and {data_model_dict[var]} < {max_val}")
            data = data.query(f"{data_model_dict[var]} >= {min_val} and {data_model_dict[var]} < {max_val}")
            return data
        except Exception as e:
            logger(f"Error applying dataframe selection on {var}: {e}", level="WARNING")
            return data

    elif isinstance(data, uproot.models.TTree.Model_TTree_v20):
        logger(f"Applying TTree selection on {var} in range {min_val} - {max_val}", level="INFO")
        df = data.arrays(library="pd")
        df = df.query(f"{data_model_dict[var]} >= {min_val} and {data_model_dict[var]} < {max_val}")
        return df

def get_tree(input_file, tables, query_signal=None):
    """
    Helper function to get correlated backgrounds tree from file
    """

    print(f"Opening file {input_file} to get correlated backgrounds trees {tables}")
    dfs_list = [[] for _ in range(len(tables))]
    with uproot.open(input_file) as f:
        for key in f.keys():
            for i_table, table in enumerate(tables):
                if table in key:
                    dfs_list[i_table].append(f[key].arrays(library="pd"))

    merged_single_dfs = []
    for df in dfs_list:
        merged_single_dfs.append(pd.concat([single_df for single_df in df], ignore_index=True))
    full_df = pd.concat(merged_single_dfs, axis=0)

    if query_signal:
        print(f"Applying query to select signal: {query_signal}")
        full_df = full_df.query(query_signal)

    print(f"Full tree with {len(full_df)} entries, columns\n: {full_df.columns.to_list()}")
    return full_df

def get_histo(input_file, sparse_dicts, cfg, pt_bin_fit_cfg):
    """
    Helper function to get histogram from file
    """

    if cfg['data_type'] == "DplusTask":
        if cfg["is_data"]:
            sparse = input_file.Get("hf-task-dplus/hSparseMass")
            dict_entry = "Data"
        else:
            sparse = input_file.Get("hf-task-dplus/hSparseMassPrompt")
            dict_entry = "RecoPrompt"
        
        # Apply selections
        print(f"\n\nsparse_dicts: {sparse_dicts}\n\n")
        sparse.GetAxis(sparse_dicts[dict_entry]['Pt']).SetRangeUser(pt_bin_fit_cfg['pt_range'][0], pt_bin_fit_cfg['pt_range'][1])
        if pt_bin_fit_cfg.get('score_bkg_max') and cfg.get('correlated_bkgs'):
            if cfg['correlated_bkgs'].get('apply_ml_score_sel'):
                sparse.GetAxis(sparse_dicts[dict_entry]['score_bkg']).SetRangeUser(0, pt_bin_fit_cfg['score_bkg_max'])
        histo = sparse.Projection(0)

    return histo

def get_data_model_dicts(config, data_type="DplusTask"):
    
    axes_dict = {}
    if data_type == "DplusTask":
        ### Data
        axes_dict['Data'] = {
            'Mass': 0,
            'Pt': 1,
            'score_bkg': 2,
            'score_FD': 4,
            'cent': 5
        }

        ### MC
        axes_dict['RecoPrompt'] = {
            'Mass': 0,
            'Pt': 1,
            'score_bkg': 2,
            'score_prompt': 3,
            'score_FD': 4,
            'cent': 5,
            'occ': 6,
        }
        axes_dict['RecoFD'] = {
            'Mass': 0,
            'Pt': 1,
            'score_bkg': 2,
            'score_prompt': 3,
            'score_FD': 4,
            'cent': 5,
            'occ': 6,
            'pt_bmoth': 7,
            'flag_bhad': 8,
        }
        axes_dict['GenPrompt'] = {
            'Pt': 0,
            'y': 1,
            'cent': 2,
            'occ': 3
        }
        axes_dict['GenFD'] = {
            'Pt': 0,
            'y': 1,
            'cent': 2,
            'occ': 3,
            'pt_bmoth': 4,
            'flag_bhad': 5,
        }

    elif data_type == "DplusCorrelator":
        axes_dict["Data"] = {
            'Mass': 'fMD',
            'Pt': 'fPtD',
            'score_bkg': 'fMlScoreBkg',
            'score_prompt': 'fMlScorePrompt',
            'score_FD': 'fMlScoreNonPrompt'
        }

    elif data_type == "DplusTree":
        print(f"Getting data model dicts for data_type: {data_type}")
        axes_dict["Data"] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality',
            'score_bkg': 'fMlScore0',
            'score_prompt': 'fMlScorePrompt',
            'score_FD': 'fMlScore1'
        }

        ### MC
        axes_dict['RecoPrompt'] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality',
            'score_bkg': 'fMlScore0',
            'score_prompt': 'fMlScorePrompt',
            'score_FD': 'fMlScore1'
        }
        axes_dict['RecoFD'] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality',
            'score_bkg': 'fMlScore0',
            'score_prompt': 'fMlScorePrompt',
            'score_FD': 'fMlScore1'
        }
        axes_dict['GenPrompt'] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality'
        }
        axes_dict['GenFD'] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality',
        }
        
    elif data_type == "PreprocessedTree":
        axes_dict["Data"] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality',
            'score_bkg': 'fMlScore0',
            'score_prompt': 'fMlScorePrompt',
            'score_FD': 'fMlScore1'
        }
        ### MC
        axes_dict['RecoPrompt'] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality',
            'score_bkg': 'fMlScore0',
            'score_prompt': 'fMlScorePrompt',
            'score_FD': 'fMlScore1'
        }
        axes_dict['RecoFD'] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality',
            'score_bkg': 'fMlScore0',
            'score_prompt': 'fMlScorePrompt',
            'score_FD': 'fMlScore1'
        }
        axes_dict['GenPrompt'] = {
            'Pt': 'Pt',
            'cent': 'Centrality'
        }
        axes_dict['GenFD'] = {
            'Pt': 'Pt',
            'cent': 'Centrality'
        }
    else:
        logger(f"Data model for data_type {data_type} not implemented yet.", level='FATAL')

    print(f"\n\nData model dicts for data_type {data_type}: {axes_dict}")
    return axes_dict

def get_pt_preprocessed_inputs(config, iPt, is_sparse_data_type):

    logger("Loading preprocessed inputs", level='INFO')

    if is_sparse_data_type:
        inputsData, inputsReco, inputsGen, axes_dict = get_pt_preprocessed_sparses(config, iPt)
    else:
        inputsData, inputsReco, inputsGen, axes_dict = get_pt_preprocessed_trees(config, iPt)

    return inputsData, inputsReco, inputsGen, axes_dict

def get_pt_preprocessed_trees(config, iPt):

    inputsData, inputsReco, inputsGen, col_dict = {}, {}, {}, {}

    pre_cfg = config['preprocess']
    col_dict = get_data_model_dicts(config, data_type="PreprocessedTree")
    ptmin = config["ptbins"][iPt]
    ptmax = config["ptbins"][iPt+1]

    infileprep = f"{config['outdir']}/preprocess/AO2D_pt_{int(ptmin*10)}_{int(ptmax*10)}.root"

    if config["operations"].get("proj_data"):
        for key, _ in pre_cfg["data"].items():
            with uproot.open(infileprep) as f:
                # Convert to pandas dataframe
                inputsData[f'_{key}'] = f.get(f'Data__{key}/treeMass').arrays(library="pd")
                # inputsData[f'_{key}'] = f.get(f'Data__{key}/treeMass')

    if config["operations"].get("proj_mc"):
        for key, _ in col_dict.items():
            if "Reco" not in key:
                continue
            with uproot.open(infileprep) as f:
                inputsReco[key] = f.get(f'MC/Reco/{key}/treeMass')

        for key, _ in col_dict.items():
            if "Gen" not in key:
                continue
            with uproot.open(infileprep) as f:
                inputsGen[key] = f.get(f'MC/Gen/{key}/treeMass')

    return inputsData, inputsReco, inputsGen, col_dict

def get_pt_preprocessed_sparses(config, iPt):

    inputsData, inputsReco, inputsGen, axes_dict = {}, {}, {}, {}

    pre_cfg = config['preprocess']
    axes_dict['Data'] = {ax: iax for iax, ax in enumerate(pre_cfg['axes_data'].keys())}
    ptmin = config["ptbins"][iPt]
    ptmax = config["ptbins"][iPt+1]

    if config.get("outdirPrep") and config["outdirPrep"] != "":
        infileprep = TFile(f"{config['outdirPrep']}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root")
    else:
        infileprep = TFile(f"{config['outdir']}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root")

    if config["operations"].get("proj_data"):
        for key, _ in pre_cfg["data"].items():
            # print(f"infileprep.ls(): {infileprep.ls()}")
            inputsData[f'_{key}'] = infileprep.Get(f'Data__{key}/hSparseMass')

    if config["operations"].get("proj_mc"):
        subdir = infileprep.Get("MC/Reco")
        for key in subdir.GetListOfKeys():
            obj = key.ReadObj()
            inputsReco[key.GetName()[1:]] = obj
            axes_dict[key.GetName()[1:]] = {ax: iax for iax, ax in enumerate(pre_cfg['axes_reco'].keys())}

        subdir = infileprep.Get("MC/Gen")
        for key in subdir.GetListOfKeys():
            obj = key.ReadObj()
            inputsGen[key.GetName()[1:]] = obj
            axes_dict[key.GetName()[1:]] = {ax: iax for iax, ax in enumerate(pre_cfg['axes_gen'].keys())}

    infileprep.Close()

    return inputsData, inputsReco, inputsGen, axes_dict

def get_inputs(config, get_data=True, get_mc=True, debug=False):
    """Load the inputs and axes infos

    Args:
        config (dict): the  config dictionary
        get_data (bool, optional): load data inputs. Defaults to True.
        get_mc (bool, optional): load mc inputs. Defaults to True.
        debug (bool, optional): print debug info. Defaults to False.

    Outputs:
        inputs: thnSparse in the  task
        inputsReco: thnSparse of reco level from the D meson task
        inputsGen: thnSparse of gen level from the D meson task
        axes_dict (dict): dictionary of the axes for each sparse
    """

    inputsData, inputsReco, inputsGen = {}, {}, {}
    cfg_prep = config['preprocess']
    is_tree_input = config['input_type'] == "Tree"
    axes_dict = get_data_model_dicts(config, data_type=cfg_prep['data_model'])
    pre_cfg = config['preprocess'] if config.get('preprocess') else config
    print("About to load inputs...")
    if get_data:
        logger(f"\t\t[Data] Loading data from: {pre_cfg['data']}")
        infile = []
        for name, dataset in pre_cfg['data'].items():
            print(f"Loading data for dataset {name} from {dataset['files']}")
            # Collect all files starting with AnalysisResults_ and ending with .root in the dataset['files'] string
            if isinstance(dataset["files"], str) and not dataset["files"].endswith(".root"):
                print("string but not root")
                list_of_files = [f for f in os.listdir(dataset["files"]) if f.endswith(".root")]
                infiledata = [TFile(os.path.join(dataset["files"], file)) for file in list_of_files]
            elif isinstance(dataset["files"], list):
                print("list")
                if len(dataset["files"]) == 1:
                    print("single file")
                    if dataset["files"][0].endswith(".root"):
                        print("single root file")
                        list_of_files = [dataset["files"][0]]
                        infiledata = [TFile(dataset["files"][0])]
                    else:
                        print("single directory")
                        list_of_files = [f for f in os.listdir(dataset["files"][0]) if f.endswith(".root")]
                        infiledata = [TFile(os.path.join(dataset["files"][0], file)) for file in list_of_files]
                elif len(dataset["files"]) > 1:
                    print("multiple files")
                    if all(file.endswith(".root") for file in dataset["files"]):
                        print("all root files")
                        print(f"\n\nLoading multiple root files for dataset {name}: {dataset['files']}")
                        list_of_files = dataset['files']
                        print(f"\n\nLoading multiple root files for dataset {name}: {list_of_files}\n\n")
                        # quit()
                        infiledata = [TFile(file) for file in dataset["files"]]
                    else:
                        print("not all root files")
                        logger("The dataset contains multiple files, but not all of them are root files. Provide a single root file or a list of root files or a directory containing all root files.", level='ERROR')
            else:
                print("single root file")
                infiledata = [TFile(dataset["files"])] if isinstance(dataset["files"], str) else [TFile(file) for file in dataset["files"]]
            print(f"Loading data sparse for {name} from {infiledata}")
            inputsData[f'_{name}'] = []
            with alive_bar(len(infiledata), title=f"[INFO]\t\t[Data] Loading data inputs for {name}") as bar:
                for infile, infilename in zip(infiledata, list_of_files):
                    if config['input_type'] == "Sparse":
                        inputsData[f'_{name}'].append(infile.Get('hf-task-dplus/hSparseMass'))
                    elif config['input_type'] == "Tree":
                        inputsData[f'_{name}'].append(build_tree_data_frame(infilename, config["preprocess"]["tables"], config["preprocess"]["cols_to_keep_data"], query_signal=None))
                    else:
                        logger(f"Unknown input type {config['input_type']}. Supported types are 'Sparse' and 'Tree'.", level='FATAL')
                    bar()
            [infile.Close() for infile in infiledata]
        print(f"Loaded inputsData: {inputsData}\n\n")
    if get_mc:
        print(f"Loading mc from: {pre_cfg['mc']}")
        infiletask = [TFile(pre_cfg['mc'])] if isinstance(pre_cfg['mc'], str) else [TFile(pre_cfg['mc']) for file in pre_cfg['mc']]
        infiletasknames = [pre_cfg['mc']] if isinstance(pre_cfg['mc'], str) else pre_cfg['mc']

        if config['input_type'] == "Sparse":
            inputsReco['RecoFD']     = [file.Get('hf-task-dplus/hSparseMassFD') for file in infiletask]
            print(f"Loaded inputsReco: {inputsReco['RecoFD']}\n\n")
            inputsReco['RecoPrompt'] = [file.Get('hf-task-dplus/hSparseMassPrompt') for file in infiletask]
            print(f"Loaded inputsReco: {inputsReco['RecoPrompt']}\n\n")
            inputsGen['GenPrompt']   = [file.Get('hf-task-dplus/hSparseMassGenPrompt') for file in infiletask]
            print(f"Loaded inputsGen: {inputsGen['GenPrompt']}\n\n")
            inputsGen['GenFD']       = [file.Get('hf-task-dplus/hSparseMassGenFD') for file in infiletask]
            print(f"Loaded inputsGen: {inputsGen['GenFD']}\n\n")
        elif config['input_type'] == "Tree":
            inputsReco['RecoFD']     = [build_tree_data_frame(file, cfg_prep["tables_mc_reco"], cfg_prep["cols_to_keep_mc_reco"], "fOriginMcRec == 1 and abs(fFlagMcMatchRec) == 1") for file in infiletasknames]
            print(f"Loaded inputsReco: {inputsReco['RecoFD']}\n\n")
            inputsReco['RecoPrompt'] = [build_tree_data_frame(file, cfg_prep["tables_mc_reco"], cfg_prep["cols_to_keep_mc_reco"], "fOriginMcRec == 2 and abs(fFlagMcMatchRec) == 1") for file in infiletasknames]
            print(f"Loaded inputsReco: {inputsReco['RecoPrompt']}\n\n")
            inputsGen['GenPrompt']   = [build_tree_data_frame(file, cfg_prep["tables_mc_gen"], cfg_prep["cols_to_keep_mc_gen"], "fOriginMcGen == 1 and abs(fFlagMcMatchGen) == 1") for file in infiletasknames]
            print(f"Loaded inputsGen: {inputsGen['GenPrompt']}\n\n")
            inputsGen['GenFD']       = [build_tree_data_frame(file, cfg_prep["tables_mc_gen"], cfg_prep["cols_to_keep_mc_gen"], "fOriginMcGen == 2 and abs(fFlagMcMatchGen) == 1") for file in infiletasknames]
            print(f"Loaded inputsGen: {inputsGen['GenFD']}\n\n")
        else:
            logger(f"Unknown input type {config['input_type']}. Supported types are 'Sparse' and 'Tree'.", level='FATAL')

        [infile.Close() for infile in infiletask]

    logger("inputs loaded", level='INFO')
    if debug:
        print('\n')
        print('###############################################################')
        for key, value in axes_dict.items():
            logger(f"{key}:", level='DEBUG')
            for sub_key, sub_value in value.items():
                logger(f"    {sub_key}: {sub_value}", level='DEBUG')
        print('###############################################################')
        print('\n')
        print("Inputs loaded:\n")
        print(f"  Data inputs: {inputsData}\n\n")
        print(f"  Reco inputs: {inputsReco}\n\n")
        print(f"  Gen inputs: {inputsGen}\n\n")

    return inputsData, inputsReco, inputsGen, axes_dict


def build_tree_data_frame(input_file, tree_names, cols_to_keep_data, query_signal=None):

    print(f"Building data frame from file {input_file} for trees {tree_names}")

    dfs = [[] for _ in range(len(tree_names))]
    with uproot.open(input_file) as f:
        for iKey, key in enumerate(f.keys()):
            for i_tree, (tree, col_to_keep) in enumerate(zip(tree_names, cols_to_keep_data)):
                if tree in key:
                    df = f[key].arrays(library='pd')
                    dfs[i_tree].append(df)

    concatenated_dfs = []
    for df in dfs:
        concatenated_dfs.append(pd.concat([single_df for single_df in df], ignore_index=True))

    merged_df = pd.concat(concatenated_dfs, axis=1)
    if query_signal is not None:
        print(f"Applying query to select signal: {query_signal}, initial entries: {len(merged_df)}")
        merged_df = merged_df.query(query_signal)
        print(f"Entries after query: {len(merged_df)}")
        print(f"cols_to_keep_data: {cols_to_keep_data}")
        merged_df = merged_df[cols_to_keep_data]

    return merged_df
