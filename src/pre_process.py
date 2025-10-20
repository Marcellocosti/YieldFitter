'''
This script is used to pre-process a/multi large AnRes.root for the BDT training:
    - split the input by pT
    - obtain the sigma from prompt enhance sample
python3 pre_process.py config_pre.yml AnRes_1.root AnRes_2.root --pre --sigma  
'''
import os
import sys
import yaml
import numpy as np
import array
import ROOT
from ROOT import TFile, TObject
import argparse
import gc
import itertools
import uproot
import pandas as pd
from alive_progress import alive_bar
import concurrent.futures
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{script_dir}/")
sys.path.append(f"{script_dir}/../utils/")
from utils import get_centrality_bins, make_dir_root_file, logger
from data_model import get_data_model_dicts, get_inputs, apply_selection, build_tree_data_frame, print_entries

def check_existing_outputs(ptmin, ptmax, outputDir, stage):

    outFilePath = f'{outputDir}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root'

    if os.path.exists(outFilePath):
        print(f"    [{stage}] Updating file: {outFilePath}")
        outFile = TFile(outFilePath, 'update')
        write_opt = TObject.kOverwrite
    else:
        print(f"    [{stage}] Creating file: {outFilePath}")
        outFile = TFile.Open(outFilePath, 'recreate')
        write_opt = 0 # Standard

    return outFile, write_opt

def process_pt_bin_data_sparses(config, ptmin, ptmax, centmin, centmax, bkg_max_cut, debugPreprocessFile, outputDir, data_inputs, data_model_vars):
    logger(f'[Data] Processing pT bin {ptmin} - {ptmax}, cent {centmin}-{centmax}')

    # Force recreate of the output file when data are reprocessed, other operations are lightweight
    outFilePath = f'{outputDir}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root'
    print(f"    [Data] Creating file: {outFilePath}")
    if config["input_type"] == "Sparse":
        outFile = TFile.Open(outFilePath, 'recreate')
        for key, dataset_inputs in data_inputs.items():
            make_dir_root_file(f'Data_{key}', outFile)

    axes_data = list(config['preprocess']["axes_data"].keys())
    proj_axes = [data_model_vars[axtokeep] for axtokeep in axes_data]
    rebin_data = list(config['preprocess']["axes_data"].values())

    for key, dataset_inputs in data_inputs.items():
        with alive_bar(len(dataset_inputs), title=f'[INFO] \t\t[Data] Processing {key}', bar='smooth') as bar:
            for i_input, input_data in enumerate(dataset_inputs):
                print_entries(input_data, "[Data] Before cuts")
                dataset_inputs[i_input] = apply_selection(input_data, data_model_vars, 'Pt', ptmin, ptmax)
                print_entries(input_data, "[Data] After pt cut")
                dataset_inputs[i_input] = apply_selection(input_data, data_model_vars, 'score_bkg', 0, bkg_max_cut)
                print_entries(input_data, "[Data] After pt cut and bkg cut")

                proj_sparse = input_data.Projection(len(proj_axes), array.array('i', proj_axes), 'O')
                proj_sparse.SetName(input_data.GetName())
                proj_sparse = proj_sparse.Rebin(array.array('i', rebin_data))
                print(f"proj_sparse.GetEntries() = {proj_sparse.GetEntries()}", flush=True)

                if i_input == 0:
                    merged_data_pt = proj_sparse.Clone()
                    proj_sparse.Delete()  # Delete the original projection to save memory
                    del proj_sparse
                    gc.collect()
                    make_dir_root_file(f'pt_{int(ptmin*10)}_{int(ptmax*10)}/{key}', debugPreprocessFile)
                    logger(f'\t[Data] Writing sparse for {key} with {merged_data_pt.GetNdimensions()} dimensions')
                    debugPreprocessFile.cd(f'pt_{int(ptmin*10)}_{int(ptmax*10)}/{key}')
                    for iDim in range(merged_data_pt.GetNdimensions()):
                        print(f"Writing axis {iDim} named {axes_data[iDim]} with {merged_data_pt.Projection(iDim).GetEntries()} entries", flush=True)
                        merged_data_pt.Projection(iDim).Write(axes_data[iDim], TObject.kOverwrite)
                else:
                    merged_data_pt.Add(proj_sparse)
                    proj_sparse.Delete()  # Delete the original projection to save memory
                    del proj_sparse
                    gc.collect()
                print(f"\t\t[Data] After adding sparse {i_input}, merged_data_pt.GetEntries() = {merged_data_pt.GetEntries()}", flush=True)
                
                bar()

        logger(f'\t[Data] Writing data for {key} with {print_entries(merged_data_pt)}')
        outFile.cd(f'Data_{key}')
        merged_data_pt.Write('hSparseMass', TObject.kOverwrite)
        merged_data_pt.Delete()
        del merged_data_pt

        print("Data written to file!")
        gc.collect()

    outFile.Close()

    logger(f'[Data] Finished processing pT bin {ptmin} - {ptmax}\n\n')

def process_pt_bin_data_trees(config, ptmin, ptmax, centmin, centmax, bkg_max_cut, debugPreprocessFile, outputDir, data_inputs, data_model_vars):
    logger(f'[Data] Processing pT bin {ptmin} - {ptmax}, cent {centmin}-{centmax}')

    # Force recreate of the output file when data are reprocessed, other operations are lightweight
    outFilePath = f'{outputDir}/preprocess/AO2D_pt_{int(ptmin*10)}_{int(ptmax*10)}.root'

    for i_dataset, (key, dataset_inputs) in enumerate(data_inputs.items()):
        with alive_bar(len(dataset_inputs), title=f'[INFO] \t\t[Data] Processing {key}', bar='smooth') as bar:
            for i_input in range(len(dataset_inputs)):
                print_entries(dataset_inputs[i_input], "[Data] Before cuts")
                dataset_inputs[i_input] = apply_selection(dataset_inputs[i_input], data_model_vars, 'Pt', ptmin, ptmax)
                print_entries(dataset_inputs[i_input], "[Data] After pt cut")
                dataset_inputs[i_input] = apply_selection(dataset_inputs[i_input], data_model_vars, 'score_bkg', 0, bkg_max_cut)
                print_entries(dataset_inputs[i_input], "[Data] After pt cut and bkg cut")

                if i_input == 0:
                    merged_data_pt = dataset_inputs[i_input]
                else:
                    merged_data_pt = merged_data_pt.Concatenate(dataset_inputs[i_input])
                bar()

        # Convert dataframe to tree and write
        if i_dataset == 0:
            print(f"File {outFilePath} does not exist, creating it.")
            mode = "recreate"
        else:
            print(f"File {outFilePath} exists, updating it.")
            mode = "update"

        with getattr(uproot, mode)(outFilePath) as f:
            print(f"Writing Data_{key}/treeMass with {len(merged_data_pt)} entries")
            f[f"Data_{key}/treeMass"] = merged_data_pt

        print("Data written to file!")
        gc.collect()

    logger(f'[Data] Finished processing pT bin {ptmin} - {ptmax}\n\n')

def process_pt_bin_mc_trees(config, ptmin, ptmax, centmin, centmax, bkg_max_cut, debugPreprocessFile, outputDir, reco_inputs, gen_inputs, data_model_vars):
    print(f'[MC] Processing pT bin {ptmin} - {ptmax}, cent {centmin}-{centmax}')
    # outFilePath = f'{outputDir}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root'
    # if os.path.exists(outFilePath):
    #     print(f"    [MC] Updating file: {outFilePath}")
    #     outFile = TFile(outFilePath, 'update')
    #     write_opt = TObject.kOverwrite
    # else:
    #     print(f"    [MC] Creating file: {outFilePath}")
    #     outFile = TFile.Open(outFilePath, 'recreate')
    #     write_opt = 0 # Standard

    outFilePath = f'{outputDir}/preprocess/AO2D_pt_{int(ptmin*10)}_{int(ptmax*10)}.root'

    # cut on pt and bkg on all the reco trees
    print(f"reco_inputs: {list(reco_inputs)}", flush=True)
    for i_dataset, (key, mc_input) in enumerate(reco_inputs.items()):
        with alive_bar(len(mc_input), title=f'[INFO] \t\t[MC Reco] Processing {key}', bar='smooth') as bar:
            for i_input in range(len(mc_input)):
                print_entries(mc_input[i_input], "[MC Reco] Before cuts")
                mc_input[i_input] = apply_selection(mc_input[i_input], data_model_vars[key], 'Pt', ptmin, ptmax)
                print_entries(mc_input[i_input], "[MC Reco] After pt cut")
                mc_input[i_input] = apply_selection(mc_input[i_input], data_model_vars[key], 'score_bkg', 0, bkg_max_cut)
                print_entries(mc_input[i_input], "[MC Reco] After pt cut and bkg cut")

                if i_input == 0:
                    merged_data_pt = mc_input[i_input]
                else:
                    merged_data_pt = merged_data_pt.Concatenate(mc_input[i_input])
                bar()

        # Convert dataframe to tree and write
        if i_dataset == 0:
            print(f"File {outFilePath} does not exist, creating it.")
            mode = "recreate"
        else:
            print(f"File {outFilePath} exists, updating it.")
            mode = "update"

        with getattr(uproot, mode)(outFilePath) as f:
            f[f"MC/Reco/{key}/treeMass"] = merged_data_pt

        print("MC Reco written to file!")
        del merged_data_pt
        gc.collect()

    for i_dataset, (key, gen_input) in enumerate(gen_inputs.items()):
        with alive_bar(len(gen_input), title=f'[INFO] \t\t[MC Gen] Processing {key}', bar='smooth') as bar:
            for i_input in range(len(gen_input)):
                print_entries(gen_input[i_input], "[MC Gen] Before cuts")
                gen_input[i_input] = apply_selection(gen_input[i_input], data_model_vars[key], 'Pt', ptmin, ptmax)
                print_entries(gen_input[i_input], "[MC Gen] After pt cut")

                if i_input == 0:
                    merged_data_pt = gen_input[i_input]
                else:
                    merged_data_pt = merged_data_pt.Concatenate(gen_input[i_input])
                bar()

        # The reconstructed MC was already processed
        print(f"File {outFilePath} exists, updating it.")
        mode = "update"

        with uproot.update(outFilePath) as f:
            f[f"MC/Gen/{key}/treeMass"] = merged_data_pt

        print("MC Gen written to file!")
        del merged_data_pt
        gc.collect()

    print(f'[MC] Finished processing pT bin {ptmin} - {ptmax}\n\n')

def process_pt_bin_mc_sparses(config, ptmin, ptmax, centmin, centmax, bkg_max_cut, debugPreprocessFile, outputDir, reco_inputs, gen_inputs, data_model_vars):
    print(f'[MC] Processing pT bin {ptmin} - {ptmax}, cent {centmin}-{centmax}')
    # outFilePath = f'{outputDir}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root'
    # if os.path.exists(outFilePath):
    #     print(f"    [MC] Updating file: {outFilePath}")
    #     outFile = TFile(outFilePath, 'update')
    #     write_opt = TObject.kOverwrite
    # else:
    #     print(f"    [MC] Creating file: {outFilePath}")
    #     outFile = TFile.Open(outFilePath, 'recreate')
    #     write_opt = 0 # Standard

    outFile, write_opt = check_existing_outputs(ptmin, ptmax, outputDir, "MC")

    axes_reco = list(config['preprocess']["axes_reco"].keys())
    rebin_reco = list(config['preprocess']["axes_reco"].values())
    axes_gen = list(config['preprocess']["axes_gen"].keys())
    rebin_gen = list(config['preprocess']["axes_gen"].values())

    # cut on pt and bkg on all the reco and gen sparses
    make_dir_root_file('MC/Reco/', outFile)
    for key, reco_input in reco_inputs.items():
        for i_input in range(len(reco_input)):
            reco_input[i_input] = apply_selection(reco_input[i_input], data_model_vars[key], 'Pt', ptmin, ptmax)
            reco_input[i_input] = apply_selection(reco_input[i_input], data_model_vars[key], 'score_bkg', 0, bkg_max_cut)
    for key, mc_input in reco_inputs.items():
        for iSparse, data_input in enumerate(mc_input):
            cloned_data_input = data_input.Clone()
            proj_axes = [data_model_vars[key][axtokeep] for axtokeep in axes_reco if axtokeep in data_model_vars[key]] # Different axes for reco and gen allowed
            proj_data_input = cloned_data_input.Projection(len(proj_axes), array.array('i', proj_axes), 'O')
            proj_data_input.SetName(f"{cloned_data_input.GetName()}_{idata_input}")
            proj_data_input = proj_data_input.Rebin(array.array('i', rebin_reco))

            if idata_input == 0:
                processed_data_input = proj_data_input.Clone()
                make_dir_root_file(f'pt_{int(ptmin*10)}_{int(ptmax*10)}/MC/Reco/{key}', debugPreprocessFile)
                debugPreprocessFile.cd(f'pt_{int(ptmin*10)}_{int(ptmax*10)}/MC/Reco/{key}')
                for iDim in range(processed_data_input.GetNdimensions()):
                    try:
                        processed_data_input.Projection(iDim).Write(axes_reco[iDim], TObject.kOverwrite)
                    except Exception as e:
                        print(f"⚠️ Exception at iDim={iDim}: {e}", flush=True)
            else:
                processed_sparse.Add(proj_sparse)

        outFile.cd('MC/Reco/')
        processed_sparse.SetName(f'h{key}')
        processed_sparse.Write(f'h{key}', write_opt)
        del processed_sparse

    make_dir_root_file('MC/Gen/', outFile)
    for key, input_type in gen_inputs.items():
        [sparse.GetAxis(data_model_vars[key]['Pt']).SetRangeUser(ptmin, ptmax) for sparse in input_type]
    for key, input_type in gen_inputs.items():
        for iSparse, sparse in enumerate(input_type):
            cloned_sparse = sparse.Clone()
            proj_axes = [data_model_vars[key][axtokeep] for axtokeep in axes_gen if axtokeep in data_model_vars[key]]
            proj_sparse = cloned_sparse.Projection(len(proj_axes), array.array('i', proj_axes), 'O')
            proj_sparse.SetName(f"{cloned_sparse.GetName()}_{iSparse}")
            proj_sparse = proj_sparse.Rebin(array.array('i', rebin_gen))

            if iSparse == 0:
                processed_sparse = proj_sparse.Clone()
                make_dir_root_file(f'pt_{int(ptmin*10)}_{int(ptmax*10)}/MC/Gen/{key}', debugPreprocessFile)
                debugPreprocessFile.cd(f'pt_{int(ptmin*10)}_{int(ptmax*10)}/MC/Gen/{key}')
                for iDim in range(processed_sparse.GetNdimensions()):
                    processed_sparse.Projection(iDim).Write(axes_gen[iDim], TObject.kOverwrite)
            else:
                processed_sparse.Add(proj_sparse)
        outFile.cd('MC/Gen/')
        processed_sparse.Write(f'h{key}', write_opt)
        del processed_sparse
    outFile.Close()
    print(f'[MC] Finished processing pT bin {ptmin} - {ptmax}\n\n')

def pre_process_data_mc(config):

    # Load the configuration
    ptmins = config['ptbins'][:-1]
    ptmaxs = config['ptbins'][1:] 
    centmin, centmax = get_centrality_bins(config['centrality'])[1]

    # Load the ThnSparse
    data_inputs, reco_inputs, gen_inputs, data_model_vars = get_inputs(config, config["operations"].get("preprocess_data", False),
                                                                       config["operations"].get("preprocess_mc", False), True)
    outputDir = config.get('outdirPrep', config['outdir'])
    os.makedirs(f'{outputDir}/preprocess', exist_ok=True)
    logger(f'Creating file {outputDir}/preprocess/DebugPreprocess.root')
    debugPreprocessFile = TFile(f'{outputDir}/preprocess/DebugPreprocess.root', 'recreate')

    bkg_maxs = config['preprocess']['bkg_cuts']
    max_workers = config['preprocess']['workers'] # hyperparameter
    if config["operations"]["preprocess_data"] and config['preprocess'].get('data'):
        logger("##### Skimming Data #####")
        ### Centrally cut on centrality and max of bkg scores
        for key, dataset_inputs in data_inputs.items():
            for i_input in range(len(dataset_inputs)):
                print_entries(dataset_inputs[i_input], "[Data] Before cuts")
                dataset_inputs[i_input] = apply_selection(dataset_inputs[i_input], data_model_vars['Data'], 'cent', centmin, centmax)
                print_entries(dataset_inputs[i_input], "[Data] After cent cut")
                dataset_inputs[i_input] = apply_selection(dataset_inputs[i_input], data_model_vars['Data'], 'score_bkg', 0, max(bkg_maxs))
                print_entries(dataset_inputs[i_input], "[Data] After cent and bkg cuts")

        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            if config['preprocess']['input_type'] == "Tree":
                tasks_data = [executor.submit(process_pt_bin_data_trees, config, ptmin, ptmax, centmin, centmax, bkg_maxs[iPt],
                                                                         debugPreprocessFile, outputDir, data_inputs,
                                                                         data_model_vars['Data']) for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs))]
            else:
                tasks_data = [executor.submit(process_pt_bin_data_sparses, config, ptmin, ptmax, centmin, centmax, bkg_maxs[iPt],
                                                                           debugPreprocessFile, outputDir, data_inputs,
                                                                           data_model_vars['Data']) for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs))]
            # Propagate exceptions
            for task in tasks_data:
                try:
                    result = task.result()  # Will raise any exception that occurred in the thread
                except Exception as e:
                    print(f"Error in thread: {e}")
                    import traceback
                    traceback.print_exc()

        logger("Finished processing data")

    if config["operations"].get("preprocess_mc") and config['preprocess'].get('mc'):
        logger("##### Skimming Monte Carlo #####")
        ### Centrally cut on centrality and max of bkg scores
        for key, reco_input in reco_inputs.items():
            print(f"key = {key}")
            for i_input in range(len(reco_input)):
                print_entries(reco_input[i_input], "[MC Reco] Before cuts")
                reco_input[i_input] = apply_selection(reco_input[i_input], data_model_vars[key], 'cent', centmin, centmax)
                print_entries(reco_input[i_input], "[MC Reco] After cent cut")
                reco_input[i_input] = apply_selection(reco_input[i_input], data_model_vars[key], 'score_bkg', 0, max(bkg_maxs))
                print_entries(reco_input[i_input], "[MC Reco] After cent and bkg cuts")
            # [apply_selection(sparse, data_model_vars[key], 'cent', centmin, centmax) for sparse in reco_input]
            # [apply_selection(sparse, data_model_vars[key], 'score_bkg', 0, max(bkg_maxs)) for sparse in reco_input]
        for key, gen_input in gen_inputs.items():
            if config['preprocess']['input_type'] == "Tree":
                logger(f"Skipping cent cut for gen trees for key {key} as the centrality column is not present", level='WARNING')
            else:
                for i_input in range(len(gen_input)):
                    print_entries(gen_input[i_input], "[MC Gen] Before cuts")
                    gen_input[i_input] = apply_selection(gen_input[i_input], data_model_vars[key], 'cent', centmin, centmax)
                    print_entries(gen_input[i_input], "[MC Gen] After cent cut")
                    # [apply_selection(sparse, data_model_vars[key], 'cent', centmin, centmax) for sparse in input_type]
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            if config['preprocess']['input_type'] == "Tree":
                tasks_mc = [executor.submit(process_pt_bin_mc_trees, config, ptmin, ptmax, centmin, centmax, bkg_maxs[iPt], 
                                                                     debugPreprocessFile, outputDir, reco_inputs, gen_inputs, 
                                                                     data_model_vars) for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs))]
            else:
                tasks_mc = [executor.submit(process_pt_bin_mc_sparses, config, ptmin, ptmax, centmin, centmax, bkg_maxs[iPt], 
                                                                       debugPreprocessFile, outputDir, reco_inputs, gen_inputs, 
                                                                       data_model_vars) for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs))]

            # Propagate exceptions
            for task in tasks_mc:
                try:
                    result = task.result()  # Will raise any exception that occurred in the thread
                except Exception as e:
                    print(f"Error in thread: {e}")
                    import traceback
                    traceback.print_exc()

        logger("Finished processing MC")

    if not config["operations"]["preprocess_data"] and not config["operations"]["preprocess_mc"]:
        logger("No data or mc pre-processing enabled. Exiting.", level='ERROR')
    debugPreprocessFile.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('config_pre', metavar='text', 
                        default='config_pre.yml', help='configuration file')
    args = parser.parse_args()

    print(f'Using configuration file: {args.config_pre}')
    with open(args.config_pre, 'r') as cfgPre:
        config = yaml.safe_load(cfgPre)

    pre_process_data_mc(config)
