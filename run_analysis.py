import os
import sys
import argparse
import yaml
import concurrent.futures
import time
import subprocess
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
# from concurrent.futures import ProcessPoolExecutor
work_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{work_dir}/utils")
from utils import check_dir, logger

paths = {
	"Preprocess": os.path.join(work_dir, "./src/pre_process.py"),
	"YamlCuts": os.path.join(work_dir, "./src/make_cutsets_cfgs.py"),
	"Projections": os.path.join(work_dir, "./src/projector.py"),
	"CorrBkgs": os.path.join(work_dir, "./src/correlated_bkgs.py"),
	"CorrBkgsSmoother": os.path.join(work_dir, "./src/correlated_bkgs_smoother.py"),
	"Efficiencies": os.path.join(work_dir, "./src/compute_efficiencies.py"),
	"GetRawYields": os.path.join(work_dir, "./src/get_raw_yields.py"),
	"CutVariation": os.path.join(work_dir, "./src/cut_variation.py"),
	"DataDrivenFraction": os.path.join(work_dir, "./src/data_driven_fraction.py"),
	"ResultsMerger": os.path.join(work_dir, "./src/results_merger.py"),
}

def make_yaml(config, outdir, cfg_type):
	logger("YAML file will be created", level="INFO")
	# check_dir(f"{outdir}/cutsets")

	cmd = (f'python3 {paths["YamlCuts"]} {config} -o {outdir} -cfg_type {cfg_type}')
	logger(f"{cmd}", level="COMMAND")
	os.system(cmd)

def project(config, outdir):
	logger("Projections will be performed", level="INFO")
	config_cutset = f"{outdir}/cutset.yml"
	cmd = (
		f"python3 {paths['Projections']} {config} -cc {config_cutset}"
	)
	logger(f"{cmd}", level="COMMAND")
	os.system(cmd)

def project_cutvar(config, outdir, nworkers=1, mCutSets=-1):
	logger("Projections will be performed", level="INFO")

	# For cut variation, we create a directory for projections
	# os.makedirs(f"{outdir}/projs", exist_ok=True)
	def run_projections(i):
		"""Run sparse projection for a given cutset index."""
		iCutSets = f"{i:02d}"
		logger(f"Processing cutset {iCutSets}...", level="INFO")

		config_cutset = f"{outdir}/cutsets/cutset_{iCutSets}.yml"
		cmd = (
			f"python3 {paths['Projections']} {config} -cc {config_cutset}"
		)
		logger(f"{cmd}", level="COMMAND")
		os.system(cmd)

	with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
		results_proj = list(executor.map(run_projections, range(mCutSets)))

def efficiencies(config, outdir, nworkers=1, mCutSets=-1):
	logger("Efficiencies will be computed", level="INFO")

	def run_efficiency(i):
		"""Run efficiency computation for a given cutset index."""
		iCutSet = f"{i:02d}"
		print(f"\033[32mProcessing cutset {iCutSet}...\033[0m")

		proj_cutset = f"{outdir}/projs/proj_{iCutSet}.root"
		cmd = (
			f"python3 {paths['Efficiencies']} {config} {proj_cutset} -b"
		)
		logger(f"{cmd}", level="COMMAND")
		os.system(cmd)

	with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
		results_eff = list(executor.map(run_efficiency, range(mCutSets)))

def get_raw_yields(config, outdir):
	print(f"\033[32m Extracting raw yields for central value ...\033[0m")
	proj_central = f"{outdir}/proj.root"
	cfg_central = f"{outdir}/cutset.yml"
	cmd = (
		f"python3 {paths['GetRawYields']} {config} {cfg_central} {proj_central}"
	)
	logger(f"{cmd}", level="COMMAND")
	os.system(cmd)

def get_raw_yields_cutvar(config, outdir, nworkers=1, mCutSets=-1):
	logger("Mass fits will be performed", level="INFO")

	def run_fit(i):
		"""Run simultaneous fit for a given cutset index."""
		iCutSets = f"{i:02d}"
		print(f"\033[32mProcessing cutset {iCutSets}...\033[0m")

		proj_cutset = f"{outdir}/projs/proj_{iCutSets}.root"
		cfg_cutset = f"{outdir}/cutsets/cutset_{iCutSets}.yml"
		cmd = (
			f"python3 {paths['GetRawYields']} {config} {cfg_cutset} {proj_cutset}"
		)
		logger(f"{cmd}", level="COMMAND")
		os.system(cmd)

	with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
		results_fit = list(executor.map(run_fit, range(mCutSets)))

def cut_variation(config, outdir, operations=None):
	# check_dir(f"{outdir}/cutVar")

	logger("Cut variation will be performed", level="INFO")

	ry_path = f"{outdir}/rawyields"
	eff_path = f"{outdir}/effs"

	cmd = (
		f"python3 {paths['CutVariation']} {config} {ry_path} {eff_path} -b"
	)
	logger(f"{cmd}", level="COMMAND")
	os.system(cmd)

def data_driven_fraction(outdir):
	logger("Data driven fraction will be performed", level="INFO")
	# check_dir(f"{outdir}/frac")
 
	cutvar_file = f"{outdir}/cutvar/frac/frac.root"
	eff_path = f"{outdir}"

	cmd = (
		f"python3 {paths['DataDrivenFraction']} {cutvar_file} {eff_path} -b"
	)
	logger(f"{cmd}", level="COMMAND")
	os.system(cmd)

def run_cut_variation(config, pt_config_path, operations, nworkers, outdir):
	#___________________________________________________________________________________________________________________________
	# make yaml file
	logger("Creating Yaml files for cut variation", level="INFO")
	make_yaml(pt_config_path, outdir, 'cutvar')
	
	mCutSets = len([f for f in os.listdir(f"{outdir}/cutsets") if os.path.isfile(os.path.join(f"{outdir}/cutsets", f))])
	logger(f"mCutSets: {mCutSets}", level="INFO")

	#___________________________________________________________________________________________________________________________
	# Projection for Data and/or MC
	logger("Projecting for cut variation", level="INFO")
	os.makedirs(f"{outdir}/projs", exist_ok=True)
	project_cutvar(pt_config_path, outdir, nworkers, mCutSets)

	# #___________________________________________________________________________________________________________________________
	# # Correlated bkgs templates
	# if operations.get('do_corr_bkgs', False):
	# 	logger("Computing corr. bkgs for cut variation", level="INFO")
	# 	os.makedirs(f"{outdir}/corrbkgs", exist_ok=True)
	# 	produce_corr_bkgs_templs_cutvar(pt_config_path, outdir, nworkers, mCutSets)
	# else:
	# 	logger("Correlated bkgs will not be included", level="WARNING")

	#___________________________________________________________________________________________________________________________
	# Efficiencies
	logger("Computing efficiencies for cut variation", level="INFO")
	efficiencies(pt_config_path, outdir, nworkers, mCutSets)

	#___________________________________________________________________________________________________________________________
	# Raw Yield extraction
	logger("Fit raw yields for cut variation", level="INFO")
	get_raw_yields_cutvar(pt_config_path, outdir, nworkers, mCutSets)

	#___________________________________________________________________________________________________________________________
	# Cut variation
	logger("Perform cut variation", level="INFO")
	cut_variation(pt_config_path, outdir, operations=operations)

def extract_raw_yields(config):

	pt_bin_configs = {}
	for pt_bin in config['pt_bins']:
		pt_bin_cfg = copy.deepcopy(config)
		pt_bin_cfg.pop('pt_bins', None)
		pt_bin_cfg['ry_setup'] = pt_bin
		pt_bin_str = f"pt_{int(pt_bin['pt_range'][0]*10)}_{int(pt_bin['pt_range'][1]*10)}"
		pt_bin_cfg['outfolder'] = f"{pt_bin_cfg['outfolder']}/{pt_bin_str}"
		os.makedirs(f"{outdir}/{pt_bin_str}", exist_ok=True)
		pt_cfg_path = f"{outdir}/{pt_bin_str}/config_{pt_bin_str}.yml"
		with open(pt_cfg_path, 'w') as file:
			yaml.dump(pt_bin_cfg, file, default_flow_style=False, sort_keys=False)
		pt_bin_configs[pt_cfg_path] = pt_bin_cfg

	# Init: pt_config_path, outdir

	operations = config['operations']

	#___________________________________________________________________________________________________________________________
	# make yaml file
	if operations.get('do_make_yaml', False):
		# make_yaml(pt_config_path, outdir, 'ry')

		with ThreadPoolExecutor(max_workers=len(pt_bin_configs.keys())) as exe:
			processes = []
			for pt_cfg_path, cfg in pt_bin_configs.items():
				processes.append(exe.submit(make_yaml, pt_cfg_path, os.path.dirname(pt_cfg_path), 'ry'))
			for proc in as_completed(processes):
				proc.result()   # raises exceptions if any

	else:
		logger("Make yaml will not be performed", level="WARNING")

	#___________________________________________________________________________________________________________________________
	# Projection for Data and/or MC
	if operations.get('do_proj_mc', False) or operations.get('do_proj_data', False):
		with ThreadPoolExecutor(max_workers=len(pt_bin_configs.keys())) as exe:
			processes = []
			for pt_cfg_path, cfg in pt_bin_configs.items():
				config_cutset = f"{outdir}/cutset.yml"
				processes.append(exe.submit(project, pt_cfg_path, os.path.dirname(pt_cfg_path)))
			for proc in as_completed(processes):
				proc.result()   # raises exceptions if any
	else:
		logger("Projections will not be performed", level="WARNING")

	# #___________________________________________________________________________________________________________________________
	# # Correlated bkgs templates
	# if operations.get('do_corr_bkgs', False):
	# 	with ThreadPoolExecutor(max_workers=1) as exe:
	# 		processes = []
	# 		for pt_cfg_path, cfg in pt_bin_configs.items():
	# 			processes.append(exe.submit(produce_corr_bkgs_templs, pt_cfg_path, os.path.dirname(pt_cfg_path)))
	# 		for proc in as_completed(processes):
	# 			proc.result()   # raises exceptions if any
	# else:
	# 	logger("Correlated bkgs will not be included", level="WARNING")

	#___________________________________________________________________________________________________________________________
	# Efficiencies
	if operations.get('do_calc_eff', False):
		print(f"\033[32mComputing efficiency of central value ...\033[0m")
		proj_central = f"{outdir}/proj.root"
		cmd = (
			f"python3 {paths['Efficiencies']} {pt_config_path} {proj_central} -b"
		)
		logger(f"{cmd}", level="COMMAND")
		os.system(cmd)
	else:
		logger("Efficiencies will not be computed", level="WARNING")

	#___________________________________________________________________________________________________________________________
	# Raw Yield extraction
	if operations.get('do_get_ry', False):
		# print(f"\033[32m Extracting raw yields for central value ...\033[0m")
		# proj_central = f"{outdir}/proj.root"
		# cmd = (
		# 	f"python3 {paths['GetRawYields']} {pt_config_path} {proj_central}"
		# )
		# logger(f"{cmd}", level="COMMAND")
		# os.system(cmd)
		with ThreadPoolExecutor(max_workers=1) as exe:
		# with ThreadPoolExecutor(max_workers=len(pt_bin_configs.keys())) as exe:
			processes = []
			for pt_cfg_path, cfg in pt_bin_configs.items():
				processes.append(exe.submit(get_raw_yields, pt_cfg_path, os.path.dirname(pt_cfg_path)))
			for proc in as_completed(processes):
				proc.result()   # raises exceptions if any
	else:
		logger("Fit raw yields will not be performed", level="WARNING")
	quit()
  
	#___________________________________________________________________________________________________________________________
	# Cut variation
	if operations.get('do_cut_var', False):
		run_cut_variation(config, pt_config_path, operations, nworkers, f"{outdir}/cutvar")
	else:
		logger("Cut variation will not be performed", level="WARNING")
  
	#___________________________________________________________________________________________________________________________
	 # Data driven fraction
	if operations.get('do_data_driven_frac', False):
		data_driven_fraction(outdir)
	else:
		logger("Data driven fraction will not be performed", level="WARNING")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Arguments')
	parser.add_argument('config', metavar='text', default='config_ry.yml', help='configuration file')
	args = parser.parse_args()

	start_time = time.time()

	# Load and copy the configuration file
	with open(args.config, 'r') as cfgFlow:
		config = yaml.safe_load(cfgFlow)

	operations = config['operations']
	nworkers = config['nworkers']
	outdir = f"{config['outdir']}/{config['outfolder']}"
	os.system(f"mkdir -p {outdir}")

	# copy the configuration file
	nfile = 0
	os.makedirs(f'{outdir}/config_ry', exist_ok=True)
	while os.path.exists(f'{outdir}/config_ry/{os.path.splitext(os.path.basename(args.config))[0]}_{nfile}.yml'):
		nfile = nfile + 1
	os.system(f'cp {args.config} {outdir}/config_ry/{os.path.splitext(os.path.basename(args.config))[0]}_{nfile}.yml')

	if operations.get('do_prep_data') or operations.get('do_prep_mc'):
		print("\033[32mINFO: Preprocess will be performed\033[0m")
		os.system(f"python3 {paths['Preprocess']} {args.config}")
	else:
		print("\033[33mWARNING: Preprocess will not be performed\033[0m")

	extract_raw_yields(config)

	if operations.get('do_results_merger', False):
		logger("Merging results across pt bins", level="INFO")
		cmd = (
			f"python3 {paths['ResultsMerger']} {outdir}"
		)
		logger(f"{cmd}", level="COMMAND")
		os.system(cmd)

	end_time = time.time()
	execution_time = end_time - start_time
	print(f"\033[34mTotal execution time: {execution_time:.2f} seconds\033[0m")
