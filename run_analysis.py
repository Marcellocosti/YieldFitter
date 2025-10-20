import os
import sys
import argparse
import yaml
import concurrent.futures
import time
import subprocess
# from concurrent.futures import ProcessPoolExecutor
work_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{work_dir}/utils")
from utils import check_dir, logger

paths = {
	"Preprocess": os.path.join(work_dir, "./src/pre_process.py"),
	"YamlCuts": os.path.join(work_dir, "./src/make_cutsets_cfgs.py"),
	"CorrBkgs": os.path.join(work_dir, "./src/correlated_bkgs.py"),
	"Projections": os.path.join(work_dir, "./src/projector.py"),
	"Efficiencies": os.path.join(work_dir, "./src/compute_efficiencies.py"),
	"GetRawYields": os.path.join(work_dir, "./src/get_raw_yields.py"),
	"CutVariation": os.path.join(work_dir, "./src/cut_variation.py"),
	"DataDrivenFraction": os.path.join(work_dir, "./src/data_driven_fraction.py"),
}

def make_yaml(config, outdir, correlated=False):
	logger("YAML file will be created", level="INFO")
	check_dir(f"{outdir}/cutsets")

	cmd = (f'python3 {paths["YamlCuts"]} {config} -o {outdir}')
	logger(f"{cmd}", level="COMMAND")
	os.system(cmd)

def produce_corr_bkgs_templs(config, outdir, nworkers, mCutSets):
	logger("Correlated backgrounds will be evaluated", level="INFO")
	os.makedirs(f"{outdir}/corrbkgs", exist_ok=True)

	def run_corr_bkgs(i):
		"""Run sparse projection for a given cutset index."""
		iCutSets = f"{i:02d}"
		logger(f"Processing cutset {iCutSets}...", level="INFO")

		config_cutset = f"{outdir}/cutsets/cutset_{iCutSets}.yml"
		cmd = (
			f"python3 {paths['CorrBkgs']} {config} {config_cutset} --final_states_only"
		)
		logger(f"{cmd}", level="COMMAND")
		os.system(cmd)

	with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
		results_corr_bkgs = list(executor.map(run_corr_bkgs, range(mCutSets)))

def project(config, outdir, nworkers, mCutSets):
	logger("Projections will be performed", level="INFO")
	os.makedirs(f"{outdir}/projs", exist_ok=True)

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

def efficiencies(config, outdir, nworkers, mCutSets):
	logger("Efficiencies will be computed", level="INFO")
	check_dir(f"{outdir}/effs")

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

def get_raw_yields(config, outdir, nworkers, mCutSets):
	logger("Mass fits will be performed", level="INFO")
	check_dir(f"{outdir}/rawyields")

	def run_fit(i):
		"""Run simultaneous fit for a given cutset index."""
		iCutSets = f"{i:02d}"
		print(f"\033[32mProcessing cutset {iCutSets}...\033[0m")

		proj_cutset = f"{outdir}/projs/proj_{iCutSets}.root"
		cmd = (
			f"python3 {paths['GetRawYields']} {config} {proj_cutset}"
		)
		logger(f"{cmd}", level="COMMAND")
		os.system(cmd)

	with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
		results_fit = list(executor.map(run_fit, range(mCutSets)))

def cut_variation(config, outdir, operations=None):
	check_dir(f"{outdir}/cutVar")

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
	check_dir(f"{outdir}/frac")
 
	cutvar_file = f"{outdir}/cutVar/cutVar.root"
	eff_path = f"{outdir}/effs"

	cmd = (
		f"python3 {paths['DataDrivenFraction']} {cutvar_file} {eff_path} -b"
	)
	logger(f"{cmd}", level="COMMAND")
	os.system(cmd)

def run_cut_variation(config, operations, nworkers, outdir):
	#___________________________________________________________________________________________________________________________
	# make yaml file
	if operations.get('make_yaml', False):
		make_yaml(config, outdir)
	else:
		logger("Make yaml will not be performed", level="WARNING")

	mCutSets = len([f for f in os.listdir(f"{outdir}/cutsets") if os.path.isfile(os.path.join(f"{outdir}/cutsets", f))])
	logger(f"mCutSets: {mCutSets}", level="INFO")

	#___________________________________________________________________________________________________________________________
	# Correlated bkgs templates
	if operations.get('produce_corr_bkgs_templs', False):
		produce_corr_bkgs_templs(config, outdir, nworkers, mCutSets)
	else:
		logger("Correlated bkgs will not be included", level="WARNING")

	#___________________________________________________________________________________________________________________________
	# Projection for MC and apply the ptweights
	if operations.get('proj_mc', False) or operations.get('proj_data', False):
		project(config, outdir, nworkers, mCutSets)
	else:
		logger("Projections will not be performed", level="WARNING")

	#___________________________________________________________________________________________________________________________
	# Efficiencies
	if operations.get('efficiencies', False):
		efficiencies(config, outdir, nworkers, mCutSets)
	else:
		logger("Efficiencies will not be computed", level="WARNING")

	#___________________________________________________________________________________________________________________________
	# Simultaneous fit
	if operations.get('get_raw_yields', False):
		get_raw_yields(config, outdir, nworkers, mCutSets)
	else:
		logger("Fit raw yields will not be performed", level="WARNING")

	#___________________________________________________________________________________________________________________________
	# Cut variation
	if operations.get('do_cut_variation'):
		cut_variation(config, outdir, operations=operations)
	else:
		logger("Cut variation will not be performed", level="WARNING")

	#___________________________________________________________________________________________________________________________
	 # Data driven fraction
	if operations.get('data_driven_fraction', False):
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
	outdir = f"{config['outdir']}/cutvar_{config['suffix']}"
	os.system(f"mkdir -p {outdir}")

	# copy the configuration file
	nfile = 0
	os.makedirs(f'{outdir}/config_ry', exist_ok=True)
	while os.path.exists(f'{outdir}/config_ry/{os.path.splitext(os.path.basename(args.config))[0]}_{config["suffix"]}_{nfile}.yml'):
		nfile = nfile + 1
	os.system(f'cp {args.config} {outdir}/config_ry/{os.path.splitext(os.path.basename(args.config))[0]}_{config["suffix"]}_{nfile}.yml')

	if operations.get('preprocess_data') or operations.get('preprocess_mc'):
		print("\033[32mINFO: Preprocess will be performed\033[0m")
		os.system(f"python3 {paths['Preprocess']} {args.config}")
	else:
		print("\033[33mWARNING: Preprocess will not be performed\033[0m")

	run_cut_variation(args.config, operations, nworkers, outdir)

	end_time = time.time()
	execution_time = end_time - start_time
	print(f"\033[34mTotal execution time: {execution_time:.2f} seconds\033[0m")
