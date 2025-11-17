"""
Script for evaluating the cross section.

Run:
    python compute_cross_section.py config.yaml
"""
import os
import re
import array
import ROOT  # pylint: disable=import-error
from ROOT import TFile, TH1F
import argparse

ALL_PT_FILES_TO_MERGE = ['eff', 'rawyield']


def merge_pt_bins(folder):
    """Merge ROOT histograms across pt-bin subfolders."""

    # --- Collect and sort pt folders numerically (ignore non-pt_ folders) ---
    pt_folders = sorted(
        [f for f in os.listdir(folder) if re.match(r"pt_\d+_\d+", f)],
        key=lambda x: int(x.split('_')[1])
    )

    # --- Build pt bin edges (convert to GeV if stored as ×10) ---
    pt_bin_edges = []
    for i, name in enumerate(pt_folders):
        _, pt_min, pt_max = name.split('_')
        pt_min, pt_max = float(pt_min) / 10, float(pt_max) / 10
        if i == 0:
            pt_bin_edges.append(pt_min)
        pt_bin_edges.append(pt_max)

    # --- Collect ROOT files to merge ---
    all_pt_files = {
        f"{name}.root": [
            TFile.Open(os.path.join(folder, sub, f"{name}.root"), 'READ')
            for sub in pt_folders
            if os.path.exists(os.path.join(folder, sub, f"{name}.root"))
        ]
        for name in ALL_PT_FILES_TO_MERGE
    }

    # --- Create output ROOT file ---
    outfile = TFile.Open(os.path.join(folder, 'summary.root'), 'RECREATE')

    # --- Merge histograms ---
    for key, files in all_pt_files.items():
        if not files:
            continue
        outdir = key.replace('.root', '')
        outfile.mkdir(outdir)
        outfile.cd(outdir)

        hist_data = {}

        for i, f in enumerate(files):
            for obj_key in f.GetListOfKeys():
                obj = f.Get(obj_key.GetName())
                if not isinstance(obj, ROOT.TH1):
                    continue
                name = obj.GetName()
                val, err = obj.GetBinContent(1), obj.GetBinError(1)
                hist_data.setdefault(name, [0.0] * len(pt_bin_edges))[i] = val
                hist_data.setdefault(f"{name}_unc", [0.0] * len(pt_bin_edges))[i] = err
            f.Close()

        for name, vals in hist_data.items():
            if name.endswith("_unc"):
                continue
            hist = TH1F(name, name, len(pt_bin_edges) - 1, array.array('d', pt_bin_edges))
            for i, val in enumerate(vals):
                hist.SetBinContent(i + 1, val)
                hist.SetBinError(i + 1, hist_data[f"{name}_unc"][i])
            hist.Write()

    outfile.Close()
    print(f"Merged file saved to {os.path.join(folder, 'summary.root')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('target_dir', metavar='text', default='./', help='target output directory')
    args = parser.parse_args()
    merge_pt_bins(args.target_dir)
