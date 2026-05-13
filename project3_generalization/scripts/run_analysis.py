#!/usr/bin/env python3
"""
Legacy analysis runner for trained PredictiveNet models.
"""

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import matplotlib.pyplot as plt
from utils.predictiveNet import PredictiveNet
from analysis.SpatialTuningAnalysis import SpatialTuningAnalysis

netfolder = 'replicate_fig1/'
netname = 'AutoencoderPred_LN-Onehot-s102'

print(f"Loading net {netname} from {netfolder} ...")
try:
    predictiveNet = PredictiveNet.loadNet(netfolder + netname)
except Exception as e:
    print(f"Failed to load the net. Ensure you have run 'sh run_single_instance.sh' completely. Error: {e}")
    sys.exit(1)

print("Running Spatial Tuning Analysis...")
STA = SpatialTuningAnalysis(predictiveNet, inputControl=True, untrainedControl=True)

print("Generating spatial tuning figures...")
savefolder = 'nets/' + netfolder + 'analysis/' + netname + '/'
import os
os.makedirs(savefolder, exist_ok=True)

try:
    STA.SpatialTuningFigure(netname=netname, savefolder=savefolder)
    STA.TCExamplesFigure(netname=netname, savefolder=savefolder)
    print(f"Analysis completed successfully. Figures saved in: {savefolder}")
except Exception as e:
    print(f"Encountered an error while saving figures: {e}")
