import sys
import matplotlib.pyplot as plt
from utils.predictiveNet import PredictiveNet
from analysis.SpatialTuningAnalysis import SpatialTuningAnalysis

# Define where the trained model is saved
netfolder = 'replicate_fig1/'
# Using the same parameters from the trainNet command:
# pRNNtype: 'AutoencoderPred_LN', namext: 'Onehot', seed: 102
netname = 'AutoencoderPred_LN-Onehot-s102'

print(f"Loading net {netname} from {netfolder} ...")
try:
    predictiveNet = PredictiveNet.loadNet(netfolder + netname)
except Exception as e:
    print(f"Failed to load the net. Ensure you have run 'sh run_single_instance.sh' completely. Error: {e}")
    sys.exit(1)

print("Running Spatial Tuning Analysis...")
STA = SpatialTuningAnalysis(predictiveNet, inputControl=True, untrainedControl=True)

# Generate and save tuning figures
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

