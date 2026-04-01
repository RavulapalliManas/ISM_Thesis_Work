# Setup and Execution Instructions

## 1. Environment Setup

Due to path string restrictions (Python virtual environments cannot be created in directories containing a colon `:` like your `Thesis:ISM work` folder), it is explicitly recommended to create your virtual environment outside of this directory, such as in your home folder.

Create and activate the virtual environment:
```bash
python3 -m venv ~/pred_learning_venv
source ~/pred_learning_venv/bin/activate
```

## 2. Install Dependencies

To ensure compatibility natively with Python 3.12 and Apple Silicon architecture without encountering `setup.py` build crashes from outdated dependencies, we patch the dependencies to use `gym=0.26.2` and `gym-minigrid=1.2.2`. 

With your virtual environment activated, install all necessary libraries by running:
```bash
pip install numpy pandas matplotlib scipy torch torchvision pynapple gym==0.26.2 gym-minigrid==1.2.2 jupyter scikit-learn
```

## 3. Running a Single Instance (Training)

The codebase originally referenced a custom, proprietary environment (`MiniGrid-LRoom-18x18-v0`), which was not provided in the repository. We have modified the launch scripts to use the standard `MiniGrid-Empty-16x16-v0` environment to compensate.

To train a single instance of the predictive network (recreating the `AutoencoderPred_LN` setup from Figure 1), execute the provided bash script:
```bash
# Make sure your venv is activated!
./run_single_instance.sh
```
*Note: This script will run the network training loops and progressively save models alongside analytical figures natively into the `nets/replicate_fig1/` output folder.*

## 4. Running the Spatial Selectivity Analysis

After the model finishes training, you can extract the spatial selectivity and tuning curves by executing the newly created analysis script:
```bash
python run_analysis.py
```

This Python script systematically fetches the newly trained model, processes it through the `SpatialTuningAnalysis` module, and dumps the output visualization PDFs directly inside the `./nets/replicate_fig1/analysis/AutoencoderPred_LN-Onehot-s102/` directory.
