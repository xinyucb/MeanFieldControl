#!/usr/bin/env python3
"""
Simple Path Dependent McKean Vlasov Control Solver Test
"""

import torch
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import numpy as np
from datetime import datetime
import pandas as pd
from plot_functions import *
# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

# Set default tensor type
d_type = torch.float32
torch.set_default_dtype(d_type)
import warnings
warnings.filterwarnings("ignore")
from Param import DefaultParams
from high_dim_path_model import HighDimPathModel
from avg_path_model import AvgPathModel

from Path_MKV_Solver import PathMKV

def main():
    """Main function to run the MKV solver test."""
    
    print("=" * 50)
    print("Path Dependent Solver Test (Low Dimension)")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(2025)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    control_types = ["Xt"] # ["X", "B"]
    d_list = [1]
    model_ = "TDAvg"
    model_dic = {"TDTer": HighDimPathModel,
                 "TDAvg": AvgPathModel}
    choice = model_dic[model_]
    for d_ in d_list: 
        for control in control_types:
            print(f"Training begins d = {str(d_)}, control =", control)
            # ================================================================= ===========
            # Parameters Setup 
            # ============================================================================
            params = DefaultParams(d=d_, control_type=control, N=5).params
            setup = choice(params, delay=0) 
            model = PathMKV(setup, params)
            # Get current time
            now = datetime.now()
            # Format as string: YYYYMMDD_HHMM
            timestamp = now.strftime("_%Y%m%d%H%M")
            # Example filename
            model.filename = model_ + "_delay0N5_" + model.filename + "_" + timestamp
            model.train(plot=False)
            now = datetime.now()
            timestamp2 = now.strftime("_%H%M")
            model.filename += timestamp2

            print(f"model filename: {model.filename}")

            # ============================================================================
            # Save the model
            # ============================================================================
            print("\nSaving model...")
            model.save_NN()
            
            # ============================================================================
            # Save results
            # ============================================================================
            print("\nSaving loss results...")
            model.save_loss()
            
            
            print(f"Loss data saved as: outputLoss/{model.filename}_val_loss.csv")
            print(f"Model saved as: outputNN/{model.filename}")

        
    print(f"\nTest completed.\n")
    
if __name__ == "__main__":
    main() 