import argparse
import os
from config import segformer_inference_config
from helper.utils import visualize_batch_overlay, filter_multi_lesions, calculate_metrics_ind
from helper.data_loader import BUSIDataset, UDIATDataset
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
# Function definitions remain the same as in your notebook

def main(args):
    # Parse dataset and model type
    selected_dataset = args.dataset
    selected_model = args.model_type
    fold_id = args.fold_id if args.fold_id is not None else segformer_inference_config[selected_dataset]["fold_id"]
    
    # Configuration
    config = segformer_inference_config[selected_dataset][selected_model]
    DATA_DIR = args.data_dir if args.data_dir else segformer_inference_config[selected_dataset]["data_path"]
    BATCH_SIZE = args.batch_size if args.batch_size else config["BATCH_SIZE"]
    MODEL_PATH = args.model_path if args.model_path else config["path"]
    POST_PROCESSING_FN = args.post_processing if args.post_processing else config["post_processing"]

    # Visualization parameters
    SAVE_FIGURES = args.save_figures if args.save_figures is not None else True
    PLOTS_PATHS = os.path.join(args.results_dir, MODEL_PATH)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(PLOTS_PATHS, exist_ok=True)

    # Load dataset
    loaded_folds = load_paths(DATA_DIR)
    if selected_dataset == "udiat":
        _, _, list_of_test_loaders, _ = get_udiat_loaders(loaded_folds, BATCH_SIZE=BATCH_SIZE)
        dataloader = list_of_test_loaders[fold_id]
    elif selected_dataset == "busi":
        _, _, dataloader, _ = get_busi_loader(loaded_folds, BATCH_SIZE=BATCH_SIZE)
    else:
        raise ValueError("Invalid dataset specified")

    # Run evaluation
    df = test_segformer(
        dataloader=dataloader,
        MODEL_NAME=MODEL_PATH,
        SAVE_DIR=PLOTS_PATHS,
        post_processing_fns=POST_PROCESSING_FN,
        SAVE_FIGURES=SAVE_FIGURES,
        plots_title_model_text=selected_model,
    )

    # Save results
    output_csv_path = f"{args.results_dir}/{MODEL_PATH}_inference_metrics_{selected_dataset}_ALL_FOLDS.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

    # Summary statistics
    df_summary = df.drop(columns=["image_path", "mask_path"])[["dice", "hd95", "iou"]].describe()
    print(df_summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run semantic segmentation evaluation.")
    parser.add_argument("--dataset", choices=["udiat", "busi"], required=True, help="Choose dataset for evaluation")
    parser.add_argument("--model_type", choices=["physformer", "baseline_model"], required=True, help="Choose model type")
    parser.add_argument("--fold_id", type=int, help="Fold ID for evaluation (if applicable)")
    parser.add_argument("--data_dir", type=str, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, help="Batch size for evaluation")
    parser.add_argument("--model_path", type=str, help="Path to the model weights")
    parser.add_argument("--post_processing", type=dict, help="Post-processing settings")
    parser.add_argument("--save_figures", type=bool, help="Whether to save figures")
    parser.add_argument("--results_dir", default="results_evaluation", help="Directory to save results")

    args = parser.parse_args()
    main(args)