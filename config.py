# 1. Set Hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
ALPHA = 0.80
# DATA_DIR =  r"D:\CMME\1_data_generation\ultrasound_scans_bezier_curve_v1\training_data"  # Replace with the actual path to your dataset
DATA_DIR = "/home/user/haris/data/combine_datasets"
# experiment_name = "bezier_curve_unet_v2_psnr_without_threshold"
experiment_name = "combine_datasets_unet_v2"
LOG_DIR = f"runs/{experiment_name}"
SAVE_DIR = f"weights/{experiment_name}"
DATA_PREP_FIXED_IMG_SIZE = (256, 256)  # Fixed size for all images in the dataset
# Config to exactly reproduce the results of the paper
segformer_inference_config = {
    "busi": {
        "data_path": "updated_dataset_folds.npz",
        "fold_id": 4,
        "physformer": {
            "path": "segformer-mit-b5_dataset_four_Fold_4freeze_encoder_FalseBUSI_FOLD_ID_4NEW_CHALLENGE_LR_2e-05_Transformations",
            "post_processing": {
                "REMOVE_SMALL_LESIONS": False,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.4,
            },
            "BATCH_SIZE": 16,
        },
        "baseline_model": {
            "path": "segformer-mit-b5_BUSI_from_scratch_BUSI_Fold_4_crisis_management",
            "threshold": 0.5,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": True,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.5,
            },
            "BATCH_SIZE": 16,
        },
    },
    "udiat": {
        "data_path": "udiat_5folds.npz",
        "fold_id": 0,
        "physformer": {
            "path": "segformer-mit-b5_dataset_four_Fold_4freeze_encoder_FalseUDIAT_FOLD_ID_0NEW_CHALLENGE_LR_2e-05_Without_Transformations",
            "post_processing": {
                "REMOVE_SMALL_LESIONS": True,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.4,
            },
            "BATCH_SIZE": 4,
        },
        "baseline_model": {
            "path": "segformer-mit-b5_UDIAT_from_scratch_Fold_0",
            "BATCH_SIZE": 16,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": True,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.5,
            },
        },
    },
}
