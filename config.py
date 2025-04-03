# 1. Set Hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
ALPHA = 0.80
# DATA_DIR =  r"D:\CMME\1_data_generation\ultrasound_scans_bezier_curve_v1\training_data"  # Replace with the actual path to your dataset
DATA_DIR = "data/combine_datasets"
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
            # "path":"segformer-mit-b5_dataset_four_Fold_4_v1",
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
        "pretrained_on_open_source_datasets_mit_b5": {
            "path": "nvidia/mit-b5",
            "threshold": 0.5,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": True,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.5,
            },
            "BATCH_SIZE": 16,
        },
        "pretrained_on_open_source_datasets_ade_b5_512": {
            "path": "nvidia/segformer-b0-finetuned-ade-512-512",
            "threshold": 0.5,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": True,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.5,
            },
            "BATCH_SIZE": 16,
        },
        "pretrained_on_open_source_datasets_cityscapes_b5_1024": {
            "path": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
            "threshold": 0.5,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": True,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.5,
            },
            "BATCH_SIZE": 16,
        },
        "pretrained_on_open_source_datasets_bresatcancer_b0": {
            "path": "PushkarA07/segformer-b0-finetuned-breastcancer-oct-1",
            "threshold": 0.5,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": True,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.5,
            },
            "BATCH_SIZE": 16,
        },
        "pretrained_on_phys_guided_synth_III": {
            # "path": "weights/segformer-mit-b5_dataset_four_Fold_0_v1",
            "path": "weights/segformer-mit-b5_combined_datasets_Fold_4",
            "threshold": 0.5,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": False,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.4,
            },
            "BATCH_SIZE": 16,
        },
        "pretrained_on_phys_guided_synth_I_II": {
            # "path": "weights/segformer-mit-b5_dataset_combined_one_two_Fold_4_v3",
            "path": "weights/segformer-mit-b5_Cubic_Bezier_Fold_4",
            "threshold": 0.5,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": False,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.4,
            },
            "BATCH_SIZE": 16,
        },
    },
    "udiat": {
        "data_path": "udiat_5folds.npz",
        "fold_id": 0,
        "physformer": {
            "path": "segformer-mit-b5_dataset_four_Fold_4freeze_encoder_FalseUDIAT_FOLD_ID_0NEW_CHALLENGE_LR_2e-05_Without_Transformations",
            # "path":"segformer-mit-b5_dataset_four_Fold_4_v1",
            "post_processing": {
                "REMOVE_SMALL_LESIONS": True,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.4,
            },
            "BATCH_SIZE": 4,
        },
        "baseline_model": {
            "path": "segformer-mit-b5_UDIAT_from_scratch_Fold_0",
            "BATCH_SIZE": 4,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": True,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.4,
            },
        },
        "pretrained_on_open_source_datasets_mit_b5": {
            "path": "nvidia/mit-b5",
            "threshold": 0.5,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": True,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.5,
            },
            "BATCH_SIZE": 4,
        },
        "pretrained_on_open_source_datasets_ade_b5_512": {
            "path": "nvidia/segformer-b0-finetuned-ade-512-512",
            "threshold": 0.5,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": True,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.5,
            },
            "BATCH_SIZE": 4,
        },
        "pretrained_on_open_source_datasets_cityscapes_b5_1024": {
            "path": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
            "threshold": 0.5,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": True,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.5,
            },
            "BATCH_SIZE": 4,
        },
        "pretrained_on_open_source_datasets_bresatcancer_b0": {
            "path": "PushkarA07/segformer-b0-finetuned-breastcancer-oct-1",
            "threshold": 0.5,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": True,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.5,
            },
            "BATCH_SIZE": 4,
        },
        "pretrained_on_phys_guided_synth_III": {
            # "path": "weights/segformer-mit-b5_dataset_four_Fold_0_v1",
            "path": "weights/segformer-mit-b5_combined_datasets_Fold_4",
            "threshold": 0.5,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": False,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.4,
            },
            "BATCH_SIZE": 4,
        },
        "pretrained_on_phys_guided_synth_I_II": {
            # "path": "weights/segformer-mit-b5_dataset_combined_one_two_Fold_4_v3",
            "path": "weights/segformer-mit-b5_Cubic_Bezier_Fold_4",
            "threshold": 0.5,
            "post_processing": {
                "REMOVE_SMALL_LESIONS": False,
                "activation_fn": "sigmoid",  # "softmax",
                "threshold": 0.4,
            },
            "BATCH_SIZE": 4,
        },
    },
}
