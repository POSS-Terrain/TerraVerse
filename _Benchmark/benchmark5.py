EXPERIMENTS = {
    "exp1": {
        # PO->JA (Train on TerraPOSS, Test on Jackal)
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final"
        ]
    },
    "exp2": {
        # RS->JA (Train on RSCD, Test on Jackal)
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final"
        ]
    },
    "exp3": {
        # JA->PO (Train on Jackal, Test on TerraPOSS)
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final"
        ]
    },
    "exp4": {
        # RS->PO (Train on RSCD, Test on TerraPOSS)
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final"
        ]
    },
    "exp5": {
        # JA->RS (Train on Jackal, Test on RSCD)
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final"
        ]
    },
    "exp6": {
        # PO->RS (Train on TerraPOSS, Test on RSCD)
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final"
        ]
    },
    "exp7": {
        # JA->VA (Train on Jackal, Test on VAST)
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test"
        ]
    },
    "exp8": {
        # RS->VA (Train on RSCD, Test on VAST)
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test"
        ]
    },
    "exp9": {
        # PO->VA (Train on TerraPOSS, Test on VAST)
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test"
        ]
    },
    "exp10": {
        # VA->JA (Train on VAST, Test on Jackal)
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/train"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final"
        ]
    },
    "exp11": {
        # VA->RS (Train on VAST, Test on RSCD)
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/train"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final"
        ]
    },
    "exp12": {
        # VA->PO (Train on VAST, Test on TerraPOSS)
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/train"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final"
        ]
    }
}




# ==============================================================================
# PART 0: 实验集配置 (EXPERIMENTS) 
# ==============================================================================
# 为了做 Few-Shot，需要：源域训练集(Base Train)、目标域训练集(抽1% Fine-tune)、目标域测试集(Test)
base_path = "/8TBHDD3/tht/TerraData"

EXPERIMENTS = {
    "exp1": { # PO->JA
        "source_train_dirs": [f"{base_path}/TerraPOSS/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/Jackal/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/Jackal/processed_data/test/local_image_final"]
    },
    "exp2": { # RS->JA
        "source_train_dirs": [f"{base_path}/RSCD/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/Jackal/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/Jackal/processed_data/test/local_image_final"]
    },
    "exp3": { # JA->PO
        "source_train_dirs": [f"{base_path}/Jackal/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/TerraPOSS/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/TerraPOSS/processed_data/test/local_image_final"]
    },
    "exp4": { # RS->PO
        "source_train_dirs": [f"{base_path}/RSCD/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/TerraPOSS/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/TerraPOSS/processed_data/test/local_image_final"]
    },
    "exp5": { # JA->RS
        "source_train_dirs": [f"{base_path}/Jackal/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/RSCD/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/RSCD/processed_data/test/local_image_final"]
    },
    "exp6": { # PO->RS
        "source_train_dirs": [f"{base_path}/TerraPOSS/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/RSCD/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/RSCD/processed_data/test/local_image_final"]
    },
    "exp7": { # JA->VA
        "source_train_dirs": [f"{base_path}/Jackal/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/train"],
        "target_test_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/test"]
    },
    "exp8": { # RS->VA
        "source_train_dirs": [f"{base_path}/RSCD/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/train"],
        "target_test_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/test"]
    },
    "exp9": { # PO->VA
        "source_train_dirs": [f"{base_path}/TerraPOSS/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/train"],
        "target_test_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/test"]
    },
    "exp10": { # VA->JA
        "source_train_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/train"],
        "target_train_dirs": [f"{base_path}/Jackal/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/Jackal/processed_data/test/local_image_final"]
    },
    "exp11": { # VA->RS
        "source_train_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/train"],
        "target_train_dirs": [f"{base_path}/RSCD/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/RSCD/processed_data/test/local_image_final"]
    },
    "exp12": { # VA->PO
        "source_train_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/train"],
        "target_train_dirs": [f"{base_path}/TerraPOSS/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/TerraPOSS/processed_data/test/local_image_final"]
    }
}