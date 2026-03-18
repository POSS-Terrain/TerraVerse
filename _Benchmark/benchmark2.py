# ========================================================================================================================================
# 训练集：
#   - S结构化城市道路：RSCD, ACDC, KITTI-360, IDD
"/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final",
"/8TBHDD3/tht/TerraData/ACDC/processed_data/train/local_image_final",
"/8TBHDD3/tht/TerraData/KITTI-360/processed_data/train/local_image_final",
"/8TBHDD3/tht/TerraData/IDD/processed_data/train/local_image_final"

#   - U非结构化自然场景：DeepScene, WildScenes, TAS500, RELLIS
"/8TBHDD3/tht/TerraData/DeepScene/processed_data/train/local_image_final",
"/8TBHDD3/tht/TerraData/WildScenes/processed_data/local_image_final",
"/8TBHDD3/tht/TerraData/TAS500/processed_data/train/local_image_final",
"/8TBHDD3/tht/TerraData/RELLIS/processed_data/train/local_image_final"

#   - H半结构化混合场景：TerraPOSS, Jackal, VAST, RUGD, GOOSE, ORFD
"/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
"/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final",
"/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/train",
"/8TBHDD3/tht/TerraData/RUGD/processed_data/train/local_image_final",
"/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/train/local_image_final",
"/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/train/local_image_final",
"/8TBHDD3/tht/TerraData/ORFD/processed_data/train/local_image_final"
# ========================================================================================================================================



# ========================================================================================================================================
# 测试集：
#   - S结构化城市道路：RSCD, KITTI-360, IDD
"/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final",
"/8TBHDD3/tht/TerraData/KITTI-360/processed_data/valid/local_image_final",
"/8TBHDD3/tht/TerraData/IDD/processed_data/valid/local_image_final"

#   - U非结构化自然场景：DeepScene, TAS500, RELLIS
"/8TBHDD3/tht/TerraData/DeepScene/processed_data/test/local_image_final",
"/8TBHDD3/tht/TerraData/TAS500/processed_data/valid/local_image_final",
"/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final"

#   - H半结构化混合场景：TerraPOSS, Jackal, VAST, RUGD, GOOSE
"/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
"/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final",
"/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
"/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final",
"/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/valid/local_image_final",
"/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/valid/local_image_final"
# ========================================================================================================================================

# exp 1
# S → U
CONFIG = {
    # S结构化城市道路
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/ACDC/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/IDD/processed_data/train/local_image_final"
    ],
    # U非结构化自然场景
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/DeepScene/processed_data/test/local_image_final",
        "/8TBHDD3/tht/TerraData/TAS500/processed_data/valid/local_image_final",
        "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final"
    ]
}

# exp 2
# S → H
CONFIG = {
    # S结构化城市道路
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/ACDC/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/IDD/processed_data/train/local_image_final"
    ],
    # H半结构化混合场景
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final",
        "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
        "/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final",
        "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/valid/local_image_final",
        "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/valid/local_image_final"
    ]
}

# exp 3
# U → S
CONFIG = {
    # U非结构化自然场景
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/DeepScene/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/WildScenes/processed_data/local_image_final",
        "/8TBHDD3/tht/TerraData/TAS500/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/RELLIS/processed_data/train/local_image_final"
    ],
    
    # S结构化城市道路
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final",
        "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/valid/local_image_final",
        "/8TBHDD3/tht/TerraData/IDD/processed_data/valid/local_image_final"
    ]
}

# exp 4
# U → H
CONFIG = {
    # U非结构化自然场景
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/DeepScene/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/WildScenes/processed_data/local_image_final",
        "/8TBHDD3/tht/TerraData/TAS500/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/RELLIS/processed_data/train/local_image_final"
    ],
    
    # H半结构化混合场景
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final",
        "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
        "/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final",
        "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/valid/local_image_final",
        "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/valid/local_image_final"
    ]
}

# exp 5
# H → S
CONFIG = {
    # H半结构化混合场景
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
        "/8TBHDD3/tht/TerraData/RUGD/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/ORFD/processed_data/train/local_image_final"
    ],
    
    # S结构化城市道路
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final",
        "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/valid/local_image_final",
        "/8TBHDD3/tht/TerraData/IDD/processed_data/valid/local_image_final"
    ]
}

# exp 6
# H → U
CONFIG = {
    # H半结构化混合场景
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
        "/8TBHDD3/tht/TerraData/RUGD/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/ORFD/processed_data/train/local_image_final"
    ],
    
    # U非结构化自然场景
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/DeepScene/processed_data/test/local_image_final",
        "/8TBHDD3/tht/TerraData/TAS500/processed_data/valid/local_image_final",
        "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final"
    ]
}





EXPERIMENTS = {
    "exp1": {
        # S结构化城市道路
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ACDC/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/IDD/processed_data/train/local_image_final"
        ],
        # U非结构化自然场景
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/DeepScene/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/TAS500/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final"
        ]
    },
    "exp2": {
        # S结构化城市道路
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ACDC/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/IDD/processed_data/train/local_image_final"
        ],
        # H半结构化混合场景
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/valid/local_image_final"
        ]
    },
    "exp3": {
            # U非结构化自然场景
            "train_dirs": [
                "/8TBHDD3/tht/TerraData/DeepScene/processed_data/train/local_image_final",
                "/8TBHDD3/tht/TerraData/WildScenes/processed_data/local_image_final",
                "/8TBHDD3/tht/TerraData/TAS500/processed_data/train/local_image_final",
                "/8TBHDD3/tht/TerraData/RELLIS/processed_data/train/local_image_final"
            ],
            
            # S结构化城市道路
            "test_dirs": [
                "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final",
                # "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/valid/local_image_final",
                "/8TBHDD3/tht/TerraData/IDD/processed_data/valid/local_image_final"
            ]
    },
    "exp4": {
        # U非结构化自然场景
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/DeepScene/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/WildScenes/processed_data/local_image_final",
            "/8TBHDD3/tht/TerraData/TAS500/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/train/local_image_final"
        ],
        
        # H半结构化混合场景
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/valid/local_image_final"
        ]
    },
    "exp5": {
        # H半结构化混合场景
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/train/local_image_final"
        ],
        
        # S结构化城市道路
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final",
            # "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/IDD/processed_data/valid/local_image_final"
        ]
    },
    "exp6": {
        # H半结构化混合场景
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/train/local_image_final"
        ],
        
        # U非结构化自然场景
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/DeepScene/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/TAS500/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final"
        ]
    }
}

