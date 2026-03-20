# ========================================================================================================================================
# Cross-Sensor/View	：Jackal，VAST

# Jackal → VAST
CONFIG = {
    # Jackal
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
    ],
    # VAST
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test"
    ]
}

# VAST → Jackal
CONFIG = {
    # VAST
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/train"
    ],
    # Jackal
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final"
    ]
}

# ========================================================================================================================================



# ========================================================================================================================================
# Cross-Platform	

# Car → AGV
CONFIG = {
    # Car： GOOSE/GOOSE-Ex, RSCD 
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/train/local_image_final",  
        "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final"
    ],
    # AGV： RELLIS, TerraPOSS, Jackal
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final",
        "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final"
    ]
}



# AGV → Car
CONFIG = {
    # AGV： RELLIS, TerraPOSS, Jackal
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/RELLIS/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
    ],
    # Car： GOOSE/GOOSE-Ex, RSCD 
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/valid/local_image_final",
        "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/valid/local_image_final",
        "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final"
    ]
}
# ========================================================================================================================================



EXPERIMENTS = {
    "exp1": {
        # Jackal
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
        ],
        # VAST
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test"
        ]
    },
    "exp2": {
        # VAST
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/train"
        ],
        # Jackal
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final"
        ]
    },
    "exp3": {
        # Car： GOOSE, RSCD, RUGD
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/train/local_image_final", 
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final", 
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/train/local_image_final"
        ],
        # AGV： RELLIS, TerraPOSS, Jackal
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final"
        ]
    },
    "exp4": {
        # AGV： RELLIS, TerraPOSS, Jackal
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
        ],
        # Car： GOOSE, RSCD, RUGD
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final"
        ]
    }
}
