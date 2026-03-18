'''
五种天气：normal、dark、foggy、snowy、sunny，
    在normal的train上训，
    在各个测试集上测
'''
# ========================================================================================================================================

CONFIG = {
    # train(normal)： TerraPOSS, Jackal
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
    ],
    
    # test：normal：TerraPOSS, Jackal
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final"
    ]
}

CONFIG = {
    # train(normal)： TerraPOSS, Jackal
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
    ],
    
    # test：dark：TerraPOSS, Jackal
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/dark/local_image_final",
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/dark/local_image_final"
    ]
}

CONFIG = {
    # train(normal)： TerraPOSS, Jackal
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
    ],
    
    # test：foggy：Jackal
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/sim_fog/local_image_final"
    ]
}

CONFIG = {
    # train(normal)： TerraPOSS, Jackal
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
    ],
    
    # test：snowy：TerraPOSS
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/snow/local_image_final"
    ]
}

CONFIG = {
    # train(normal)： TerraPOSS, Jackal
    "train_dirs": [
        "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
    ],
    
    # test：sunny：Jackal
    "test_dirs": [
        "/8TBHDD3/tht/TerraData/Jackal/processed_data/sim_sun/local_image_final"
    ]
}





EXPERIMENTS = {
    "exp1": {
        # train(normal)： TerraPOSS, Jackal
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
        ],
        
        # test：normal：TerraPOSS, Jackal
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final"
        ]
    },
    "exp2": {
        # train(normal)： TerraPOSS, Jackal
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
        ],
        
        # test：dark：TerraPOSS, Jackal
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/dark/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/dark/local_image_final"
        ]
    },
    "exp3": {
        # train(normal)： TerraPOSS, Jackal
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
        ],
        
        # test：foggy：Jackal
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/sim_fog/local_image_final"
        ]
    },
    "exp4": {
        # train(normal)： TerraPOSS, Jackal
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
        ],
        
        # test：snowy：TerraPOSS
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/snow/local_image_final"
        ]
    },
    "exp5": {
        # train(normal)： TerraPOSS, Jackal
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
        ],
        
        # test：sunny：Jackal
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/sim_sun/local_image_final"
        ]
    }
}
