# General vars intended to be globals.

tomopy_recon_algorithm_kwargs = {
    "art": ["num_gridx", "num_gridy", "num_iter"],
    "bart": ["num_gridx", "num_gridy", "num_iter", "num_block", "ind_block"],
    "fbp": ["num_gridx", "num_gridy", "filter_name", "filter_par"],
    "gridrec": ["num_gridx", "num_gridy", "filter_name", "filter_par"],
    "mlem": ["num_gridx", "num_gridy", "num_iter"],
    "osem": ["num_gridx", "num_gridy", "num_iter", "num_block", "ind_block"],
    "ospml_hybrid": [
        "num_gridx",
        "num_gridy",
        "num_iter",
        "reg_par",
        "num_block",
        "ind_block",
    ],
    "ospml_quad": [
        "num_gridx",
        "num_gridy",
        "num_iter",
        "reg_par",
        "num_block",
        "ind_block",
    ],
    "pml_hybrid": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
    "pml_quad": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
    "sirt": ["num_gridx", "num_gridy", "num_iter"],
    "tv": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
    "grad": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
    "tikh": ["num_gridx", "num_gridy", "num_iter", "reg_data", "reg_par"],
}

tomopy_align_algorithm_kwargs = {
    "art": ["num_gridx", "num_gridy", "num_iter"],
    "bart": ["num_gridx", "num_gridy", "num_iter", "num_block", "ind_block"],
    "mlem": ["num_gridx", "num_gridy", "num_iter"],
    "osem": ["num_gridx", "num_gridy", "num_iter", "num_block", "ind_block"],
    "ospml_hybrid": [
        "num_gridx",
        "num_gridy",
        "num_iter",
        "reg_par",
        "num_block",
        "ind_block",
    ],
    "ospml_quad": [
        "num_gridx",
        "num_gridy",
        "num_iter",
        "reg_par",
        "num_block",
        "ind_block",
    ],
    "pml_hybrid": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
    "pml_quad": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
    "sirt": ["num_gridx", "num_gridy", "num_iter"],
    "tv": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
    "grad": ["num_gridx", "num_gridy", "num_iter", "reg_par"],
    "tikh": ["num_gridx", "num_gridy", "num_iter", "reg_data", "reg_par"],
}

astra_cuda_recon_algorithm_kwargs = {
    "FBP CUDA": False,
    "SIRT CUDA": False,
    "SART CUDA": False,
    "CGLS CUDA": False,
    "MLEM CUDA": False,
    "SIRT Plugin": False,
    "SIRT 3D": False,
}

astra_cuda_recon_algorithm_underscores = {
    "FBP_CUDA": False,
    "SIRT_CUDA": False,
    "SART_CUDA": False,
    "CGLS_CUDA": False,
    "MLEM_CUDA": False,
    "SIRT_Plugin": False,
    "SIRT_3D": False,
}

tomopy_filter_names = {
    "none",
    "shepp",
    "cosine",
    "hann",
    "hamming",
    "ramlak",
    "parzen",
    "butterworth",
}

extend_description_style = {"description_width": "auto", "font_family": "Helvetica"}
