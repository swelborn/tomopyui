# General vars intended to be globals.
from tomopy.recon.algorithm import allowed_recon_kwargs as tomopy_algorithm_kwargs

tomopy_algorithm_kwargs  # keep here to avoid automatic removal of import
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

cuda_import_dict = {"cupy": "cuda_enabled"}
