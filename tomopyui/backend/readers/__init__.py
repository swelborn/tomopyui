from .ALS import RawProjectionsBase, RawProjectionsHDF5_ALS832, Metadata_ALS_832_Prenorm, Metadata_ALS_832_Raw
from .APS import RawProjectionsBase, RawProjectionsHDF5_APS, Metadata_APS_Prenorm, Metadata_APS_Raw
from .SSRL import RawProjectionsTiff_SSRL62B, RawProjectionsXRM_SSRL62C, Metadata_SSRL62B_Raw, Metadata_SSRL62B_Prenorm, Metadata_SSRL62B_Raw_Projections, Metadata_SSRL62B_Raw_References, Metadata_SSRL62C_Prenorm, Metadata_SSRL62C_Raw, Metadata_SSRL62B_Raw

def get_metadata_instance(metadata_type: str):
    if metadata_type == "SSRL62C_Normalized":
        metadata_instance = Metadata_SSRL62C_Prenorm()
    if metadata_type == "SSRL62C_Raw":
        metadata_instance = Metadata_SSRL62C_Raw()
    if metadata_type == "SSRL62B_Normalized":
        metadata_instance = Metadata_SSRL62B_Prenorm()
    if metadata_type == "SSRL62B_Raw":
        raw_prj = Metadata_SSRL62B_Raw_Projections()
        raw_ref = Metadata_SSRL62B_Raw_References()
        metadata_instance = Metadata_SSRL62B_Raw(raw_prj, raw_ref)
    if metadata_type == "ALS832_Normalized":
        metadata_instance = Metadata_ALS_832_Prenorm()
    if metadata_type == "ALS832_Raw":
        metadata_instance = Metadata_ALS_832_Raw()
    
    return metadata_instance
