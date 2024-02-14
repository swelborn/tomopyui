import pytest
from tomopyui.widgets.imports import PrenormUploader
from tomopyui.widgets.imports import Import_ALS832
from tomopyui.widgets.center import Center
from tomopyui.widgets.analysis import Align, Recon
from tomopyui.backend.runanalysis import RunRecon, RunAlign
from unittest.mock import patch, MagicMock
import pathlib

this_file_dir = pathlib.Path(__file__).resolve().parent
import os
import multiprocessing

os.environ["num_cpu_cores"] = str(multiprocessing.cpu_count())


@pytest.fixture
def import_als_832():
    importer = Import_ALS832()
    return importer


@pytest.fixture
def prenorm_uploader(import_als_832):
    with patch(
        "tomopyui.widgets.imports.PrenormUploader.filechooser", create=True
    ) as mock_filechooser:

        mock_filechooser.selected_path = "/mock/path"
        mock_filechooser.selected_filename = "dummy_file.npy"
        mock_filechooser.register_callback = MagicMock()

        uploader = PrenormUploader(import_als_832)
        uploader.filedir = pathlib.Path(this_file_dir / "data")
        uploader.filename = pathlib.Path(
            this_file_dir / "data" / "normalized_projections.hdf5"
        )
        uploader.projections.import_metadata(
            pathlib.Path(this_file_dir / "data" / "import_metadata.json")
        )
        uploader.projections.metadatas = []
        uploader.projections.metadatas.append(uploader.projections.metadata)
        uploader.imported_metadata = True
        uploader.projections.import_file_projections(uploader)
        uploader._update_quicksearch_from_filechooser = MagicMock()

        yield uploader


def test_prenorm_uploader_init(prenorm_uploader):
    assert prenorm_uploader.projections.metadata != None


@pytest.fixture
def center(prenorm_uploader):
    center = Center(prenorm_uploader.Import)
    return center


@pytest.fixture
def align(prenorm_uploader, center):
    align = Align(prenorm_uploader.Import, center)
    align.projections = prenorm_uploader.projections
    return align


@pytest.fixture
def recon(prenorm_uploader, center):
    recon = Recon(prenorm_uploader.Import, center)
    recon.projections = prenorm_uploader.projections
    return recon


@pytest.fixture
def run_recon_instance(recon):
    return RunRecon(recon)


# Define a list or a function that generates combinations of parameters to test
methods = {
    "art": True,
    "bart": True,
    "mlem": True,
    "osem": True,
    "ospml_hybrid": True,
    "ospml_quad": True,
    "pml_hybrid": True,
    "pml_quad": True,
    "sirt": True,
    "tv": True,
    "grad": True,
    "tikh": True,
    "FBP CUDA": True,
    "SIRT CUDA": True,
    "SART CUDA": True,
    "CGLS CUDA": True,
    "MLEM CUDA": True,
    "SIRT Plugin": True,
    "SIRT 3D": True,
}

alignment_parameters = [
    {"downsample": True, "ds_factor": 1, "num_iter": 1},
    {"downsample": True, "ds_factor": 2, "num_iter": 1},
    {"downsample": False, "ds_factor": 1, "num_iter": 1},
]


@pytest.fixture(params=alignment_parameters)
def align_config(align, request):
    align.downsample = request.param["downsample"]
    align.ds_factor = request.param["ds_factor"]
    align.num_iter = request.param["num_iter"]
    align.metadata.metadata["methods"] = methods
    return align


@pytest.fixture(params=alignment_parameters)
def recon_config(recon, request):
    align.downsample = request.param["downsample"]
    align.ds_factor = request.param["ds_factor"]
    align.num_iter = request.param["num_iter"]
    align.use_multiple_centers = request.param["use_multiple_centers"]
    align.metadata.metadata["methods"] = methods
    return align


@pytest.fixture
def run_align_instance(align_config):
    with patch("tomopyui.backend.runanalysis.RunAlign.make_wd") as mock_make_wd, patch(
        "tomopyui.backend.runanalysis.RunAlign.save_overall_metadata"
    ) as save_overall_metadata:
        runalign = RunAlign(align_config, auto_run=False)
        runalign.metadata = align_config.metadata
        mock_make_wd.return_value = None
        runalign.save_data_after = MagicMock()
        runalign.save_overall_metadata = MagicMock()
        runalign.metadata.save_metadata = MagicMock()
        yield runalign


# def test_run_recon(run_recon, run_config):
#     try:
#         run_recon_instance.run()
#         process_successful = True
#     except Exception as e:
#         print(f"Error during reconstruction process: {e}")
#         process_successful = False
#     assert process_successful


def test_align_all_methods(run_align_instance):
    print(run_align_instance.__dict__)
    run_align_instance.run()
    assert run_align_instance.prjs is not None
