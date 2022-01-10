#!/usr/bin/env python

from tomopy.recon import algorithm as tomopy_algorithm
import astra


def recon_sirt_plugin(prj, angles, num_iter=1, rec=None, center=None):
    # Init tomo in sinogram order
    sinograms = tomopy_algorithm.init_tomo(prj, 0)
    num_proj = sinograms.shape[1]
    num_y = sinograms.shape[0]
    num_x = sinograms.shape[2]
    # assume angles used are the same as parent tomography
    proj_geom = astra.create_proj_geom("parallel3d", 1, 1, num_y, num_x, angles)
    if center is not None:
        center_shift = -(center - num_x / 2)
        proj_geom = astra.geom_postalignment(proj_geom, (center_shift,))
    vol_geom = astra.create_vol_geom(num_x, num_x, num_y)
    sinograms_id = astra.data3d.create("-sino", proj_geom, sinograms)
    rec_id = astra.data3d.create("-vol", vol_geom, rec)
    projector_id = astra.create_projector("cuda3d", proj_geom, vol_geom)
    astra.plugin.register(astra.plugins.SIRTPlugin)
    cfg = astra.astra_dict("SIRT-PLUGIN")
    cfg["ProjectionDataId"] = sinograms_id
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectorId"] = projector_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iter)
    rec_sirt = astra.data3d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(sinograms_id)
    return rec_sirt


def recon_sirt_3D(prj, angles, num_iter=1, rec=None, center=None):
    # Todo: allow this to handle batches.
    # Init tomo in sinogram order
    sinograms = tomopy_algorithm.init_tomo(prj, 0)
    num_proj = sinograms.shape[1]
    num_y = sinograms.shape[0]
    num_x = sinograms.shape[2]
    # assume angles used are the same as parent tomography
    # create projection geometry with shape of
    proj_geom = astra.create_proj_geom("parallel3d", 1, 1, num_y, num_x, angles)
    # shifts the projection geometry so that it will reconstruct using the
    # correct center.
    if center is not None:
        center_shift = -(center - num_x / 2)
        proj_geom = astra.geom_postalignment(proj_geom, (center_shift,))
    vol_geom = astra.create_vol_geom(num_x, num_x, num_y)
    sinograms_id = astra.data3d.create("-sino", proj_geom, sinograms)
    rec_id = astra.data3d.create("-vol", vol_geom, rec)
    reco_alg = "SIRT3D_CUDA"
    cfg = astra.astra_dict(reco_alg)
    cfg["ProjectionDataId"] = sinograms_id
    cfg["ReconstructionDataId"] = rec_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iter)
    rec_sirt = astra.data3d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(sinograms_id)
    return rec_sirt


def recon_cgls_3D_allgpu(prj, angles, num_iter=1, rec=None, center=None):
    # Todo: allow this to handle batches.
    # Init tomo in sinogram order
    sinograms = tomopy_algorithm.init_tomo(prj, 0)
    num_proj = sinograms.shape[1]
    num_y = sinograms.shape[0]
    num_x = sinograms.shape[2]
    # assume angles used are the same as parent tomography
    # create projection geometry with shape of
    proj_geom = astra.create_proj_geom("parallel3d", 1, 1, num_y, num_x, angles)
    # shifts the projection geometry so that it will reconstruct using the
    # correct center.
    if center is not None:
        center_shift = -(center - num_x / 2)
        proj_geom = astra.geom_postalignment(proj_geom, (center_shift,))
    vol_geom = astra.create_vol_geom(num_x, num_x, num_y)
    sinograms_id = astra.data3d.create("-sino", proj_geom, sinograms)
    rec_id = astra.data3d.create("-vol", vol_geom, rec)
    reco_alg = "CGLS3D_CUDA"
    cfg = astra.astra_dict(reco_alg)
    cfg["ProjectionDataId"] = sinograms_id
    cfg["ReconstructionDataId"] = rec_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iter)
    rec_cgls = astra.data3d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(sinograms_id)
    return rec_cgls
