from tomopy.prep.alignment import scale as scale_tomo
import tomocupy.recon as tcrecon


def align_joint(TomoAlign, Align=None):
    
    # ensure it only runs on 1 thread for CUDA
    os.environ["TOMOPY_PYTHON_THREADS"] = "1"
    # Initialize variables from metadata for ease of reading:
    num_iter = TomoAlign.metadata["opts"]["num_iter"]
    init_tomo_shape = TomoAlign.prj_for_alignment.shape
    downsample = TomoAlign.metadata["opts"]["downsample"]            
    pad = TomoAlign.metadata["opts"]["pad"]
    method_str = list(TomoAlign.metadata["methods"].keys())[0]
    upsample_factor = TomoAlign.metadata["opts"]["upsample_factor"]
    num_batches = TomoAlign.metadata["opts"]["batch_size"] # change to num_batches
    center = TomoAlign.metadata["center"]

    # Needs scaling for skimage float operations.
    TomoAlign.prj_for_alignment, scl = scale_tomo(TomoAlign.prj_for_alignment)

    # Initialization of reconstruction dataset
    tomo_shape = TomoAlign.prj_for_alignment.shape
    TomoAlign.recon = np.empty(
        (tomo_shape[1], tomo_shape[2], tomo_shape[2]), dtype=np.float32
    )

    # add progress bar for method. roughly a full-loop progress bar.
    # with TomoAlign.method_bar_cm:
    #     method_bar = tqdm(
    #         total=num_iter,
    #         desc=options["method"],
    #         display=True,
    #     )

    # Initialize shift arrays
    TomoAlign.sx = np.zeros((init_tomo_shape[0]))
    TomoAlign.sy = np.zeros((init_tomo_shape[0]))
    TomoAlign.conv = np.zeros((num_iter))

    # start iterative alignment
    for n in range(num_iter):
        _rec = TomoAlign.recon

        if TomoAlign.metadata["methods"]["SIRT_CUDA"]["Faster"]:
            TomoAlign.recon = tcrecon.recon_sirt_3D(
                TomoAlign.prj_for_alignment, 
                TomoAlign.tomo.theta,
                num_iter=1,
                rec=_rec,
                center=center)
        elif TomoAlign.metadata["methods"]["SIRT_CUDA"]["Fastest"]:
            TomoAlign.recon = tcrecon.recon_sirt_3D_allgpu(
                TomoAlign.prj_for_alignment, 
                TomoAlign.tomo.theta, 
                num_iter=2,
                rec=_rec, 
                center=center)
        else:
            # Options go into kwargs which go into recon()
            kwargs = {}
            options = {
                "proj_type": "cuda",
                "method": method_str,
                "num_iter": 1
                }
            kwargs["options"] = options

            TomoAlign.recon = algorithm.recon(
                TomoAlign.prj_for_alignment,
                TomoAlign.tomo.theta,
                algorithm=wrappers.astra,
                init_recon=_rec,
                center=center,
                ncore=None,
                **kwargs,
            )
        # update progress bar
        # method_bar.update()

        # break up reconstruction into batches along z axis
        TomoAlign.recon = np.array_split(TomoAlign.recon, num_batches, axis=0)
        # may not need a copy.
        _rec = TomoAlign.recon.copy()
        
        # initialize simulated projection cpu array
        sim = []

        # begin simulating projections using astra.
        # this could probably be made more efficient, right now I am not 
        # certain if I need to be deleting every time.
        
        with TomoAlign.output1_cm:
            TomoAlign.output1_cm.clear_output()
            simulate_projections(_rec, sim, center, TomoAlign.tomo.theta)
            # del _rec
            sim = np.concatenate(sim, axis=1)

            # only flip the simulated datasets if using normal tomopy algorithm
            # can remove if it is flipped in the algorithm
            if (
                TomoAlign.metadata["methods"]["SIRT_CUDA"]["Faster"] == False
                and TomoAlign.metadata["methods"]["SIRT_CUDA"]["Fastest"] == False
            ):
                sim = np.flip(sim, axis=0)

            # Cross correlation
            shift_cpu = []
            batch_cross_correlation(TomoAlign.prj_for_alignment, 
                sim, shift_cpu,
                num_batches, upsample_factor, subset_correlation=False, blur=False, pad=pad_ds)
            TomoAlign.shift = np.concatenate(shift_cpu, axis=1)

            # Shifting 
            (TomoAlign.prj_for_alignment, 
            TomoAlign.sx, 
            TomoAlign.sy, 
            TomoAlign.shift, 
            err, 
            pad_ds, 
            center) = warp_prj_shift_cp(
                TomoAlign.prj_for_alignment, 
                TomoAlign.sx, 
                TomoAlign.sy, 
                TomoAlign.shift, 
                num_batches,
                pad_ds,
                center, 
                downsample_factor=downsample_factor   
            )
            TomoAlign.conv[n] = np.linalg.norm(err)
        with TomoAlign.output2_cm:
            TomoAlign.output2_cm.clear_output(wait=True)
            TomoAlign.plotIm(sim)
            TomoAlign.plotSxSy(downsample_factor)
            print(f"Error = {np.linalg.norm(err):3.3f}.")

        TomoAlign.recon = np.concatenate(TomoAlign.recon, axis=0)
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

    # TomoAlign.recon = np.concatenate(TomoAlign.recon, axis=0)
    # Re-normalize data
    # method_bar.close()
    TomoAlign.prj_for_alignment *= scl
    TomoAlign.recon = circ_mask(TomoAlign.recon, 0)
    if downsample:
        TomoAlign.sx = TomoAlign.sx / downsample_factor
        TomoAlign.sy = TomoAlign.sy / downsample_factor
        TomoAlign.shift = TomoAlign.shift / downsample_factor
    
    pad = tuple([x / downsample_factor for x in pad_ds])
    # make new dataset and pad/shift it for the next round
    new_prj_imgs = deepcopy(TomoAlign.tomo.prj_imgs)
    new_prj_imgs, pad = pad_projections(new_prj_imgs, pad, 1)
    new_prj_imgs = warp_prj_cp(new_prj_imgs, TomoAlign.sx, TomoAlign.sy, num_batches, pad, use_corr_prj_gpu=False)
    new_prj_imgs = trim_padding(new_prj_imgs)
    TomoAlign.tomo = td.TomoData(
        prj_imgs=new_prj_imgs, metadata=TomoAlign.metadata["importmetadata"]["tomo"]
    )
    return TomoAlign

def plotIm(TomoAlign, sim, projection_num=50):
    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.imshow(TomoAlign.prj_for_alignment[projection_num], cmap="gray")
    ax1.set_axis_off()
    ax1.set_title("Projection Image")
    ax2.imshow(sim[projection_num], cmap="gray")
    ax2.set_axis_off()
    ax2.set_title("Re-projected Image")
    plt.show()

def plotSxSy(TomoAlign, downsample_factor):
    plotrange = range(TomoAlign.prj_for_alignment.shape[0])
    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.set(xlabel= "Projection number",ylabel="Pixel shift (not downsampled)")
    ax1.plot(plotrange, TomoAlign.sx/downsample_factor)
    ax1.set_title("Sx")
    ax2.plot(plotrange, TomoAlign.sy/downsample_factor)
    ax2.set_title("Sy")
    ax2.set(xlabel= "Projection number",ylabel="Pixel shift (not downsampled)")
    plt.show()






def transform_parallel(prj, sx, sy, shift, metadata):
num_theta = prj.shape[0]
err = np.zeros((num_theta + 1, 1))
shift_y_condition = (
    metadata["opts"]["pad"][1] * metadata["opts"]["downsample_factor"]
)
shift_x_condition = (
    metadata["opts"]["pad"][0] * metadata["opts"]["downsample_factor"]
)

def transform_algorithm(prj, shift, sx, sy, m):
    shiftm = shift[:, m]
    # don't let it shift if the value is larger than padding
    if (
        np.absolute(sx[m] + shiftm[1]) < shift_x_condition
        and np.absolute(sy[m] + shiftm[0]) < shift_y_condition
    ):
        sx[m] += shiftm[1]
        sy[m] += shiftm[0]
        err[m] = np.sqrt(shiftm[0] * shiftm[0] + shiftm[1] * shiftm[1])

        # similarity transform shifts in (x, y)
        # tform = transform.SimilarityTransform(translation=(shiftm[1], shiftm[0]))
        # prj[m] = transform.warp(prj[m], tform, order=5)

        # found that ndi is much faster than the above warp
        # uses opposite convention
        shift_tuple = (shiftm[0], shiftm[1])
        shift_tuple = tuple([-1*x for x in shift_tuple])
        prj[m] = ndi.shift(prj[m], shift_tuple, order=5)

Parallel(n_jobs=-1, require="sharedmem")(
    delayed(transform_algorithm)(prj, shift, sx, sy, m)
    for m in range(num_theta)
    # for m in tnrange(num_theta, desc="Transformation", leave=True)
)
return prj, sx, sy, err


def warp_projections(prj, sx, sy, metadata):
num_theta = prj.shape[0]
err = np.zeros((num_theta + 1, 1))
shift_y_condition = (
    metadata["opts"]["pad"][1]
)
shift_x_condition = (
    metadata["opts"]["pad"][0] 
)

def transform_algorithm_warponly(prj, sx, sy, m):
    # don't let it shift if the value is larger than padding
    if (
        np.absolute(sx[m]) < shift_x_condition
        and np.absolute(sy[m]) < shift_y_condition
    ):
        # similarity transform shifts in (x, y)
        # see above note for ndi switch
        # tform = transform.SimilarityTransform(translation=(sx[m], sy[m]))
        # prj[m] = transform.warp(prj[m], tform, order=5)

        shift_tuple = (sy[m], sx[m])
        shift_tuple = tuple([-1*x for x in shift_tuple])
        prj[m] = ndi.shift(prj[m], shift_tuple, order=5)

Parallel(n_jobs=-1, require="sharedmem")(
    delayed(transform_algorithm_warponly)(prj, sx, sy, m)
    for m in range(num_theta)
    # for m in tnrange(num_theta, desc="Transformation", leave=True)
)
return prj


def init_new_from_prior(prior_tomoalign, metadata):
prj_imgs = deepcopy(prior_tomoalign.tomo.prj_imgs)
new_tomo = td.TomoData(
    prj_imgs=prj_imgs, metadata=metadata["importmetadata"]["tomo"]
)
new_align_object = TomoAlign(
    new_tomo,
    metadata,
    alignment_wd=prior_tomoalign.alignment_wd,
    alignment_wd_child=prior_tomoalign.alignment_wd_child,
)
return new_align_object


def trim_padding(prj):
# https://stackoverflow.com/questions/54567986/python-numpy-remove-empty-zeroes-border-of-3d-array
xs, ys, zs = np.where(prj > 1e-7)

minxs = np.min(xs)
maxxs = np.max(xs)
minys = np.min(ys)
maxys = np.max(ys)
minzs = np.min(zs)
maxzs = np.max(zs)

# extract cube with extreme limits of where are the values != 0
result = prj[minxs : maxxs + 1, minys : maxys + 1, minzs : maxzs + 1]
# not sure why +1 here.

return result


def simulate_projections(rec, sim, center, theta):
for batch in range(len(rec)):
# for batch in tnrange(len(rec), desc="Re-projection", leave=True):
    _rec = rec[batch]
    vol_geom = astra.create_vol_geom(
        _rec.shape[1], _rec.shape[1], _rec.shape[0]
    )
    phantom_id = astra.data3d.create("-vol", vol_geom, data=_rec)
    proj_geom = astra.create_proj_geom(
        "parallel3d",
        1,
        1,
        _rec.shape[0],
        _rec.shape[1],
        theta,
    )
    if center is not None:
        center_shift = -(center - _rec.shape[1]/2)
        proj_geom = astra.geom_postalignment(proj_geom, (center_shift,))
    projections_id, _sim = astra.creators.create_sino3d_gpu(
        phantom_id, proj_geom, vol_geom
    )
    _sim = _sim.swapaxes(0, 1)
    sim.append(_sim)
    astra.data3d.delete(projections_id)
    astra.data3d.delete(phantom_id)

def batch_cross_correlation(prj, sim, shift_cpu, num_batches, upsample_factor, 
                        blur=True, rin=0.5, rout=0.8, subset_correlation=False,
                        mask_sim=True, pad=(0,0)):
# TODO: the sign convention for shifting is bad here. 
# To fix this, change to 
# shift_gpu = phase_cross_correlation(_sim_gpu, _prj_gpu...)
# convention right now:
# if _sim is down and to the right, the shift tuple will be (-, -) 
# before going positive. 
# split into arrays for batch.
_prj = np.array_split(prj, num_batches, axis=0)
_sim = np.array_split(sim, num_batches, axis=0)

for batch in range(len(_prj)):
# for batch in tnrange(len(_prj), desc="Cross-correlation", leave=True):
    # projection images have been shifted. mask also shifts.
    # apply the "moving" mask to the simulated projections
    # simulated projections have data outside of the mask.
    if subset_correlation:
         _prj_gpu = cp.array(_prj[batch]
            [
                :,
                2*pad[1]:-2*pad[1]:1,
                2*pad[0]:-2*pad[0]:1
            ], 
            dtype=cp.float32
            )
         _sim_gpu = cp.array(_sim[batch]
            [
                :,
                2*pad[1]:-2*pad[1]:1,
                2*pad[0]:-2*pad[0]:1
            ], 
            dtype=cp.float32
            )
    else:
        _prj_gpu = cp.array(_prj[batch], dtype=cp.float32)
        _sim_gpu = cp.array(_sim[batch], dtype=cp.float32)

    if mask_sim:
        _sim_gpu = cp.where(_prj_gpu < 1e-7, 0, _sim_gpu)

    if blur:
        _prj_gpu = blur_edges_cp(_prj_gpu, rin, rout)
        _sim_gpu = blur_edges_cp(_sim_gpu, rin, rout)

    # e.g. lets say sim is (-50, 0) wrt prj. This would correspond to
    # a shift of [+50, 0]
    # In the warping section, we have to now warp prj by (-50, 0), so the 
    # SAME sign of the shift value given here.
    shift_gpu = phase_cross_correlation(
        _sim_gpu,
        _prj_gpu,
        upsample_factor=upsample_factor,
        return_error=False,
    )
    shift_cpu.append(cp.asnumpy(shift_gpu))
# shift_cpu = np.concatenate(shift_cpu, axis=1)

def blur_edges_cp(prj, low=0, high=0.8):
"""
Blurs the edge of the projection images using cupy.

Parameters
----------
prj : ndarray
    3D stack of projection images. The first dimension
    is projection axis, second and third dimensions are
    the x- and y-axes of the projection image, respectively.
low : scalar, optional
    Min ratio of the blurring frame to the image size.
high : scalar, optional
    Max ratio of the blurring frame to the image size.

Returns
-------
ndarray
    Edge-blurred 3D stack of projection images.
"""
if type(prj) is np.ndarray:
    prj_gpu = cp.array(prj, dtype=cp.float32)
else:
    prj_gpu = prj
dx, dy, dz = prj_gpu.shape
rows, cols = cp.mgrid[:dy, :dz]
rad = cp.sqrt((rows - dy / 2) ** 2 + (cols - dz / 2) ** 2)
mask = cp.zeros((dy, dz))
rmin, rmax = low * rad.max(), high * rad.max()
mask[rad < rmin] = 1
mask[rad > rmax] = 0
zone = cp.logical_and(rad >= rmin, rad <= rmax)
mask[zone] = (rmax - rad[zone]) / (rmax - rmin)
prj_gpu *= mask
return prj_gpu








    # warning I don't think I fixed sign convention here.
    def transform_parallel(prj, sx, sy, shift, metadata):
    num_theta = prj.shape[0]
    err = np.zeros((num_theta + 1, 1))
    shift_y_condition = (
        metadata["opts"]["pad"][1] * metadata["opts"]["downsample_factor"]
    )
    shift_x_condition = (
        metadata["opts"]["pad"][0] * metadata["opts"]["downsample_factor"]
    )

    def transform_algorithm(prj, shift, sx, sy, m):
        shiftm = shift[:, m]
        # don't let it shift if the value is larger than padding
        if (
            np.absolute(sx[m] + shiftm[1]) < shift_x_condition
            and np.absolute(sy[m] + shiftm[0]) < shift_y_condition
        ):
            sx[m] += shiftm[1]
            sy[m] += shiftm[0]
            err[m] = np.sqrt(shiftm[0] * shiftm[0] + shiftm[1] * shiftm[1])

            # similarity transform shifts in (x, y)
            # tform = transform.SimilarityTransform(translation=(shiftm[1], shiftm[0]))
            # prj[m] = transform.warp(prj[m], tform, order=5)

            # found that ndi is much faster than the above warp
            # uses opposite convention
            shift_tuple = (shiftm[0], shiftm[1])
            shift_tuple = tuple([-1*x for x in shift_tuple])
            prj[m] = ndi.shift(prj[m], shift_tuple, order=5)

    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(transform_algorithm)(prj, shift, sx, sy, m)
        for m in range(num_theta)
        # for m in tnrange(num_theta, desc="Transformation", leave=True)
    )
    return prj, sx, sy, err