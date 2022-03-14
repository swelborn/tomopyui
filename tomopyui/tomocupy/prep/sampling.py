import cupy as cp
import numpy as np
from cupyx.scipy import ndimage as ndi_cp

def shrink_projections(imagestack, high_energy, low_energy, num_batches, order=3):
    shrink_ratio = low_energy / high_energy
    _imagestack = np.array_split(imagestack, num_batches, axis=0)
    zoomed_image_cpu = []
    for batch in _imagestack:
        batch_gpu = cp.array(batch, dtype=cp.float32)
        zoomed_image_gpu = ndi_cp.zoom(batch_gpu, (1, shrink_ratio, shrink_ratio), order=order)
        zoomed_image_cpu.append(cp.asnumpy(zoomed_image_gpu))
    
    zoomed_image_cpu = np.concatenate(zoomed_image_cpu, axis=0)
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    return zoomed_image_cpu