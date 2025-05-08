import torch
import numpy as np
import sys
sys.path.append("../noise-estimation/utils") # to import PDF
from noise import PDF


def convert_batches_to_pdf(batch_array):
    stacked_batch_array = torch.stack(batch_array, dim=0)
    batch_array_tensor = stacked_batch_array.view(-1, *stacked_batch_array.shape[2:])

    num_batches = batch_array_tensor.shape[0]
    num_bins = 20

    lo = batch_array_tensor.min().item()
    hi = batch_array_tensor.max().item()
    #lo_k = int(0.95 * batch_array_tensor.numel())
    #top_vals_lo, _ = torch.topk(batch_array_tensor.flatten(), lo_k)
    #lo = top_vals_lo[-1]
    #hi_k = int(0.05 * batch_array_tensor.numel())
    #top_vals_hi, _ = torch.topk(batch_array_tensor.flatten(), hi_k)
    #hi = top_vals_hi[-1]

    pdf_ranges = torch.linspace(lo, hi, num_bins+1)
    pdf = torch.zeros((num_bins, *batch_array_tensor.shape[1:]), dtype=torch.float32)
    p_zero = torch.zeros(batch_array_tensor.shape[1:], dtype=torch.float32)

    to_add = torch.full(batch_array_tensor.shape, 1/num_batches, dtype=pdf.dtype, device=pdf.device)

    index = ((batch_array_tensor - (lo - 1)) * num_bins / (hi - lo + 2)).long()
    index = index.clamp(min=0, max=num_bins-1)

    batch_array_flat = batch_array_tensor.reshape(num_batches, -1)
    index_flat = index.reshape(num_batches, -1)
    to_add_flat = to_add.reshape(num_batches, -1)
    to_add_flat = to_add_flat * (batch_array_flat != 0)

    to_add_p_zero = (batch_array_flat == 0).to(p_zero.dtype) * 1/num_batches

    pdf_flat = pdf.reshape(pdf.shape[0], -1)       
    pdf_flat.scatter_add_(0, index_flat, to_add_flat)

    p_zero_flat = p_zero.reshape(-1)  
    p_zero_flat += to_add_p_zero.sum(dim=0)

    #eps = 0.05
    #for batch1 in range(num_batches1):
    #    for batch2 in range(num_batches2):
    #        for (i, val) in np.ndenumerate(batch_array[batch1][batch2]):
    #            if abs(val) < eps:
    #                p_zero[i] += 1/num_batches
    #            else:
    #                pdf[(int((val - (lo - 1)) * num_bins / (hi - lo + 2)),) + i] += 1/num_batches
    
    return PDF(pdf_ranges, pdf, p_zero)


def convert_range_to_zp_scale(q_range, n_bits):
    scale = (q_range[1] - q_range[0]) / (2**n_bits)
    zp = (q_range[0] - q_range[0]) / scale
    return zp, scale