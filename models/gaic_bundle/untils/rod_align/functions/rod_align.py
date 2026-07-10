import torch

# Pure-PyTorch port of the original CUDA RoDAlign kernel
# (see untils/rod_align/src/rod_align_kernel.cu). No custom extension / nvcc
# is required. Faithful, vectorized transcription of RODAlignForward.
#
# Kernel semantics (per output element n, c, ph, pw):
#   bin_h = (H - 1.001) / (AH - 1) ; bin_w = (W - 1.001) / (AW - 1)   # ROI-independent
#   h = ph * bin_h ; w = pw * bin_w         # a fixed grid over the whole feature map
#   hstart = min(floor(h), H-2) ; wstart = min(floor(w), W-2)
#   sw = x1*scale ; sh = y1*scale ; ew = x2*scale ; eh = y2*scale
#   if (sh <= h <= eh) and (sw <= w <= ew):  out = 0     # inside the crop box
#   else: bilinear sample at (h, w)
#
# i.e. RoD ("Region of Discard") resamples the entire feature map on a fixed
# grid and blanks out whatever falls inside the crop ROI -> "whole-image
# feature minus the crop region". torchvision has no equivalent, so this is a
# direct reimplementation from the CUDA source.


def rod_align(features, rois, aligned_height, aligned_width, spatial_scale):
    """features: (B, C, H, W); rois: (N, 5) as [batch_ind, x1, y1, x2, y2]."""
    B, C, H, W = features.shape
    N = rois.shape[0]
    AH = int(aligned_height)
    AW = int(aligned_width)
    device = features.device
    dtype = features.dtype

    rois = rois.to(device=device, dtype=dtype)
    batch_ind = rois[:, 0].long()                    # (N,)
    sw = rois[:, 1] * spatial_scale                  # (N,)
    sh = rois[:, 2] * spatial_scale
    ew = rois[:, 3] * spatial_scale
    eh = rois[:, 4] * spatial_scale

    bin_h = (H - 1.001) / (AH - 1.0)                 # scalars, ROI-independent
    bin_w = (W - 1.001) / (AW - 1.0)

    ph = torch.arange(AH, device=device, dtype=dtype)
    pw = torch.arange(AW, device=device, dtype=dtype)
    h = ph * bin_h                                    # (AH,)
    w = pw * bin_w                                    # (AW,)

    hstart = torch.clamp(torch.floor(h), min=0, max=H - 2).long()   # (AH,)
    wstart = torch.clamp(torch.floor(w), min=0, max=W - 2).long()   # (AW,)
    h_ratio = h - hstart.to(dtype)                   # (AH,)
    w_ratio = w - wstart.to(dtype)                   # (AW,)

    feat_n = features[batch_ind]                     # (N, C, H, W)

    h_idx0 = hstart[None, None, :, None].expand(N, C, AH, W)
    h_idx1 = (hstart + 1)[None, None, :, None].expand(N, C, AH, W)
    rows0 = torch.gather(feat_n, 2, h_idx0)          # (N, C, AH, W)
    rows1 = torch.gather(feat_n, 2, h_idx1)

    w_idx0 = wstart[None, None, None, :].expand(N, C, AH, AW)
    w_idx1 = (wstart + 1)[None, None, None, :].expand(N, C, AH, AW)
    c00 = torch.gather(rows0, 3, w_idx0)             # (N, C, AH, AW)
    c01 = torch.gather(rows0, 3, w_idx1)
    c10 = torch.gather(rows1, 3, w_idx0)
    c11 = torch.gather(rows1, 3, w_idx1)

    hr = h_ratio[None, None, :, None]                # (1,1,AH,1)
    wr = w_ratio[None, None, None, :]                # (1,1,1,AW)
    out = (c00 * (1.0 - hr) * (1.0 - wr) + c01 * (1.0 - hr) * wr
           + c10 * hr * (1.0 - wr) + c11 * hr * wr)

    # zero every grid point that lies inside the crop ROI box
    hcond = (h[None, :] >= sh[:, None]) & (h[None, :] <= eh[:, None])   # (N, AH)
    wcond = (w[None, :] >= sw[:, None]) & (w[None, :] <= ew[:, None])   # (N, AW)
    inside = hcond[:, None, :, None] & wcond[:, None, None, :]          # (N,1,AH,AW)
    return out * (~inside).to(dtype)


class RoDAlignFunction(object):
    """Drop-in replacement for the old autograd.Function (see roi_align.py)."""

    @staticmethod
    def apply(features, rois, aligned_height, aligned_width, spatial_scale):
        return rod_align(features, rois, aligned_height, aligned_width, spatial_scale)
