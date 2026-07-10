import torch

# Pure-PyTorch port of the original CUDA RoIAlign kernel
# (see untils/roi_align/src/roi_align_kernel.cu). No custom extension / nvcc
# is required. The math below is a faithful, vectorized transcription of
# ROIAlignForward so that outputs match the compiled kernel numerically and
# the pretrained weights stay valid.
#
# Kernel semantics (per output element n, c, ph, pw):
#   sw = x1 * scale ; sh = y1 * scale ; ew = x2 * scale ; eh = y2 * scale
#   roi_w = max(ew - sw + 1, 0) ; roi_h = max(eh - sh + 1, 0)   # note the +1
#   bin_w = roi_w / (AW - 1) ; bin_h = roi_h / (AH - 1)         # grid vertices
#   w = pw * bin_w + sw ; h = ph * bin_h + sh                   # single sample
#   if h<0 or h>=H or w<0 or w>=W: out = 0
#   else: bilinear sample at (h, w) with
#         hstart = min(floor(h), H-2), wstart = min(floor(w), W-2)
#
# This is crop-and-resize style (one bilinear sample per grid vertex), which
# is NOT the same as torchvision.ops.roi_align (bin averaging), hence the
# dedicated reimplementation.


def roi_align(features, rois, aligned_height, aligned_width, spatial_scale):
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

    roi_w = torch.clamp(ew - sw + 1.0, min=0.0)      # (N,)  -- the +1 is intentional
    roi_h = torch.clamp(eh - sh + 1.0, min=0.0)
    bin_w = roi_w / (AW - 1.0)                        # (N,)
    bin_h = roi_h / (AH - 1.0)

    pw = torch.arange(AW, device=device, dtype=dtype)  # (AW,)
    ph = torch.arange(AH, device=device, dtype=dtype)  # (AH,)
    w = pw[None, :] * bin_w[:, None] + sw[:, None]     # (N, AW)
    h = ph[None, :] * bin_h[:, None] + sh[:, None]     # (N, AH)

    valid_w = (w >= 0) & (w < W)                        # (N, AW)
    valid_h = (h >= 0) & (h < H)                        # (N, AH)

    # hstart = min(floor(h), H-2). The kernel never clamps below 0, but the
    # valid branch guarantees h >= 0 there; invalid entries are masked to 0
    # afterwards, so the extra lower clamp only keeps gather indices in range.
    wstart = torch.clamp(torch.floor(w), min=0, max=W - 2).long()   # (N, AW)
    hstart = torch.clamp(torch.floor(h), min=0, max=H - 2).long()   # (N, AH)
    w_ratio = w - wstart.to(dtype)                     # (N, AW)
    h_ratio = h - hstart.to(dtype)                     # (N, AH)

    feat_n = features[batch_ind]                       # (N, C, H, W)

    # gather along H, then along W -> the four bilinear corners
    h_idx0 = hstart[:, None, :, None].expand(N, C, AH, W)
    h_idx1 = (hstart + 1)[:, None, :, None].expand(N, C, AH, W)
    rows0 = torch.gather(feat_n, 2, h_idx0)            # (N, C, AH, W)
    rows1 = torch.gather(feat_n, 2, h_idx1)

    w_idx0 = wstart[:, None, None, :].expand(N, C, AH, AW)
    w_idx1 = (wstart + 1)[:, None, None, :].expand(N, C, AH, AW)
    c00 = torch.gather(rows0, 3, w_idx0)              # (N, C, AH, AW)
    c01 = torch.gather(rows0, 3, w_idx1)
    c10 = torch.gather(rows1, 3, w_idx0)
    c11 = torch.gather(rows1, 3, w_idx1)

    hr = h_ratio[:, None, :, None]                     # (N,1,AH,1)
    wr = w_ratio[:, None, None, :]                     # (N,1,1,AW)
    out = (c00 * (1.0 - hr) * (1.0 - wr) + c01 * (1.0 - hr) * wr
           + c10 * hr * (1.0 - wr) + c11 * hr * wr)

    mask = (valid_h[:, None, :, None] & valid_w[:, None, None, :]).to(dtype)
    return out * mask


class RoIAlignFunction(object):
    """Drop-in replacement for the old autograd.Function.

    All ops in ``roi_align`` are differentiable w.r.t. ``features``, so no
    manual backward is needed; ``apply`` keeps the original call signature.
    """

    @staticmethod
    def apply(features, rois, aligned_height, aligned_width, spatial_scale):
        return roi_align(features, rois, aligned_height, aligned_width, spatial_scale)
