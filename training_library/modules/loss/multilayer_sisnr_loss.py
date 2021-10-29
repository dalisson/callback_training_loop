'''
Implements the multlayer si_snr as seen on swave paper

'''
import torch
from .sinsnr_loss import sisnr_loss as loss_func


def multilayer_sisnr_loss(yhat, y):

    sources, lengths = y
    lengths.to(0)
    loss = 0
    cnt = len(yhat)
    # apply a loss function after each layer
    with torch.autograd.set_detect_anomaly(False):
        for c_idx, est_src in enumerate(yhat):
            coeff = ((c_idx+1)*(1/cnt))
            # SI-SNR loss
            sisnr_loss, _, est_src, _ = loss_func(
                    sources, est_src, lengths)
            loss += (coeff * sisnr_loss)
        loss /= cnt

    return loss
