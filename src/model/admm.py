import torch.nn as nn
import torch
import math

from src.model.admm_utils import crop, decrop, H, Ht, Psi, Psit, soft_thresholding

class ADMM(nn.Module):
    def __init__(self, n_iters, learnable, mu_init = 1e-4, tau_init = 2e-4):
        super().__init__()

        self.n_iters = n_iters
        self.learnable = learnable

        if learnable:
            self.log_mu1 = nn.Parameter(torch.ones(n_iters) * math.log(mu_init))
            self.log_mu2 = nn.Parameter(torch.ones(n_iters) * math.log(mu_init))
            self.log_mu3 = nn.Parameter(torch.ones(n_iters) * math.log(mu_init))
            self.log_tau = nn.Parameter(torch.ones(n_iters) * math.log(tau_init))
        else:
            self.register_buffer('log_mu1', torch.ones(n_iters) * math.log(mu_init))
            self.register_buffer('log_mu2', torch.ones(n_iters) * math.log(mu_init))
            self.register_buffer('log_mu3', torch.ones(n_iters) * math.log(mu_init))
            self.register_buffer('log_tau', torch.ones(n_iters) * math.log(tau_init))

    def _get_params(self, step):
        mu1, mu2, mu3, tau = self.log_mu1[step].exp(), self.log_mu2[step].exp(), self.log_mu3[step].exp(), self.log_tau[step].exp()
        threshold = tau if self.learnable else tau / mu2
        return mu1, mu2, mu3, threshold

    def _step(self, vars, ops, mu1, mu2, mu3, threshold):
        u_new = soft_thresholding(Psi(vars["x"]) + vars["alpha2"] / mu2, threshold)
        v_new = (vars["alpha1"] + mu1 * H(vars["x"], ops["P"]) + ops["Ctb"]) / (ops["CtC"] + mu1)
        w_new = torch.maximum(vars["alpha3"] / mu3 + vars["x"], torch.zeros_like(vars["x"]))

        r_k = mu3 * w_new - vars["alpha3"] + Psit(mu2 * u_new - vars["alpha2"]) + Ht(mu1 * v_new - vars["alpha1"], ops["P"])
        denom = mu1 * ops["P_norm"] + mu2 * ops["Psi_gram_norm"] + mu3

        x_new = torch.fft.irfft2(torch.fft.rfft2(r_k) / denom, s = r_k.shape[-2:])
        alpha1_new = vars["alpha1"] + mu1 * (H(x_new, ops["P"]) - v_new)
        alpha2_new = vars["alpha2"] + mu2 * (Psi(x_new) - u_new)
        alpha3_new = vars["alpha3"] + mu3 * (x_new - w_new)

        return {
            "u": u_new,
            "v": v_new,
            "w": w_new,
            "x": x_new,
            "alpha1": alpha1_new,
            "alpha2": alpha2_new,
            "alpha3": alpha3_new
        }

    def _precompute(self, lensless, psf):
        padded_h, padded_w = lensless.shape[-2] * 2, lensless.shape[-1] * 2
        padded_psf = decrop(psf, padded_h, padded_w)

        P = torch.fft.rfft2(torch.fft.ifftshift(padded_psf, dim = (-2, -1)))

        shift = torch.zeros_like(padded_psf)
        shift[..., 0, 0] = 1.0
        shift_x_fft = torch.fft.rfft2(torch.roll(shift, shifts = -1, dims = -1) - shift)
        shift_y_ftt = torch.fft.rfft2(torch.roll(shift, shifts = -1, dims = -2) - shift)

        return {
            "P" : P,
            "P_norm": P.abs() ** 2,
            "Psi_gram_norm": shift_x_fft.abs() ** 2 + shift_y_ftt.abs() ** 2,
            "CtC": decrop(torch.ones_like(psf), padded_h, padded_w),
            "Ctb": decrop(lensless, padded_h, padded_w)
        }
        

    def forward(self, lensless, psf, **kwargs):
        ops = self._precompute(lensless, psf)

        h, w = lensless.shape[-2:]
        padded_h, padded_w = h * 2, w * 2

        vars = {
            "u": torch.zeros(2, *lensless.shape[:-2], padded_h, padded_w, device = lensless.device, dtype = lensless.dtype),
            "v": torch.zeros(*lensless.shape[:-2], padded_h, padded_w, device = lensless.device, dtype = lensless.dtype),
            "w": torch.zeros(*lensless.shape[:-2], padded_h, padded_w, device = lensless.device, dtype = lensless.dtype),
            "x": torch.zeros(*lensless.shape[:-2], padded_h, padded_w, device = lensless.device, dtype = lensless.dtype),
            "alpha1": torch.zeros(*lensless.shape[:-2], padded_h, padded_w, device = lensless.device, dtype = lensless.dtype),
            "alpha2": torch.zeros(2, *lensless.shape[:-2], padded_h, padded_w, device = lensless.device, dtype = lensless.dtype),
            "alpha3": torch.zeros(*lensless.shape[:-2], padded_h, padded_w, device = lensless.device, dtype = lensless.dtype)
        }

        for step in range(self.n_iters):
            mu1, mu2, mu3, threshold = self._get_params(step)
            vars = self._step(vars, ops, mu1, mu2, mu3, threshold)

        x = crop(vars["x"], h, w)

        return {"output": x}