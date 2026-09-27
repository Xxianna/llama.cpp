#!/usr/bin/env python3
"""Algo 2: pooled activations -> SwiGLU per (layer L, lookahead k), trained online with Adam.

Input: mean of the current token's MoE inputs of layers 0..L and the previous token's L+1..end (RMS-normalised).
Output: residual on the router of L+k applied to x_L (down zero-initialised, so it starts at router quality).
Loss: MSE to L+k's real router logits (--loss mse) or BCE on its real top-k one-hot (--loss bce).
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import cli, load, report


def main():
    ap = cli(__doc__)
    ap.add_argument("--ahead", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["mse", "bce"], default="mse")
    ap.add_argument("--input", choices=["pool", "x"], default="pool")
    a = ap.parse_args()
    d = load(a.data, need_x=True)
    dev = "cuda"
    X = d["X"].to(dev)                                                        # fp16 [NL, T, D]
    Y, W, real, K = d["Y"].to(dev), d["W"].to(dev), d["real"].to(dev), d["k"]
    NL, T, D = X.shape
    E = W.shape[1]
    oh = torch.zeros(NL, T, E, device=dev).scatter_(2, real, 1.0)
    for k in range(1, a.ahead + 1):
        src, tl = torch.arange(NL - k, device=dev), torch.arange(k, NL, device=dev)
        n = len(src)
        t0 = time.time()
        up = (torch.randn(n, D, a.hidden, device=dev) / D ** 0.5).requires_grad_()
        gate = (torch.randn(n, D, a.hidden, device=dev) / D ** 0.5).requires_grad_()
        down = torch.zeros(n, a.hidden, E, device=dev, requires_grad=True)
        opt = torch.optim.Adam([up, gate, down], lr=a.lr)
        Wt = W[tl]
        mask = (torch.arange(NL, device=dev)[None, :] <= src[:, None]).float().unsqueeze(-1)  # [n, NL, 1]
        pred = torch.empty(n, T, E, device=dev)
        for t in range(T):
            xs = X[src, t].float()
            if a.input == "pool":
                cur, prev = X[:, t].float(), (X[:, t - 1].float() if t else torch.zeros(NL, D, device=dev))
                inp = (mask * cur[None] + (1 - mask) * prev[None]).mean(1)
            else:
                inp = xs
            inp = inp * torch.rsqrt(inp.square().mean(-1, keepdim=True) + 1e-6)
            h = F.silu(torch.bmm(inp.unsqueeze(1), gate)) * torch.bmm(inp.unsqueeze(1), up)
            out = torch.einsum("ned,nd->ne", Wt, xs) + torch.bmm(h, down).squeeze(1)
            pred[:, t] = out.detach()
            if a.loss == "mse":
                loss = (out - Y[tl, t]).square().mean()
            else:
                loss = F.binary_cross_entropy_with_logits(out, oh[tl, t])
            opt.zero_grad()
            loss.backward()
            opt.step()
        pi = torch.topk(pred, K, dim=-1).indices
        hm = np.full((NL, T), np.nan)
        hm[k:] = (pi.unsqueeze(-1) == real[tl].unsqueeze(-2)).any(-1).float().mean(-1).cpu().numpy()
        report(f"L+{k} swiglu {a.input} h{a.hidden} {a.loss} lr {a.lr}", hm, d["bounds"], f"  ({time.time() - t0:.1f} s)", d["gen"])


if __name__ == "__main__":
    main()
