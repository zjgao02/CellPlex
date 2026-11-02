"""Specialist H&E -> multiplex models (Eva, GigaTIME) behind one interface for scripts/finetune_specialist.py."""
import cv2
import torch
import torch.nn.functional as F


class Specialist:
    """train_loss(img, t, m) on a batch of windows and predict_image(he) -> (P, H, W) maps at the input resolution.
    keep: the panel proteins the model predicts (indices); mpp: its working resolution; W: its window in px."""

    def __init__(self, kind, cohort, proteins, mpp_native, zeroshot, dev, ckpt=None):
        self.kind, self.proteins, self.m0, self.dev = kind, proteins, mpp_native, dev
        if kind == 'eva':
            from . import eva as E
            E.setup()
            self.E = E
            self.keep, self.out = E.names(cohort, proteins)               # before the model reads Eva's marker table
            self.net = E.build('cuda', ckpt)
            self.mpp, self.W = E.MPP, E.WIN
            self.mask_in = torch.zeros(len(E.HE), (E.WIN // self.net.token_size) ** 2, device=dev)
        else:
            from . import gigatime as G
            G.setup()
            self.G = G
            m, self.keep, self.cols = G.build(cohort, proteins, zeroshot, ckpt)
            self.net = m.to(dev)
            self.mpp, self.W = G.MPP, G.CROP
            self.mean, self.std = G.MEAN.to(dev), G.STD.to(dev)
        self.L = int(round(self.W * self.mpp / mpp_native))                # window in native px

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)

    def parameters(self):
        return [p for p in self.net.parameters() if p.requires_grad]

    def train_loss(self, img, t, m):
        if self.kind == 'eva':
            x = (1. - img.float() / 255.).permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1)
            y = F.interpolate(t.permute(0, 3, 1, 2).float() / 255., size=self.W, mode='bilinear', align_corners=False)
            k = F.interpolate(m[:, None].float(), size=self.W, mode='bilinear', align_corners=False)[:, 0] > 0.99
            with torch.autocast('cuda', dtype=torch.bfloat16):
                rec, _, _ = self.net(x, marker_in=[self.E.HE] * len(x), marker_out=[self.out] * len(x), infer_mask=self.mask_in)
            w = k.float()[..., None]
            return (((rec.float() - y.permute(0, 2, 3, 1)) ** 2) * w).sum() / (w.sum() * y.shape[1]).clamp(min=1)
        x, y, k = self.G.prep(img, t, m, self.dev, self.mean, self.std)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = self.net(x)
        return self.G.bce_dice(logits.float(), y, k)[0]

    @torch.no_grad()
    def predict_image(self, he):
        """(H, W, 3) uint8 H&E at native resolution -> (len(keep), H, W) predicted maps"""
        H, W = he.shape[:2]
        img = cv2.resize(he, (int(round(W * self.m0 / self.mpp)), int(round(H * self.m0 / self.mpp))),
                         interpolation=cv2.INTER_AREA if self.mpp > self.m0 else cv2.INTER_LINEAR)
        self.net.eval()
        x = torch.from_numpy(img).to(self.dev).permute(2, 0, 1)[None].float() / 255.
        if self.kind == 'eva':
            p = self.E.sliding(self.net, 1. - x, self.out, 112, 8)
        else:
            p = self.G.sliding(self.net, (x - self.mean) / self.std, 128, 128)
            if self.cols is not None:
                p = p[self.cols]
        self.net.train()
        return F.interpolate(p[None].float(), size=(H, W), mode='bilinear', align_corners=False)[0]
