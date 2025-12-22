from torch import nn
import torch.nn.functional as F
import torch


class LoG(nn.Module):
    def __init__(self, device):
        super(LoG, self).__init__()

        self.tmpl = torch.tensor(
            [[-2, -4, -4, -4, -2],
             [-4,  0,  8,  0, -4],
             [-4,  8, 24,  8, -4],
             [-4,  0,  8,  0, -4],
             [-2, -4, -4, -4, -2]]
        ).to(device).float()

        ws, ws = self.tmpl.shape
        self.tmpl = self.tmpl.reshape(1, 1, 1, ws, ws)
        self.pad = ws // 2

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        x = x.unsqueeze(0)

        x = F.conv3d(x, self.tmpl)

        x = x.squeeze(0).squeeze(0)

        return x


class SeTLoss(nn.Module):
    def __init__(self, lmda, device):
        super(SeTLoss, self).__init__()

        self.lmda = lmda
        self.eps = 1e-6

        self.log = LoG(device)
        self.norm = lambda x: (x ** 2).sum()

    def forward(self, **kwargs):
        anm_mask = kwargs['mask']
        x = kwargs['x']
        decoder_outputs = kwargs['y']
        if isinstance(decoder_outputs, (tuple, list)):
            y = decoder_outputs[0]
        else:
            y = decoder_outputs
        num_anm = anm_mask.count()

        bg_mask = anm_mask.not_op()
        num_bg = bg_mask.count()

        log_y = self.log(y)

        as_loss = self.norm(anm_mask.dot_prod(log_y)) / (num_anm + self.eps)

        br_loss = self.norm(bg_mask.dot_prod(x - y)) / num_bg

        set_loss = br_loss + self.lmda * as_loss

        return set_loss


class MSRLoss(nn.Module):
    def __init__(self):
        super(MSRLoss, self).__init__()

        self.norm = lambda x: (x ** 2).sum()

    def forward(self, **kwargs):
        x = kwargs['x']
        decoder_outputs = kwargs['y']
        y = decoder_outputs
        scale = 1
        layers = []

        if isinstance(decoder_outputs, (tuple, list)):
            for _do in decoder_outputs:
                _rows = _do.shape[0] // scale
                _cols = _do.shape[1] // scale
                _x_down = F.interpolate(
                    x.permute(2, 0, 1).unsqueeze(0),
                    size=(_rows, _cols), mode='bilinear'
                )
                _do_down = F.interpolate(
                    _do.permute(2, 0, 1).unsqueeze(0),
                    size=(_rows, _cols), mode='bilinear'
                )
                _layer = self.norm(_x_down - _do_down) / (_rows * _cols)
                layers.append(_layer)
                scale *= 2
            msr_loss = sum(layers) / len(layers)
        else:
            rows = decoder_outputs.shape[0]
            cols = decoder_outputs.shape[1]
            layer = self.norm(x - decoder_outputs) / (rows * cols)
            msr_loss = layer
        return msr_loss


class SamLoss(nn.Module):
    def __init__(self):
        super(SamLoss, self).__init__()
        self.eps = 1e-6
        
    def forward(self, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        flat_x = x.reshape(-1, x.shape[-1])
        flat_y = y.reshape(-1, y.shape[-1])
        
        dot_product = (flat_x * flat_y).sum(dim=1)
        norm_x = torch.norm(flat_x, dim=1)
        norm_y = torch.norm(flat_y, dim=1)
        
        cos_theta = dot_product / (norm_x * norm_y + self.eps)
        cos_theta = torch.clamp(cos_theta, -1 + self.eps, 1 - self.eps)
        sam = torch.acos(cos_theta).mean()
        return sam


class TotalLoss(nn.Module):
    def __init__(self, lmda, device):
        super(TotalLoss, self).__init__()

        self.set_loss = SeTLoss(lmda, device)
        self.msr_loss = MSRLoss()
        self.sam_loss = SamLoss()

    def forward(self, **kwargs):
        rows, cols, bands = kwargs['x'].shape
        total_loss = self.set_loss(**kwargs) + self.msr_loss(**kwargs)
        total_loss /= bands
        
        return total_loss
