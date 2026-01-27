"""
TransGCF (MS_GELRSA3) - Three-layer Gating Fusion Structure
This is the main framework mentioned in the paper, named TransGCF
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from einops import rearrange


def build_grid_edge_index(nh, nw, device):
    """Build 4-connected grid edge index for GCN."""
    edges = []
    for i in range(nh):
        for j in range(nw):
            idx = i * nw + j
            if i + 1 < nh:
                edges.append([idx, (i + 1) * nw + j])
            if j + 1 < nw:
                edges.append([idx, i * nw + (j + 1)])
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


def patch_divide(x, step, ps):
    """Crop image into patches."""
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()
    return crop_x, nh, nw


def scaled_dot_product_attention(q, k, v):
    """Scaled dot-product attention."""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image."""
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1

    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2

    return output


class PreNorm(nn.Module):
    """Normalization layer before applying function."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class dwconv(nn.Module):
    """Depthwise convolution."""
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features),
            nn.GELU()
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    """Convolutional Feed-Forward Network."""
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size=None):
        x = self.fc1(x)
        x = self.act(x)
        if x_size is not None:
            x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    """Attention module."""
    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        out = scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class LRSA(nn.Module):
    """Lightweight Local Region Self-Attention (LRSA)."""
    def __init__(self, dim, qk_dim=36, mlp_dim=96, heads=4):
        super().__init__()
        self.layer = nn.ModuleList([
            PreNorm(dim, Attention(dim, heads, qk_dim)),
            PreNorm(dim, ConvFFN(dim, mlp_dim))
        ])

    def forward(self, x):
        """x: [B, C, H, W]"""
        ps = 16
        step = ps - 2

        crop_x, nh, nw = patch_divide(x, step, ps)
        b, n, c, ph, pw = crop_x.shape

        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')
        attn, ff = self.layer

        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c -> b n c h w', b=b, n=n, h=ph, w=pw)

        x = patch_reverse(crop_x, x, step, ps)

        _, _, h, w = x.shape
        x_reshape = rearrange(x, 'b c h w-> b (h w) c')
        x_out = ff(x_reshape, x_size=(h, w)) + x_reshape
        x_out = rearrange(x_out, 'b (h w) c->b c h w', h=h)
        return x_out


class GraphEnhancedLRSA(nn.Module):
    """Single-scale LRSA + Patch GCN."""
    def __init__(self, dim, ps=16, step=None, qk_dim=36, mlp_dim=96, heads=4, gcn_dim=None):
        super().__init__()
        self.ps, self.step = ps, (ps - 2) if step is None else step
        self.attn_block = PreNorm(dim, Attention(dim, heads, qk_dim))
        self.ffn_block = PreNorm(dim, ConvFFN(dim, mlp_dim))
        gcn_dim = gcn_dim or dim
        self.gcn1 = GCNConv(dim, gcn_dim)
        self.gcn2 = GCNConv(gcn_dim, dim)

    def forward(self, x):
        """x: (B,C,H,W)"""
        B, C, H, W = x.shape
        patches, nh, nw = patch_divide(x, self.step, self.ps)
        B, N, C, ph, pw = patches.shape

        tokens = rearrange(patches, 'b n c h w -> (b n) (h w) c')
        tokens = self.attn_block(tokens) + tokens
        tokens = rearrange(tokens, '(b n) (h w) c -> b n c h w', b=B, n=N, h=ph, w=pw)
        patches = tokens

        patch_emb = patches.mean((-2, -1))
        refined = []
        for b in range(B):
            edge_index = build_grid_edge_index(nh, nw, x.device)
            out = F.relu(self.gcn1(patch_emb[b], edge_index))
            out = self.gcn2(out, edge_index)
            refined.append(out)
        refined = torch.stack(refined, dim=0)

        patches = patches + refined[..., None, None]

        x = patch_reverse(patches, x, self.step, self.ps)

        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x = self.ffn_block(x_flat) + x_flat
        x = rearrange(x, 'b (h w) c -> b c h w', h=H)
        return x


class MS_GELRSA(nn.Module):
    """Multi-Scale Graph-Enhanced LRSA."""
    def __init__(self, dim, patch_sizes=(16, 32), **kwargs):
        super().__init__()
        self.branches = nn.ModuleList([
            GraphEnhancedLRSA(dim, ps=ps, **kwargs) for ps in patch_sizes
        ])
        self.fuse = nn.Conv2d(dim * len(patch_sizes), dim, kernel_size=1)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        x = torch.cat(outs, dim=1)
        return self.fuse(x)


class SpectralSpatialAttention(nn.Module):
    """Spectral-Spatial Attention module."""
    def __init__(self, channels):
        super().__init__()
        self.spectral_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        spectral_weights = self.spectral_att(x)
        spatial_weights = self.spatial_att(x)
        return x * spectral_weights * spatial_weights


class HFEB(nn.Module):
    """High-Frequency Enhancement Block."""
    def __init__(self, in_dim):
        super().__init__()
        self.mid_dim = in_dim
        self.dim = in_dim
        self.act = nn.Sigmoid()
        
        self.last_fc = nn.Conv2d(3*self.mid_dim, self.dim, 1)
        self.channel_adjust = nn.ModuleDict({
            'd_conv': nn.Conv2d(in_dim, self.mid_dim, 1),
            'hfe': nn.Conv2d(in_dim, self.mid_dim, 1),
            'scale2': nn.Conv2d(in_dim, self.mid_dim, 1)
        })
        
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.scale2 = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='nearest'),
            nn.MaxPool2d(3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
            
        self.conv = nn.Conv2d(self.mid_dim, self.mid_dim, 3, 1, 1)
        self.d_conv = nn.Conv2d(self.mid_dim, self.mid_dim, 3, 1, dilation=3, padding=3)
        self.ss_att = SpectralSpatialAttention(self.mid_dim)

    def forward(self, x):
        short = x
        
        d_conv = self.act(self.d_conv(self.scale2(self.channel_adjust['d_conv'](x))))
        hfe = self.act(self.channel_adjust['hfe'](self.max_pool(self.channel_adjust['hfe'](x))))
        scale2 = self.act(self.channel_adjust['scale2'](self.scale2(self.channel_adjust['scale2'](x))))

        x = torch.cat([d_conv, hfe, scale2], dim=1)
        x = short + self.last_fc(x)
        x = self.ss_att(x)
        return x


class GCNBranch(nn.Module):
    """Graph Convolutional Network branch for spectral processing."""
    def __init__(self, bands, hidden_dim=32):
        super(GCNBranch, self).__init__()
        self.conv1 = GCNConv(bands, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, bands)
        self.relu = nn.ReLU()

        edges = []
        for i in range(bands - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        edge_index = torch.LongTensor(edges).t()
        
        self.register_buffer('edge_index', edge_index)

    def forward(self, x):
        """x: [B, C, H, W]"""
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        feat = x.reshape(-1, C)
        
        feat = self.conv1(feat, self.edge_index)
        feat = self.relu(feat)
        feat = self.conv2(feat, self.edge_index)
        
        feat = feat.reshape(B, H, W, C)
        feat = feat.permute(0, 3, 1, 2)
        return feat


class GatingNetwork(nn.Module):
    """Gating network for feature fusion."""
    def __init__(self, channels):
        super(GatingNetwork, self).__init__()
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        cat_feat = torch.cat([x, y], dim=1)
        gating_map = self.sigmoid(self.conv(cat_feat))
        fused = x * gating_map + y * (1 - gating_map)
        return fused


class TransGCF(nn.Module):
    """
    TransGCF (MS_GELRSA3): Three-layer Gating Fusion Structure
    This framework integrates MS_GELRSA, HFEB, and GCNBranch through gating networks
    """
    def __init__(self, shape):
        super(TransGCF, self).__init__()
        rows, cols, bands = shape
        self.name = 'TransGCF'
        
        self.encoder_lrsa = MS_GELRSA(dim=bands, qk_dim=36, mlp_dim=96, heads=4)
        self.encoder_hfeb = HFEB(in_dim=bands)
        self.encoder_gcn = GCNBranch(bands=bands, hidden_dim=32)
        self.gating_enc1 = GatingNetwork(channels=bands)
        self.gating_enc2 = GatingNetwork(channels=bands)
        
        self.middle_lrsa = MS_GELRSA(dim=bands, qk_dim=36, mlp_dim=96, heads=4)
        self.middle_hfeb = HFEB(in_dim=bands)
        self.middle_gcn = GCNBranch(bands=bands, hidden_dim=32)
        self.gating_mid1 = GatingNetwork(channels=bands)
        self.gating_mid2 = GatingNetwork(channels=bands)
        
        self.decoder_lrsa = MS_GELRSA(dim=bands, qk_dim=36, mlp_dim=96, heads=4)
        self.decoder_hfeb = HFEB(in_dim=bands)
        self.decoder_gcn = GCNBranch(bands=bands, hidden_dim=32)
        self.gating_dec1 = GatingNetwork(channels=bands)
        self.gating_dec2 = GatingNetwork(channels=bands)
        
        self.in_conv = nn.Conv2d(bands, bands, 1)
        self.mid_conv1 = nn.Conv2d(bands, bands, 1)
        self.mid_conv2 = nn.Conv2d(bands, bands, 1)
        self.out_conv = nn.Conv2d(bands, bands, 1)
    
    def forward(self, x):
        x = x.permute(2, 0, 1).unsqueeze(0)
        x = self.in_conv(x)
        
        enc_lrsa_feat = self.encoder_lrsa(x)
        enc_hfeb_feat = self.encoder_hfeb(x)
        enc_gcn_feat = self.encoder_gcn(x)
        
        enc_fusion1 = self.gating_enc1(enc_lrsa_feat, enc_hfeb_feat)
        encoder_out = self.gating_enc2(enc_fusion1, enc_gcn_feat)
        encoder_out = self.mid_conv1(encoder_out)
        
        mid_lrsa_feat = self.middle_lrsa(encoder_out)
        mid_hfeb_feat = self.middle_hfeb(encoder_out)
        mid_gcn_feat = self.middle_gcn(encoder_out)
        
        mid_fusion1 = self.gating_mid1(mid_lrsa_feat, mid_hfeb_feat)
        middle_out = self.gating_mid2(mid_fusion1, mid_gcn_feat)
        middle_out = self.mid_conv2(middle_out)
        
        dec_lrsa_feat = self.decoder_lrsa(middle_out)
        dec_hfeb_feat = self.decoder_hfeb(middle_out)
        dec_gcn_feat = self.decoder_gcn(middle_out)
        
        dec_fusion1 = self.gating_dec1(dec_lrsa_feat, dec_hfeb_feat)
        decoder_out = self.gating_dec2(dec_fusion1, dec_gcn_feat)
        out = self.out_conv(decoder_out)
        
        return out.squeeze(0).permute(1, 2, 0)


def count_parameters(model, trainable_only=True):
    """Return total number of parameters (default: trainable parameters only)."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    H, W, C = 100, 100, 64
    net = TransGCF((H, W, C))
    print(f"Total trainable params: {count_parameters(net):,}")
