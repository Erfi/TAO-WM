import torch.distributions as D
from torch.distributions.kl import register_kl

from taowm.utils.custom_distributions import TanhNormal


@register_kl(TanhNormal, TanhNormal)
def kl_tanhnormal_tanhnormal(p, q):
    return D.kl.kl_divergence(p.normal, q.normal)
