import torch
import torch.distributions as D


class TanhNormal(D.Distribution):
    """
    TanhNormal distribution:
    Sample z ~ Normal(mu, sigma)
    Return x = tanh(z)
    """

    arg_constraints = {}
    support = D.constraints.interval(-1.0, 1.0)
    has_rsample = True

    def __init__(self, loc, scale, epsilon=1e-6):
        self.loc = loc
        self.scale = scale
        self.normal = D.Normal(loc, scale)
        self.epsilon = epsilon
        super().__init__(self.normal.batch_shape, self.normal.event_shape)

    def rsample(self, sample_shape=torch.Size()):
        z = self.normal.rsample(sample_shape)
        return torch.tanh(z)

    def log_prob(self, value):
        # inverse transform
        z = torch.atanh(value.clamp(-1 + self.epsilon, 1 - self.epsilon))
        log_prob = self.normal.log_prob(z)
        # adjustment term from change of variables
        log_prob -= torch.log(1 - value.pow(2) + self.epsilon)
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def mean(self):
        # mean of tanh(Normal(mu, sigma)) has no closed form
        # return tanh(mu) as an approximation
        return torch.tanh(self.loc)
