import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers).double()

def copy(from_seq, to_seq, start_index, end_index):
    if end_index < 0:
        end_index += len(from_seq)
    for i in range(start_index, end_index):
        to_seq[i].load_state_dict(from_seq[i].state_dict())

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

# PPO

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float64)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

class MLPBetaActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.act_dim = act_dim
        self.alpha_beta_net = mlp([obs_dim] + list(hidden_sizes) + [2*act_dim], activation, output_activation=nn.Softplus)

    def _distribution(self, obs):
        alpha, beta = torch.split(self.alpha_beta_net(obs), self.act_dim, dim=-1)
        return Beta(alpha, beta)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Beta distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

def load_PPO(path):
    args = torch.load(path)
    ac = MLPActorCriticPPO(args["obs_dim"], args["act_dim"], args["hidden_sizes"], args["activation"])
    ac.pi.load_state_dict(args["pi_state_dict"])
    ac.v.load_state_dict(args["v_state_dict"])
    return ac

class MLPActorCriticPPO(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # policy builder
        self.pi = MLPBetaActor(obs_dim, act_dim, hidden_sizes, activation)
        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)
        self.hidden_sizes = hidden_sizes
        self.activation = activation

    def export(self):
        return {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "pi_state_dict": self.pi.state_dict(),
            "v_state_dict": self.v.state_dict()
        }

    def save(self, path):
        torch.save(self.export(), path)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

# SAC

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim).double()
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim).double()
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        #pi_action = (pi_action + 1) / 2
        #pi_action = nn.ReLU()(pi_action)

        return pi_action, logp_pi

class MLPActorCriticSAC(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit

        self.hidden_sizes = hidden_sizes
        self.activation = activation

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
            #return (a.numpy() + 1) / 2

    def export(self):
        return {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "act_limit": self.act_limit,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "pi_state_dict": self.pi.state_dict(),
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict()
        }\

    def save(self, path):
        torch.save(self.export(), path)

def load_SAC(path):
    args = torch.load(path)
    ac = MLPActorCriticSAC(args["obs_dim"], args["act_dim"], args["act_limit"], args["hidden_sizes"], args["activation"])
    ac.pi.load_state_dict(args["pi_state_dict"])
    ac.q1.load_state_dict(args["q1_state_dict"])
    ac.q2.load_state_dict(args["q2_state_dict"])
    return ac
