from abc import ABC, abstractmethod
from torch.linalg import norm
from torch import nn
import torch, time, math
import numpy as np

# Ref: https://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation
class RunningStats:
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def std(self):
        return math.sqrt(self.variance())

defaultConfig = {
    "lr": 0.001,
    "alpha": 0.9,
    "gamma": 0.9,
    "lamb": 0.8,
    "ridge": 0,
    "epsilon": 1e-7,
    "quantile": 0.75
}

class Approximator():
    def __init__(self, vectorizer, config):
        if config is None:
            config = dict()
        for attr, value in defaultConfig.items():
            if attr in config:
                setattr(self, attr, config[attr])
            else:
                setattr(self, attr, value)
        self.stats = RunningStats()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=self.alpha, eps=self.epsilon)

    def zeroEligiblity(self):
        self.Es = [torch.zeros_like(w) for w in self.model.parameters()]

    def updateEligibility(self, S, A):
        self.model.zero_grad()
        q = self.forward(self.vectorize(S, A))
        q.backward()
        for i, w in enumerate(self.model.parameters()):
            self.Es[i] = self.gamma*self.lamb*self.Es[i] + w.grad
        return q.item()

    def updateWeights(self, delta):
        self.stats.push(abs(delta))
        theta = self.stats.mean() + self.stats.std()*math.sqrt(2)*torch.erf(torch.tensor([2*self.quantile-1])).item()
        if abs(delta) <= theta:
            for i, w in enumerate(self.model.parameters()):
                w.grad = -delta*self.Es[i] + self.ridge*w
        else:
            for i, w in enumerate(self.model.parameters()):
                w.grad = -theta*np.sign(delta)*self.Es[i] + self.ridge*w
        self.optimizer.step()

    def getMaxAction(self, S, maxit=10000, qreltol=-1, qrelcount=100, vreltol=-1):
        vS = self.V.vectorizeState(S)
        vA = np.random.uniform(-1, 1, self.V.reducedM)
        rvA = torch.zeros(self.V.reducedM, dtype=torch.double)
        v = torch.tensor(np.concatenate((vS, vA)), requires_grad=True, dtype=torch.double)
        it = 0
        for w in self.model.parameters():
            w.requires_grad = False
        preQ = None
        preV = torch.clone(v[self.V.n:])
        qrc = 0
        while maxit < 0 or (maxit > 0 and it < maxit):
            if v.grad is not None:
                v.grad.data.zero_()
            q = self.forward(v)
            q.backward()
            with torch.no_grad():
                gt = v.grad[self.V.n:]
                if it % 100 == 0:
                    torch.set_printoptions(precision=10)
                    print("At it", it, "gt", gt, "full", v.grad, flush=True)
                rvA = self.alpha*rvA + (1-self.alpha)*gt*gt
                v[self.V.n:] = v[self.V.n:] + 0.1*self.lr*gt/(torch.sqrt(rvA)+self.epsilon)
                v[self.V.n:] = torch.clamp(v[self.V.n:], -1, 1)
            if qreltol > 0 and preQ is not None:
                if (q.item()-preQ) / abs(preQ) < qreltol:
                    qrc += 1
                    if qrc >= qrelcount:
                        break
                else:
                    qrc = 0
            if vreltol > 0:
                if norm(v[self.V.n:]-preV) / norm(preV) < vreltol:
                    break
            preQ = q.item()
            preV = torch.clone(v[self.V.n:])
            if it % 100 == 0:
                print("At it", it, "q", q.item(), "action", v[self.V.n:], flush=True)
            it += 1
        for w in self.model.parameters():
            w.requires_grad = True
        return self.V.defeaturizeReducedAction(v[self.V.n:].detach().numpy())

    def vectorize(self, S, A):
        return torch.tensor(self.V.vectorize(S, A), dtype=torch.double)

    def save(self, fileName):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, fileName)

    def load(self, fileName):
        checkpoint = torch.load(fileName)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def forward(self, v):
        return self.model(v)

class LinearApproximator(Approximator):
    def __init__(self, vectorizer, config=None):
        self.V = vectorizer
        self.nInput = self.V.n + self.V.reducedM
        self.model = nn.Linear(self.nInput, 1, bias=False).double()
        super(LinearApproximator, self).__init__(vectorizer, config)


class NNApproximator(Approximator):
    def __init__(self, vectorizer, config=None):
        self.V = vectorizer
        self.nInput = self.V.n + self.V.reducedM
        self.model = nn.Sequential(
          nn.Linear(self.nInput, 24, bias=True),
          nn.LeakyReLU(),
          nn.Linear(24, 12, bias=True),
          nn.LeakyReLU(),
          nn.Linear(12, 1, bias=True),
          nn.Linear(1, 1, bias=True)
        ).double()
        super(NNApproximator, self).__init__(vectorizer, config)
