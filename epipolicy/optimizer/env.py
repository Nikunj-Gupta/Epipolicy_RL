from ..core.epidemic import construct_epidemic
from .vectorizer import EpiVectorizer, EpiVectorizerS1, EpiVectorizerS2
import numpy as np

init_sample_size = 3000
eps = 1e-8

class Env:
    def __init__(self, session):
        self.epi = construct_epidemic(session)
        self.vectorizer = EpiVectorizerS2(self.epi, self)
        # self.vectorizer = EpiVectorizerS1(self.epi)
        self.obs_dim = len(self.vectorizer.obs_domain)
        self.act_dim = len(self.vectorizer.reduced_act_domain)
        self.current_state = None
        self.mean = 0
        self.M2 = 0
        self.sample_count = 0
        for i in range(init_sample_size):
            action = np.random.uniform(size=self.act_dim)
            _, r, done, _ = self.env_step(action)
            self.add_sample(r)
            if done:
                self.reset()
        self.reset()

    def add_sample(self, x):
        self.sample_count += 1
        delta = x - self.mean
        self.mean += delta / self.sample_count
        self.M2 += delta * (x - self.mean)

    def get_std(self):
        if self.sample_count == 0:
            return 0
        return np.sqrt(self.M2 / self.sample_count)

    def reset(self):
        self.prev_state = None
        self.current_state = self.epi.reset()
        return self.vectorizer.vectorize_state(self.current_state)

    def env_step(self, action):
        raise NotImplementedError

    def step(self, action):
        state, r, done, info = self.env_step(action)
        normed_r = (r - self.mean) / (self.get_std() + eps)
        return state, normed_r, done, info

class PPOEnv(Env):
    def env_step(self, reduced_action):
        action = self.vectorizer.defeaturize_reduced_action(reduced_action.copy())
        self.prev_state = self.current_state
        self.current_state, r, done = self.epi.step(action)
        return self.vectorizer.vectorize_state(self.current_state), r, done, None
    def test_step(self, reduced_action):
        action = self.vectorizer.defeaturize_reduced_action(reduced_action.copy())
        self.prev_state = self.current_state
        self.current_state, r, done = self.epi.step(action)
        return self.vectorizer.vectorize_state(self.current_state), r, done, None

class InitPPOEnv(Env):
    def env_step(self, reduced_action):
        action = self.vectorizer.defeaturize_reduced_action(reduced_action.copy())
        self.prev_state = self.current_state
        self.current_state, r, done = self.epi.step(action)
        T = 330
        r = 0
        if self.current_state.t < T:
            r -= (reduced_action[0]-1)**2
            r -= (reduced_action[1]-1)**2
        else:
            r -= reduced_action[0]**2
            r -= reduced_action[1]**2
        return self.vectorizer.vectorize_state(self.current_state), r, done, None

class SACEnv(Env):
    def env_step(self, reduced_action):
        reduced_action = (reduced_action + 1) / 2
        action = self.vectorizer.defeaturize_reduced_action(reduced_action.copy())
        self.prev_state = self.current_state
        self.current_state, r, done = self.epi.step(action)
        return self.vectorizer.vectorize_state(self.current_state), r, done, None
    def test_step(self, reduced_action):
        action = self.vectorizer.defeaturize_reduced_action(reduced_action.copy())
        self.prev_state = self.current_state
        self.current_state, r, done = self.epi.step(action)
        return self.vectorizer.vectorize_state(self.current_state), r, done, None
