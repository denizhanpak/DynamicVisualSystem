import gym
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt

class dvs_wrapper(gym.ObservationWrapper):
    def __init__(self, env, std_deviation=0.01, threshold = 0.0):
        self.sd = std_deviation
        self.threshold = threshold
        super().__init__(env)
        self.env = gym.wrappers.FrameStack(env,2)
        self.observation_space = gym.spaces.Box(shape=(2,), low=0, high=255)

    def create_grayscale(self, image):
        rv = image.sum(axis=2)/3 * 255
        return np.expand_dims(rv,axis=2)

    def blur(self, image):
        return gaussian_filter(image, sigma=self.sd)
    
    def quantize(self, diff):
        diff[np.abs(diff) <= self.threshold] = 0
        diff[diff > self.threshold] = 1
        diff[diff < -self.threshold] = 1
        return diff

    def observation(self, obs):
        current = self.create_grayscale(obs[0])
        prev = self.create_grayscale(obs[1])
        current = self.blur(current)
        prev = self.blur(prev)
        dc = self.quantize(current - prev)
        return dc

env = gym.make("BreakoutNoFrameskip-v4")
env = dvs_wrapper(env)
obs,_ = env.reset()
for i in range(20):
    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
plt.imshow(obs, cmap="gray")
plt.show()