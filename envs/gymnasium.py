import os
from typing import Union, Optional
from dataclasses import dataclass
import gymnasium as gym
from enum import Enum, EnumMeta
import numpy as np
import torch

import cv2

try:
    from envs.non_stationary.half_cheetah_inter_episode_changes import *
except ImportError:
    import sys
    sys.path.insert(0, os.getcwd()) 
    from envs.non_stationary.half_cheetah_inter_episode_changes import *

FORCED_EPISODE_END_GYM_ENVIRONMENTS = {
    "Ant-v5",
    "Hopper-v5",
    "HumanoidStandup-v5",
    "Humanoid-v5",
    # "InvertedDoublePendulum-v5",
    # "InvertedPendulum-v5",
    "Walker2d-v5",
}


class TorchBox(gym.spaces.Box):

    def __init__(self,
                 low: torch.Tensor,
                 high=torch.Tensor,
                 shape=tuple,
                 dtype = torch.dtype):
        self.low = self.low_repr = low
        self.high = self.high_repr = high
        self._shape = shape
        self.dtype = dtype

    def __getattr__(self, item):
        if item not in ["shape"]:
            raise NotImplementedError("Probably not implemented")
        return self.__getattribute__(item)


class TorchTuple(gym.spaces.Tuple):

    def __init__(self,
                 spaces):
        self.spaces = tuple(spaces)

    def __getattr__(self, item):
        raise NotImplementedError("Probably not implemented")


class TorchEnvWrapper(gym.Wrapper):

    def __init__(self,
                 env,
                 dtype: torch.dtype = torch.float32):
        super(TorchEnvWrapper, self).__init__(env)
        self._dtype = dtype
        self._obs_are_images = env.obs_are_images

        self.action_space = TorchBox(low=torch.from_numpy(env.action_space.low).to(self._dtype),
                                     high=torch.from_numpy(env.action_space.high).to(self._dtype),
                                     shape=env.action_space.shape,
                                     dtype=self._dtype)

        obs_spaces = []
        for o, is_image in zip(env.observation_space, self._obs_are_images):
            if is_image:
                assert o.dtype == np.uint8, "Images need to be uint8"
                obs_spaces.append(TorchBox(low=torch.from_numpy(o.low),
                                           high=torch.from_numpy(o.high),
                                           shape=(o.shape[2], o.shape[0], o.shape[1]),
                                           dtype=torch.uint8))
            else:
                obs_spaces.append(TorchBox(low=torch.from_numpy(o.low).to(self._dtype),
                                           high=torch.from_numpy(o.high).to(self._dtype),
                                           shape=o.shape,
                                           dtype=self._dtype))
        self.observation_space = TorchTuple(obs_spaces)

    # @property
    # def obs_are_images(self):
    #     return self._obs_are_images

    def _obs_to_torch(self, obs: list[np.ndarray]) -> list[torch.Tensor]:
        torch_obs = []
        for o, is_image in zip(obs, self._obs_are_images):
            if is_image:
                torch_obs.append(torch.from_numpy(np.ascontiguousarray(np.transpose(o, axes=(2, 0, 1)))))
            else:
                torch_obs.append(torch.from_numpy(o).to(self._dtype))
        return torch_obs

    def _dict_to_torch(self, np_dict: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        torch_dict = {}
        for k, v in np_dict.items():
            if k == "obs_valid":
                torch_dict[k] = [torch.BoolTensor([b]) for b in v]
            elif k == "loss_mask":
                torch_dict[k] = [torch.from_numpy(lm) for lm in v]
            elif np.isscalar(v):
                torch_dict[k] = torch.Tensor([v]).to(self._dtype)
            else:
                torch_dict[k] = torch.from_numpy(v).to(self._dtype)
        return torch_dict

    def reset(self):
        np_obs, np_info = self.env.reset()
        return self._obs_to_torch(np_obs), self._dict_to_torch(np_info)

    def step(self, action: torch.Tensor):
        np_action = action.detach().cpu().numpy().astype(self.env.action_space.dtype)
        np_obs, scalar_reward, done, np_info = self.env.step(action=np_action)
        reward = torch.Tensor([scalar_reward]).to(self._dtype)
        done = torch.Tensor([done]).to(torch.bool)
        return self._obs_to_torch(np_obs), reward, done, self._dict_to_torch(np_info)


class _WhiteNoise:

    def __init__(self,
                 mu: np.ndarray,
                 sigma: np.ndarray):
        self._mu = mu
        self._sigma = sigma

    def __call__(self) -> np.ndarray:
        return np.random.normal(size=self._mu.shape) * self._sigma

    def reset(self):
        pass


class TorchEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._dtype = torch.float32

    def _dict_to_torch(self, np_dict: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        torch_dict = {}
        for k, v in np_dict.items():
            if k == "obs_valid":
                torch_dict[k] = [torch.BoolTensor([b]) for b in v]
            elif k == "loss_mask":
                torch_dict[k] = [torch.from_numpy(lm) for lm in v]
            elif np.isscalar(v):
                torch_dict[k] = torch.Tensor([v])
            else:
                torch_dict[k] = torch.from_numpy(v)
        return torch_dict

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return torch.tensor(obs).unsqueeze(0), torch.tensor([reward]).to(self._dtype), torch.tensor([done]).to(torch.bool), self._dict_to_torch(info)

    def reset(self, seed=0, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return torch.tensor(obs).unsqueeze(0), self._dict_to_torch(info)

    @property
    def obs_are_images(self):
        return self.env.obs_are_images

    @property
    def observation_space(self):
        return TorchTuple([self.env.observation_space])

    @property
    def action_dim(self):
        return self.env.action_space.shape[0]

    @property
    def action_space(self):
        return TorchBox(low=torch.from_numpy(self.env.action_space.low).float(),
                                     high=torch.from_numpy(self.env.action_space.high).float(),
                                     shape=self.env.action_space.shape,
                                     dtype=torch.float32)


class PixelNormalization(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _pixel_normalization(self, obs):
        return obs / 255.0 - 0.5

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._pixel_normalization(obs), reward, terminated, truncated, info

    def reset(self, seed=0, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._pixel_normalization(obs), info


class ChannelFirstEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_shape = self.env.observation_space.shape[-1:] + self.env.observation_space.shape[:2]

    def _permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor as neural networks take as first dimensions the channel dimension C
        observation = np.transpose(observation, (2, 0, 1))
        return observation

    def observation(self, observation):
        observation = self._permute_orientation(observation)
        return observation

    @property
    def obs_are_images(self):
        return self.env.env.obs_are_images
    
    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=0, high=255, shape=self.observation_shape, dtype=np.uint8
        )
    
    @property
    def action_space(self):
        return self.env.env.action_space


class OldGymInterface(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info

    @property
    def obs_are_images(self):
        return self.env.obs_are_images

    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space


@dataclass
class GymConfigs:
    env = "HalfCheetah-v5"
    env_type = ""
    action_repeat = 2  # env default
    symbolic = False
    forced_episode_end = False
    seed = 0
    img_size = (64, 64)
    obs_type = "image"

@dataclass
class GymNonStationaryInter:
    env = ""
    env_type = "gym_non_stationary_inter"
    obs_type = "proprio"
    action_repeat = 2  # env default
    seed = 0
    img_size = (64, 64)
    exclude_current_positions_from_observation = True
    forced_episode_end = True
    forward_reward_weight = 1.0
    frame_skip = 5
    # remove additional penalization for the magnitude of the actions since this would have a negative impact on agent's adaption capability.
    # E.g., when certain wind-friction is acting against the robot it needs to apply higher magnitude actions which will be penalized. 
    # Not sure, try with smaller as in the meta-learning non-stationary benchmark: https://arxiv.org/pdf/1803.11347.pdf Apeendix D: Reward Functions
    ctrl_cost_weight = 0.05
    reset_noise_scale = 0.1
    max_episode_steps = 1000
    target_velocity = 1.5 
    default_input_shape = 17

    # non-stationary general config parameters
    observe_non_stationarity = False
    use_obs_velocity = True
    observe_target_velocity = False
    change_phase = None

    # non-stationarity w.r.t. wind-friction changes
    average_wind_friction = 0.0
    amplitude_wind_friction = 10.0
    frequency_wind_friction_1 = 0.001
    frequency_wind_friction_2 = 0.001
    
    # non-stationarity w.r.t velocity changes
    average_velocity = 0.0
    amplitude_velocity = 10.0    
    frequency_velocity_1 = 0.001
    frequency_velocity_2 = 0.001


class GymMBRLEnv(gym.Env):

    def __init__(self,
                 seed,
                 config: Union[GymConfigs, GymNonStationaryInter],):


        super(GymMBRLEnv, self).__init__()

        self._seed = seed
        self._env = config.env
        self._action_repeat = self._env.default_action_repeat if config.action_repeat < 0 else config.action_repeat
        self._img_size = config.img_size
        self._current_step = 0
        self._symbolic = config.obs_type == "proprio"
        self._env_type = config.env_type

        match config.env_type:
            case "gym_non_stationary_intra" | "gym_non_stationary_inter":
                self._env = make_custom(config)

            case "gym_stationary":
                if self._symbolic:
                    print(f"CREATE PROPRIO GYM STATIONARY ENV: {config.env}")
                    if config.forced_episode_end:
                        self._env = gym.make(config.env, terminate_when_unhealthy=True)
                    else:
                        if config.env in FORCED_EPISODE_END_GYM_ENVIRONMENTS:
                            self._env = gym.make(config.env, terminate_when_unhealthy=False)
                        else:
                            self._env = gym.make(config.env) 
                            
                elif config.forced_episode_end:
                    self._env = gym.make(config.env, render_mode="rgb_array", terminate_when_unhealthy=True) 
                else:
                    if config.env in FORCED_EPISODE_END_GYM_ENVIRONMENTS:
                        self._env = gym.make(config.env, render_mode="rgb_array", terminate_when_unhealthy=False) 
                    else:
                        self._env = gym.make(config.env, render_mode="rgb_array") 

                self.action_space.seed(seed=config.seed)
            case _:
                raise TypeError(f"Environment type does not exist: {config.env_type}")

        self.action_space.seed(seed=config.seed)

    # use instead of get_obs -> ResizeImg and NormalizeImg wrappers from gymnasium

    def reset(self, seed=0, options=None):
        self._current_step = 0
        observation, info = self._env.reset(seed=self._seed, options=options)
        #print(f"OBSERVATION FROM RESET ENV: {observation}")
        #observation = observation[0].astype(np.float32)
        #print(f"OBSERVATION FROM RESET ENV AFTER TYPE CONVERTION: {observation}")

        if not self._symbolic:
            observation = self.render()
            observation = cv2.resize(
                    observation, (64, 64), interpolation=cv2.INTER_AREA
                ),
            observation = observation[0].astype(np.float32)

        key_obs = "image" if self._symbolic == False else "proprio"
        return {key_obs: observation, "is_first": True, "is_terminal": False}
        
    def step(self, action: np.ndarray):
        reward = 0.0
        self._current_step += 1

        if not isinstance(action, np.ndarray):
            action = np.array(action)

        for k in range(self._action_repeat):
            observation, reward_k, terminated, truncated, info  = self._env.step(action)
            reward += reward_k

            if terminated or truncated:
                break

        if not self._symbolic:
            observation = self.render()

            observation = cv2.resize(
                    observation, (64, 64), interpolation=cv2.INTER_AREA
                ),

            observation = observation[0].astype(np.float32)
            
        key_obs = "image" if self._symbolic == False else "proprio"
        return {key_obs: observation, "is_first": False, "is_terminal": terminated or truncated}, reward, terminated or truncated, info

    @property
    def obs_are_images(self) -> list[bool]:
        if not self._symbolic:
            return [True]
        else:
            return [False]

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0].float() if self._symbolic else (64, 64, 3)

    @property
    def action_dim(self):
        return self._env.action_space.shape[0]

    def _get_img_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)


    @property
    def observation_spec(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self.observation_size, dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def observation_space(self):
        if not self._symbolic:
            return gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        else: 
            return self._env.observation_space

    @property
    def state_space(self):
        return NotImplemented


    @property
    def action_space(self):
        return gym.spaces.Box(low=self._env.action_space.low,
                              high=self._env.action_space.high,
                              shape=self._env.action_space.shape,
                              dtype=float)

    @property
    def max_seq_length(self) -> int:
        return 1000 // self._action_repeat
    
    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


class GymEnvFactory:

    def build(seed, config: GymConfigs):

        base_env = GymMBRLEnv(seed, config)
        if config.obs_type == "image":
            #base_env = PixelNormalization(base_env) 
            #base_env = gym.wrappers.ResizeObservation(base_env, config.img_size)
            #base_env = ChannelFirstEnv(base_env)
            pass
        #base_env = OldGymInterface(base_env)
        #base_env = TorchEnv(base_env)

        return base_env

def make_custom(config):
    print(f"ENVIRONMENT TYPE: {config.env}")
    match config.env:
        case "HalfCheetahWithinEpisodeSineWindEnv":
            env = HalfCheetahWithinEpisodeSineWindEnv(config)
        case "HalfCheetahWithinEpisodeSineVelEnv":
            env = HalfCheetahWithinEpisodeSineVelEnv(config)
        case "HalfCheetahWithinEpisodeSineWindVelEnv":
            env = HalfCheetahWithinEpisodeSineWindVelEnv(config)
        case "HalfCheetahAcrossEpisodeSineWindEnv":
            env = HalfCheetahAcrossEpisodeSineWindEnv(config)
        case "HalfCheetahAcrossEpisodeSineVelEnv":
            env = HalfCheetahAcrossEpisodeSineVelEnv(config)
        case "HalfCheetahAcrossEpisodeSineWindVelEnv":
            env = HalfCheetahAcrossEpisodeSineWindVelEnv(config)
        case "HalfCheetahAcrossEpisodeMixSineWindEnv":
            env = HalfCheetahAcrossEpisodeMixSineWindEnv(config)
        case "HalfCheetahAcrossEpisodeMixSineVelEnv":
            env = HalfCheetahAcrossEpisodeMixSineVelEnv(config)
        case "HalfCheetahAcrossEpisodeMixSineWindVelEnv":
            env = HalfCheetahAcrossEpisodeMixSineWindVelEnv(config)
        case "HalfCheetahAcrossEpisodeCrippleEnv":
            env = HalfCheetahAcrossEpisodeCrippleEnv(config)
        case _:
            TypeError("Environment type does not exist")
    
    return env