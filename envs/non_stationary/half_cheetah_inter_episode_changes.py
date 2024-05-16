from dataclasses import dataclass
import numpy as np
import wandb

import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box


@dataclass
class GymNonStationaryInter:
    env = ""
    env_type = "gym_non_stationary_inter"
    obs_type = ""
    action_repeat = 2  # env default
    symbolic = True
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


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class HalfCheetahAcrossEpisodeSineWindEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Inter-episode Wind Friction Change according to a sine function"""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self,
                config: GymNonStationaryInter,):
        self._xml_file = '/home/i53/student/gospodinov/dreamerv3-torch/envs/non_stationary/assets/half_cheetah_wind.xml'
        utils.EzPickle.__init__(self,
                                self._xml_file,
                                config.frame_skip,
                                DEFAULT_CAMERA_CONFIG,
                                config.forward_reward_weight,
                                config.ctrl_cost_weight,
                                config.reset_noise_scale,
                                config.exclude_current_positions_from_observation,
                                )

        self._seed = config.seed
        self._max_episode_steps = config.max_episode_steps
        self._forward_reward_weight = config.forward_reward_weight
        self._ctrl_cost_weight = config.ctrl_cost_weight
        self._reset_noise_scale = config.reset_noise_scale
        self._exclude_current_positions_from_observation = config.exclude_current_positions_from_observation
        self._use_obs_velocity = config.use_obs_velocity
        self._observe_target_velocity = config.observe_target_velocity
        # keep track of the number of steps in an episode used to initialize a time limit (self.max_episode_steps)
        self._env_step_counter = 0

        # Non-stationary
        self._change_phase = config.change_phase
        self._observe_non_stationarity = config.observe_non_stationarity
        # used to keep track of the number of steps in an episode so that sine function vary according to this at every step
        self._sine_step_counter = 0 
        self._wind_frc = 0.0
        self._avg_wind_frc = config.average_wind_friction
        self._amplitude_wind_frc = config.amplitude_wind_friction
        self._phase_wind_frc = np.random.randint(100)
        self._frequency_wind_frc = config.frequency_wind_friction_1 

        # FOR LOGGING -> REFACTOR THIS LATER!
        self._episode_forward_velocities = []
        self._log = False 

        # adapt observation shape according to user preferences
        default_input_shape = config.default_input_shape
        if self._observe_non_stationarity:
            default_input_shape += 1
        if self._observe_target_velocity:
            default_input_shape += 1
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                            shape=(default_input_shape,), dtype=np.float64)
        print('Step-wise Wind Friction Change as a sine function:\n', 
              f'Average wind frinction: {self._avg_wind_frc}\n', 
              f'Amplitude wind friction: {self._amplitude_wind_frc}\n',
              f'Frequency wind friction: {self._frequency_wind_frc}\n',
              f'Phase wind friction: {self._phase_wind_frc}\n',
              f"Change type wind friction frequency: {self._change_phase}\n",
              f"Observe non_stationarity: {self._observe_non_stationarity}\n")

        mujoco_env.MujocoEnv.__init__(self, self._xml_file, config.frame_skip, self.observation_space)
        self.exclude_wind_fric_from_action_space()

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self._env_step_counter += 1

        # As wind-force is realized as an additional actuator, we need to include it into the action vector.
        action = np.concatenate([action, [self._wind_frc]])

        # DO NOT avoid calculating velocity using finite differences in positions as in the original implementation: 
        # https://github.com/Farama-Foundation/Gymnasium/blob/9c812af180150d7a16774c3bbbf24b7669b49f7b/gymnasium/envs/mujoco/half_cheetah_v5.py#L225. 
        # Taking the velocity directly from the data buffer assume that the velocity is constant during the simulation step which is wrong:
        # https://github.com/Farama-Foundation/Gymnasium/issues/1021
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_reward(x_velocity, action)
        truncated = True if self._env_step_counter >= self._max_episode_steps else False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            **reward_info,
        }
        self._episode_forward_velocities.append(x_velocity)

        # done=False as in this environment there is no forced episode end (e.g., episode end can only happen because of truncation).
        return observation, reward, False, truncated, info

    def reset(self, seed, options=None):
        return super().reset(seed=seed, options=options)

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self._env_step_counter = 0

        self.set_state(qpos, qvel)

        match self._change_phase:
            case "incremental":
                self._phase_wind_frc += 1
            case "random":
                self._phase_wind_frc = np.random.randint(100)
            case _:
                pass 

        self.reset_task()
        self._sine_step_counter += 1
        self.log()

        return self._get_obs()

    def _get_reward(self, x_velocity, action):
        forward_reward = self._forward_reward_weight * x_velocity 
        # Do not penalize the agent for the additional action dimension used for the wind force (realized as an actuator).
        ctrl_cost = self.control_cost(action[:-1])

        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return reward, reward_info

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        if self._use_obs_velocity:
            if self._observe_non_stationarity:
                observation = np.concatenate((position, velocity, [self._wind_frc])).ravel()
            else:
                observation = np.concatenate((position, velocity)).ravel()
        else:
            if self._observe_non_stationarity:
                observation = np.concatenate((position, [self._wind_frc])).ravel()
            else:
                observation = position.ravel()

        return observation

    def reset_task(self):
        """Change the wind friction magnitude across episodes.
        The involvement over time is discribed as a sine wave."""
        self._wind_frc = self._amplitude_wind_frc * np.sin(self._frequency_wind_frc * self._sine_step_counter + self._phase_wind_frc) + self._avg_wind_frc 

    def seed(self, seed=0):
        self._seed = seed

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def exclude_wind_fric_from_action_space(self):
        """The additional actuator used to realize the wind friction should not be part of the action space of the agent!"""
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)[:-1]
        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=np.float32)

    def log(self):
        if self._log:
            wandb.log({"forward_velocity": np.array(self._episode_forward_velocities).mean(),
                       "wind_friction": self._wind_frc,})
            self._episode_forward_velocities = []

    def set_logging(self, logging):
        self._log = logging
        print(f"WANDB LOGGING CHANGED TO: {logging}!")


class HalfCheetahAcrossEpisodeSineVelEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Inter-episode Velocity Change according to a sine function"""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self,
                config: GymNonStationaryInter,):
        self._xml_file = '/home/i53/student/gospodinov/dreamerv3-torch/envs/non_stationary/assets/half_cheetah.xml'
        utils.EzPickle.__init__(self,
                                self._xml_file,
                                config.frame_skip,
                                DEFAULT_CAMERA_CONFIG,
                                config.forward_reward_weight,
                                config.ctrl_cost_weight,
                                config.reset_noise_scale,
                                config.exclude_current_positions_from_observation,
                                )

        self._seed = config.seed
        self._max_episode_steps = config.max_episode_steps
        self._forward_reward_weight = config.forward_reward_weight
        self._ctrl_cost_weight = config.ctrl_cost_weight
        self._reset_noise_scale = config.reset_noise_scale
        self._exclude_current_positions_from_observation = config.exclude_current_positions_from_observation
        self._use_obs_velocity = config.use_obs_velocity
        self._observe_target_velocity = config.observe_target_velocity
        # keep track of the number of steps in an episode used to initialize a time limit (self.max_episode_steps)
        self._env_step_counter = 0

        # Non-stationary
        self._change_phase = config.change_phase
        self._observe_non_stationarity = config.observe_non_stationarity
        # used to keep track of the number of steps in an episode so that sine function vary according to this at every step
        self._sine_step_counter = 0 
        # The initial target velocity, later on the target velocity is changed according to a sine function
        self._target_vel = config.target_velocity
        # the phase of the wind friction function set randomly
        self._phase_vel = np.random.randint(100)
        # this changes the middle line of the sine function
        self._avg_vel = config.average_velocity
        self._amplitude_vel = config.amplitude_velocity
        self._frequency_vel = config.frequency_velocity_1

        # FOR LOGGING -> REFACTOR THIS LATER!
        self._episode_forward_velocities = []
        self._log = False 

        # adapt observation shape according to user preferences
        default_input_shape = config.default_input_shape
        if self._observe_non_stationarity:
            default_input_shape += 1
        if self._observe_target_velocity:
            default_input_shape += 1
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                            shape=(default_input_shape,), dtype=np.float64)
        print("Step-wise Velocity Change:\n",
              f"Target velocity: {self._target_vel}\n", 
              f"Average velocity: {self._avg_vel}\n", 
              f"Amplitude velocity: {self._amplitude_vel}\n",
              f"Velocity frequency: {self._frequency_vel}\n",
              f"Change type velocity frequency: {self._change_phase}\n",
              f"Observe non_stationarity: {self._observe_non_stationarity}\n")

        mujoco_env.MujocoEnv.__init__(self, self._xml_file, config.frame_skip, self.observation_space)

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self._env_step_counter += 1

        # DO NOT avoid calculating velocity using finite differences in positions as in the original implementation: 
        # https://github.com/Farama-Foundation/Gymnasium/blob/9c812af180150d7a16774c3bbbf24b7669b49f7b/gymnasium/envs/mujoco/half_cheetah_v5.py#L225. 
        # Taking the velocity directly from the data buffer assume that the velocity is constant during the simulation step which is wrong:
        # https://github.com/Farama-Foundation/Gymnasium/issues/1021
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_reward(x_velocity, action)
        truncated = True if self._env_step_counter >= self._max_episode_steps else False
        info = {
            'x_position': self.data.qpos[0],
            'x_velocity': x_velocity,
            **reward_info,
        }
        self._episode_forward_velocities.append(x_velocity)

        # done=False as in this environment there is no forced episode end (e.g., episode end can only happen because of truncation).
        return observation, reward, False, truncated, info

    def reset(self, seed, options=None):
        return super().reset(seed=seed, options=options)

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self._env_step_counter = 0

        self.set_state(qpos, qvel)

        match self._change_phase:
            case "incremental":
                self._phase_vel += 1
            case "random":
                self._phase_vel = np.random.randint(100)
            case _:
                pass

        self.reset_task()
        self._sine_step_counter += 1
        self.log()

        return self._get_obs()

    def _get_reward(self, x_velocity, action):
        forward_reward = -self._forward_reward_weight * abs(x_velocity - self._target_vel) 
        ctrl_cost = self.control_cost(action)
        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return reward, reward_info

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        if self._use_obs_velocity:
            if self._observe_non_stationarity:
                observation = np.concatenate((position, velocity, [self._target_vel])).ravel()
            else:
                observation = np.concatenate((position, velocity)).ravel()
        else:
            if self._observe_non_stationarity:
                observation = np.concatenate((position, [self._target_vel])).ravel()
            else:
                observation = position.ravel()

        return observation

    def reset_task(self):
        """Change the target velocity across episodes.
        The involvement over time is discribed as a sine wave."""
        self._target_vel = self._amplitude_vel * np.sin(self._frequency_vel * self._sine_step_counter + self._phase_vel) + self._avg_vel

    def seed(self, seed=0):
        self._seed = seed

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def get_task_state(self):
        return self._target_vel

    def log(self):
        if self._log:
            wandb.log({"forward_velocity": np.array(self._episode_forward_velocities).mean(),
                       "target_velocity": self._target_vel,})
            self._episode_forward_velocities = []

    def set_logging(self, logging):
        self._log = logging
        print(f"WANDB LOGGING CHANGED TO: {logging}!")


class HalfCheetahAcrossEpisodeSineWindVelEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Inter-episode Velocity Change and Wind friction change according to separate sine functions"""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self,
                config: GymNonStationaryInter,):
        self._xml_file = '/home/i53/student/gospodinov/dreamerv3-torch/envs/non_stationary/assets/half_cheetah_wind.xml'        
        utils.EzPickle.__init__(self,
                                self._xml_file,
                                config.frame_skip,
                                DEFAULT_CAMERA_CONFIG,
                                config.forward_reward_weight,
                                config.ctrl_cost_weight,
                                config.reset_noise_scale,
                                config.exclude_current_positions_from_observation,
                                )

        self._seed = config.seed
        self._max_episode_steps = config.max_episode_steps
        self._forward_reward_weight = config.forward_reward_weight
        self._ctrl_cost_weight = config.ctrl_cost_weight
        self._reset_noise_scale = config.reset_noise_scale
        self._exclude_current_positions_from_observation = config.exclude_current_positions_from_observation
        self._use_obs_velocity = config.use_obs_velocity
        self._observe_target_velocity = config.observe_target_velocity
        # keep track of the number of steps in an episode used to initialize a time limit (self.max_episode_steps)
        self._env_step_counter = 0

        # Non-stationary parameters
        self._change_phase = config.change_phase
        self._observe_non_stationarity = config.observe_non_stationarity
        # used to keep track of the number of steps in an episode so that sine function vary according to this at every step
        self._sine_step_counter = 0

        # Wind friction
        self._wind_frc = 0.0
        self._avg_wind_frc = config.average_wind_friction
        self._amplitude_wind_frc = config.amplitude_wind_friction
        self._frequency_wind_frc = config.frequency_wind_friction_1
        self._phase_wind_frc = np.random.randint(100)

        # Target velocity
        self._target_vel = config.target_velocity
        self._avg_vel = config.average_velocity
        self._amplitude_vel = config.amplitude_velocity
        self._frequency_vel = config.frequency_velocity_1
        self._phase_vel = np.random.randint(100) 

        print(f"Step-wise Wind Friction and Velocity Change changed as a sine wave function\n", 
              f"Change: Average wind frinction: {self._avg_wind_frc}\n", 
              f"Amplitude wind friction: {self._amplitude_wind_frc}\n", 
              f"Frequency wind friction: {self._frequency_wind_frc}\n",
              f"Target velocity: {self._target_vel}\n", 
              f"Average velocity: {self._avg_vel}\n", 
              f"Amplitude velocity: {self._amplitude_vel}\n",
              f"Frequency velocity: {self._frequency_vel}\n",
              f"Change type frequencies: {self._change_phase}\n",
              f"Observe non_stationarity: {self._observe_non_stationarity}\n")

        # FOR LOGGING -> REFACTOR THIS LATER!
        self._episode_forward_velocities = []
        self._log = False 

        default_input_shape = config.default_input_shape
        # In this setting, we observe the wind_frc as well as the target velocity as non-stationarity
        if self._observe_non_stationarity:
            default_input_shape += 2
        if self._observe_target_velocity:
            default_input_shape += 1
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                            shape=(default_input_shape,), dtype=np.float64)
        
        mujoco_env.MujocoEnv.__init__(self, self._xml_file, config.frame_skip, self.observation_space)
        self.exclude_wind_fric_from_action_space()

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self._env_step_counter += 1

        # As wind-force is realized as an additional actuator, we need to include it into the action vector.
        action = np.concatenate([action, [self._wind_frc]])

        # DO NOT avoid calculating velocity using finite differences in positions as in the original implementation: 
        # https://github.com/Farama-Foundation/Gymnasium/blob/9c812af180150d7a16774c3bbbf24b7669b49f7b/gymnasium/envs/mujoco/half_cheetah_v5.py#L225. 
        # Taking the velocity directly from the data buffer assume that the velocity is constant during the simulation step which is wrong:
        # https://github.com/Farama-Foundation/Gymnasium/issues/1021
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        # Do not penalize the agent for the additional action dimension used for the wind force (realized as an actuator).
        observation = self._get_obs()
        reward, reward_info = self._get_reward(x_velocity, action)
        truncated = True if self._env_step_counter >= self._max_episode_steps else False
        x_position_after = self.data.qpos[0]
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            **reward_info,
        }
        self._episode_forward_velocities.append(x_velocity)

        # done=False as in this environment there is no forced episode end (e.g., episode end can only happen because of truncation).
        return observation, reward, False, truncated, info

    def reset(self, seed, options=None):
        return super().reset(seed=seed, options=options)

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self._env_step_counter = 0

        self.set_state(qpos, qvel)

        match self._change_phase:
            case "incremental":
                self._phase_wind_frc += 1
                self._phase_vel += 1
            case "random":
                self._phase_wind_frc = np.random.randint(100)
                self._phase_vel = np.random.randint(100)             
            case _:
                pass   

        self.reset_task()
        self._sine_step_counter += 1
        self.log()

        return self._get_obs()

    def _get_reward(self, x_velocity, action):
        forward_reward = -self._forward_reward_weight * abs(x_velocity - self._target_vel) 
        # Do not penalize the agent for the additional action dimension used for the wind force (realized as an actuator).
        ctrl_cost = self.control_cost(action[:-1])
        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return reward, reward_info

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def reset_task(self):
        """Change the wind friction magnitude and target velocity across episodes.
        The involvements over time are discribed as a sine wave."""        
        self._wind_frc = self._amplitude_wind_frc * np.sin(self._frequency_wind_frc * self._sine_step_counter + self._phase_wind_frc) + self._avg_wind_frc
        self._target_vel = self._amplitude_vel * np.sin(self._frequency_vel * self._sine_step_counter + self._phase_vel) + self._avg_vel

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        if self._use_obs_velocity:
            if self._observe_non_stationarity:
                observation = np.concatenate((position, velocity, [self._wind_frc, self._target_vel])).ravel()
            else:
                observation = np.concatenate((position, velocity)).ravel()
        else:
            if self._observe_non_stationarity:
                observation = np.concatenate((position, [self._wind_frc, self._target_vel])).ravel()
            else:
                observation = position.ravel()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def get_task_state(self):
        return [self._wind_frc, self._target_vel]

    def exclude_wind_fric_from_action_space(self):
        """The additional actuator used to realize the wind friction should not be part of the action space of the agent!"""
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)[:-1]
        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=np.float32)

    def log(self):
        if self._log:
            wandb.log({"forward_velocity": np.array(self._episode_forward_velocities).mean(),
                       "wind_friction": self._wind_frc,
                       "target_velocity": self._target_vel,})
            self._episode_forward_velocities = []

    def set_logging(self, logging):
        self._log = logging
        print(f"WANDB LOGGING CHANGED TO: {logging}!")


class HalfCheetahAcrossEpisodeMixSineWindEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Inter-episode Wind Friction Change as a mix of two sine functions. """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self,
                config: GymNonStationaryInter):
        self._xml_file = '/home/i53/student/gospodinov/dreamerv3-torch/envs/non_stationary/assets/half_cheetah_wind.xml'
        utils.EzPickle.__init__(self,
                                self._xml_file,
                                config.frame_skip,
                                DEFAULT_CAMERA_CONFIG,
                                config.forward_reward_weight,
                                config.ctrl_cost_weight,
                                config.reset_noise_scale,
                                config.exclude_current_positions_from_observation,
                                )

        self._seed = config.seed
        self._max_episode_steps = config.max_episode_steps
        self._forward_reward_weight = config.forward_reward_weight
        self._ctrl_cost_weight = config.ctrl_cost_weight
        self._reset_noise_scale = config.reset_noise_scale
        self._exclude_current_positions_from_observation = config.exclude_current_positions_from_observation
        self._use_obs_velocity = config.use_obs_velocity
        self._observe_target_velocity = config.observe_target_velocity
        # keep track of the number of steps in an episode used to initialize a time limit (self.max_episode_steps)
        self._env_step_counter = 0

        # Non-stationary parameters
        self._change_phase = config.change_phase
        self._observe_non_stationarity = config.observe_non_stationarity
        # used to keep track of the number of steps in an episode so that sine function vary according to this at every step
        self._sine_step_counter = 0 
        self._wind_frc = 0.0
        self._avg_wind_frc = config.average_wind_friction 
        self._amplitude_wind_frc = config.amplitude_wind_friction
        self._phase_wind_frc_1 = np.random.randint(100)
        self._phase_wind_frc_2 = np.random.randint(100)
        self._frequency_wind_frc_1 = config.frequency_wind_friction_1 
        self._frequency_wind_frc_2 = config.frequency_wind_friction_2 

        # adapt observation shape according to user preferences
        default_input_shape = config.default_input_shape
        if self._observe_non_stationarity:
            default_input_shape += 1
        if self._observe_target_velocity:
            default_input_shape += 1
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                            shape=(default_input_shape,), dtype=np.float64)
        
        # FOR LOGGING -> REFACTOR THIS LATER!
        self._episode_forward_velocities = []
        self._log = False 

        print('Step-wise Wind Friction Change as a mix of two sine functions:\n', 
              f'Average wind frinction: {self._avg_wind_frc}\n', 
              f'Amplitude wind friction: {self._amplitude_wind_frc}\n',
              f'Frequency wind friction 1: {self._frequency_wind_frc_1}\n',
              f'Frequency wind friction 2: {self._frequency_wind_frc_2}\n',
              f'Phase wind friction 1: {self._phase_wind_frc_1}\n',
              f'Phase wind friction 2: {self._phase_wind_frc_2}\n',
              f"Change type wind friction frequency: {self._change_phase}\n",
              f"Observe non_stationarity: {self._observe_non_stationarity}\n")

        mujoco_env.MujocoEnv.__init__(self, self._xml_file, config.frame_skip, self.observation_space)
        self.exclude_wind_fric_from_action_space()

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self._env_step_counter += 1

        # As wind-force is realized as an additional actuator, we need to include it into the action vector.
        action = np.concatenate([action, [self._wind_frc]])

        # DO NOT avoid calculating velocity using finite differences in positions as in the original implementation: 
        # https://github.com/Farama-Foundation/Gymnasium/blob/9c812af180150d7a16774c3bbbf24b7669b49f7b/gymnasium/envs/mujoco/half_cheetah_v5.py#L225. 
        # Taking the velocity directly from the data buffer assume that the velocity is constant during the simulation step which is wrong:
        # https://github.com/Farama-Foundation/Gymnasium/issues/1021
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_reward(x_velocity, action)
        truncated = True if self._env_step_counter >= self._max_episode_steps else False 
        info = {
            'x_position': self.data.qpos[0],
            'x_velocity': x_velocity,
            **reward_info,
        }
        self._episode_forward_velocities.append(x_velocity)

        # done=False as in this environment there is no forced episode end (e.g., episode end can only happen because of truncation).
        return observation, reward, False, truncated, info

    def reset(self, seed, options=None):
        return super().reset(seed=seed, options=options)

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self._env_step_counter = 0

        self.set_state(qpos, qvel)

        match self._change_phase:
            case "incremental":
                self._phase_wind_frc_1 += 1
                self._phase_wind_frc_2 += 1
            case "random":
                self._phase_wind_frc_1 = np.random.randint(100)
                self._phase_wind_frc_2 = np.random.randint(100)            
            case _:
                pass 

        self.reset_task()
        self._sine_step_counter += 1
        self.log() 

        return self._get_obs()

    def _get_reward(self, x_velocity, action):
        forward_reward = self._forward_reward_weight * x_velocity 
        # Do not penalize the agent for the additional action dimension used for the wind force (realized as an actuator).
        ctrl_cost = self.control_cost(action[:-1])
        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return reward, reward_info

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        if self._use_obs_velocity:
            if self._observe_non_stationarity:
                observation = np.concatenate((position, velocity, [self._wind_frc])).ravel()
            else:
                observation = np.concatenate((position, velocity)).ravel()
        else:
            if self._observe_non_stationarity:
                observation = np.concatenate((position, [self._wind_frc])).ravel()
            else:
                observation = position.ravel()

        return observation

    def reset_task(self):
        """Change the wind friction magnitude across episodes.
        The evolvement over time is discribed as a mix of two sine waves."""
        self._wind_frc = self._amplitude_wind_frc * np.sin(self._frequency_wind_frc_1 * self._sine_step_counter + self._phase_wind_frc_1) + self._avg_wind_frc + \
                         self._amplitude_wind_frc * np.sin(self._frequency_wind_frc_2 * self._sine_step_counter + self._phase_wind_frc_2) + self._avg_wind_frc

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def get_task_state(self):
        return self._wind_frc

    def seed(self, seed=0):
        self._seed = seed

    def exclude_wind_fric_from_action_space(self):
        """The additional actuator used to realize the wind friction should not be part of the action space of the agent!"""
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)[:-1]
        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=np.float32)

    def log(self):
        if self._log:
            wandb.log({"forward_velocity": np.array(self._episode_forward_velocities).mean(),
                       "wind_friction": self._wind_frc,})
            self._episode_forward_velocities = []

    def set_logging(self, logging):
        self._log = logging
        print(f"WANDB LOGGING CHANGED TO: {logging}!")


class HalfCheetahAcrossEpisodeMixSineVelEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Inter-episode Velocity Change according to a mix of two sine functions"""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self,
                config: GymNonStationaryInter,):
        self._xml_file = '/home/i53/student/gospodinov/dreamerv3-torch/envs/non_stationary/assets/half_cheetah.xml'
        utils.EzPickle.__init__(self,
                                self._xml_file,
                                config.frame_skip,
                                DEFAULT_CAMERA_CONFIG,
                                config.forward_reward_weight,
                                config.ctrl_cost_weight,
                                config.reset_noise_scale,
                                config.exclude_current_positions_from_observation,
                                )

        self._seed = config.seed
        self._max_episode_steps = config.max_episode_steps
        self._forward_reward_weight = config.forward_reward_weight
        self._ctrl_cost_weight = config.ctrl_cost_weight
        self._reset_noise_scale = config.reset_noise_scale
        self._exclude_current_positions_from_observation = config.exclude_current_positions_from_observation
        self._use_obs_velocity = config.use_obs_velocity
        self._observe_target_velocity = config.observe_target_velocity
        # keep track of the number of steps in an episode used to initialize a time limit (self.max_episode_steps)
        self._env_step_counter = 0

        # Non-stationary
        self._change_phase = config.change_phase
        self._observe_non_stationarity = config.observe_non_stationarity
        # used to keep track of the number of steps in an episode so that sine function vary according to this at every step
        self._sine_step_counter = 0 
        # The initial target velocity, later on the target velocity is changed according to a sine function
        self._target_vel = config.target_velocity
        # this changes the middle line of the sine function
        self._avg_vel = config.average_velocity
        self._amplitude_vel = config.amplitude_velocity
        # the phase of each wind friction function set randomly
        self._phase_vel_1 = np.random.randint(100)
        self._phase_vel_2 = np.random.randint(100)
        # the frequencies of each wind friction function
        self._frequency_vel_1 = config.frequency_velocity_1
        self._frequency_vel_2 = config.frequency_velocity_2

        # adapt observation shape according to user preferences
        default_input_shape = config.default_input_shape
        if self._observe_non_stationarity:
            default_input_shape += 1
        if self._observe_target_velocity:
            default_input_shape += 1
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                            shape=(default_input_shape,), dtype=np.float64)
        
        # FOR LOGGING -> REFACTOR THIS LATER!
        self._episode_forward_velocities = []
        self._log = False 

        print("Step-wise Velocity Change as a mix of two sine functions:\n",
              f"Target velocity: {self._target_vel}\n", 
              f"Average velocity: {self._avg_vel}\n", 
              f"Amplitude velocity: {self._amplitude_vel}\n",
              f"Velocity frequency 1: {self._frequency_vel_1}\n",
              f"Velocity frequency 2: {self._frequency_vel_2}\n",
              f"Phase frequency 1: {self._phase_vel_1}\n",
              f"Phase frequency 2: {self._phase_vel_2}\n",
              f"Change type velocity frequency: {self._change_phase}\n",
              f"Observe non_stationarity: {self._observe_non_stationarity}\n")

        mujoco_env.MujocoEnv.__init__(self, self._xml_file, config.frame_skip, self.observation_space)

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self._env_step_counter += 1

        # DO NOT avoid calculating velocity using finite differences in positions as in the original implementation: 
        # https://github.com/Farama-Foundation/Gymnasium/blob/9c812af180150d7a16774c3bbbf24b7669b49f7b/gymnasium/envs/mujoco/half_cheetah_v5.py#L225. 
        # Taking the velocity directly from the data buffer assume that the velocity is constant during the simulation step which is wrong:
        # https://github.com/Farama-Foundation/Gymnasium/issues/1021
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_reward(x_velocity, action)
        truncated = True if self._env_step_counter >= self._max_episode_steps else False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            **reward_info,
        }
        self._episode_forward_velocities.append(x_velocity)

        # done=False as in this environment there is no forced episode end (e.g., episode end can only happen because of truncation).
        return observation, reward, False, truncated, info

    def reset(self, seed, options=None):
        return super().reset(seed=seed, options=options)

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self._env_step_counter = 0

        self.set_state(qpos, qvel)

        match self._change_phase:
            case "incremental":
                self._phase_vel_1 += 1
                self._phase_vel_2 += 1
            case "random":
                self._phase_vel_1 = np.random.randint(100)      
                self._phase_vel_2 = np.random.randint(100)
            case _:
                pass

        self.reset_task()
        self._sine_step_counter += 1
        self.log()

        return self._get_obs()

    def _get_reward(self, x_velocity, action):
        forward_reward = -self._forward_reward_weight * abs(x_velocity - self._target_vel) 
        ctrl_cost = self.control_cost(action)
        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return reward, reward_info

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        if self._use_obs_velocity:
            if self._observe_non_stationarity:
                observation = np.concatenate((position, velocity, [self._target_vel])).ravel()
            else:
                observation = np.concatenate((position, velocity)).ravel()
        else:
            if self._observe_non_stationarity:
                observation = np.concatenate((position, [self._target_vel])).ravel()
            else:
                observation = position.ravel()

        return observation

    def reset_task(self):
        """Change the target value magnitude across episodes.
        The involvement over time is discribed as a mix of two sine waves."""

        self._target_vel = self._amplitude_vel * np.sin(self._frequency_vel_1 * self._sine_step_counter + self._phase_vel_1) + self._avg_vel + \
                           self._amplitude_vel * np.sin(self._frequency_vel_2 * self._sine_step_counter + self._phase_vel_2) + self._avg_vel

    def seed(self, seed=0):
        self._seed = seed

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def get_task_state(self):
        return self._target_vel

    def log(self):
        if self._log:
            wandb.log({"forward_velocity": np.array(self._episode_forward_velocities).mean(),
                       "target_velocity": self._target_vel,})
            self._episode_forward_velocities = []

    def set_logging(self, logging):
        self._log = logging
        print(f"WANDB LOGGING CHANGED TO: {logging}!")


class HalfCheetahAcrossEpisodeMixSineWindVelEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Inter-episode Velocity Change and Wind friction change according to two separate sine functions for each change"""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self,
                config: GymNonStationaryInter,):
        self._xml_file = '/home/i53/student/gospodinov/dreamerv3-torch/envs/non_stationary/assets/half_cheetah_wind.xml'        
        utils.EzPickle.__init__(self,
                                self._xml_file,
                                config.frame_skip,
                                DEFAULT_CAMERA_CONFIG,
                                config.forward_reward_weight,
                                config.ctrl_cost_weight,
                                config.reset_noise_scale,
                                config.exclude_current_positions_from_observation,
                                )

        self._seed = config.seed
        self._max_episode_steps = config.max_episode_steps
        self._forward_reward_weight = config.forward_reward_weight
        self._ctrl_cost_weight = config.ctrl_cost_weight
        self._reset_noise_scale = config.reset_noise_scale
        self._exclude_current_positions_from_observation = config.exclude_current_positions_from_observation
        self._use_obs_velocity = config.use_obs_velocity
        self._observe_target_velocity = config.observe_target_velocity
        # keep track of the number of steps in an episode used to initialize a time limit (self.max_episode_steps)
        self._env_step_counter = 0

        # Non-stationary parameters
        self._change_phase = config.change_phase
        self._observe_non_stationarity = config.observe_non_stationarity
        # used to keep track of the number of steps in an episode so that sine function vary according to this at every step
        self._sine_step_counter = 0

        # Wind friction
        self._wind_frc = 0.0
        self._avg_wind_frc = config.average_wind_friction
        self._amplitude_wind_frc = config.amplitude_wind_friction
        self._frequency_wind_frc_1 = config.frequency_wind_friction_1
        self._frequency_wind_frc_2 = config.frequency_wind_friction_2
        self._phase_wind_frc_1 = np.random.randint(100)
        self._phase_wind_frc_2 = np.random.randint(100)

        # Target velocity
        self._target_vel = config.target_velocity
        self._avg_vel = config.average_velocity
        self._amplitude_vel = config.amplitude_velocity
        self._frequency_vel_1 = config.frequency_velocity_1
        self._frequency_vel_2 = config.frequency_velocity_2
        self._phase_vel_1 = np.random.randint(100) 
        self._phase_vel_2 = np.random.randint(100) 

        # FOR LOGGING -> REFACTOR THIS LATER!
        self._episode_forward_velocities = []
        self._log = False 

        print(f"Step-wise Wind Friction and Velocity, both changed using a mix of two sine functions\n", 
              f"Change: Average wind frinction: {self._avg_wind_frc}\n", 
              f"Amplitude wind friction: {self._amplitude_wind_frc}\n", 
              f"Frequency wind friction 1: {self._frequency_wind_frc_1}\n",
              f"Frequency wind friction 2: {self._frequency_wind_frc_2}\n",
              f"Phase wind friction 1: {self._phase_wind_frc_1}\n",
              f"Phase wind friction 2: {self._phase_wind_frc_2}\n",
              f"Target velocity: {self._target_vel}\n", 
              f"Average velocity: {self._avg_vel}\n", 
              f"Amplitude velocity: {self._amplitude_vel}\n",
              f"Frequency velocity 1: {self._frequency_vel_1}\n",
              f"Frequency velocity 2: {self._frequency_vel_2}\n",
              f"Phase velocity 1: {self._phase_vel_1}\n",
              f"Phase velocity 2: {self._phase_vel_2}\n",
              f"Change type frequencies: {self._change_phase}\n",
              f"Observe non_stationarity: {self._observe_non_stationarity}\n")

        default_input_shape = config.default_input_shape
        # In this setting, we observe the wind_frc as well as the target velocity as non-stationarity
        if self._observe_non_stationarity:
            default_input_shape += 2
        if self._observe_target_velocity:
            default_input_shape += 1
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                            shape=(default_input_shape,), dtype=np.float64)
        
        mujoco_env.MujocoEnv.__init__(self, self._xml_file, config.frame_skip, self.observation_space)
        self.exclude_wind_fric_from_action_space()

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self._env_step_counter += 1

        # As wind-force is realized as an additional actuator, we need to include it into the action vector.
        action = np.concatenate([action, [self._wind_frc]])

        # DO NOT avoid calculating velocity using finite differences in positions as in the original implementation: 
        # https://github.com/Farama-Foundation/Gymnasium/blob/9c812af180150d7a16774c3bbbf24b7669b49f7b/gymnasium/envs/mujoco/half_cheetah_v5.py#L225. 
        # Taking the velocity directly from the data buffer assume that the velocity is constant during the simulation step which is wrong:
        # https://github.com/Farama-Foundation/Gymnasium/issues/1021
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_reward(x_velocity, action)
        truncated = True if self._env_step_counter >= self._max_episode_steps else False
        x_position_after = self.data.qpos[0]
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            **reward_info,
        }
        self._episode_forward_velocities.append(x_velocity)

        # done=False as in this environment there is no forced episode end (e.g., episode end can only happen because of truncation).
        return observation, reward, False, truncated, info

    def reset(self, seed, options=None):
        return super().reset(seed=seed, options=options)

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self._env_step_counter = 0

        self.set_state(qpos, qvel)
        match self._change_phase:
            case "incremental":
                self._phase_wind_frc_1 += 1
                self._phase_wind_frc_2 += 1
                self._phase_vel_1 += 1
                self._phase_vel_2 += 1
            case "random":
                self._phase_wind_frc_1 = np.random.randint(100)
                self._phase_wind_frc_2 = np.random.randint(100)
                self._phase_vel_1 = np.random.randint(100)      
                self._phase_vel_2 = np.random.randint(100)             
            case _:
                pass   

        self.reset_task()
        self._sine_step_counter += 1
        self.log()

        observation = self._get_obs()
        return observation

    def _get_reward(self, x_velocity, action):
        forward_reward = -self._forward_reward_weight * abs(x_velocity - self._target_vel) 
        # Do not penalize the agent for the additional action dimension used for the wind force (realized as an actuator).
        ctrl_cost = self.control_cost(action[:-1])
        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return reward, reward_info

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        if self._use_obs_velocity:
            if self._observe_non_stationarity:
                observation = np.concatenate((position, velocity, [self._wind_frc, self._target_vel])).ravel()
            else:
                observation = np.concatenate((position, velocity)).ravel()
        else:
            if self._observe_non_stationarity:
                observation = np.concatenate((position, [self._wind_frc, self._target_vel])).ravel()
            else:
                observation = position.ravel()

        return observation

    def reset_task(self):
        """Change the wind friction magnitude and target velocity across episodes.
        The involvements over time are discribed as a mix of two sine waves for each change."""        
        self._wind_frc = self._amplitude_wind_frc * np.sin(self._frequency_wind_frc_1 * self._sine_step_counter + self._phase_wind_frc_1) + self._avg_wind_frc + \
                         self._amplitude_wind_frc * np.sin(self._frequency_wind_frc_2 * self._sine_step_counter + self._phase_wind_frc_2) + self._avg_wind_frc

        self._target_vel = self._amplitude_vel * np.sin(self._frequency_vel_1 * self._sine_step_counter + self._phase_vel_1) + self._avg_vel + \
                           self._amplitude_vel * np.sin(self._frequency_vel_2 * self._sine_step_counter + self._phase_vel_2) + self._avg_vel
        
    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def get_task_state(self):
        return [self._wind_frc, self._target_vel]

    def seed(self, seed=0):
        self._seed = seed

    def exclude_wind_fric_from_action_space(self):
        """The additional actuator used to realize the wind friction should not be part of the action space of the agent!"""
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)[:-1]
        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=np.float32)

    def log(self):
        if self._log:
            wandb.log({"forward_velocity": np.array(self._episode_forward_velocities).mean(),
                       "wind_friction": self._wind_frc,
                       "target_velocity": self._target_vel,})
            self._episode_forward_velocities = []

    def set_logging(self, logging):
        self._log = logging
        print(f"WANDB LOGGING CHANGED TO: {logging}!")


class HalfCheetahAcrossEpisodeCrippleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Inter-episode actuator disabling of the half cheetah model"""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self,
                config: GymNonStationaryInter,):
        self._xml_file = '/home/i53/student/gospodinov/dreamerv3-torch/envs/non_stationary/assets/half_cheetah.xml'
        utils.EzPickle.__init__(self,
                                self._xml_file,
                                config.frame_skip,
                                DEFAULT_CAMERA_CONFIG,
                                config.forward_reward_weight,
                                config.ctrl_cost_weight,
                                config.reset_noise_scale,
                                config.exclude_current_positions_from_observation,
                                )

        self._seed = config.seed
        self._max_episode_steps = config.max_episode_steps
        self._forward_reward_weight = config.forward_reward_weight
        self._ctrl_cost_weight = config.ctrl_cost_weight
        self._reset_noise_scale = config.reset_noise_scale
        self._exclude_current_positions_from_observation = config.exclude_current_positions_from_observation
        self._use_obs_velocity = config.use_obs_velocity
        self._observe_target_velocity = config.observe_target_velocity
        self._observe_non_stationarity = config.observe_non_stationarity
        # keep track of the number of steps in an episode used to initialize a time limit (self.max_episode_steps)
        self._env_step_counter = 0

        # FOR LOGGING -> REFACTOR THIS LATER!
        self._episode_forward_velocities = []
        self._log = False 

        # adapt observation shape according to user preferences
        default_input_shape = config.default_input_shape
        if self._observe_non_stationarity:
            default_input_shape += 1
        if self._observe_target_velocity:
            default_input_shape += 1
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                            shape=(default_input_shape,), dtype=np.float64)

        mujoco_env.MujocoEnv.__init__(self, self._xml_file, config.frame_skip, self.observation_space)

        # non-stationary related attributes. They must be defined after initialization of MujocoEnv.
        self._crippled_actuator_name_id = 6
        # Explicitly define the actuator names for easier access later. 
        # This is needed because the half-cheetah model is an underactuated model, all joints names (including the actuators and floor of the environment) are defined as follows: 
        # ['bfoot', 'bshin', 'bthigh', 'ffoot', 'floor', 'fshin', 'fthigh', 'head', 'torso']
        self._actuator_names = ['bfoot', 'bshin', 'bthigh', 'ffoot', 'fshin', 'fthigh']
        # Define a mask that will disable the action control w.r.t. a particular actuator
        self._disable_actuator_mask = np.ones(self.action_space.shape)
        # The color of all joints and environment floor: ['bfoot', 'bshin', 'bthigh', 'ffoot', 'floor', 'fshin', 'fthigh', 'head', 'torso']
        self._init_geom_rgba = self.model.geom_rgba.copy()

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self._env_step_counter += 1
        # Apply the mask that will disable the action control w.r.t. a particular actuator
        action = self._disable_actuator_mask * action

        # DO NOT avoid calculating velocity using finite differences in positions as in the original implementation: 
        # https://github.com/Farama-Foundation/Gymnasium/blob/9c812af180150d7a16774c3bbbf24b7669b49f7b/gymnasium/envs/mujoco/half_cheetah_v5.py#L225. 
        # Taking the velocity directly from the data buffer assume that the velocity is constant during the simulation step which is wrong:
        # https://github.com/Farama-Foundation/Gymnasium/issues/1021
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_reward(x_velocity, action)
        truncated = True if self._env_step_counter >= self._max_episode_steps else False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            **reward_info,
        }
        self._episode_forward_velocities.append(x_velocity)

        # done=False as in this environment there is no forced episode end (e.g., episode end can only happen because of truncation).
        return observation, reward, False, truncated, info

    def reset(self, seed, options=None):
        return super().reset(seed=seed, options=options)

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self._env_step_counter = 0
        self.set_state(qpos, qvel)
        #self.reset_task()
        self.log()

        return self._get_obs()

    def _get_reward(self, x_velocity, action):
        forward_reward = self._forward_reward_weight * x_velocity 
        ctrl_cost = self.control_cost(action)

        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return reward, reward_info

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        if self._use_obs_velocity:
            if self._observe_non_stationarity:
                observation = np.concatenate((position, velocity, [self._crippled_actuator_name_id])).ravel()
            else:
                observation = np.concatenate((position, velocity)).ravel()
        else:
            if self._observe_non_stationarity:
                observation = np.concatenate((position, [self._crippled_actuator_name_id])).ravel()
            else:
                observation = position.ravel()
        
        return observation

    def reset_task(self):
        """Choose one actuator randomly and change its color to red.
        If crippled_actuator_name_id, then no actuator gets disabled."""

        self._crippled_actuator_name_id = np.random.randint(0, len(self._actuator_names))
        if not self._crippled_actuator_name_id == len(self._actuator_names):
            self._crippled_actuator_name = self._actuator_names[self._crippled_actuator_name_id]
            print(f"Disable actuator: {self._crippled_actuator_name}")
            # DO NOT take joint/actuator id's from mujoco.geom()!!!
            # The actual id's used in MuJoCo are different and can be found by using the following helper function.
            crippled_actuator_mujoco_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self._crippled_actuator_name)

            # update actuator mask
            self._disable_actuator_mask = np.ones(self.action_space.shape)
            self._disable_actuator_mask[self._crippled_actuator_name_id] = 0.0
            # update actuator color
            geom_rgba = self._init_geom_rgba.copy()
            geom_rgba[crippled_actuator_mujoco_id, :3] = np.array([1, 0, 0])
            self.model.geom_rgba = geom_rgba

    def seed(self, seed=0):
        self._seed = seed

    def log(self):
        if self._log:
            wandb.log({"forward_velocity": np.array(self._episode_forward_velocities).mean()})
            self._episode_forward_velocities = []

    def set_logging(self, logging):
        self._log = logging
        print(f"WANDB LOGGING CHANGED TO: {logging}!")

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def unit_test(self):
        """By deactivation certain joints the corresponding actuator forcess and actuator forces in action space should be zero"""
        print('Actuator forces in action space:', self.data.actuator_force)
        print('Actuator forces:', self.data.qfrc_actuator)
        print(f"Actuator velocities: {self.data.actuator_velocity}")