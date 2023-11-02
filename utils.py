"""
################################################################################################################
################################################################################################################
This code is based on the MushroomRL code, which can be found at https://github.com/MushroomRL/mushroom-rl.#####
Modifications have been made by A. Ali. ########################################################################
################################################################################################################
################################################################################################################
"""
# Warning: Do not make any modifications to the code in "inverted_pendulum.xml" or "utils.py".

import time
import warnings
from enum import Enum

import glfw
import mujoco
import numpy as np
import scipy.stats as st


class Environment(object):
    @classmethod
    def register(cls):
        """
        The register function is used to register a new environment.

        :param cls: Pass the class object of the environment
        :return: The class it decorates
        """

        env_name = cls.__name__

        if env_name not in Environment._registered_envs:
            Environment._registered_envs[env_name] = cls

    @staticmethod
    def list_registered():
        """
        The list_registered function returns a list of the names of all registered environments.
        This is useful for printing out a list of available environments to the user.

        :return: A list of registered environments
        """

        return list(Environment._registered_envs.keys())

    @staticmethod
    def make(env_name, *args, **kwargs):
        """
        The make function is a convenience function that instantiates an environment
        object and returns the result of calling its generate method.  It takes either
        a single argument, which is the name of the environment to be created, or a
        list containing all arguments except for the first one.  The first argument can
        also be a module in which case it will look for an attribute matching that name.

        :param env_name: Specify the environment to be generated
        :param *args: Pass a non-keyworded, variable-length argument list
        :param **kwargs: Pass keyworded variable length of arguments to a function
        :return: A list of the environments that are generated
        """
        if "." in env_name:
            env_data = env_name.split(".")
            env_name = env_data[0]
            args = env_data[1:] + list(args)

        env = Environment._registered_envs[env_name]

        if hasattr(env, "generate"):
            return env.generate(*args, **kwargs)
        else:
            return env(*args, **kwargs)

    def __init__(self, mdp_info):
        """
        The __init__ function is called automatically when a new instance of the class is created.
        It initializes the attributes of an object, and sets up any default behavior that we want to happen automatically when a new object is created.


        :param self: Reference the class instance
        :param mdp_info: Store the mdp information
        :return: The mdpinfo object
        """

        self._mdp_info = mdp_info

    def seed(self, seed):
        """
        The seed function is used to set the random seed of the environment.
        This allows you to reproduce your results by setting a fixed seed and running
        the same code multiple times.  This is helpful, for example, if you are trying
        to determine whether an agent has learned a given behavior or if it's just luck.

        :param self: Access the attributes and methods of the class in which it is used
        :param seed: Set the random seed of numpy and torch
        :return: The seed that was set
        """

        if hasattr(self, "env") and hasattr(self.env, "seed"):
            self.env.seed(seed)
        else:
            warnings.warn(
                "This environment has no custom seed. "
                "The call will have no effect. "
                "You can set the seed manually by setting numpy/torch seed"
            )

    def reset(self, state=None):
        """
        The reset function is called at the beginning of each trial.

        :param self: Access the attributes and methods of the class in python
        :param state=None: Reset the environment to a random initial state
        :return: The initial state of the environment
        """

        raise NotImplementedError

    def step(self, action):
        """
        The step function should return the next state, reward, and done flag given an action.
        The action is a number from 0 to 8 indicating which of the 9 possible moves are being played.
        The step function should also update the internal Q-table.

        :param self: Access the attributes and methods of the class in python
        :param action: Pass the agent's action to the environment so that it can progress to the next state
        :return: A tuple of four elements:
        """

        raise NotImplementedError

    def render(self):
        """
        The render function should return a string containing the ReST
        markup for the section.


        :param self: Access the class attributes and methods
        :return: The string representation of the object
        """
        raise NotImplementedError

    def stop(self):
        """
        The stop function stops the robot from moving.


        :param self: Refer to the object itself
        :return: None
        """
        pass

    @property
    def info(self):
        """
        The info function returns a dictionary containing the MDP's metadata.
        The keys of this dictionary are:
            * name: The name of the MDP.
            * description: A short description of what the MDP is about.
            * states_specification: A specification for each state in terms of its
                features and their possible values, as well as an optional label to be
                used when displaying it to humans (e.g., &quot;s0&quot; or &quot;state 0&quot;). This is
                stored as a list with one entry per state, where each entry is itself a


        :param self: Refer to the object itself
        :return: The mdp info dictionary
        """
        return self._mdp_info

    @staticmethod
    def _bound(x, min_value, max_value):
        """
        The _bound function takes a value x and the min_value and max_value of a range.
        It returns the maximum of min_value and minimum of max_value with respect to x.

        :param x: Pass the input data to the function
        :param min_value: Set a lower bound on the output of the function
        :param max_value: Set the upper limit of the interval
        :return: The maximum of the minimum value and the minimum of the maximum value
        """
        return np.maximum(min_value, np.minimum(x, max_value))

    _registered_envs = dict()

class Policy(object): # TODO: investigate for Task 4.2 b
    # This is a deterministic linear policy with continuous action space
    def __init__(self, theta, observation_space, action_space):
        """
        The __init__ function is called automatically every time the class is instantiated.
        The first argument of every class method, including init, is always a reference to the current instance of the class.
        By convention, this argument is named self. The role of init as you can see from its name (initialize) and function signature
        is to initialize attributes that are specific to each instance.

        :param self: Reference the class instance
        :param theta: Store the weights and the d parameter is used to store the bias
        :param observation_space: Define the size of the input layer
        :param action_space: Specify the size of the action space, which is needed to reshape the weights vector into a matrix
        :return: The weights and the bias of the model
        """
        self.action_space = action_space
        dimensions = observation_space.shape[0] * action_space.shape[0]
        self.weights = theta[0:dimensions].reshape(
            observation_space.shape[0], action_space.shape[0]
        )
        self.d = theta[dimensions:None]

    def get_action(self, observations):
        """
        The get_action function takes in observations and returns the action that is
        clipped to within the action space. This function also updates the weights of
        the linear model using a learning rate, self.lr, and a decay term, self.decay.

        :param self: Access the variables and methods of the class in which it is used
        :param observations: Get the current state of the environment
        :return: The action that is calculated by the observations @ self
        """
        action = np.clip(
            observations @ self.weights + self.d,
            self.action_space.low,
            self.action_space.high,
        )
        return action


class MDPInfo:
    def __init__(self, observation_space, action_space, gamma, horizon):
        """
        The __init__ function is called when an object of the class is instantiated.
        It initializes all the variables that are defined in it and can be used later
        in other methods. In this case, we initialize a few attributes:

        :param self: Refer to the object itself
        :param observation_space: Define the observation space of the environment
        :param action_space: Define the action space of the environment
        :param gamma: Calculate the discount factor
        :param horizon: Define the length of the horizon
        :return: Nothing
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.horizon = horizon
        self.save_attributes = dict()

        self.add_save_attr(
            observation_space="space",
            action_space="space",
            gamma="primitive",
            horizon="primitive",
        )

    def add_save_attr(self, **attr_dict):
        """
        The add_save_attr function adds attributes to the save_attributes dictionary.
        The save_attributes dictionary is used to store information about a class that will be saved in a pickle file.
        This function allows you to add new attributes and values to the save_attributes dictionary.

        :param self: Refer to the object itself
        :param **attr_dict: Specify the attributes that you want to save
        :return: A dictionary of attributes that are to be saved
        """
        self.save_attributes.update(attr_dict)

    @property
    def size(self):
        """
        The size function returns the size of the observation space + action space.
        This is useful for calculating how many neurons should be in a neural network,
        as well as other things such as number of weights and biases.

        :param self: Access the attributes and methods of the class in python
        :return: The sum of the sizes of the observation and action spaces
        """
        return self.observation_space.size + self.action_space.size

    @property
    def shape(self):
        """
        The shape function is used to return the shape of the observation space and action space.
        The shape function returns a tuple that contains both the observation space and action space.
        This is done by concatenating them together.

        :param self: Access the object's variables
        :return: The shape of the observation space and action space combined
        """
        return self.observation_space.shape + self.action_space.shape


class ObservationType(Enum):
    # BODY_POS: (3,) x, y, z position of the body
    # BODY_ROT: (4,) quaternion of the body
    # BODY_VEL: (6,) first angular velocity around x, y, z. Then linear velocity for x, y, z
    # JOINT_POS: (1,) rotation of the joint OR (7,) position, quaternion of a free joint
    # JOINT_VEL: (1,) velocity of the joint OR (6,) FIRST linear then angular velocity !different to BODY_VEL!
    # SITE_POS: (3,) x, y, z position of the body
    # SITE_ROT: (9,) rotation matrix of the site

    __order__ = "BODY_POS BODY_ROT BODY_VEL JOINT_POS JOINT_VEL SITE_POS SITE_ROT"
    BODY_POS = 0
    BODY_ROT = 1
    BODY_VEL = 2
    JOINT_POS = 3
    JOINT_VEL = 4
    SITE_POS = 5
    SITE_ROT = 6


class PlotRewards:
    def get_mean_and_confidence(self, data):
        """
        Compute the mean and 95% confidence interval

        Args:
            data (np.ndarray): Array of experiment data of shape (n_runs, n_epochs).

        Returns:
            The mean of the dataset at each epoch along with the confidence interval.

        """
        mean = np.mean(data, axis=0)
        se = st.sem(data, axis=0)
        n = len(data)
        interval, _ = st.t.interval(0.95, n - 1, scale=se)
        return mean, interval

    def plot_mean_conf(self, data, ax, color='blue', line='-', facecolor=None, alpha=0.4, label=None):
        """
        Method to plot mean and confidence interval for data on matplotlib axes.

        Args:
            data (np.ndarray): Array of experiment data of shape (n_runs, n_epochs);
            ax (plt.Axes): matplotlib axes where to create the curve;
            color (str, 'blue'): matplotlib color identifier for the mean curve;
            line (str, '-'): matplotlib line type to be used for the mean curve;
            facecolor (str, None): matplotlib color identifier for the confidence interval;
            alpha (float, 0.4): transparency of the confidence interval;
            label (str, one): legend label for the plotted curve.


        """
        facecolor = color if facecolor is None else facecolor

        mean, conf = self.get_mean_and_confidence(np.array(data))
        upper_bound = mean + conf
        lower_bound = mean - conf

        ax.plot(mean, color=color, linestyle=line, label=label)
        ax.fill_between(np.arange(np.size(mean)), upper_bound, lower_bound, facecolor=facecolor, alpha=alpha)


class ObservationHelper:
    def __init__(self, observation_spec, model, data, max_joint_velocity):
        """
        The __init__ function is called when an object of the class is instantiated.
        It initializes all the variables and attributes that are present in a new instance
        of this class. In our case, it sets up some lists to store information about which
        joints we care about, as well as their observation space boundaries.

        :param self: Reference the class instance
        :param observation_spec: Specify the observation space
        :param model: Access the mujoco model
        :param data: Get the current state of the model
        :param max_joint_velocity: Set the maximum velocity of each joint
        :return: The observation space
        """
        if len(observation_spec) == 0:
            raise AttributeError(
                "No Environment observations were specified. "
                "Add at least one observation to the observation_spec."
            )

        self.obs_low = []
        self.obs_high = []
        self.joint_pos_idx = []
        self.joint_vel_idx = []
        self.joint_mujoco_idx = []

        self.obs_idx_map = {}

        self.build_omit_idx = {}

        self.observation_spec = observation_spec

        if max_joint_velocity is not None:
            max_joint_velocity = iter(max_joint_velocity)

        current_idx = 0
        key = 0
        for name, ot in observation_spec:
            assert key not in self.obs_idx_map.keys(), (
                'Found duplicate key in observation specification: "%s"' % key
            )
            obs_count = len(self.get_state(data, name, ot))
            self.obs_idx_map[key] = list(range(current_idx, current_idx + obs_count))
            self.build_omit_idx[key] = []
            if obs_count == 1 and ot == ObservationType.JOINT_POS:
                self.joint_pos_idx.append(current_idx)
                self.joint_mujoco_idx.append(model.joint(name).id)
                if model.joint(name).limited:
                    self.obs_low.append(model.joint(name).range[0])
                    self.obs_high.append(model.joint(name).range[1])
                else:
                    self.obs_low.append(-np.inf)
                    self.obs_high.append(np.inf)

            elif obs_count == 1 and ot == ObservationType.JOINT_VEL:
                self.joint_vel_idx.append(current_idx)
                if max_joint_velocity is None:
                    max_vel = np.inf
                else:
                    max_vel = next(max_joint_velocity)

                self.obs_low.append(-max_vel)
                self.obs_high.append(max_vel)
            else:
                self.obs_low.extend([-np.inf] * obs_count)
                self.obs_high.extend([np.inf] * obs_count)

            current_idx += obs_count
            key += 1
        self.obs_low = np.array(self.obs_low)
        self.obs_high = np.array(self.obs_high)

    def remove_obs(self, key, index):
        """
        The remove_obs function removes an observation from the environment.
        The remove_obs function takes two arguments: key and index. The key is a string that specifies which observation to remove, and the index is an integer specifying which element of the specified observation to remove.

        :param self: Reference the class object
        :param key: Identify the observation that is to be removed
        :param index: Remove the observations from the observation space
        :return: The indices of the observations that were removed
        """
        indices = self.obs_idx_map[key]
        adjusted_index = index - len(self.build_omit_idx[key])

        self.obs_low = np.delete(self.obs_low, indices[adjusted_index])
        self.obs_high = np.delete(self.obs_high, indices[adjusted_index])
        cutoff = indices.pop(adjusted_index)

        for obs_list in self.obs_idx_map.values():
            for idx in range(len(obs_list)):
                if obs_list[idx] > cutoff:
                    obs_list[idx] -= 1

        for i in range(len(self.joint_pos_idx)):
            if self.joint_pos_idx[i] > cutoff:
                self.joint_pos_idx[i] -= 1

        for i in range(len(self.joint_vel_idx)):
            if self.joint_vel_idx[i] > cutoff:
                self.joint_vel_idx[i] -= 1

        self.build_omit_idx[key].append(index)

    def add_obs(self, key, length, min_value=-np.inf, max_value=np.inf):
        """
        The add_obs function adds a new observation to the environment.
        The key is used as a reference for the observation, and should be unique.
        The length of the observation is how many dimensions it has, and min_value/max_value are lists containing that number of minimums and maximums respectively.

        :param self: Reference the class object
        :param key: Define the name of the observation
        :param length: Specify the number of elements in the observation vector
        :param min_value=-np.inf: Set the lower bound of the observation
        :param max_value=np.inf: Set the upper bound of the observation space
        :return: The index of the observation
        """

        self.obs_idx_map[key] = list(
            range(len(self.obs_low), len(self.obs_low) + length)
        )

        if hasattr(min_value, "__len__"):
            self.obs_low = np.append(self.obs_low, min_value)
        else:
            self.obs_low = np.append(self.obs_low, [min_value] * length)

        if hasattr(max_value, "__len__"):
            self.obs_high = np.append(self.obs_high, max_value)
        else:
            self.obs_high = np.append(self.obs_high, [max_value] * length)

    def get_from_obs(self, obs, key):
        """
        The get_from_obs function is a helper function that allows the agent to access observations from the environment.
        It does this by returning a slice of the observation array, which is writeable. This means that we can modify
        the values in this slice without changing the original observation array.

        :param self: Access the attributes and methods of the class in a method
        :param obs: Store the observations
        :param key: Specify which parameter we are interested in
        :return: The data from the observation that is specified by the key
        """
        # Cannot use advanced indexing because it returns a copy.....
        # We want this data to be writeable
        return obs[self.obs_idx_map[key][0] : self.obs_idx_map[key][-1] + 1]

    def get_joint_pos_from_obs(self, obs):
        """
        The get_joint_pos_from_obs function returns the joint positions of the robot from an observation.
        The function takes in a single argument, obs, which is assumed to be a list of observations returned by the environment.
        The function then returns a numpy array containing only those observations corresponding to joints.

        :param self: Access the variables and methods of the class in which it is used
        :param obs: Get the joint positions from the observation
        :return: The joint positions of the robot
        """
        return obs[self.joint_pos_idx]

    def get_joint_vel_from_obs(self, obs):
        """
        The get_joint_vel_from_obs function returns the joint velocities of the robot.
        The function takes in an observation from the environment and returns a list of
        the velocity values for each joint.

        :param self: Access the variables and methods of the class in python
        :param obs: Get the joint velocities
        :return: The velocity of the joints
        """
        return obs[self.joint_vel_idx]

    def get_obs_limits(self):
        """
        The get_obs_limits function returns the observation limits for the environment.

        :param self: Reference the object itself
        :return: The lower and upper limits of the observation space
        """
        return self.obs_low, self.obs_high

    def get_joint_pos_limits(self):
        """
        The get_joint_pos_limits function returns the lower and upper joint position limits of the robot.
        The function takes no arguments, but it does return two values:
            - low_limits: a list of all the lower joint position limits for each joint in the robot.
                The length of this list is equal to number_of_joints (the total number of joints in our simulated robot).
                Each value in this list is an integer between 0 and 2*pi representing radians between 0 and 360 degrees.

        :param self: Access the variables and methods inside a class
        :return: The lower and upper limits of the joint positions
        """
        return self.obs_low[self.joint_pos_idx], self.obs_high[self.joint_pos_idx]

    def get_joint_vel_limits(self):
        """
        The get_joint_vel_limits function returns the joint velocity limits for the robot.
        The function takes no arguments and returns a list of two elements,
        the lower and upper joint velocity limits. The values are obtained from self.obs_low
        and self.obs_high, which are set in the __init__ method.

        :param self: Access the variables and methods inside the class
        :return: The lower and upper bound of the joint velocities
        """
        return self.obs_low[self.joint_vel_idx], self.obs_high[self.joint_vel_idx]

    def build_obs(self, data):
        """
        The build_obs function is a helper function that takes in the data from the
        environment and returns an array of observations. The build_obs function also
        omits any values that are not needed for training, such as the player's health.


        :param self: Access the class attributes
        :param data: Get the data from the database
        :return: The observations for the current time step
        """

        observations = []
        key = 0
        for name, o_type in self.observation_spec:
            omit = np.array(self.build_omit_idx[key])
            obs = self.get_state(data, name, o_type)
            if len(omit) != 0:
                obs = np.delete(obs, omit)
            observations.append(obs)
            key += 1
        return np.concatenate(observations)

    def get_state(self, data, name, o_type):
        """
        The get_state function returns the observation for a given state.

        :param self: Access the class variables
        :param data: Access the data from the mujoco simulator
        :param name: Specify the body or joint that is observed
        :param o_type: Specify the type of observation
        :return: The observation of the environment
        """

        if o_type == ObservationType.BODY_POS:
            obs = data.body(name).xpos
        elif o_type == ObservationType.BODY_ROT:
            obs = data.body(name).xquat
        elif o_type == ObservationType.BODY_VEL:
            obs = data.body(name).cvel
        elif o_type == ObservationType.JOINT_POS:
            obs = data.joint(name).qpos
        elif o_type == ObservationType.JOINT_VEL:
            obs = data.joint(name).qvel
        elif o_type == ObservationType.SITE_POS:
            obs = data.site(name).xpos
        elif o_type == ObservationType.SITE_ROT:
            obs = data.site(name).xmat
        else:
            raise ValueError("Invalid observation type")

        return np.atleast_1d(obs)

    def get_all_observation_keys(self):
        """
        The get_all_observation_keys function returns a list of all the observation keys that are currently in the
        observation dictionary. This function is useful for debugging and/or creating user interfaces.

        :param self: Refer to the object of the class
        :return: A list of all the observation keys in the dataset
        """
        return list(self.obs_idx_map.keys())


class MujocoGlfwViewer:
    def __init__(self, model, dt, width=1920, height=1080, start_paused=False):
        """
        The __init__ function is called when an instance of the class is created.
        It initializes variables that are common to all instances of the class, and it
        can take arguments that get placed into these variables. In this case, we're
        taking in a model (which will be used later) and setting up some basic
        variables like width, height, etc.

        :param self: Reference the class instance from within the class
        :param model: Set the model that will be rendered
        :param dt: Define the time step between frames
        :param width=1920: Specify the width of the window
        :param height=1080: Set the height of the window
        :param start_paused=False: Indicate whether the simulation should start in paused state or not
        :return: None
        """
        self.button_left = False
        self.button_right = False
        self.button_middle = False
        self.last_x = 0
        self.last_y = 0
        self.dt = dt
        self.frames = 0
        self.start_time = time.time()
        glfw.init()
        self._loop_count = 0
        self._time_per_render = 1 / 60.0
        self._paused = start_paused
        self._window = glfw.create_window(
            width=width, height=height, title="MuJoCo", monitor=None, share=None
        )
        glfw.make_context_current(self._window)
        glfw.set_mouse_button_callback(self._window, self.mouse_button)
        glfw.set_cursor_pos_callback(self._window, self.mouse_move)
        glfw.set_key_callback(self._window, self.keyboard)
        glfw.set_scroll_callback(self._window, self.scroll)

        self._model = model

        self._scene = mujoco.MjvScene(model, 1000)
        self._scene_option = mujoco.MjvOption()

        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self._camera)

        self._viewport = mujoco.MjrRect(0, 0, width, height)
        self._context = mujoco.MjrContext(model, mujoco.mjtFontScale(100))

        self.rgb_buffer = np.empty((width, height, 3), dtype=np.uint8)

    def mouse_button(self, window, button, act, mods):
        """
        The mouse_button function is called whenever the user clicks a mouse button.
        It sets the state of self.button_left, self.button_right, and self.button_middle to True or False depending on whether that particular button was pressed or not during this call of mouse_button.

        :param self: Reference the class instance from within the function
        :param window: Get the window object from glfw
        :param button: Check which button is being pressed
        :param act: Check if the button is pressed or released
        :param mods: Check if the shift key is pressed
        :return: The mouse button pressed
        """
        self.button_left = (
            glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self.button_right = (
            glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )
        self.button_middle = (
            glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        )

        self.last_x, self.last_y = glfw.get_cursor_pos(self._window)

    def mouse_move(self, window, x_pos, y_pos):
        """
        The mouse_move function is called when the user moves their mouse.
        It takes in a window, x_pos, and y_pos as arguments.
        The x_pos and y_pos are used to determine how much the camera should move in each direction.


        :param self: Access the class attributes
        :param window: Get the window size
        :param x_pos: Get the x-coordinate of the mouse position
        :param y_pos: Determine whether the mouse is moved up or down
        :return: None
        """
        if not self.button_left and not self.button_right and not self.button_middle:
            return

        dx = x_pos - self.last_x
        dy = y_pos - self.last_y
        self.last_x = x_pos
        self.last_y = y_pos

        width, height = glfw.get_window_size(self._window)

        mod_shift = (
            glfw.get_key(self._window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(self._window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        if self.button_right:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if mod_shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif self.button_left:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if mod_shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        mujoco.mjv_moveCamera(
            self._model, action, dx / width, dy / height, self._scene, self._camera
        )

    def keyboard(self, window, key, scancode, act, mods):
        """
        The keyboard function is called whenever the user hits a key in the window.
        The function receives four parameters:
            - window: The glfw.Window object that was created when GLFW initialized the context.
            - key: An integer representing which key was hit (glfw.KEY_*). See [the GLFW docs](http://docs.racket-lang.org/glfw3/latest/constant-values-and-properties/#key) for more details on what keys are available and their values, but note that some of these may not be available on all platforms or systems, so you

        :param self: Access the class attributes
        :param window: Get the window handle
        :param key: Check which key was pressed
        :param scancode: Distinguish between multiple keys that are pressed simultaneously
        :param act: Distinguish between key press and release
        :param mods: Check if the shift key is pressed
        :return: Nothing
        """
        if act != glfw.RELEASE:
            return

        if key == glfw.KEY_SPACE:
            self._paused = not self._paused

        if key == glfw.KEY_C:
            self._scene_option.flags[
                mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
            ] = not self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]
            self._scene_option.flags[
                mujoco.mjtVisFlag.mjVIS_CONSTRAINT
            ] = not self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT]

        if key == glfw.KEY_T:
            self._scene_option.flags[
                mujoco.mjtVisFlag.mjVIS_TRANSPARENT
            ] = not self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]

    def scroll(self, window, x_offset, y_offset):
        """
        The scroll function is used to zoom in and out of the camera view.
        It takes in a window object, x_offset, and y_offset as arguments.
        The function then calls mjv_moveCamera() with the appropriate parameters to
        zoom in or out depending on whether y_offset is positive or negative.

        :param self: Access the class attributes
        :param window: Get the size of the window
        :param x_offset: Zoom in or out
        :param y_offset: Zoom in or out
        :return: The x and y offsets
        """
        mujoco.mjv_moveCamera(
            self._model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0,
            0.05 * y_offset,
            self._scene,
            self._camera,
        )

    def render(self, data):
        """
        The render function is the main loop of MuJoCo rendering. It takes in a
        data dictionary, and renders it to the screen. The render function also
        handles keyboard events (e.g., pausing, quitting). If you want to modify your
        own code's interaction with the screen, start by modifying this function.

        :param self: Access the class variables
        :param data: Pass in the state of the physics simulation
        :return: None
        """
        def render_inner_loop(self):
            render_start = time.time()

            mujoco.mjv_updateScene(
                self._model,
                data,
                self._scene_option,
                None,
                self._camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                self._scene,
            )

            self._viewport.width, self._viewport.height = glfw.get_window_size(
                self._window
            )
            mujoco.mjr_render(self._viewport, self._scene, self._context)

            glfw.swap_buffers(self._window)
            glfw.poll_events()

            self.frames += 1

            if glfw.window_should_close(self._window):
                self.stop()
                exit(0)

            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )

        if self._paused:
            while self._paused:
                render_inner_loop(self)

        self._loop_count += self.dt / self._time_per_render
        while self._loop_count > 0:
            render_inner_loop(self)
            self._loop_count -= 1

    def stop(self):
        """
        The stop function is used to close the window.


        :param self: Access the attributes and methods of the class
        :return: The window object
        """
        glfw.destroy_window(self._window)


class Box:
    def __init__(self, low, high, shape=None):
        """
        The __init__ function initializes the class.
        It takes as input a low and high value, which are used to define the space of this Box.
        The shape is also required if it is not defined by the low and high values.

        :param self: Refer to the object itself
        :param low: Set the lower bound of the random variable
        :param high: Define the upper limit of the random variable
        :param shape=None: Determine if the user has defined a shape
        :return: None
        """

        self.save_attributes = dict()
        if shape is None:
            self._low = low
            self._high = high
            self._shape = low.shape
        else:
            self._low = low
            self._high = high
            self._shape = shape
            if np.isscalar(low) and np.isscalar(high):
                self._low += np.zeros(shape)
                self._high += np.zeros(shape)

        assert self._low.shape == self._high.shape

        self.add_save_attr(_low="numpy", _high="numpy")

    def add_save_attr(self, **attr_dict):
        """
        The add_save_attr function adds attributes to the save_attributes dictionary.
        The save_attributes dictionary is used to store information about a class that
        is not stored in the database, but needs to be accessed by other functions. For example,
        if you have a class representing an item in an inventory, it might make sense to create
        a function that returns how many items are in stock at any given time. However, this
        information doesn't need to be stored with the item's data every time it is saved -- only
        whenver its value changes.

        :param self: Refer to the object itself
        :param **attr_dict: Add new attributes to the save_attributes dictionary
        :return: The updated attribute dictionary
        """
        self.save_attributes.update(attr_dict)

    @property
    def low(self):
        """
        The low function returns the lowest value in a list.

        :param self: Access variables that belongs to the class
        :return: The value of the _low attribute
        """
        return self._low

    @property
    def high(self):
        """
        The high function returns a dictionary of the following form:
        {'&lt;parameter name&gt;': &lt;high value&gt;, ...}


        :param self: Access variables that belongs to the class
        :return: The value of the high attribute
        """
        return self._high

    @property
    def shape(self):
        """
        The shape function returns the shape of a tensor.

        :param self: Access the attributes and methods of the class in python
        :return: The shape of the array
        """
        return self._shape

    def _post_load(self):
        """
        The _post_load function is called after the data has been loaded.
        It sets the shape attribute of the class to be equal to self._low.shape,
        which is a tuple containing information about how many dimensions there are and what their size is.

        :param self: Refer to the object itself
        :return: The shape of the array that is being loaded
        """
        self._shape = self._low.shape


class MuJoCo(Environment):
    def __init__(
        self,
        file_name,
        actuation_spec,
        observation_spec,
        gamma,
        horizon,
        timestep=None,
        n_substeps=1,
        n_intermediate_steps=1,
        additional_data_spec=None,
        collision_groups=None,
        max_joint_vel=None,
        **viewer_params
    ):
        """
        The __init__ function is the constructor of a class. It is called when an
        instance of the class is created. The __init__ function can take arguments, but
        the first argument should always be self (see below). The __init__ function can
        have a return value which should be None or the same type as the class.

        :param self: Reference the object in which it is called
        :param file_name: Specify the path to the xml file of your model
        :param actuation_spec: Specify which actuators are used in the simulation
        :param observation_spec: Specify the names of the observations that are returned by the environment
        :param gamma: Compute the discounted rewards
        :param horizon: Define the length of an episode
        :param timestep=None: Indicate that the timestep is not
        :param n_substeps=1: Specify the number of substeps that are
        :param n_intermediate_steps=1: Specify how many intermediate
        :param additional_data_spec=None: Specify additional data that is not part of the observation
        :param collision_groups=None: Specify the geom groups that
        :param max_joint_vel=None: Avoid the warning message:
        :param **viewer_params: Pass parameters to the mujoco viewer
        :return: The mdpinfo object
        """
        # Create the simulation
        self._model = mujoco.MjModel.from_xml_path(file_name)
        if timestep is not None:
            self._model.opt.timestep = timestep
            self._timestep = timestep
        else:
            self._timestep = self._model.opt.timestep

        self._data = mujoco.MjData(self._model)

        self._n_intermediate_steps = n_intermediate_steps
        self._n_substeps = n_substeps
        self._viewer_params = viewer_params
        self._viewer = None
        self._obs = None

        # Read the actuation spec and build the mapping between actions and ids
        # as well as their limits
        if len(actuation_spec) == 0:
            self._action_indices = [i for i in range(0, len(self._data.actuator_force))]
        else:
            self._action_indices = []
            for name in actuation_spec:
                self._action_indices.append(self._model.actuator(name).id)

        low = []
        high = []
        for index in self._action_indices:
            if self._model.actuator_ctrllimited[index]:
                low.append(self._model.actuator_ctrlrange[index][0])
                high.append(self._model.actuator_ctrlrange[index][1])
            else:
                low.append(-np.inf)
                high.append(np.inf)
        self.action_space = Box(np.array(low), np.array(high))

        # Read the observation spec to build a mapping at every step. It is
        # ensured that the values appear in the order they are specified.
        self.obs_helper = ObservationHelper(
            observation_spec, self._model, self._data, max_joint_velocity=max_joint_vel
        )

        self.observation_space = Box(*self.obs_helper.get_obs_limits())

        # Pre-process the additional data to allow easier writing and reading
        # to and from arrays in MuJoCo
        self.additional_data = {}
        if additional_data_spec is not None:
            for key, name, ot in additional_data_spec:
                self.additional_data[key] = (name, ot)

        # Pre-process the collision groups for "fast" detection of contacts
        self.collision_groups = {}
        if collision_groups is not None:
            for name, geom_names in collision_groups:
                self.collision_groups[name] = {
                    mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                    for geom_name in geom_names
                }

        # Finally, we create the MDP information and call the constructor of
        # the parent class
        mdp_info = MDPInfo(self.observation_space, self.action_space, gamma, horizon)

        mdp_info = self._modify_mdp_info(mdp_info)

        # set the warning callback to stop the simulation when a mujoco warning occurs
        mujoco.set_mju_user_warning(self.user_warning_raise_exception)

        super().__init__(mdp_info)

    def seed(self, seed):
        """
        The seed function is used to initialize the pseudorandom number generator.
        This allows us to reproduce the results from our script later on by using the same seed and thus generate the same sequence of random numbers used in our script.


        :param self: Refer to the object of the class
        :param seed: Set the random number generator
        :return: The seed value
        """
        np.random.seed(seed)

    def reset(self, obs=None):
        """
        The reset function resets the simulation to a clean slate. It is called at the beginning of an episode, and returns an observation that is passed to step().
        The reset function accepts one argument (obs), which can be used as a hint about what observations should look like.
        If it's implemented, it must return a numpy array with shape (n,) + OBS_SHAPE where n is any integer. The value of each entry in the array will be passed to step() as the corresponding argument.

        :param self: Reference the object itself
        :param obs=None: Pass the observation to the reset function
        :return: The observation
        """
        mujoco.mj_resetData(self._model, self._data)
        self.setup()

        self._obs = self._create_observation(self.obs_helper.build_obs(self._data))
        return self._modify_observation(self._obs)

    def step(self, action):
        """
        The step function takes an action and returns the next observation, reward, whether the episode is done (absorbing state), and a dict of misc information.

        :param self: Access the class attributes
        :param action: Compute the control action
        :return: The observation, the reward and a flag indicating whether the episode is over
        """
        cur_obs = self._obs.copy()

        action = self._preprocess_action(action)

        self._step_init(cur_obs, action)

        for i in range(self._n_intermediate_steps):

            ctrl_action = self._compute_action(cur_obs, action)
            self._data.ctrl[self._action_indices] = ctrl_action

            self._simulation_pre_step()

            mujoco.mj_step(self._model, self._data, self._n_substeps)

            self._simulation_post_step()

            cur_obs = self._create_observation(self.obs_helper.build_obs(self._data))

        self._step_finalize()

        absorbing = self.is_absorbing(cur_obs)
        reward = self.reward(self._obs, action, cur_obs, absorbing)

        self._obs = cur_obs
        return self._modify_observation(cur_obs), reward, absorbing, {}

    def render(self):
        """
        The render function is meant to be used in conjunction with the other functions
        in this module. It is not meant to be called directly by the user.


        :param self: Access the instance of the class
        :return: The viewer object
        """
        if self._viewer is None:
            self._viewer = MujocoGlfwViewer(self._model, self.dt, **self._viewer_params)

        self._viewer.render(self._data)

    def stop(self):
        """
        The stop function stops the viewer.


        :param self: Refer to the object itself
        :return: None
        """
        if self._viewer is not None:
            self._viewer.stop()
            del self._viewer
            self._viewer = None

    def _modify_mdp_info(self, mdp_info):
        """
        The _modify_mdp_info function is called by the MDP method to modify the mdp_info dictionary.
        It should return a modified version of this dictionary, which will be used in place of mdp_info.
        This function is useful for adding or removing items from the mdp_info dictionary before it is passed to Gromacs.

        :param self: Access the class instance inside of a method
        :param mdp_info: Pass information to the mdp class
        :return: The mdp_info dictionary
        """
        return mdp_info

    def _create_observation(self, obs):
        """
        The _create_observation function is called by the environment to create a new observation.

        :param self: Refer to the object of the class
        :param obs: Store the current observation
        :return: The observation
        """

        return obs

    def _modify_observation(self, obs):
        """
        The _modify_observation function is a helper function that modifies the observation
        by adding noise to it. The noise added is based on the current value of self.noise_scale,
        which starts at 1 and decreases as episodes progress.

        :param self: Access the class attributes
        :param obs: Store the observation that is received from the environment
        :return: The observation
        """

        return obs

    def _preprocess_action(self, action):
        """
        The _preprocess_action function is called by the step function to preprocess an action.
        It takes in a raw action and returns a processed action. This allows for customizing what the agent sees as actions.

        :param self: Access the attributes and methods of the class in a class method
        :param action: Convert the action into a form that is usable by the environment
        :return: The action that the agent takes
        """

        return action

    def _step_init(self, obs, action):
        """
        The _step_init function is called at the beginning of each episode.
        It does not perform any computations, but rather just sets up data structures
        that will be needed later on in the episode. In this case, it initializes a
        list to store all of the observations and actions that are received during an
        episode.

        :param self: Access the attributes of the class
        :param obs: Store the current state of the environment
        :param action: Pass the action to be performed by the agent
        :return: The action that the agent will take
        """

        pass

    def _compute_action(self, obs, action):
        """
        The _compute_action function is used to compute the action that should be taken by the agent.
        It takes in an observation, and returns a single number representing the action that should be taken.
        The function can also take in a second argument, which is not used by this particular implementation of DQN.

        :param self: Access the variables and methods of the class
        :param obs: Get the current state of the environment
        :param action: Compute the action to be taken in that step
        :return: The action that the agent takes given a particular observation
        """

        return action

    def _simulation_pre_step(self):
        """
        The _simulation_pre_step function is called before the simulation step.
        It can be used to update the state of a model, for example by updating
        the position of a particle or changing its velocity. The function is called
        before any other code in the simulation step.

        :param self: Access the class instance
        :return: The number of infected people in the population
        """

        pass

    def _simulation_post_step(self):
        """
        The _simulation_post_step function is called after the simulation's main
        loop has been executed. It can be used to perform any necessary post-processing
        of data, such as writing results to a file or performing analysis.

        :param self: Refer to the object itself
        :return: The results of the simulation
        """

        pass

    def _step_finalize(self):
        """
        The _step_finalize function is called after the step has been executed.
        It can be used to clean up any temporary files or processes that were created by the step.


        :param self: Access the class attributes
        :return: The step_outputs, which is a list of the outputs from each step
        """

        pass

    def _read_data(self, name):
        """
        The _read_data function reads the data from the observation helper.
        It takes in a name, which is used to get the id and type of data from
        the additional_data dictionary. The function then returns an array of
        the read data.

        :param self: Make the function a member of the class
        :param name: Identify the data that is read
        :return: The data from the observation helper
        """

        data_id, otype = self.additional_data[name]
        return np.array(self.obs_helper.get_state(self._data, data_id, otype))

    def _write_data(self, name, value):
        """
        The _write_data function writes the value of a given variable to the
        corresponding data buffer. The function takes two arguments: name, which is
        the name of the variable and value, which is its corresponding value. It then
        writes this information into the appropriate data buffer.

        :param self: Access the class variables
        :param name: Identify the data in the observation
        :param value: Set the joint position or velocity
        :return: None
        """

        data_id, otype = self.additional_data[name]
        if otype == ObservationType.JOINT_POS:
            self._data.joint(data_id).qpos = value
        elif otype == ObservationType.JOINT_VEL:
            self._data.joint(data_id).qvel = value
        else:
            data_buffer = self.obs_helper.get_state(self._data, data_id, otype)
            data_buffer[:] = value

    def _check_collision(self, group1, group2):
        """
        The _check_collision function checks if there is a collision between two groups of objects.
        It does this by checking the contact data structure for any contacts that involve an object in group 1 and an object in group 2.
        If it finds such a contact, it returns True (meaning that there was indeed a collision).  If no collisions are found, False is returned.

        :param self: Access the data of the class
        :param group1: Specify the first group of objects that are in collision
        :param group2: Check for collisions between two different groups
        :return: True if the two objects are colliding, false otherwise
        """

        ids1 = self.collision_groups[group1]
        ids2 = self.collision_groups[group2]

        for coni in range(0, self._data.ncon):
            con = self._data.contact[coni]

            collision = con.geom1 in ids1 and con.geom2 in ids2
            collision_trans = con.geom1 in ids2 and con.geom2 in ids1

            if collision or collision_trans:
                return True

        return False

    def _get_collision_force(self, group1, group2):
        """
        The _get_collision_force function returns the force vector and torque vector of a collision between two objects.
        The function takes in two parameters: group 1 and group 2. The first parameter is the name of an object, such as &quot;robot&quot;.
        The second parameter is another object, such as &quot;table&quot;. The function then searches through all contacts to find any contact
        between the specified objects. If there are no contacts between those two objects, it returns an array of zeros with size 6 (for
        the force) and 3 (for the torque). Otherwise, it finds which contact geom belongs to which object using its ID number and then

        :param self: Access the class attributes
        :param group1: Specify which collision groups the contact force should be calculated for
        :param group2: Specify which collision group should be considered
        :return: A 6-dimensional array of the contact force between two geoms
        """

        ids1 = self.collision_groups[group1]
        ids2 = self.collision_groups[group2]

        c_array = np.zeros(6, dtype=np.float64)
        for con_i in range(0, self._data.ncon):
            con = self._data.contact[con_i]

            if (
                con.geom1 in ids1
                and con.geom2 in ids2
                or con.geom1 in ids2
                and con.geom2 in ids1
            ):

                mujoco.mj_contactForce(self._model, self._data, con_i, c_array)
                return c_array

        return c_array

    def reward(self, obs, action, next_obs, absorbing):
        """
        The reward function is a function of the observation and action. It returns a float value,
        which can be any real number. The reward function is used to define the goal in most RL problems.

        :param self: Access the agent's variables
        :param obs: Define the state of the environment
        :param action: Determine the action taken by the agent in a given state
        :param next_obs: Determine the goal
        :param absorbing: Indicate that the episode has ended
        :return: The reward for the given transition
        """

        raise NotImplementedError

    def is_absorbing(self, obs):
        """
        The is_absorbing function is used to determine whether the current state of the environment is absorbing.
        An absorbing state is one that ends an episode. This function should return a Boolean value, True if the
        current state of the environment corresponds to an absorbing state, False otherwise.

        :param self: Access the attributes and methods of the class in which it is used
        :param obs: Check if the state is absorbing
        :return: A boolean value
        """

        raise NotImplementedError

    def setup(self):
        """
        The setup function is used to define the following:

        :param self: Reference the object itself
        :return: A dictionary with the following keys:
        """

        pass

    def get_all_observation_keys(self):
        """
        The get_all_observation_keys function returns a list of all the observation keys that are currently in use.
        The function is used to ensure that no two observations have the same key.

        :param self: Access the class variables
        :return: A list of all the observation keys that can be used in get_observation
        """

        return self.obs_helper.get_all_observation_keys()

    @property
    def dt(self):
        """
        The dt function returns the timestep of the simulation.
        The timestep is equal to _timestep * _n_intermediate_steps * _n_substeps.

        :param self: Access the class attributes
        :return: The timestep multiplied by the number of intermediate steps and substeps
        """
        return self._timestep * self._n_intermediate_steps * self._n_substeps

    @staticmethod
    def user_warning_raise_exception(warning):
        """
        The user_warning_raise_exception function is a helper function that raises an exception
        if the user's code triggers a warning from MuJoCo. The purpose of this function is to make
        it easier for users to figure out which warnings they are triggering by running their code.


        :param warning: Check if the warning is one of the three known warnings that can occur during simulation
        :return: The warning
        """

        if "Pre-allocated constraint buffer is full" in warning:
            raise RuntimeError(warning + "Increase njmax in mujoco XML")
        elif "Pre-allocated contact buffer is full" in warning:
            raise RuntimeError(warning + "Increase njconmax in mujoco XML")
        elif "Unknown warning type" in warning:
            raise RuntimeError(warning + "Check for NaN in simulation.")
        else:
            raise RuntimeError("Got MuJoCo Warning: " + warning)
