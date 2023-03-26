"""
This file is where your agent's logic is kept. Define a bidding policy, factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error e.g. print("message", file=sys.stderr)
"""

import os.path as osp
import sys
import numpy as np
import torch as th
from stable_baselines3.ppo import PPO
from lux.config import EnvConfig
from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper
import os

# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary. Make sure to exclude the .zip extension here
MODEL_WEIGHTS_RELATIVE_PATH = "./best_model"

class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)
        self.policy = PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))

        self.controller = SimpleUnitDiscreteController(self.env_cfg)

    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        return dict(faction="AlphaStrike", bid=0)

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        if obs["teams"][self.player]["metal"] == 0:
            return dict()
        potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
        potential_spawns_set = set(potential_spawns)
        done_search = False

        ice_diff = np.diff(obs["board"]["ice"])
        pot_ice_spots = np.argwhere(ice_diff == 1)
        if len(pot_ice_spots) == 0:
            pot_ice_spots = potential_spawns
        trials = 5
        while trials > 0:
            pos_idx = np.random.randint(0, len(pot_ice_spots))
            pos = pot_ice_spots[pos_idx]

            area = 3
            for x in range(area):
                for y in range(area):
                    check_pos = [pos[0] + x - area // 2, pos[1] + y - area // 2]
                    if tuple(check_pos) in potential_spawns_set:
                        done_search = True
                        pos = check_pos
                        break
                if done_search:
                    break
            if done_search:
                break
            trials -= 1
        spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        if not done_search:
            pos = spawn_loc

        #metal = obs["teams"][self.player]["metal"]
        return dict(spawn=pos, metal=min(obs["teams"][self.player]["metal"],300), water=min(obs["teams"][self.player]["water"],300))

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # first convert observations using the same observation wrapper you used for training
        # note that SimpleUnitObservationWrapper takes input as the full observation for both players and returns an obs for players

        raw_obs = dict(player_0=obs, player_1=obs)
        shared_obs = raw_obs[self.player]
        obs = SimpleUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        obs = obs[self.player]

        obs = th.from_numpy(obs).float()

        with th.no_grad():

            # to improve performance, we have a rule based action mask generator for the controller used
            # which will force the agent to generate actions that are valid only.
            action_mask = (
                th.from_numpy(self.controller.action_masks(self.player, raw_obs))
                .unsqueeze(0)
                .bool()
            )

            #if len(action_mask.shape) >= 3:
            #    action_mask = action_mask[0,:,0]
            #    action_mask = action_mask[None, :]
            #    obs = obs[:,0]
            
            actions = dict()
            units = shared_obs["units"][self.player]
            for unit_idx,unit_id in enumerate(units.keys()):
                # SB3 doesn't support invalid action masking. So we do it ourselves here
                
                if len(obs.shape) == 1:
                    obs = obs[:,None]
                
                features = self.policy.policy.features_extractor(obs[:,unit_idx].unsqueeze(0))
                x = self.policy.policy.mlp_extractor.shared_net(features)
                logits = self.policy.policy.action_net(x) # shape (1, N) where N=12 for the default controller

                unit_mask = action_mask[0,:,unit_idx]
                unit_mask = unit_mask[None, :]
                logits[~unit_mask] = -1e8 # mask out invalid actions
                dist = th.distributions.Categorical(logits=logits)
                unit_actions = dist.sample().cpu().numpy() # shape (1, 1)
                actions[unit_id] = unit_actions[0]

        # use our controller which we trained with in train.py to generate a Lux S2 compatible action
        lux_action = self.controller.action_to_lux_action(
            self.player, raw_obs, actions
        )

        # commented code below adds watering lichen which can easily improve your agent
        factories = shared_obs["factories"][self.player]
        for unit_id in factories.keys():
            factory = factories[unit_id]
            if step > 800 and factory["cargo"]["water"] > 100:
                lux_action[unit_id] = 2 # water and grow lichen at 200 steps
        return lux_action
