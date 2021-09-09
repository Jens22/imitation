"""Loads CartPole-v1 demonstrations and trains BC, GAIL, and AIRL models on that data.
"""

import pathlib
import pickle
import tempfile
import os
import time

import stable_baselines3 as sb3

from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger, util

eval_n_timesteps = int(2e4)  # Min timesteps to evaluate, optional.
eval_n_episodes = None  # Num episodes to evaluate, optional.
sample_until = rollout.make_sample_until(eval_n_timesteps, eval_n_episodes)
venv = util.make_vec_env("Wheelchair-v0", n_envs=1)

tempdir = tempfile.TemporaryDirectory(prefix="wheelchair")
tempdir_path = pathlib.Path(tempdir.name)
print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

# Train BC on expert data.
# BC also accepts as `expert_data` any PyTorch-style DataLoader that iterates over
# dictionaries containing observations and actions.
save_path = os.path.join("wheelchair", "bc") 
os.makedirs(save_path, exist_ok=True)
save_file = os.path.join(save_path, "discrim.pt")

logger.configure(tempdir_path / "bc/")

#bc_policy = bc.reconstruct_policy(policy_path=save_file)
#trajs = rollout.generate_trajectories(bc_policy, venv, sample_until)

try:
	bc_policy = bc.reconstruct_policy(policy_path=save_file)
	trajs = rollout.generate_trajectories(bc_policy, venv, sample_until)
finally:
    	print ("over and out")
    	venv.close()
