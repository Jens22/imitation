# Train PPO agent on cartpole and collect expert demonstrations. Tensorboard logs saved in `quickstart/rl/`
python -m imitation.scripts.expert_demos with fast cartpole log_dir=quickstart/rl/

# Train GAIL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial with fast gail cartpole rollout_path=quickstart/rl/rollouts/final.pkl

# Train AIRL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial with fast airl cartpole rollout_path=quickstart/rl/rollouts/final.pkl

#Turtlebot3
# Train GAIL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python3 -m imitation.scripts.train_adversarial with gail turtlebot rollout_path=turtlebot3/expert/final.pkl

python3 -m imitation.scripts.eval_policy with turtlebot policy_path=output/train_adversarial/Turtlebot3-v0/20210607_124503_f9193a/checkpoints/final/gen_policy/

#wheelchair
python3 -m imitation.scripts.train_adversarial with gail wheelchair rollout_path=wheelchair/expert/final.pkl

python3 -m imitation.scripts.eval_policy with env_name=Wheelchair-v0 policy_path=output/train_adversarial/Wheelchair-v0/20210623_171057_95a59b/checkpoints/final/gen_policy/
