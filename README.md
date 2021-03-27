**Status:** Archive (code is provided as-is, no updates expected)

# Potential-based Reward Shaping for Multi-Agent Reinforcement Learning

This is the code for implementing the SAM algorithm presented in the paper:
`Shaping Advice in Deep Multi-Agent Reinforcement Learning`
<!-- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf). -->
It is configured to be run in conjunction with environments from the
[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).
Different from the original MPE environment, we modified the reward structure such that the rewards are sparse.
Note: this codebase has been restructured since the original paper, and the results may
vary from those reported in the paper.

## Installation

- To install, `cd` into the root directory and type `pip install -e .`

- Known dependencies: Python (3.5.4), OpenAI gym (0.10.5), tensorflow (1.9.0), numpy (1.15.2)

## Case study: Run SAM-NonUniform in simple-spread Environments

- To run SAM-NonUniform in simple spread, `cd` into the `experiments` directory and run `train.py`:

``python train_spread.py --scenario=simple_spread --num-episodes=60000 --save-dir=./logs/simple_spread/``

- To visualize the play if the saved model is in `./logs/simple_spread/`:

``python train.py --scenario=simple_spread --num-episodes=60000 --display --load-dir=./logs/simple_spread/``

- Here are examples for running SAM on other environments:

``python train_tag.py --scenario=simple_tag --num-adversaries=3 --num-episodes=60000 --save-dir=./logs/simple_tag/``

``python train_adv.py --scenario=simple_adversary --num-adversaries=1 --num-episodes=60000 --save-dir=./logs/simple_adversary``

- For comparison, you can run IRCR or MADDPG alone on MPE with saprse reward:

``python train_IRCR_spread.py --scenario=simple_spread --num-episodes=60000 --save-dir=./logs/simple_spread_IRCR/``

``python train.py --scenario=simple_spread --num-episodes=60000 --save-dir=./logs/simple_spread_MADDPG_alone/``

## Command-line options

### Environment options

- `--scenario`: defines which environment in the MPE is to be used (default: `"simple"`)

- `--max-episode-len` maximum length of each episode for the environment (default: `25`)

- `--num-episodes` total number of training episodes (default: `60000`)

- `--num-adversaries`: number of adversaries in the environment (default: `0`)

### Core training parameters

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

### Checkpointing

- `--exp-name`: name of the experiment, used as the file name to save all results (default: `None`)

- `--save-dir`: directory where intermediate training results and model will be saved (default: `"/tmp/policy/"`)

- `--save-rate`: model is saved every time this number of episodes has been completed (default: `1000`)

- `--load-dir`: directory where training state and model are loaded from (default: `""`)

### Evaluation

- `--restore`: restores previous training state stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), and continues training (default: `False`)

- `--display`: displays to the screen the trained policy stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), but does not continue training (default: `False`)

- `--benchmark-dir`: directory where benchmarking data is saved (default: `"./benchmark_files/"`)

- `--plots-dir`: directory where training curves are saved (default: `"./learning_curves/"`)

## Code structure

- `./experiments/train.py`: contains code for SAM-Uniform and training MADDPG on the MPE

- `./maddpg/trainer/maddpg.py`: core code for SAM-NonUniform and the MADDPG algorithm

- `./maddpg/trainer/replay_buffer.py`: replay buffer code for MADDPG



## Acknowledgement

The code of MADDPG is based on the publicly available implementation of [MADDPG](https://github.com/openai/maddpg)

