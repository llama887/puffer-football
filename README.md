# Puffer Football

Puffer Football is a high-throughput, headless, multi-agent fork of
[Google Research Football](https://github.com/google-research/football) for
self-play with [PufferLib](https://github.com/PufferAI/PufferLib). It keeps the
original game and adds a vectorized 22-agent environment, training on
automatically randomized scenes (ADR), policy evaluation, and Torch/Slurm
jobs.

## Setup with UV (recommended)

The tested Python version is 3.10. On Ubuntu/Debian, install the native build
dependencies first:

```shell
sudo apt-get update
sudo apt-get install -y git cmake build-essential pkg-config \
  libgl1-mesa-dev libegl1-mesa-dev libglu1-mesa-dev \
  libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev \
  libboost-all-dev
```

Then install the project into a repo-local UV environment:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/llama887/puffer-football.git
cd puffer-football

uv python install 3.10
uv venv --python 3.10
source .venv/bin/activate

# Install valid dependencies and this editable package with UV first.
uv pip install 'pip<24' 'setuptools==65.5.0' 'wheel==0.38.4' \
  'numpy<2' pygame opencv-python psutil absl-py \
  'gymnasium>=0.29.1' 'pufferlib>=3.0,<3.1'
CMAKE_ARGS='-DCMAKE_POLICY_VERSION_MINIMUM=3.5' \
  uv pip install --no-deps --no-build-isolation -e .

# gym 0.21 has invalid legacy metadata that UV intentionally rejects, so it
# must be installed last and is the only dependency installed through pip.
python -m pip install --no-build-isolation 'gym==0.21.0'
```

The last command compiles the C++ game engine and may take several minutes.
Verify the installation with:

```shell
python -m gfootball.env.puffer_env_test
python -m gfootball.examples.benchmark_fast_mode --steps=500 --repeats=3
```

### Torch HPC

On NYU Torch, the setup script creates a repo-local UV `.venv` that inherits
the already validated `football-fast` CUDA/PufferLib environment. This avoids
duplicating the large Torch/CUDA stack under the scratch quota. UV owns the
venv, and the setup script links this checkout into it as editable source:

```shell
cd /scratch/$USER/repos/football
bash scripts/setup_uv_hpc.sh
sbatch sbatch/test_fast_env.sbatch
```

Set `TOOLCHAIN_PREFIX=/path/to/environment` if the validated environment is
somewhere other than `/scratch/$USER/.conda/envs/football-fast`.

Future evaluation and training jobs use this `.venv` automatically:

```shell
sbatch sbatch/evaluate_policy.sbatch /path/to/checkpoint.pt
sbatch sbatch/train_regularized.sbatch
```

## About the upstream environment

The base environment is an RL environment built on the open-source Gameplay
Football game. It was created by the Google Brain team for research purposes.

Useful links:

* [Run in Colab](https://colab.research.google.com/github/google-research/football/blob/master/gfootball/colabs/gfootball_example_from_prebuild.ipynb) - start training in less that 2 minutes.
* [Google Research Football Paper](https://arxiv.org/abs/1907.11180)
* [GoogleAI blog post](https://ai.googleblog.com/2019/06/introducing-google-research-football.html)
* [Google Research Football on Cloud](https://towardsdatascience.com/reproducing-google-research-football-rl-results-ac75cf17190e)
* [GRF Kaggle competition](https://www.kaggle.com/c/google-football) - take part in the competition playing games against others, win prizes and become the GRF Champion!


We'd like to thank Bastiaan Konings Schuiling, who authored and open-sourced the original version of this game.


## Quick Start

### In colab

Open our example [Colab](https://colab.research.google.com/github/google-research/football/blob/master/gfootball/colabs/gfootball_example_from_prebuild.ipynb), that will allow you to start training your model in less than 2 minutes.

This method doesn't support game rendering on screen - if you want to see the game running, please use the method below.

### Using Docker

This is the recommended way for Linux-based systems to avoid incompatible package versions.
Instructions are available [here](gfootball/doc/docker.md).

### On your computer

#### 1. Install required packages
#### Linux
```shell
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

python3 -m pip install --upgrade pip setuptools psutil wheel
```

#### macOS
First install [brew](https://brew.sh/). It should automatically install Command Line Tools.
Next install required packages:

```shell
brew install git python3 cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-python3

python3 -m pip install --upgrade pip setuptools psutil wheel
```


#### Windows
Install [Git](https://git-scm.com/download/win) and [Python 3](https://www.python.org/downloads/).
Update pip in the Command Line (here and for the **next steps** type `python` instead of `python3`)
```commandline
python -m pip install --upgrade pip setuptools psutil wheel
```


#### 2. Install GFootball
#### Option a. From PyPi package (recommended)
```shell
python3 -m pip install gfootball
```

#### Option b. Installing from sources using GitHub repository 
(On Windows you have to install additional tools and set an environment variable, see 
[Compiling Engine](gfootball/doc/compile_engine.md#windows) for detailed instructions.)

```shell
git clone https://github.com/google-research/football.git
cd football
```

Optionally you can use [virtual environment](https://docs.python.org/3/tutorial/venv.html):

```shell
python3 -m venv football-env
source football-env/bin/activate
```

Next, build the game engine and install dependencies:

```shell
python3 -m pip install .
```
This command can run for a couple of minutes, as it compiles the C++ environment in the background.
If you face any problems, first check [Compiling Engine](gfootball/doc/compile_engine.md) documentation and search GitHub issues.


#### 3. Time to play!
```shell
python3 -m gfootball.play_game --action_set=full
```
Make sure to check out the [keyboard mappings](#keyboard-mappings).
To quit the game press Ctrl+C in the terminal.

### Fast headless self-play with PufferLib

Install the optional interface and create a CPU-vectorized environment:

```shell
python3 -m pip install -e '.[pufferlib]'
```

```python
from gfootball.env.puffer_env import make_vector_env

env = make_vector_env(num_envs=8, num_workers=8)
```

Each match exposes all 22 players as Puffer agents with four stacked 115-float
observations and the default 19-action set. Every episode is an automatic
domain randomization (ADR) scene, `gfootball/scenarios/11_vs_11_adr.py`: eight
settings drawn from ranges the trainer grows and shrinks (see
[ADR](#automatic-domain-randomization-adr)); without them the env plays the
start scene, a carrier at the ball 0.1 from the goal line with one blocker in
front. Headless fast mode preserves the original ten physics phases while
skipping trace, video, dump, animated officials, redundant Python copies, and
unused bot tactics. The ADR scenes also disable movement magnetism,
automatic pass/shot aiming and power, and automatic standing interference.
Stock scenarios and rendered evaluation keep the original features.

#### Training on Torch

`sbatch/train_selfplay.sbatch` runs recurrent shared-policy PPO self-play on
ADR scenes.  It asks for 48 CPUs and one GPU of any type (Torch allows 48
CPUs per L40S-class GPU, only 32 per H100) and runs `CPUs - 2` environment
workers (46).  Its defaults are the entity-observation transformer with the
September 29 tuning study's trial-3 settings, which every run since October 2
has used.  Submit it as an array to train one seed per task; single-seed
results on this task are inside the seed noise:

```shell
cd /scratch/$USER/repos/football
sbatch --array=0-2 --account=<account> \
  --export=ALL,TOTAL_TIMESTEPS=5000000000,SEED_BASE=20261002 \
  sbatch/train_selfplay.sbatch
```

Queue times differ a lot between Slurm accounts.  `--test-only` prints an
estimated start without submitting, so check each account you can use first:

```shell
for A in <account> <account> ...; do
  sbatch --test-only --account=$A --cpus-per-task=48 --gres=gpu:1 \
    --time=24:00:00 --wrap=true 2>&1 | tail -1
done
```

`sbatch/smoke_train.sbatch` trains seven epochs on CPUs only, with an
evaluation every three, to check the whole loop end to end.

Every setting is an environment variable passed with `--export=ALL,...`.
The throughput-related ones (defaults in brackets):

| Variable | Meaning |
|---|---|
| `ENVS_PER_WORKER` [1] | matches stepped by each worker process |
| `GRAPH_UPDATE` [1] | capture one whole-rollout PPO step (forward, loss, backward, clip, Adam) as a CUDA graph per rollout and replay it `UPDATE_EPOCHS` times |
| `GRAPH_ACTOR` [1] | replay the acting forward and sampling as one CUDA graph |
| `OVERLAP_COLLECTION` [1] | keep the environments stepping while the update runs (data at most one update stale; the stored log-prob keeps PPO's ratio exact) |
| `COMPILE` [0] | `torch.compile` the BPTT forward; redundant with the graphed update |
| `GPU_FILLER` [1] | keep reported GPU utilization up with a single-block spin kernel on a side stream |

Steady-state trained samples per second, one L40S + 48 CPUs, simple115 with
the MLP (`--benchmark-epochs`, evaluation excluded; measured before ADR, on
the old level-0 scenes):

| Workers x matches per worker | Trained samples/s | Peak memory |
|---|---|---|
| 46 x 1 (default) | 121k-126k | 6 GB |
| 46 x 2 | 130k | 12 GB |
| 46 x 4 | 127k | 17 GB |
| 46 x 8 | 109k | 28 GB |
| 92 x 1 | 99k | 10 GB |
| 46 x 16 | 98k | 50 GB |
| before the graphed update and overlap, 46 x 1 | 65k | 9 GB |

With whole-rollout steps (the update graph captured once per layout
capacity and replayed), 46 x 1 trains 121k-124k samples/s at 7, 16 or 32
`UPDATE_EPOCHS` alike: the update overlaps collection completely, so more
optimizer steps per rollout cost no throughput.  `OBSERVATION=entities` with
the MLP runs at the same speed.  `NETWORK=transformer` is update-bound: 85k
samples/s at 7 update epochs, 39k at 16 (one whole-rollout step is ~47 ms
of attention in bfloat16 against ~260 ms to collect a rollout).

More matches than CPUs gains at most a few percent, because collection
(~275k steps/s) already outruns the update, and a bigger batch grows the
update with it; more worker processes than CPUs is slower.  More matches also
means a bigger batch and more optimizer steps per epoch, which changes
learning, not just speed.

Every optimizer step trains on the whole rollout (one 32-step segment per
agent, ~0.6 GB on the GPU at 46 matches), so there is no minibatch size:
`UPDATE_EPOCHS` is the number of optimizer steps per rollout.  The LSTM runs
each training window as one packed cuDNN call: segments are cut at episode
ends into pieces that each start from zero memory, exactly as the actor saw
them.

With `ANNEAL_LR=1` the learning rate anneals on a cosine over training
progress (agent steps done / `TOTAL_TIMESTEPS`), reaching exactly zero at the
end; the default keeps it constant.  It lives in a GPU tensor updated in
place, so the graphed update follows it.

#### Observation and network

`OBSERVATION=entities` (the default) replaces egocentric simple115v2 with
per-entity rows (`gfootball/env/entity_observation.py`) in a normalized
egocentric frame: the ball and all 22 players as offsets and distances from
the agent, velocities, who has the ball, keepers, offside positions and
tiredness, plus offsets from the agent to both goals and both touchlines, its
held sticky buttons (direction, sprint, dribble), the game mode and the steps
left before the episode times out.  No feature is an absolute pitch position;
axes stay aligned with the pitch (+x toward the attacked goal) because the
movement actions are pitch directions.
`NETWORK=transformer` (the default; needs `OBSERVATION=entities`)
replaces the MLP trunk with two self-attention layers over the player, ball
and context tokens, masked to present players and independent of player
order; the LSTM and heads are unchanged.  No actions are masked: the engine
buffers a kick press for up to 20 steps and it steers the player to the ball
meanwhile, so no kick is provably a no-op from the observation.

#### Reward shaping

`BALL_POTENTIAL=k` adds potential-based reward shaping on the ball's progress:
every step pays `gamma * Phi(s') - Phi(s)` with `Phi = k * x`, where `x` is
the ball's position toward the goal that side attacks (in [-1, 1]), and
`Phi = 0` in the absorbing state.  The other team gets exactly `-Phi`, so
shaping is zero-sum like the score, and by Ng, Harada and Russell (1999,
Theorem 1) it leaves the optimal policy unchanged.  The trainer's `--gamma`
is used for the whole potential difference; it must match PPO's discount for
that guarantee.  Rewards are stored unclipped: clipping to [-1, 1] would cut
the terminal refund of the potential and break the cancellation.  Thus the
**discounted** sum of shaping rewards is exactly `-Phi(initial_state)`
regardless of trajectory or outcome.  This preserves the underlying
discounted objective; it does not guarantee PPO convergence or faster
learning.  Evaluation never uses shaping.

`PLAYER_POTENTIAL=k` additionally shapes the distance of the closest teammate
to the ball (including the goalkeeper, ignoring absent players), in pitch
half-length units with the engine's x/y scales accounted for.  Unlike the
ball term it is not zero-sum.  It defaults to zero.  See
[Ng, Harada and Russell (1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).
`PLAYER_POTENTIAL_FADE_SPEED=v` fades the player term to zero as the ball's
speed reaches `v` (ball-direction units: a dribbled ball stays under about
0.27, a struck one is about 1), so a kick costs no potential.

#### Automatic domain randomization (ADR)

Training scenes come from automatic domain randomization (OpenAI, "Solving
Rubik's Cube with a Robot Hand", 2019, Section 5; `gfootball/adr.py`).  Eight
scene settings each get a range that starts at a single start value: ball
distance from the goal line, ball lateral position, goal-side blockers, fetch
distance to a still ball, the defence's head start in the race for it,
attackers ahead of the ball, how far the formations are from packed around
the ball (0) toward real match formations (1), and the share of players
already holding a direction.  Every training episode draws each setting
uniformly from its range, so easy scenes stay in the mix.  Every episode
ends as soon as the ball goes out of play: with the magnet off the engine
never completes a set piece for agent-controlled players, so a goal kick,
corner or throw-in would otherwise freeze the match until the timeout.

`ADR_PROBE_FRACTION` (0.2) of episodes are probes: one setting is pinned at
one edge of its range, and the attack plays against the probe defence, whose
rows are hidden from training.  After `ADR_BUFFER_SIZE` (64) probes at an
edge, a success rate of at least `ADR_HIGH_THRESHOLD` (0.6) moves that edge
outward one step and at most `ADR_LOW_THRESHOLD` (0.3) moves it inward.

The probe defence matters.  With `ADR_PROBE_DEFENCE=random` (uniform random
defenders, which score the same as defenders standing still), the October 4
runs pushed every defensive setting to its limit by epoch ~12k of 154k:
against random defenders, five blockers cost the attack nothing (0.61 with 0
or 5) and real formations even look easier.  Against a trained defence the
same edges separate (formation 0.55 vs 0.34, head start 0.56 vs 0.25).  So
the default, `ADR_PROBE_DEFENCE=self`, grows the probe defence with the run:
the learner's starting weights are saved to
`<run>/checkpoints/adr_probe_defence.pt` and, every
`ADR_PROBE_REFRESH_INTERVAL` epochs (1000, a multiple of
`EVALUATION_INTERVAL`), the learner attacks it and its own defence on the
evaluation scenes.  Once the learner scores at least
`ADR_PROBE_REFRESH_MARGIN` (0.1) more against the probe defence, the learner's
current weights replace it (copies are kept as
`adr_probe_defence_epoch<N>.pt`); the env workers reload it at their next
probe.  `ADR_PROBE_DEFENCE` can also be a policy file, copied to the same
place and refreshed the same way; `FROZEN_DEFENCE_HORIZON` sets the memory
window such a defence trained by another run used (0: whole episodes).
`random` and `idle` are never refreshed: they stay random and idle for the
whole run.

To start the probe defence from the October 4 ADR seed-0 final
(`defences/adr_probe_defence_init.pt`, as the October 7 runs did) instead,
from the repo root:

```shell
sbatch --array=0-2 --account=<account> --job-name=football-tf-adr-probe \
  --export=ALL,TOTAL_TIMESTEPS=5000000000,SEED_BASE=20261002,ADR_PROBE_DEFENCE=$PWD/defences/adr_probe_defence_init.pt \
  sbatch/train_selfplay.sbatch
```

Every `EVALUATION_INTERVAL` (100) epochs the log prints an `ADR {json}`
line: both edges of every range (`adr/<setting>/low`, `.../high`), the ADR
entropy (mean log range width), `vs_random/difficulty_<d>/success_rate`
against uniform random defenders at each `EVALUATION_DIFFICULTIES` test
difficulty (0.25, 0.5, 1.0; `EVALUATION_EPISODES` episodes each), and on
refresh epochs `adr_probe/vs_probe`, `adr_probe/vs_self` and
`adr_probe/replaced`.  A test difficulty `d` draws every setting uniformly
from its start value out to `d` of the way to its valid limits
(`set_difficulty` in `gfootball/adr.py`): the scenes depend on `d` and the
seed alone, never on the ranges a run has reached, so the numbers compare
across runs.  A policy probe defence (`self` or a file) costs about 5% of
throughput (the transformer runs on the env workers' CPUs during probes).

Results of the October runs, against the fixed defender on the old level
5 / 9 / 14 scenes (three seeds, mean of checkpoints 140k, 147k and final):

| Run | L5 | L9 | L14 | L14 by seed |
|---|---|---|---|---|
| Level curriculum (control) | 0.70 | 0.27 | 0.26 | 0.13 / 0.08 / 0.56 |
| ADR, random probes (October 4) | 0.59 | 0.32 | 0.19 | 0.21 / 0.18 / 0.19 |

The level curriculum these were compared with has been removed; the commit
before the removal is tagged `pre-adr-only`.  The ADR policies only score
with the ball starting between the posts, and the ball's lateral range never
grew past about 0.06 (2026-10-09 analysis, outside this repo).

#### Scoring runs against a fixed defence

`scripts/attack_vs_defence.py` plays a checkpoint's attack against a chosen
defence on the same scenes every time, and `sbatch/attack_vs_defence.sbatch`
runs one row of a config file per array task.  Runs are scored with 128
episodes, seed 4242, against the fixed defender `defences/fixed_defender.pt`
(the September 29 simple115 MLP, seed 1, final; on Torch it was
`football-selfplay-18794847-1/checkpoints/yxu10gg8/model_153674.pt`).  For
example, at the three test difficulties, from the repo root:

```shell
FIXED=$PWD/defences/fixed_defender.pt
RUN=/scratch/fyy2003/experiments/football-selfplay-19372742-0/checkpoints/pumwbkuo
for D in 0.25 0.5 1.0; do
  echo "final_d$D $RUN/model_153663.pt $FIXED adr:difficulty:$D"
done > /scratch/$USER/configs.txt
sbatch --array=0-2 --account=<account> \
  --export=ALL,CONFIGS=/scratch/$USER/configs.txt,OUT=/scratch/$USER/scores \
  sbatch/attack_vs_defence.sbatch
```

Each task writes `OUT/<label>.json` and prints a `SUMMARY {json}` line with
the goal rate (`scored`) and how the defence played: `regained` (it won the
ball), `goal_side` (outfield defenders between the ball and their goal while
the attack has it) and `pressure` (nearest defender's distance to the ball).
The defence can also be `self`, `random` or `idle`, and the scene
`adr:start`, `adr:full` or `adr:pin:<setting>:<0|1>` (an ADR probe);
`ADR_LOG=<training log>` takes the ADR ranges from that run's last `ADR`
line.  `VIDEO=1` renders every episode to webm, which needs a GPU node for
EGL: add `--gres=gpu:1 --partition=h100_tandon,a100_tandon`.

# Contents #

* [Running training](#training-agents-to-play-GRF)
* [Playing the game](#playing-the-game)
    * [Keyboard mappings](#keyboard-mappings)
    * [Play vs built-in AI](#play-vs-built-in-AI)
    * [Play vs pre-trained agent](#play-vs-pre-trained-agent)
    * [Trained checkpoints](#trained-checkpoints)
* [Environment API](gfootball/doc/api.md)
* [Observations & Actions](gfootball/doc/observation.md)
* [Scenarios](gfootball/doc/scenarios.md)
* [Multi-agent support](gfootball/doc/multi_agent.md)
* [Running in docker](gfootball/doc/docker.md)
* [Saving replays, logs, traces](gfootball/doc/saving_replays.md)
* [Imitation Learning](gfootball/doc/imitation.md)

## Training agents to play GRF

### Run training
In order to run TF training, you need to install additional dependencies

- Update PIP, so that tensorflow 1.15 is available: `python3 -m pip install --upgrade pip setuptools wheel`
- TensorFlow: `python3 -m pip install tensorflow==1.15.*` or
  `python3 -m pip install tensorflow-gpu==1.15.*`, depending on whether you want CPU or
  GPU version;
- Sonnet and psutil: `python3 -m pip install dm-sonnet==1.* psutil`;
- OpenAI Baselines:
  `python3 -m pip install git+https://github.com/openai/baselines.git@master`.

Then:

- To run example PPO experiment on `academy_empty_goal` scenario, run
  `python3 -m gfootball.examples.run_ppo2 --level=academy_empty_goal_close`
- To run on `academy_pass_and_shoot_with_keeper` scenario, run
  `python3 -m gfootball.examples.run_ppo2 --level=academy_pass_and_shoot_with_keeper`

In order to train with nice replays being saved, run
`python3 -m gfootball.examples.run_ppo2 --dump_full_episodes=True --render=True`

In order to reproduce PPO results from the paper, please refer to:

- gfootball/examples/repro_checkpoint_easy.sh
- gfootball/examples/repro_scoring_easy.sh

## Playing the game

Please note that playing the game is implemented through an environment, so human-controlled players use the same interface as the agents.
One important implication is that there is a single action per 100 ms reported to the environment, which might cause a lag effect when playing.


### Keyboard mappings
The game defines following keyboard mapping (for the `keyboard` player type):

* `ARROW UP` - run to the top.
* `ARROW DOWN` - run to the bottom.
* `ARROW LEFT` - run to the left.
* `ARROW RIGHT` - run to the right.
* `S` - short pass in the attack mode, pressure in the defense mode.
* `A` - high pass in the attack mode, sliding in the defense mode.
* `D` - shot in the attack mode, team pressure in the defense mode.
* `W` - long pass in the attack mode, goalkeeper pressure in the defense mode.
* `Q` - switch the active player in the defense mode.
* `C` - dribble in the attack mode.
* `E` - sprint.

### Play vs built-in AI
Run `python3 -m gfootball.play_game --action_set=full`. By default, it starts
the base scenario and the left player is controlled by the keyboard. Different
types of players are supported (gamepad, external bots, agents...). For possible
options run `python3 -m gfootball.play_game -helpfull`.

### Play vs pre-trained agent

In particular, one can play against agent trained with `run_ppo2` script with
the following command (notice no action_set flag, as PPO agent uses default
action set):
`python3 -m gfootball.play_game --players "keyboard:left_players=1;ppo2_cnn:right_players=1,checkpoint=$YOUR_PATH"`

### Trained checkpoints
We provide trained PPO checkpoints for the following scenarios:

  - [11_vs_11_easy_stochastic](https://storage.googleapis.com/gfootball-public-bucket/trained_model_11_vs_11_easy_stochastic),
  - [academy_run_to_score_with_keeper](https://storage.googleapis.com/gfootball-public-bucket/trained_model_academy_run_to_score_with_keeper_v2).

In order to see the checkpoints playing, run
`python3 -m gfootball.play_game --players "ppo2_cnn:left_players=1,policy=gfootball_impala_cnn,checkpoint=$CHECKPOINT" --level=$LEVEL`,
where `$CHECKPOINT` is the path to downloaded checkpoint. Please note that the checkpoints were trained with Tensorflow 1.15 version. Using 
different Tensorflow version may result in errors. The easiest way to run these checkpoints is through provided `Dockerfile_examples` image.
See [running in docker](gfootball/doc/docker.md) for details (just override the default Docker definition with `-f Dockerfile_examples` parameter).

In order to train against a checkpoint, you can pass 'extra_players' argument to create_environment function.
For example extra_players='ppo2_cnn:right_players=1,policy=gfootball_impala_cnn,checkpoint=$CHECKPOINT'.
