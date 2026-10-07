# Puffer Football

Puffer Football is a high-throughput, headless, multi-agent fork of
[Google Research Football](https://github.com/google-research/football) for
self-play with [PufferLib](https://github.com/PufferAI/PufferLib). It keeps the
original game and adds a vectorized 22-agent environment, curriculum training,
policy evaluation, and Torch/Slurm jobs.

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
observations and the default 19-action set. The default curriculum begins with
the ball and attackers near a goal. Each worker advances after the attacking
side scores in 60% of its last 20 episodes, gradually adding defenders and
moving the ball to midfield until it reaches stock 11v11. Headless fast mode
preserves the original ten physics phases while skipping trace, video, dump,
animated officials, redundant Python copies, and unused bot tactics. The
curriculum also disables movement magnetism, automatic pass/shot aiming and
power, and automatic standing interference. Stock scenarios and rendered
evaluation keep the original features.

#### Training on Torch

`sbatch/train_selfplay.sbatch` runs recurrent shared-policy PPO self-play on
the `11_vs_11_advantage` curriculum.  It asks for 48 CPUs and one GPU of any
type (Torch allows 48 CPUs per L40S-class GPU, only 32 per H100) and runs
`CPUs - 2` environment workers (46).  Submit it as an array to train one seed
per task; single-seed results on this task are inside the seed noise:

```shell
cd /scratch/$USER/repos/football
sbatch --array=0-2 --account=<account> sbatch/train_selfplay.sbatch
python scripts/summarize_runs.py sbatch/logs/football-selfplay-<array id>-*.out
```

Queue times differ a lot between Slurm accounts.  `--test-only` prints an
estimated start without submitting, so check each account you can use first:

```shell
for A in <account> <account> ...; do
  sbatch --test-only --account=$A --cpus-per-task=48 --gres=gpu:1 \
    --time=24:00:00 --wrap=true 2>&1 | tail -1
done
```

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

Steady-state trained samples per second, one L40S + 48 CPUs, level 0,
trial-36 settings below (`--benchmark-epochs`, promotion excluded):

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

The learning rate anneals (`ANNEAL_LR=1`) on a cosine over training progress
(agent steps done / `TOTAL_TIMESTEPS`), reaching exactly zero at the end.
It lives in a GPU tensor updated in place, so the graphed update follows it.

#### Observation and network

`OBSERVATION=entities` replaces egocentric simple115v2 with per-entity rows
(`gfootball/env/entity_observation.py`) in a normalized egocentric frame: the
ball and all 22 players as offsets and distances from the agent, velocities,
who has the ball, keepers, offside positions and tiredness, plus offsets from
the agent to both goals and both touchlines, its held sticky buttons
(direction, sprint, dribble), the game mode and the steps left before the
episode times out.  No feature is an absolute pitch position; axes stay
aligned with the pitch (+x toward the attacked goal) because the movement
actions are pitch directions.
`NETWORK=transformer` (needs `OBSERVATION=entities`)
replaces the MLP trunk with two self-attention layers over the player, ball
and context tokens, masked to present players and independent of player
order; the LSTM and heads are unchanged.  No actions are masked: the engine
buffers a kick press for up to 20 steps and it steers the player to the ball
meanwhile, so no kick is provably a no-op from the observation.

#### Curriculum, gate and metrics

Promotion to the next level is gated on held-out evaluation against a
*frozen* snapshot of the policy taken on entering the level, so the score
target is fixed rather than the same network's improving defence
(`FROZEN_DEFENCE_GATE=0` restores the live self-play gate).  A level passes
at 60% overall success with at least 40% on the two weakest spawn templates
(`CURRICULUM_SUCCESS_THRESHOLD`; `1.0` keeps training on one level).
`FROZEN_DEFENCE_INIT=/path/policy.pt` makes the level-0 gate play that fixed
policy instead of the untrained one.  Each gate result is also logged per
level as `level_entry/level_<L>/success_rate` (and `.../worst_two_...`): the
opponent is fixed while a level trains, so a flat line marks a stalled level.
Greedy and live self-play diagnostic evaluations run on every 4th check
(`DIAGNOSTIC_PROMOTION_EVERY`), and a hopeless gate evaluation stops after a
quarter of its episodes.  Evaluation scenes are keyed on each worker's
(seed, episode) pair, so a 256-episode gate plays 256 distinct scenes.

Goal-side blockers fade in by probability across levels instead of arriving
one whole defender at a time; the side a lone blocker takes is a per-episode
coin flip, so mirror-image templates are equally hard.  Every episode ends
as soon as the ball goes out of play: with the magnet off the engine never
completes a set piece for agent-controlled players, so a goal kick, corner or
throw-in would otherwise freeze the match until the timeout.

`SPAWN=uniform` replaces the curriculum with uniformly random scenes: the
ball and outfield players anywhere on the pitch, keepers inside their own
penalty area, every attacker onside, offside on, 400-step episodes.  Use it
with `CURRICULUM_SUCCESS_THRESHOLD=1.0`.

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
learning.  Promotion evaluation never uses shaping.

`PLAYER_POTENTIAL=k` additionally shapes the distance of the closest teammate
to the ball (including the goalkeeper, ignoring absent players), in pitch
half-length units with the engine's x/y scales accounted for.  Unlike the
ball term it is not zero-sum.  Both scales default to zero.  See
[Ng, Harada and Russell (1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).
`PLAYER_POTENTIAL_FADE_SPEED=v` fades the player term to zero as the ball's
speed reaches `v` (ball-direction units: a dribbled ball stays under about
0.27, a struck one is about 1), so a kick costs no potential.

Dead-ball practice (all off by default; on three transformer seeds none
clearly beat the plain run below, whose seed spread was as large as any gap):
`WIDTH_SPAWN_FRACTION` spreads that share of training episodes' outfield
players across the pitch width; `MIXED_SHAPE=1` puts some attackers ahead of
the ball, onside, from level 1 (the gate plays it too); and
`LOOSE_BALL_FRACTION` opens that share of training episodes as a race for a
still ball, with some players already holding a direction.

#### Replicating the best run so far

The strongest run before ADR is the entity-observation transformer on the
level curriculum with ball shaping only: the October 2 control arm, seed 2
(`/scratch/fyy2003/experiments/football-selfplay-19011722-2`, 5B agent steps,
about 24 h on one GPU and 48 CPUs).  It entered level 16 of 0-20 and, against
the fixed defender (see [Scoring runs](#scoring-runs-against-a-fixed-defence)),
scored 0.87 / 0.33 / 0.56 at levels 5 / 9 / 14, the mean of checkpoints 140k,
147k and final.  Its final checkpoint is
`checkpoints/x3bep4me.pt` in that run directory.  The settings are the
September 29 study's best transformer (trial 3) without annealing:

```shell
cd /scratch/$USER/repos/football
sbatch --array=2 --account=<account> --job-name=football-tf-control \
  --export=ALL,TOTAL_TIMESTEPS=5000000000,SEED_BASE=20261002,OBSERVATION=entities,NETWORK=transformer,LEARNING_RATE=0.000543,ENT_COEF=0.00159,GAMMA=0.995,GAE_LAMBDA=0.927,CLIP_COEF=0.132,VF_COEF=0.428,UPDATE_EPOCHS=3,BALL_POTENTIAL=0.5,ANNEAL_LR=0 \
  sbatch/train_selfplay.sbatch
```

Array task 2 with `SEED_BASE=20261002` is seed 20261004.  The run used commit
24eea79; everything added since is off by default, so the current code trains
the same way (it only adds the random-defender evaluation to the logs).  It cannot be replayed bit for bit (asynchronous collection
takes whichever workers are ready first), and the seed matters as much as
any setting: the same command scored 0.72 / 0.26 / 0.13 on task 0 and
0.52 / 0.21 / 0.08 on task 1.  Run `--array=0-2` and compare the means
(0.70 / 0.27 / 0.26 for this arm) rather than chasing one seed.  When
launching from a study snapshot without its own `.venv`, add
`ENV_PREFIX=/scratch/$USER/repos/football/.venv` to the exports.

#### Automatic domain randomization (ADR)

`ADR=1` replaces the level curriculum with automatic domain randomization
(OpenAI, "Solving Rubik's Cube with a Robot Hand", 2019, Section 5;
`gfootball/adr.py`).  Eight scene settings each get a range that starts at
the level-0 value: ball distance from the goal line, ball lateral position,
goal-side blockers, fetch distance to a still ball, the defence's head start
in the race for it, attackers ahead of the ball, how far the formations are
from the packed level-0 shape (0) toward real match formations (1), and the
share of players already holding a direction.  Every training episode draws
each setting uniformly from its range, so easy scenes stay in the mix.

`ADR_PROBE_FRACTION` (0.2) of episodes are probes: one setting is pinned at
one edge of its range, and the attack plays against the probe defence, whose
rows are hidden from training.  After `ADR_BUFFER_SIZE` (64) probes at an
edge, a success rate of at least `ADR_HIGH_THRESHOLD` (0.6) moves that edge
outward one step and at most `ADR_LOW_THRESHOLD` (0.3) moves it inward.  The
level gate is off.

The probe defence matters.  With the default `ADR_PROBE_DEFENCE=random`
(uniform random defenders, which score the same as defenders standing
still), the October 4 runs pushed every defensive setting to its limit by
epoch ~12k of 154k: against random defenders, five blockers cost the attack
nothing (0.61 with 0 or 5) and real formations even look easier.  Against a
trained defence the same edges separate (formation 0.55 vs 0.34, head start
0.56 vs 0.25).  So `ADR_PROBE_DEFENCE` can be a policy file instead.  It is
copied to `<run>/checkpoints/adr_probe_defence.pt` and, every
`ADR_PROBE_REFRESH_INTERVAL` epochs (1000, a multiple of the 100-epoch log
interval), the learner attacks it and its own defence on the
`RANDOM_BASELINE_LEVELS` scenes.  Once the learner scores at least
`ADR_PROBE_REFRESH_MARGIN` (0.1) more against the probe defence, the learner's
current weights replace it (copies are kept as
`adr_probe_defence_epoch<N>.pt`); the env workers reload it at their next
probe.

To run ADR with a snapshot probe defence, seeded like the run above, starting
the probe defence from the October 4 ADR seed-0 final:

```shell
sbatch --array=0-2 --account=<account> --job-name=football-tf-adr-probe \
  --export=ALL,TOTAL_TIMESTEPS=5000000000,SEED_BASE=20261002,OBSERVATION=entities,NETWORK=transformer,LEARNING_RATE=0.000543,ENT_COEF=0.00159,GAMMA=0.995,GAE_LAMBDA=0.927,CLIP_COEF=0.132,VF_COEF=0.428,UPDATE_EPOCHS=3,BALL_POTENTIAL=0.5,ANNEAL_LR=0,ADR=1,ADR_PROBE_DEFENCE=/scratch/fyy2003/experiments/football-adr-probe-snapshot-20261007/probe_defence_init.pt \
  sbatch/train_selfplay.sbatch
```

Leave out `ADR_PROBE_DEFENCE` for the random-probe version.  Every 100
epochs the log prints an `ADR {json}` line: both edges of every range
(`adr/<setting>/low`, `.../high`), the ADR entropy (mean log range width),
`vs_random/level_<L>/success_rate` against uniform random defenders at
`RANDOM_BASELINE_LEVELS` (5, 9, 14; every run logs these, ADR or not), and on
refresh epochs `adr_probe/vs_probe`, `adr_probe/vs_self` and
`adr_probe/replaced`.  The snapshot probe defence costs about 5% of
throughput (the transformer runs on the env workers' CPUs during probes).

Results so far, against the fixed defender at levels 5 / 9 / 14 (three seeds,
mean of checkpoints 140k, 147k and final):

| Run | L5 | L9 | L14 | L14 by seed |
|---|---|---|---|---|
| Level curriculum (control) | 0.70 | 0.27 | 0.26 | 0.13 / 0.08 / 0.56 |
| ADR, random probes (October 4) | 0.59 | 0.32 | 0.19 | 0.21 / 0.18 / 0.19 |

Random-probe ADR is steadier across seeds and the only change so far that
cut dead-ball episodes (by 35-40%), but it is weaker at level 5: once the
ranges saturate, scenes near the goal are rare.  The snapshot-probe runs
(October 7) are in
`/scratch/fyy2003/experiments/football-adr-probe-snapshot-20261007`.

#### Scoring runs against a fixed defence

Gate numbers are against each run's own snapshots, so they do not compare
across runs.  `scripts/attack_vs_defence.py` plays a checkpoint's attack
against a chosen defence on the same scenes every time, and
`sbatch/attack_vs_defence.sbatch` runs one row of a config file per array
task.  Runs above are scored at levels 5, 9 and 14, 128 episodes, seed
4242, against the fixed defender
`/scratch/fyy2003/experiments/football-selfplay-18794847-1/checkpoints/yxu10gg8/model_153674.pt`
(the September 29 simple115 MLP, seed 1, final), for checkpoints 140000,
147000 and the final one.  For example, the best run's final checkpoint:

```shell
FIXED=/scratch/fyy2003/experiments/football-selfplay-18794847-1/checkpoints/yxu10gg8/model_153674.pt
RUN=/scratch/fyy2003/experiments/football-selfplay-19011722-2/checkpoints/x3bep4me
for L in 5 9 14; do
  echo "best_L$L $RUN/model_153670.pt $FIXED level:$L"
done > /scratch/$USER/configs.txt
sbatch --array=0-2 --account=<account> \
  --export=ALL,CONFIGS=/scratch/$USER/configs.txt,OUT=/scratch/$USER/scores \
  sbatch/attack_vs_defence.sbatch
```

With seed 4242 this reproduces the recorded scores exactly (0.930 at level 5
for that checkpoint).  Each task writes `OUT/<label>.json` and prints a
`SUMMARY {json}` line with
the goal rate (`scored`) and how the defence played: `regained` (it won the
ball), `goal_side` (outfield defenders between the ball and their goal while
the attack has it) and `pressure` (nearest defender's distance to the ball).
The defence can also be `self`, `random` or `idle`, and the scene
`adr:start`, `adr:full` or `adr:pin:<setting>:<0|1>` (an ADR probe);
`ADR_LOG=<training log>` takes the ADR ranges from that run's last `ADR`
line.  `VIDEO=1` renders every episode to webm, which needs a GPU node for
EGL: add `--gres=gpu:1 --partition=h100_tandon,a100_tandon`.

#### Tuning level 0 with Optuna

`scripts/tune_optuna.py` searches PPO settings (learning rate and annealing,
entropy, discount, GAE lambda, clip, value coefficient, optimizer steps per
rollout, matches per worker, ball potential, and the model: observation and
trunk) with TPE, training and scoring on
level 0 only against a fixed opponent, and prunes the bottom quarter of
trials after ~50M agent steps.  Optuna runs from its own small venv so the
trainer's environment is untouched:

```shell
uv venv /scratch/$USER/envs/optuna && \
  uv pip install --python /scratch/$USER/envs/optuna/bin/python optuna
# A study directory holds repo/ (a copy of this source with third_party and
# resources) and opponents/level4.pt (the fixed gate opponent).
STUDY=/scratch/$USER/experiments/<study>
/scratch/$USER/envs/optuna/bin/python $STUDY/repo/scripts/tune_optuna.py init --study $STUDY
sbatch --array=0-39%10 --account=<account> --export=ALL,OPTUNA_STUDY=$STUDY \
  $STUDY/repo/sbatch/tune_optuna.sbatch
/scratch/$USER/envs/optuna/bin/python $STUDY/repo/scripts/tune_optuna.py summarize --study $STUDY
```

The September 25 study's winner (trial 36) is `LEARNING_RATE=5.04e-4
ANNEAL_LR=1 ENT_COEF=1.36e-3 GAMMA=0.999 GAE_LAMBDA=0.947 CLIP_COEF=0.153
VF_COEF=0.918 UPDATE_EPOCHS=7 BALL_POTENTIAL=0.571` with 128-segment
minibatches (about 55 optimizer steps per rollout), from before whole-rollout
steps; the next study retunes for them.

The September 29 study (whole-rollout steps, 46 x 1, 48 trials, the model
searched too) is the launcher's default: trial 22, 0.921 level-0 success
(September 25 winner re-run: 0.854), `LEARNING_RATE=7.06e-4 ANNEAL_LR=1
ENT_COEF=1.29e-3 GAMMA=0.999 GAE_LAMBDA=0.960 CLIP_COEF=0.119 VF_COEF=0.774
UPDATE_EPOCHS=7 BALL_POTENTIAL=0.585` with simple115 and the MLP.  The best
entity-observation runs were within noise of it (0.906, MLP) and the
transformer reached 0.836; the model mattered least of all settings on level
0 (importance 0.02, against 0.33 for the ball potential), so the models are
compared on the full curriculum.

#### Earlier shaping screen

The bounded screen in `scripts/tune_shaping.py` compares eight configurations
over the same three seeds, at 50M agent steps each (1.2B total). It includes
unshaped and ball-only controls, player scales 0.1/0.3/1.0, ball scales
0.3/1.0, learning rates 0.0001/0.0003, and entropy coefficients 0.001/0.003
as targeted variations, not a full Cartesian grid. Discount stays 0.997.
Final checkpoints are evaluated without shaping or early abort for 256
episodes at each of levels 4, 6, and 7, using identical evaluation seeds and
the same copied frozen opponents for every candidate. The primary ranking
is mean goal success across these levels and seeds; weak-scenario success
and seed variability are also reported. This is an initial screening budget,
not evidence of full-run convergence.

Create a study directory containing `repo/` (a copy of this source, with
`third_party` and `resources` available), `opponents/level4.pt`, `level6.pt`,
`level7.pt`, and `results/`. Submit
`sbatch --array=0-23%3 --export=ALL,SHAPING_STUDY=/absolute/study sbatch/tune_shaping.sbatch`.
After the array finishes, run
`python scripts/tune_shaping.py summarize --study /absolute/study`.
It writes `summary.json` and selects a screening winner only after all 24
full-budget trials have produced benchmark results. Keep source and opponent
snapshots fixed for the entire study. The September 21 study is stored under
`/scratch/fyy2003/experiments/football-player-ball-shaping-20260921`.

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
