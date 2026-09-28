#!/usr/bin/env python3
"""Optuna search over PPO settings, trained and scored on level 0 only.

Why: the shaping study scored policies on levels 4/6/7 although every trial
finished training at level 2-4, and probes of its best checkpoint showed a
near-uniform policy that almost never holds the ball and scores only with an
opening first-touch strike. This search optimises level 0 alone before any
later level is considered.

How a trial works (one trial per Slurm array task):
  * Optuna samples the PPO settings (TPE, multivariate, constant liar so
    parallel workers do not sample the same point).
  * The unchanged trainer (repo/sbatch/train_selfplay.sbatch) runs with the
    curriculum threshold at 1.0 (256/256 wins), so it stays on level 0; a
    trial that ever reaches that is stopped as mastered. It runs with its
    gate opponent fixed to opponents/level4.pt instead of the untrained
    policy. Its periodic gate evaluation (256 sampled episodes every ~2.1M
    agent steps) is therefore level-0 success against one fixed defence.
  * Each evaluation is streamed from the trainer's PROMOTION log lines and
    reported to Optuna as the mean of the last three evaluations (smoothing
    out 256-episode noise). A percentile pruner stops trials that sit in the
    bottom quarter after a long warm-up, because the same settings can take
    very different times to learn level 0 depending on the seed.
  * The trial value is level-0 success averaged over its final three
    evaluations (768 episodes).

How it differs from scripts/tune_shaping.py: that script runs a fixed grid,
trains through the curriculum, and scores on later levels afterwards. This
one searches adaptively, never leaves level 0, scores during training, and
stops weak trials early so the survivors can train four times longer.

This script only needs optuna and the standard library; the trainer runs in
its own environment. Actions: init (once), run (per array task), summarize.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import optuna

STUDY_NAME = 'level0'
MAX_STEPS = 200_000_000
# Gate evaluations happen every PROMOTION_EPOCHS * 21120 agent steps at one
# environment per worker; the interval is divided by the environments per
# worker so evaluations stay ~2.1M agent steps apart in every trial.
PROMOTION_EPOCHS = 100
SMOOTHING = 3
SEED_BASE = 20261000
OPPONENT = 'opponents/level4.pt'
# The trainer's Python environment; the study's repo snapshot has no .venv of
# its own.  Override with FOOTBALL_ENV_PREFIX.
FOOTBALL_PYTHON_ENV = os.environ.get(
    'FOOTBALL_ENV_PREFIX',
    '/scratch/{}/repos/football/.venv'.format(os.environ.get('USER', '')))
# The shaping-study winner; enqueued as trial 0 so it is the reference point.
BASELINE = dict(
    learning_rate=3e-4, anneal_lr=0, ent_coef=1e-3, gamma=0.997,
    gae_lambda=0.98, clip_coef=0.2, vf_coef=0.5, update_epochs=4,
    minibatch_segments=32, envs_per_worker=1, ball_potential=1.0)


def storage(study_dir):
  """Journal file storage: safe for many Slurm tasks on a network filesystem."""
  return optuna.storages.JournalStorage(
      optuna.storages.journal.JournalFileBackend(
          str(study_dir / 'optuna_journal.log'),
          lock_obj=optuna.storages.journal.JournalFileOpenLock(
              str(study_dir / 'optuna_journal.log'))))


def load_study(study_dir):
  """Open the shared study with the sampler and pruner every worker uses.

  The pruner only acts after 24 evaluations (~50M agent steps) and only on
  trials in the bottom quarter, because learning speed on level 0 varies a
  lot between seeds of the same settings.
  """
  return optuna.load_study(
      study_name=STUDY_NAME, storage=storage(study_dir),
      sampler=optuna.samplers.TPESampler(
          n_startup_trials=10, multivariate=True, constant_liar=True),
      pruner=optuna.pruners.PercentilePruner(
          25.0, n_startup_trials=8, n_warmup_steps=24, interval_steps=4,
          n_min_trials=4))


def suggest(trial):
  """Sample one full set of PPO settings; the search space lives only here."""
  return dict(
      learning_rate=trial.suggest_float('learning_rate', 1e-4, 2e-3, log=True),
      anneal_lr=trial.suggest_categorical('anneal_lr', [0, 1]),
      ent_coef=trial.suggest_float('ent_coef', 1e-5, 1e-2, log=True),
      gamma=trial.suggest_categorical('gamma', [0.99, 0.995, 0.997, 0.999]),
      gae_lambda=trial.suggest_float('gae_lambda', 0.9, 0.99),
      clip_coef=trial.suggest_float('clip_coef', 0.1, 0.3),
      vf_coef=trial.suggest_float('vf_coef', 0.25, 1.0),
      update_epochs=trial.suggest_int('update_epochs', 1, 8),
      minibatch_segments=trial.suggest_categorical(
          'minibatch_segments', [16, 32, 64, 128]),
      envs_per_worker=trial.suggest_categorical('envs_per_worker', [1, 2]),
      ball_potential=trial.suggest_float('ball_potential', 0.0, 2.0))


def trainer_environment(study_dir, params, seed, steps):
  """Translate sampled settings into the trainer's environment variables.

  Everything that is not searched is pinned here, including the two level-0
  controls: a curriculum threshold of 1.0 (the environment rejects anything
  higher; only a perfect evaluation would promote) and the fixed gate
  opponent. Early abort is off so every evaluation plays all 256 episodes.
  """
  settings = dict(
      LEARNING_RATE=params['learning_rate'], ANNEAL_LR=params['anneal_lr'],
      ENT_COEF=params['ent_coef'], GAMMA=params['gamma'],
      GAE_LAMBDA=params['gae_lambda'], CLIP_COEF=params['clip_coef'],
      VF_COEF=params['vf_coef'], UPDATE_EPOCHS=params['update_epochs'],
      MINIBATCH_SEGMENTS=params['minibatch_segments'],
      ENVS_PER_WORKER=params['envs_per_worker'],
      BALL_POTENTIAL=params['ball_potential'], PLAYER_POTENTIAL=0.0,
      SEED=seed, TOTAL_TIMESTEPS=steps, START_LEVEL=0,
      CURRICULUM_SUCCESS_THRESHOLD=1.0,
      FROZEN_DEFENCE_GATE=1, FROZEN_DEFENCE_INIT=str(study_dir / OPPONENT),
      PROMOTION_INTERVAL=PROMOTION_EPOCHS // params['envs_per_worker'],
      PROMOTION_EPISODES=256, PROMOTION_EARLY_ABORT_MARGIN=0,
      GREEDY_PROMOTION_EPISODES=64, SELFPLAY_PROMOTION_EPISODES=64,
      SCORED_PROMOTION_LEVELS=21, ENV_NAME='11_vs_11_advantage',
      BPTT_HORIZON=32, HIDDEN_SIZE=256, FRAME_STACK=1,
      ASYNC_COLLECTION=1, ASYNC_PROMOTION=1, COMPILE=1, GPU_FILLER=1,
      LOCAL_GAME_DATA=1)
  environment = dict(os.environ, **{k: str(v) for k, v in settings.items()})
  environment.update(REPO=str(study_dir / 'repo'),
                     ENV_PREFIX=FOOTBALL_PYTHON_ENV,
                     WANDB_GROUP=study_dir.name)
  return settings, environment


def run_trial(study_dir, steps):
  """Ask for one trial, train it on level 0, report as it learns, tell result.

  The trainer is started in its own process group so a pruned or cancelled
  trial can be stopped completely. A trial that crashes or is cancelled is
  told FAIL and writes its partial curve, so it never counts as a result.
  """
  study = load_study(study_dir)
  trial = study.ask()
  params = suggest(trial)
  seed = SEED_BASE + trial.number
  settings, environment = trainer_environment(study_dir, params, seed, steps)
  environment['WANDB_TAG'] = 'trial-{}'.format(trial.number)
  print('OPTUNA_TRIAL ' + json.dumps(
      dict(number=trial.number, params=params, settings=settings)), flush=True)
  process = subprocess.Popen(
      ['bash', str(study_dir / 'repo/sbatch/train_selfplay.sbatch')],
      env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
      text=True, bufsize=1, start_new_session=True)

  def stop_trainer():
    if process.poll() is None:
      os.killpg(process.pid, signal.SIGTERM)
      try:
        process.wait(timeout=60)
      except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()

  def on_signal(signum, _frame):
    raise KeyboardInterrupt('signal {}'.format(signum))

  signal.signal(signal.SIGTERM, on_signal)
  curve = []
  state = 'running'
  try:
    for line in process.stdout:
      sys.stdout.write(line)
      if not line.startswith('PROMOTION '):
        continue
      metrics = json.loads(line[len('PROMOTION '):])
      if metrics['level'] != 0:
        raise RuntimeError('trial left level 0: {}'.format(metrics['level']))
      curve.append({key: metrics.get(key) for key in (
          'promotion_success_rate', 'greedy_promotion_success_rate',
          'selfplay_promotion_success_rate', 'promotion_possession_fraction',
          'promotion_shot_fraction', 'promotion_mean_episode_length',
          'promotion_policy_entropy_fraction',
          'promotion_worst_two_template_success_rate', 'optimizer_steps')})
      smoothed = statistics.mean(
          point['promotion_success_rate'] for point in curve[-SMOOTHING:])
      step = len(curve) - 1
      trial.report(smoothed, step)
      print('OPTUNA_REPORT ' + json.dumps(dict(
          number=trial.number, step=step, smoothed=smoothed,
          raw=curve[-1]['promotion_success_rate'])), flush=True)
      if trial.should_prune():
        state = 'pruned'
        stop_trainer()
        break
      if metrics.get('advanced'):
        # Only 256/256 against the fixed opponent passes a threshold of 1.0:
        # level 0 is mastered, so stop before training moves to level 1.
        state = 'mastered'
        stop_trainer()
        break
    if state == 'mastered':
      state = 'complete'
    elif state == 'running':
      if process.wait() != 0:
        raise RuntimeError('trainer exited with {}'.format(process.returncode))
      if len(curve) < SMOOTHING:
        raise RuntimeError('too few level-0 evaluations: {}'.format(len(curve)))
      state = 'complete'
  except BaseException as error:
    state = 'fail'
    stop_trainer()
    print('OPTUNA_FAIL {!r}'.format(error), flush=True)
  value = (statistics.mean(point['promotion_success_rate']
                           for point in curve[-SMOOTHING:])
           if state == 'complete' else None)
  (study_dir / 'results').mkdir(exist_ok=True)
  (study_dir / 'results/{}.json'.format(trial.number)).write_text(json.dumps(
      dict(number=trial.number, state=state, value=value, seed=seed,
           params=params, settings=settings, curve=curve,
           slurm_job=os.environ.get('SLURM_JOB_ID'),
           finished=time.strftime('%Y-%m-%dT%H:%M:%S')), indent=2) + '\n')
  if state == 'complete':
    study.tell(trial, value)
  elif state == 'pruned':
    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
  else:
    study.tell(trial, state=optuna.trial.TrialState.FAIL)
    sys.exit(1)


def init(study_dir):
  """Create the shared study once and queue the baseline as trial 0."""
  study = optuna.create_study(
      study_name=STUDY_NAME, storage=storage(study_dir), direction='maximize')
  study.enqueue_trial(BASELINE)
  print('created study {} in {}'.format(STUDY_NAME, study_dir))


def summarize(study_dir):
  """Rank finished trials and estimate which settings matter.

  Parameter importances need several completed trials; with a noisy objective
  they are a hint, not a conclusion. The top settings still need a multi-seed
  confirmation before they are adopted.
  """
  study = load_study(study_dir)
  trials = study.get_trials(deepcopy=False)
  counts = {}
  for trial in trials:
    counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
  completed = sorted(
      (t for t in trials if t.state == optuna.trial.TrialState.COMPLETE),
      key=lambda t: t.value, reverse=True)
  importances = None
  if len(completed) >= 8:
    importances = optuna.importance.get_param_importances(study)
  report = dict(
      counts=counts,
      top=[dict(number=t.number, value=t.value, params=t.params)
           for t in completed[:10]],
      baseline=next((dict(number=t.number, value=t.value, state=t.state.name)
                     for t in trials if t.number == 0), None),
      importances=importances)
  (study_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))


def main():
  """Dispatch init / run / summarize for one study directory."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('action', choices=('init', 'run', 'summarize'))
  parser.add_argument('--study', type=Path, required=True)
  parser.add_argument('--steps', type=int, default=MAX_STEPS)
  args = parser.parse_args()
  study_dir = args.study.resolve()
  if args.action == 'init':
    init(study_dir)
  elif args.action == 'run':
    run_trial(study_dir, args.steps)
  else:
    summarize(study_dir)


if __name__ == '__main__':
  main()
