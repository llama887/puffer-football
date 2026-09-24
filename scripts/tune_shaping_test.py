"""Small checks for paired configurations and incomplete-study handling."""

import json
from pathlib import Path
import tempfile

from tune_shaping import CONFIGS, SEEDS, STEPS, configuration, summarize


def test_study():
  """Do not choose a winner until every full-budget paired trial succeeds."""
  assert len({tuple(configuration(i)[1].items()) for i in range(24)}) == 24
  for config_id in range(len(CONFIGS)):
    assert [configuration(3 * config_id + i)[1]['SEED'] for i in range(3)] == list(SEEDS)
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / 'results').mkdir()
    assert summarize(root)['selected'] is None
    for index in range(24):
      config_id, config = configuration(index)
      metrics = dict(promotion_success_rate=0.1 + 0.01 * config_id,
                     promotion_worst_two_template_success_rate=0.05)
      row = dict(index=index, config_id=config_id, config=config,
                 global_step=STEPS, benchmark_episodes=256,
                 benchmark={str(level): metrics for level in (4, 6, 7)})
      (root / 'results/{}.json'.format(index)).write_text(json.dumps(row))
      if index < 23:
        assert summarize(root)['selected'] is None
    assert summarize(root)['selected']['config_id'] == 7
    row['global_step'] = 1
    (root / 'results/23.json').write_text(json.dumps(row))
    assert summarize(root)['selected'] is None


if __name__ == '__main__':
  test_study()
  print('ok: paired configurations, ranking, and incomplete budget checks')
