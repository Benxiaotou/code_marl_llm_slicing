"""Frozen MAPPO teacher adapters: live P0 model and existing P0 MAPPO records."""
import random
from pathlib import Path

import numpy as np

from p1a.local_obs import AGENTS
from p1b.core import (ContractError, file_sha256, normalize_teacher_actions,
                      read_json, read_jsonl)


class P0FrozenTeacher:
    """Use the already-validated MAPPOPolicy from local p0_runner.py.

    This class never calls train() or alters model parameters. It requires the
    user's actual local P0 code and checkpoint; compatibility is checked at
    startup rather than silently substituting a rule/fixed policy.
    """
    def __init__(self, config, checkpoint_path, *, action_space='env'):
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError('MAPPO checkpoint not found: '+str(checkpoint_path))
        try:
            from p0_runner import MAPPOPolicy
        except (ImportError, AttributeError) as exc:
            raise RuntimeError('Need the previously working p0_runner.py MAPPOPolicy') from exc
        self.policy = MAPPOPolicy(config, str(checkpoint_path))
        self.source_info = {
            'kind': 'live_frozen_mappo', 'verified_mappo': True,
            'checkpoint': str(checkpoint_path),
            'checkpoint_sha256': file_sha256(checkpoint_path),
            'action_space': action_space,
        }
        self.action_space = action_space

    def act(self, obs, *, original_scene_id=None, seed=1):
        # Seeds independent from response latency. For stochastic checkpoints,
        # additionally specify original evaluator deterministic/test mode.
        step_seed = int(seed) + int(original_scene_id or 0)
        random.seed(step_seed)
        np.random.seed(step_seed)
        try:
            import torch
        except ImportError:
            raise RuntimeError('XuanCe MAPPO teacher needs PyTorch')
        torch.manual_seed(step_seed)
        with torch.no_grad():
            decision = self.policy.act(obs)
        if isinstance(decision, dict) and decision.get('fallback_used'):
            raise ContractError('Teacher used fallback: not MAPPO behavior')
        if getattr(decision, 'fallback_used', False):
            raise ContractError('Teacher used fallback: not MAPPO behavior')
        return normalize_teacher_actions(decision, assumed_space=self.action_space)

    def close(self):
        vec = getattr(self.policy, 'vec', None)
        if vec is not None and hasattr(vec, 'close'):
            vec.close()


class RecordedMAPPO:
    """Replay previously generated MAPPO actions; env metrics are recomputed.

    Data origin is checked using run manifest/summary. No inference happens.
    Suitable for auditing P0 runs, with `kind=recorded_MAPPO_replay` labeling.
    """
    def __init__(self, run_dir, *, action_space='env', allow_unverified=False):
        root = Path(run_dir).resolve()
        if root.is_dir():
            matches = [root / 'mappo_records.jsonl', root / 'records.jsonl']
            path = next((x for x in matches if x.is_file()), None)
            if path is None:
                raise FileNotFoundError('No MAPPO records.jsonl in '+str(root))
            manifests = [root / 'manifest.json', root / 'summary.json']
            run_metas = [read_json(x) for x in manifests if x.is_file()]
        else:
            path = root
            if not path.is_file():
                raise FileNotFoundError(str(path))
            run_metas = []
        labels = [str(m.get('policy', '') or m.get('method', '')).lower()
                  for m in run_metas]
        labels = [x for x in labels if x]
        good_label = lambda s: s == 'mappo' or s.startswith('mappo_') or 'frozen_mappo' in s
        verified = bool(labels) and all(good_label(label) for label in labels)
        label = ','.join(labels) or 'unknown'
        if not verified and not allow_unverified:
            raise ContractError('Teacher run lacks policy=mappo provenance. '
                                'Provide original P0 summary.json or use '
                                '--allow-unverified-source for DEV ONLY.')
        rows = {}
        for r in read_jsonl(path):
            if not isinstance(r, dict):
                raise ContractError('Teacher record is not an object')
            if r.get('status') not in ('ok', 'success', None):
                raise ContractError('Teacher record is failed; cannot use as teacher')
            if r.get('fallback_used') or r.get('is_fallback'):
                raise ContractError('Teacher run contains fallback actions')
            fields = ('original_scene_id', 'original_scene_index', 'scene_id', 'scene_index')
            sid = next((r[k] for k in fields if k in r), None)
            if sid is None:
                raise ContractError('Teacher record missing original scene ID; keys={}'.format(sorted(r)))
            sid = int(sid)
            if sid in rows:
                raise ContractError('Duplicate source scene ID: {}'.format(sid))
            rows[sid] = r
        if not rows:
            raise ContractError('Empty teacher run')
        self.rows = rows
        self.action_space = action_space
        self.source_info = {
            'kind': 'recorded_MAPPO_replay', 'verified_mappo': bool(verified),
            'source_file': str(path), 'source_file_sha256': file_sha256(path),
            'run_policy_label': label or 'unknown',
            'action_space': action_space,
        }

    def act(self, obs, *, original_scene_id, seed=1):
        del obs, seed
        if original_scene_id not in self.rows:
            raise ContractError('No MAPPO action for scene {}'.format(original_scene_id))
        row = self.rows[original_scene_id]
        return normalize_teacher_actions(row, assumed_space=self.action_space)

    def close(self):
        pass
