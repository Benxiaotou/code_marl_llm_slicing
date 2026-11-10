# P2-D replay v1.1: legacy pickle initialization order fix

Root cause: The P2-D v1 script called p1c.build_policy() before creating
its first evaluation environment. The project p1c_runner.py does the opposite:
create_eval_env() installs the `__main__.InitVnetworkTrain` compatibility
aliases before constructing a MAPPO teacher. With P2-D v1, XuanCe
make_envs() hit a pickle.load() before aliases were available.

Fix: Before p1c.build_policy(), v1.1 calls the ORIGINAL project's
p1a_runner._config() and p1a_runner._compat_pickle_main_classes() helpers.
It checks the expected class in __main__. No sample, checkpoint, Qwen/GPT
prompt, environment, or previous run is changed; no API access is needed.

Windows PowerShell in the project root:

  Expand-Archive .\P2D_10scene_MAPPO_GPT_counterfactual_replay_v1_1_pickle_fix.zip -DestinationPath .\p2d_replay -Force
  python .\p2d_replay\p2d_counterfactual_replay.py --preflight
  python .\p2d_replay\p2d_counterfactual_replay.py `
    --project-root . `
    --archive .\GPT61_MAPPO_10scene_diagnosis.zip `
    --dataset-manifest p1b_data\gpt61_dev_seed7601.json `
    --library p2b_libraries\p2b_standard_1000_v2 `
    --seed 1 `
    --output-dir gpt61_runs\dev7601_cross_replay_v1_1

The last path is new to avoid overwriting partial outputs. The script
stops before crossovers if MM/GG do not match stored results.

Status: static checks, ZIP preflight and mocked compatibility tests run.
Real GEANT scoring is NOT possible without local environment files.
