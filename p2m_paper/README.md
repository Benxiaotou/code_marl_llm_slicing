# Paper supplement: fixed3x4 attribution and untouched confirmation

User authorized the paper-strengthening experiments. Both stages are fixed
before looking at their scores; the independent stage runs all12 arms
regardless of old-test outcomes. No parameter/prompt/model selection occurs.

## Design

Three original decision policies: frozen MAPPO checkpoint, original
Standard1000 nearest-case kNN, original P2-A Standard1000 Qwen. For each:
raw, link-only, node-only, link+node. Corrected policies are newly named
ablation methods; original baseline definitions/results are unchanged.

Link target: nearest3 public TRAIN cases using role-local link features,
unweighted mean, alpha0.5. Node: unchanged bounded_release.adjust_local,
max ratio increment0.025, extra projected overload <=0.01 own capacity.
Only own240D observation, own4D action and own TRAIN retrieval enter the
postprocessor. Other policy actions/outcomes and peer inputs never enter.
Original g action is preserved by all postprocessors.

Stage1 uses existing1536 scenes (6601/6701/6801), cached MAPPO/Qwen/kNN
actions and prior audited4-arm scores. It recomputes8 arms/scene:7 new
ablations and original kNN as a control. Additional controls reproduce
4 archived arms at first/last scene of each seed, plus all12 first-scene
post-controls at each seed's end. No API calls in this stage.

Stage2 independently generates1536 new scenes, seeds
2026110511/2026110512/2026110513,512 each, split test. Same original GEANT
generator/config/checkpoint. One fresh Qwen draw/scene shared across four
postprocessors. All12 arms scored.6144 internalQwen calls,8 concurrent,
temperature0.1,max_tokens4096,timeout120s,streaming,no retries/fallback.
No GPT calls. Additional API fee estimate$0 under user-provided unlimited
internal Qwen quota; not independently verified. Missing token usage must
be reported unknown, never zero.

Three prespecified meanV claims: Qwen_both minus MAPPO_raw, Qwen_both minus
MAPPO_both, Qwen_both minus kNN_both.10000 paired bootstrap resamples
stratified by generation seed,RNG20261105. Each claim receives a
98.333333% Bonferroni interval for family size3, in addition to descriptive
95% intervals. All other contrasts/metrics are descriptive. Within each
policy, report raw/link/node/both and V interaction
`both-link-node+raw`. No optional stopping or arm selection.

## Inputs and resources

New files are only in p2m_paper; original environment/rewards/metrics,
migration/routing,datasets,checkpoints and official outputs are untouched.
Read relevant AGENTS.md and PROJECT_HANDOFF.md; inspected original validated
helpers in p2j_validation/run_validation.py,p2l_cached_test/run_cached.py,
p2k_formal/run_formal.py,p1a_runner.py,p1c/policy.py,p1c/retrieval.py,
p1b/teacher.py,p1b_generate.py,p2a/policy.py,p2a/rerank.py,
p1a/policy.py,p1b/core.py,p2i_node_boundary/bounded_release.py,
p2g_autonomous/jobs/teacher_link_multicohort.py.

Exact old inputs: p2l_cached_test/run_20261105T023605_900720Z/{protocol.json,
summary.json,audit.json,records.jsonl}; existing dataset manifests
p1b_data/p2c_test_seed<seed>.json;48 kNN chunks
p2c_runs/qwen8_test_v1/p2c_test_seed<seed>/knn_std1000/chunk_<start>_032/
{manifest.json,summary.json,records.jsonl}. Related prior frozen source
hashes and exact public TRAIN library paths come from
p2j_validation/run_20261105T014642_470733Z/protocol.json. Runtime only reads
specific public TRAIN files/checkpoint/development controls through the
validated helpers. No recursive scans or raw data/log dumps.

Each run writes exclusive timestamped outputs. One512-scene environment per
stage at a time. Cached stage minimum8GiB free RAM; fresh minimum12GiB;
disk minimum12GiB. Fresh samples about5GB plus structured records/prompts;
no original files overwritten. Credential values never printed; launcher
only enables internalQwen credentials for fresh mode and clears GPT keys.

## Commands and checks

```powershell
& 'python' -B -m unittest discover -s .\p2m_paper -p test_contract.py -v
& .\p2m_paper\run.ps1 -Mode cached
& .\p2m_paper\run.ps1 -Mode fresh
```

Four contract tests actually passed before launch. They check factorial
dimension separation,immutability,g preservation,ratio bounds,no
cross-policy action coupling,kNN equivalence and fixed unseen-seed design.
Saved-output auditor reconstructs all12 actions/scene,rechecks TRAIN targets,
metrics and summary/contrast point estimates,source/dataset hashes; fresh
audit also reconstructs every Qwen prompt/parsed response and request journal.
Original scene-reset/reward identities/IR definitions are checked by
unchanged score helpers. A COMPLETE run needs summary.json,audit.json and
RESULTS.md; partial failures preserve STOPPED.json,never silently retry API.

Old tests were previously inspected; they are attribution analysis,not an
untouched final test. Fresh tests remain in the same GEANT distribution;
no cross-topology or repeated model-draw stability claim. No agent
communication is implemented. V is original accounting utility,not measured
end-to-end throughput. Average improvements do not guarantee per-scene or
per-participant improvement.

## Completed cached stage

`cached_20261105T025855_555782Z`: exit0,COMPLETE,audit passed. All1536 kNN
controls,24 historical probes,36 post-controls passed;18432 metric rows and
18432 actions checked.0API calls,$0. MeanV MAPPO_node344.943035,
MAPPO_both343.987580,Qwen_both343.579617. CorrectedMAPPO is a necessary
stronger comparator; originalQwen-versus-rawMAPPO gain cannot all be
attributed to the LLM. See run RESULTS.md and ANALYSIS.md for full evidence.
The untouched confirmation stage also completed; see below.

Additional reporting artifacts: plot_results.py generated the cached-stage
ablation_effects.png/pdf and a source-hash manifest. The PNG was visually
inspected; labels,axes,intervals and descriptive-inference caveat are readable.
PAPER_NOTES.md records methods and interpretation limits; it changes no code
or experimental protocol. To create plots for a completed run (new files):

```powershell
& 'python' -B .\p2m_paper\plot_results.py .\p2m_paper\cached_20261105T025855_555782Z
```

Plot generation refuses to overwrite existing plots. The command above was
already executed successfully for the cached run; use the future completed
fresh run path when generating its plots.

## Completed untouched confirmation

`fresh_20261105T025939_435887Z`: exit0,COMPLETE,audit passed. Three new512-scene
datasets generated with fixed seeds.18432 scores/actions,6144 reconstructed
prompts/responses,1536 MAPPO controls,36 post-controls and input/source hashes
passed.6144 successful internalQwen requests,no failures/retries/fallback,noGPT.
All6144 usage values missing: tokens unknown,not zero. ExtraAPIfee estimate$0.
Single-role request latency mean5.924s,median5.853s,p95 6.707s.

Qwen_both V344.178028 vs rawMAPPO341.423768: +2.754259(+0.8067%),95%CI
[1.171117,4.339738],family3 BonferroniCI[.818918,4.699339]. However matched
MAPPO_both V343.787866 gives Qwen-minus-MAPPO +.390161,BonferroniCI
[-1.275360,2.011263]. MAPPO_node meanV344.888470 remains highest of12.
Qwen has higher admission but lower cooperation than correctedMAPPO.
Both favorable and limiting results are preserved in the run's ANALYSIS.md.

New plots generated successfully and PNG visually checked. Runtime library
versions and descriptive API statistics saved. summarize_paper.py created
PAPER_SUPPLEMENT_REPORT.md with both stages,all12 methods and adjusted claims.
Its first execution encountered a relative __file__ path in Python3.8 before
any report write; corrected to resolve() and rerun successfully. No scoring,
inference,protocol or original result changed, and no requests repeated.

Additional reporting sources: plot_results.py,summarize_paper.py,PAPER_NOTES.md.
No process remains active,no further experiment started. Read completed report:

```powershell
Get-Content -LiteralPath .\p2m_paper\PAPER_SUPPLEMENT_REPORT.md
```

Full fresh report: fresh_20261105T025939_435887Z/RESULTS.md; interpretation:
fresh_20261105T025939_435887Z/ANALYSIS.md. Both runs and all original results
remain intact. Existing main.py/.gitignore modifications predate this task
and were left untouched. Code/source/input inventory and reproduction commands
are listed above; those experiment commands create NEW runs and fresh mode
makes another6144 requests,so do not rerun merely to read completed results.
