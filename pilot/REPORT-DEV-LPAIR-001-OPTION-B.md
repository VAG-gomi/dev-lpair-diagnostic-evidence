# DEV-LPAIR-001 Option B — Executor Report

## Unofficial side-by-side pilot: L-SHARED vs L-SVA, W10 + W20 only

**Label: DEV-LPAIR-001 only. NOT EXP1-COMB. NOT SWEEP. NOT SPIKE.**
**Completeness: INCOMPLETE. No comparative winner is declared.**
**Authorisation: owner 2026-10-29, own words via tap: "I authorise Option B for DEV-LPAIR-001, as specified in the preflight report §5, scratch only, implementation checks first, stop on failure or incomplete, no rerun without separate decision."**
**Executor: subagent, reporting to parent for verification; this report is not delivered to the owner directly, and it is not a project ruling.**

Lead finding, before any detail: the pilot did NOT complete. A VM restart killed the final training job mid-run. That job was NOT rerun, NOT resumed, and NOT substituted, as instructed. All results below are for completed jobs only, and no comparative winner follows from them, even where completed-job counts differ.

---

## 1. Implementation-check outcomes — separately from experimental results

Checks were run BEFORE any training, in scratch only, file `checks.py`, output `checks_output.txt`.

- Total: 24 checks, 0 failures on the final run.
- Checks covered: output shapes and parameter counts (shared W10 3,986 / W20 13,586; SVA W10 7,378 / W20 26,578); encoder layout grid²+5, one-hot, door slot, blank-feature convention (phase/remaining 0.0, K=0 before knowledge, K=1 after convergence, features matching ScheduleModel exactly), prev-door at reset = reset-tick door bit and updating to the previous tick's bit, ScheduleModel observe exactly once per encode; permitted inputs (no goal/geometry/next_door/period/open_len direct read, tick only via `model.observe(tick, door)`); target recomputation (shared action target uses recomputed target-selector argmax, not a stored act; SVA selector target uses best head value, not selector max; terminal D=1 gives reward alone); masked losses (only chosen outputs carry gradient); greedy path deterministic, act in 0..2, action in 0..4; replay stores all required fields (x, act, a, r, x2, done, crushed, K/D/raw R/C at S and S′, exploration flags); flat action index act*5+a round-trip.
- One check initially failed, and the failure was in the TEST, not the implementation: the masked-action test used a target equal to the output (0.0), so its gradient was legitimately zero and the non-zero count was 1, not 2. The test was fixed to use targets 1.0 and 0.5, the mechanism was not changed, and the full check suite was rerun: 24/24 PASS. This deviation is reported here, not hidden.
- Blank-feature convention: PROPOSED in the L memo as phase/remaining = 0.0 when K=0. Checking against `agent_wrapped.py` CONFIRMED it matches wrapped code exactly (wrapped encoder sets phase/remaining only when both are available, otherwise leaves them 0.0 and K=0). No ambiguity arose, no stop was required on this point.
- Passing these checks is implementation evidence only, NOT experimental evidence. It establishes that the scratch code implements the specified equations, shapes, and input contract as tested; it does not establish that either construction learns well.

## 2. Exact implementations and configuration identities

Scratch workspace only: `~/workspace/dev-lpair-001/`
- `lpair.py` — encoder, both networks, target functions, trainer, evaluator, diagnostic.
- `checks.py`, `checks_output.txt` — implementation checks.
- `run_pilot.py`, `run_pilot.log` — sequential runner and its per-job log.
- `records_dev_lpair_001.jsonl` — incremental records, 55 lines (see §3).
- `weights_L-SHARED_W10_*.npz` (3), `weights_L-SHARED_W20_*.npz` (3), `weights_L-SVA_W10_*.npz` (3), `weights_L-SVA_W20_0.npz`, `weights_L-SVA_W20_1.npz` (2) — audit weights for completed models only. No weights file exists for the missing model (see §3).

Source files relied on, read-only, sibling HEAD `2e2f029fffd4161bb978b38f2762209b82837b3e`:
- `TEMPLATE/nn/exp1_run.py` SHA256 `829a17e4072bd3371cd296aac4cae63bfebef10ee68e1209a526b168491d2dbd`
- `TEMPLATE/world_exp1.py` SHA256 `d6a8bb24c04aa6429d6fe75c58aa372ac37c9ec4210d80401c1163f5a503e9c8`
- `TEMPLATE/nn/agent_wrapped.py` SHA256 `f727699e6bdce58bca71fc8b0fe4013b601fad4adb53ae77ba798d4e6fddd08e`
- `TEMPLATE/nn/wrapper.py` SHA256 `e3546347f2ee51270cceb5e70c6bbd7f2f64452c497ac5102fc1a03ff7adf6a0`
- `TEMPLATE/nn/nnet.py` SHA256 `a31046a8c4f31485b9f8d171f2cd76c0fc967041323c2d0297e5c322e792de2b`
- `TEMPLATE/nn/agent_nn.py` SHA256 `b7405daf175f0767a34df57b95483219bffaf5097abda2260450ecc8b8fa7d34`
- `TEMPLATE/nn/agent_mem.py` was also read, for the prev-door convention only. No SHA256 is claimed for it in this report, because no SHA for that file was recorded in the executor's inspection output; the source list above is limited to files whose SHA256 was printed in inspection.

Configuration, pilot only, NOT fully accepted official L settings:
- Worlds: W10, W20 trained; held-out W10h (5,2), W20h (10,5). W30, M1, M2, M3 excluded entirely.
- Seeds: 0, 1, 2, matched across constructions. Each seed-world model trained independently, no weight transfer.
- Training: 1000 episodes per seed per world, episode cap 200 ticks, schedule 20/10, reward step −0.01 / goal +1.0 / crush −1.0, done = reached only, crush/cap not terminal.
- Evaluation: 20 episodes per job, one per start phase 0..19, frozen weights, greedy selector and greedy within-head action, ScheduleModel may update during an eval job, fresh encoder per (agent, eval world) job, ScheduleModel blank at job start, persists across that job's 20 episodes, no training on held-out.
- Learning: Adam lr 0.001, discount 0.99, replay 10000, batch 32, epsilon linear 1.0 → 0.05 over 1000 episodes (agent_nn.epsilon formula, so ep 999 epsilon ≈ 0.05095, not exactly 0.05 — reported as a formula choice, not a deviation from the spec's stated schedule), target sync every 200 updates. L-SVA: each network its own optimiser and target network, synced at its own 200-update count.
- Networks: hand-written NumPy only, tanh hidden, linear output, float64, as in nnet.py. Encoder: position one-hot + door D + phase (ticks_since_opening/20.0) + remaining (open_ticks_remaining/10.0) + known K + prev-door = grid²+5 (W10 105, W20 405). ScheduleModel sees only (tick, door), once per encode.
- L-SHARED: one MLP [input, 32, 18]; selector target = target selector max next-state value; action target = recompute next-state act via target selector, maximise over 5 actions in that selected target head; sum two masked MSE losses, one Adam update; terminal reward alone.
- L-SVA: selector MLP [input, 32, 3], action MLP [input, 32, 15]; selector target = reward + discounted maximum action-head value at next state, using target action network (max over acts of max over actions in that head); action target = same as L-SHARED; separate masked-loss Adam updates; terminal reward alone.
- Implementation choices, reported, not silently made: (a) SVA init seeds: selector seed = seed, action seed = seed+10000, to avoid identical hidden-layer init draws that would occur with the same seed for both networks (same fan_in/fan_out, overlapping output draws). This does not change any target, loss, or input equation. (b) Exploration draws: a single Random(seed) stream, selector draw first then action draw per step, for both constructions — spec says independent draws, this implements them sequentially from one matched stream, not from two separate RNG streams. (c) Replay sampling: a single RandomState(seed) batch per step, used for both losses in SVA and for the summed loss in SHARED, matched across constructions. (d) Diagnostic states: states recorded during each eval job, diagnostic computed immediately after that job from those recorded states, offline, no env step, no weight update. Raw states are NOT persisted as separate files; only diagnostic summaries are in the JSONL. Diagnostic therefore cannot be recomputed from the JSONL alone without the saved weights and a fresh eval run (which, given frozen weights and fresh encoder, would reproduce the same eval trajectory, but that reproduction was not performed and is not claimed here).

## 3. Job ledger and completeness status — INCOMPLETE

Planned: 12 trainings + 24 evaluations = 36 jobs, plus 24 diagnostic records, total 60 records.
Completed: 11 trainings + 22 evaluations + 22 diagnostics = 55 records, 0 duplicates (55 unique job_ids).
Missing, NOT rerun, NOT resumed, NOT substituted:
- Training: `DEV-LPAIR-001:train:L-SVA:W20:2`
- Evaluation: `DEV-LPAIR-001:eval:L-SVA:W20:2:W20` (trained)
- Evaluation: `DEV-LPAIR-001:eval:L-SVA:W20:2:W20h` (held-out)
- Diagnostics: `DEV-LPAIR-001:eval:L-SVA:W20:2:W20:diagnostic` and `DEV-LPAIR-001:eval:L-SVA:W20:2:W20h:diagnostic`
- Weights: no `weights_L-SVA_W20_2.npz` exists.

Cause, with evidence and its limit: the pilot process disappeared during the missing training job. Last record written 2026-10-29 07:17:54 UTC (L-SVA W20 seed 1 evaluations). At 07:21:20 UTC the VM uptime was 1 minute, evidencing a VM restart at about 07:20 UTC. No pilot process was running after that check, the run log ends mid-training with no error line and no "PILOT COMPLETE" line, and no record or weights file was written for the missing job (training records are written only after a training job completes, so a mid-training kill leaves no partial training record — that absence is expected, not a missing partial record to be recovered). The specific signal that killed the process (SIGTERM or other) is NOT recorded in available logs and is NOT claimed here; only the restart itself is evidenced by uptime. The exec timeout (3600 s) was not the cause: it would have expired at about 07:34 UTC, after the restart.
Failures/restarts otherwise: none in completed jobs. No duplicate records. No job was rerun by the executor. No official repo file was edited, committed, or pushed (see §8).

Training jobs completed, wall time and training-episode reached count (1000 episodes each):
- L-SHARED W10: seed 0: 108.2 s, 644 reached, 116,442 updates; seed 1: 107.4 s, 639, 113,077; seed 2: 113.9 s, 606, 120,246.
- L-SHARED W20: seed 0: 276.2 s, 450, 149,722; seed 1: 290.1 s, 399, 156,050; seed 2: 268.9 s, 472, 147,205.
- L-SVA W10: seed 0: 156.7 s, 671, 112,500; seed 1: 163.4 s, 656, 117,454; seed 2: 160.3 s, 653, 116,711.
- L-SVA W20: seed 0: 495.6 s, 363, 157,366; seed 1: 439.7 s, 517, 139,631; seed 2: MISSING (see above).
- Completed training wall total: 2,580.4 s (43.0 min). Completed evaluation wall total: 2.74 s. Total completed wall: about 43.1 min. This is measured for completed jobs only; it is NOT a full-pilot runtime, and no full-pilot runtime is claimed. The missing SVA W20 training would have added roughly 440–500 s on the pattern of its two completed seeds, but that figure is an extrapolation from completed seeds, not a measurement, and is not added to a claimed total.

## 4. Results by world and seed — raw counts beside percentages, completed jobs only

Evaluation episodes: 20 per job. Percentages below are reached/20 for a single job, or reached/denominator for pooled completed jobs. Pooled denominators differ between constructions because one SVA job is missing; pooled comparisons must therefore be read with §3, not as matched totals.

### Trained-world evaluation

| Construction | World | Seed 0 | Seed 1 | Seed 2 | Pooled completed |
| --- | --- | --- | --- | --- | --- |
| L-SHARED | W10 | 20/20 (100%) | 20/20 (100%) | 13/20 (65%) | 53/60 (88.3%) |
| L-SHARED | W20 | 4/20 (20%) | 0/20 (0%) | 20/20 (100%) | 24/60 (40.0%) |
| L-SHARED | W10+W20 pooled completed | — | — | — | 77/120 (64.2%) |
| L-SVA | W10 | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) | 60/60 (100%) |
| L-SVA | W20 | 19/20 (95%) | 20/20 (100%) | MISSING | 39/40 (97.5%, seeds 0–1 only) |
| L-SVA | W10+W20 pooled completed | — | — | — | 99/100 (99.0%, denominator 100, not 120) |

Matched per-seed trained differences, L-SVA minus L-SHARED, completed pairs only:
- W10 seed 0: 0; seed 1: 0; seed 2: +7 (20−13).
- W20 seed 0: +15 (19−4); seed 1: +20 (20−0); seed 2: UNMATCHED (shared 20/20, SVA missing).
- Matched trained total, 5 pairs: shared 57/100, SVA 99/100, difference +42 on a denominator of 100 matched episodes. This is a completed-jobs arithmetic fact, NOT a comparative winner claim (see §7: pilot incomplete, held-out primary question tied at zero, trained difference is not the primary question, and 3 seeds is a small sample).

### Held-out evaluation — primary question, completed jobs only

| Construction | World | Seed 0 | Seed 1 | Seed 2 | Pooled completed |
| --- | --- | --- | --- | --- | --- |
| L-SHARED | W10h | 0/20 | 0/20 | 0/20 | 0/60 |
| L-SHARED | W20h | 0/20 | 0/20 | 0/20 | 0/60 |
| L-SHARED | Held-out pooled completed | — | — | — | 0/120 (0%) |
| L-SVA | W10h | 0/20 | 0/20 | 0/20 | 0/60 |
| L-SVA | W20h | 0/20 | 0/20 | MISSING | 0/40 (0%, seeds 0–1 only) |
| L-SVA | Held-out pooled completed | — | — | — | 0/100 (0%, denominator 100, not 120) |

Matched per-seed held-out differences, completed pairs only: 0 on every completed pair (5 pairs, 100 matched episodes, both 0). The held-out result for completed jobs is therefore a tie at zero, not an advantage for either construction. It does not establish that either construction generalises, and it does not establish that the missing job would also have been zero — that job is unknown, not zero.

### Training-episode reached counts (for context, not evaluation)

- L-SHARED: W10 644+639+606 = 1,889/3,000; W20 450+399+472 = 1,321/3,000.
- L-SVA: W10 671+656+653 = 1,980/3,000; W20 seeds 0–1: 363+517 = 880/2,000, seed 2 missing.
- Training reached counts are not evaluation success rates and are not used as a comparison metric in §5 beyond this context line.

### Crushes and episode lengths (evaluation)

- L-SHARED trained eval: crushes 0 total, ticks mean per job: W10 28.5 / 23.9 / 85.0, W20 172.0 / 200.0 / 56.5.
- L-SHARED held-out eval: crushes 0 total, ticks mean 200.0 on every held-out job (all episodes ran to cap, none reached).
- L-SVA trained eval: crushes 0 total, ticks mean: W10 34.2 / 40.6 / 26.4, W20 66.7 / 47.9.
- L-SVA held-out eval: crushes 83 total, all in one job (W10 seed 1 held-out: 83 crushes, 0 reached, ticks mean 200.0); all other completed held-out jobs crushes 0, ticks mean 200.0.
- Training crushes (completed jobs): L-SHARED W10 266, W20 220; L-SVA W10 292, W20 seeds 0–1: 150. Training crush counts are totals over 1000 episodes per job, not rates, and are not directly comparable across constructions without the missing job.
- Interpretation limit, stated, not hidden: evaluation episodes within a job may not be independent, because ScheduleModel knowledge persists during evaluation. Pooled episode counts are therefore not treated as independent samples, and no statistical certainty is manufactured from them.

## 5. Side-by-side comparison — performance, stability, act coverage, diagnostics, runtime (completed jobs only)

Performance:
- Held-out (primary): tie at zero on all completed jobs, both constructions, both worlds, all completed seeds. No preliminary held-out advantage is observed, and none is claimed.
- Trained: on completed matched pairs, SVA is higher (99/100 vs 57/100 matched, see §4), driven by W20 seeds 0–1 (+15, +20) and W10 seed 2 (+7), with W10 seeds 0–1 tied at 20/20. Trained performance is not the primary question, it is not a held-out result, and it is based on 5 matched pairs only, with the sixth pair missing. It must not be read as "SVA is better overall".
- Stability across seeds/worlds: neither construction is stable in the trained sense. Shared trained eval ranges 0/20 to 20/20 across seeds within W20, and 13/20 to 20/20 within W10. SVA trained eval is 20/20 on all completed W10 jobs and 19–20/20 on completed W20 jobs, but that stability statement is limited to completed jobs and does not include the missing W20 seed 2 job, which could have been different. Held-out stability is trivially stable at zero for both, which is not a positive stability finding — it is a failure to reach on any held-out job.
- No advantage is concentrated in a single seed for held-out, because held-out is zero everywhere completed. For trained, SVA's matched advantage is present on 3 of 5 matched pairs (W10 seed 2, W20 seeds 0–1) and tied on 2 (W10 seeds 0–1), so it is not a single-seed artefact within completed pairs, but that pattern is still based on 5 pairs and an incomplete pilot, and is not a winner claim.

Act coverage:
- Training: no act collapse in training step counts. Act steps are roughly equal thirds for both constructions (for example, shared W10 total steps per act across 3 seeds: about 116,236 / 116,789 / 116,833; SVA W10: about 115,471 / 115,440 / 115,847; shared W20: about 150,960 / 150,331 / 151,779; SVA W20 seeds 0–1: about 98,775 / 99,361 / 98,923). Episodes using each act are almost all episodes (for example, 3000/3000 per act in most training sets, with a single 2999 exception in shared W10 seed 0 act 2 and in SVA W10 act sets). This reflects epsilon exploration during training, not greedy selector behaviour, and must not be read as "the selector does not collapse" — greedy eval act usage below is the relevant selector behaviour, and it is skewed.
- Evaluation (greedy selector): act usage is skewed in most jobs, for both constructions, with act 2 (ACT-2, neutral index, no meaning assigned) used least in many trained jobs (for example, shared trained pooled: 4,704 / 5,515 / 1,099 steps; SVA trained pooled completed: 1,961 / 1,991 / 364 steps). Held-out eval act usage is also skewed and varies by job (for example, shared W10 seed 1 held-out: 774 / 390 / 2,836; SVA W10 seed 2 held-out: 3,984 / 11 / 5). No act meaning is assigned from these patterns, as instructed. No coverage threshold was invented, and none is claimed as passed or failed. The skewed greedy usage, combined with zero held-out success, means act usage alone does not explain held-out performance in either direction.
- Head starvation in the training sense (a head receiving too little training) is NOT observed in training step counts, because epsilon exploration distributes training steps across acts. Head starvation in the greedy-eval sense (a head rarely selected when greedy) IS observed in many eval jobs (act 2 rarely selected), but that is a description of greedy behaviour, not a training-starvation finding, and it does not by itself explain held-out failure.

Diagnostics (offline, recorded eval states, separately trained/held-out, counts/denominators, no thresholds):
- Same-state head diagnostic: heads differ on a substantial fraction of recorded states in most jobs, for both constructions. Pairwise agreement varies widely by job (for example, shared W10 seed 0 trained: head0–head1 228/570, head0–head2 210/570, head1–head2 495/570; shared W20 seed 1 trained and held-out both: 68/4000, 61/4000, 95/4000 — the identical diagnostic in that case is consistent with the agent never reaching the wall, so trained and held-out state sequences coincide, because the door-row difference only matters after crossing, which the agent never does; this is reported as an observation, not as a diagnostic error). Held-out diagnostic n_states is 4000 whenever an eval job fails all episodes (20 episodes × 200 ticks), and smaller when episodes reach (for example, shared W10 seed 0 trained: 570 states). Per-head action distributions are in the JSONL diagnostic records, with counts and denominators, and are not reproduced in full here to keep this report compact — parent should read them from the JSONL, not from a summary alone.
- Diagnostic limit, preserved: head differences on the same states establish only that final frozen heads choose different actions on those recorded states. They do NOT establish that training caused the differences, that the differences are useful, or that they caused (or would cause) held-out success. Held-out diagnostics showing head differences alongside 0/20 held-out success are a direct example of that limit in this pilot.
- L-SVA selector-vs-head-value agreement, descriptive only, no threshold, distinct from the head diagnostic: trained agreement 259/685 (37.8%), 249/812 (30.7%), 166/528 (31.4%), 311/1333 (23.3%), 405/958 (42.3%); held-out agreement 1760/4000 (44.0%), 1032/4000 (25.8%), 2263/4000 (56.6%), 1040/4000 (26.0%), 1946/4000 (48.7%). Mean selector margins 0.0024–0.0136, mean head-value margins 0.0046–0.0097. Agreement near 33% in several jobs is close to chance for 3 acts, and margins are small in absolute Q terms. This is descriptive only: it does NOT prove the selector is weak, useful, or causally responsible for any result, and it does NOT prove value-alignment succeeded or failed as a mechanism — it shows that, on recorded eval states, selector argmax and head-value argmax coincided only about a quarter to a half of the time, with small margins. No threshold was applied, and none should be applied retrospectively to these numbers.

Losses (training, mean over updates, for context only):
- L-SHARED mean selector loss per job: W10 0.0011 / 0.0010 / 0.0012, W20 0.0006 / 0.0005 / 0.0006; mean action loss: W10 0.0003 / 0.0004 / 0.0006, W20 0.0003 / 0.0007 / 0.0003.
- L-SVA mean selector loss: W10 0.0012 / 0.0015 / 0.0022, W20 seeds 0–1: 0.0004 / 0.0002; mean action loss: W10 0.0005 / 0.0004 / 0.0005, W20 seeds 0–1: 0.0003 / 0.0002.
- Losses are small and similar in magnitude across constructions. They are not a learning-quality metric, they are not compared as a winner criterion, and no conclusion is drawn from their similarity or difference.

Runtime (measured, completed jobs only):
- Training: shared W10 total 329.5 s, shared W20 total 835.2 s, SVA W10 total 480.4 s, SVA W20 seeds 0–1 total 935.3 s. Per-job SVA training is slower than shared on matched worlds (W10: about 157–163 s vs 107–114 s, ratio about 1.45×; W20: 440–496 s vs 269–290 s, ratio about 1.6–1.8× on completed seeds). This matches the direction expected from two networks vs one, but the ratio is measured only on completed jobs and is not a general runtime claim.
- Evaluation: negligible in all jobs (0.02–0.33 s per job, total 2.74 s completed).
- Total completed wall: 2,583.1 s (43.1 min). Preflight Option B estimate was about 1.00 h conservative (with allowance); completed jobs alone are under that, but the full pilot (including the missing job) was not measured, so no claim is made that the full pilot would have finished under 1 hour.

## 6. Limitations and unresolved questions

- Incomplete pilot: the single most important limitation. Any comparative statement must carry "incomplete, 33/36 jobs" (11 trainings, 22 evaluations, 22 diagnostics completed; 3 jobs and 2 diagnostic records missing). The missing job is a whole seed (SVA W20 seed 2), not a random episode, so pooled SVA totals are on a smaller denominator and matched comparisons are on 5 pairs, not 6.
- Small sample: 3 seeds, 20 episodes per eval job, and eval episodes within a job are not independent (ScheduleModel persists). No statistical certainty is claimed from pooled counts.
- Two changes at once: L-SHARED vs L-SVA differ in network sharing AND selector target together. No result here, even if complete, could isolate which change caused any difference. No causal claim about individual components is made.
- 1000 episodes, not 2000: pilot training is half the official EXP1-COMB episode count, and W30 is excluded. Results here are not comparable to official EXP1-COMB results, and the official bar is not applied as pass/fail (see §7).
- Diagnostic states not persisted: diagnostic summaries are in JSONL, raw states are not. Recomputation requires saved weights and a fresh eval run, which was not performed.
- Implementation choices not in spec text: SVA init seeds (seed vs seed+10000), single RNG stream for selector/action draws, epsilon formula at ep 999 ≈ 0.05095, diagnostic computed immediately after each eval job from its recorded states. Each is reported in §2, none changed the specified target, loss, input, or reward equations.
- Test-fix deviation: one implementation check initially failed due to a test target equal to output; test fixed, mechanism unchanged, full suite rerun PASS. Reported in §1.
- Side effect disclosed and cleaned: importing reference harness modules created untracked `__pycache__` directories in the sibling clone (`TEMPLATE/__pycache__/`, `TEMPLATE/nn/__pycache__/`). Those directories were removed by the executor after the pilot, and sibling `git status` is clean afterwards. No tracked file was edited. Workbench status shows only the two pre-existing untracked files (`nn/luck_notified.marker`, `nn/results_prb2.jsonl`) that were present before the pilot.
- Unresolved questions: (a) Would SVA W20 seed 2 have trained/evaluated like its seeds 0–1, or differently? Unknown, not guessed. (b) Why did held-out success remain zero for both constructions on W10h/W20h at 1000 episodes, when trained success was high on W10 and, for SVA, on W20? No mechanism breakdown is established by this pilot; the pilot was not designed as a failure-anatomy study, and G4-style anatomy is excluded from the official phase and was not performed here. (c) Is low selector-vs-head-value agreement a property of SVA's target, of small Q margins, of training length, or of something else? Unknown from descriptive agreement rates alone. (d) Would 2000 episodes, W30, or different seeds change any pattern? Unknown, not tested, and not authorised by this pilot.

## 7. Bounded conclusion — INCOMPLETE

**Conclusion: INCOMPLETE, not preliminary advantage, not no clear difference, not mixed evidence as a complete-pilot verdict.**

Reasons, in order:
1. Completeness fails: 33/36 jobs, 55/60 records. A comparative result requires all 36 jobs, no duplicates, valid required records. That condition is not met.
2. Held-out primary question, completed jobs: tie at zero (shared 0/120, SVA 0/100 completed, matched pairs 0/100). No held-out advantage is observed, and no held-out winner can be declared, especially with a missing seed.
3. Trained difference on completed matched pairs (SVA higher) does not convert an incomplete pilot into a preliminary advantage claim, because trained performance is not the primary question, the pilot is incomplete, and the sample is 5 matched pairs with non-independent eval episodes.
4. No result here establishes original multiple-act recovery, emergent acts, Uncensored Mindset achievement, or universality. No result here is official EXP1-COMB evidence, and the official success bar is NOT applied as pass/fail to either construction: for contextual reference only, the official bar would require held-out ≥20% absolute AND ≥50% of own trained pooled rate on 300/300 denominators, which this pilot does not meet in design (different seeds/episodes, no outside worlds) and does not meet in held-out counts (0 on completed jobs) — but that contextual note is NOT a declaration that either construction "failed" the official bar, because this pilot is not the official experiment.

If the parent or owner wishes to use the completed-job numbers at all, the only bounded statement supported is: "In DEV-LPAIR-001 Option B, as far as it ran before a VM restart killed the final job, held-out success was 0 on every completed job for both constructions, trained success on completed matched pairs was higher for L-SVA than L-SHARED, and the pilot as a whole is incomplete, so no comparative winner follows."

## 8. Recommended next decision — PROPOSAL only

PROPOSAL, not a decision, not an authorisation, for parent/owner to consider or reject:

- Do NOT rerun the missing job alone as if it were a completion of this pilot, without an explicit new authorisation that states how a single-job rerun would be labelled, whether it would be pooled with the pre-restart records, and how the restart interruption would be disclosed. The executor's instruction was explicit: no rerun without separate decision, and that instruction is preserved here.
- If a complete comparison is still wanted, the cleanest next step, as a PROPOSAL, is a fresh authorisation for a complete rerun of Option B from scratch (all 36 jobs, new scratch records, new label or a clearly versioned rerun label, never pooled with the interrupted run's records), with implementation checks rerun first, and with a stated plan for what happens if a restart occurs again (for example, a pre-stated rule that a restart makes that run incomplete and stops it, as in this run, rather than resuming it). Alternatively, the owner may decide that the held-out tie at zero on completed jobs, combined with the incompleteness, is sufficient reason not to rerun at all, and to return to the official EXP1-COMB decision track (remaining decisions: shared vs separate construction, full L-settings acceptance, bar-evidence vs mechanism definition, §7E) without further unofficial pilots.
- No official EXP1-COMB change, spike, full run, ablation, extra seeds, extended training, second-domain build, or subsequent experiment is proposed by this report as authorised. Any of those would need its own explicit owner authorisation.

## 9. Confirmations and file list

Confirmations:
- No official repository file was edited, committed, or pushed. Sibling HEAD remains `2e2f029`, status clean after removal of import-created `__pycache__` directories (disclosed in §6). Workbench HEAD remains `9b00849`, status shows only pre-existing untracked files.
- No failed or incomplete job was rerun, resumed, or substituted by the executor.
- No record was pooled with SWEEP, SPIKE, EXP1-COMB, or any other programme. All pilot records are labelled DEV-LPAIR-001 only, in scratch only.
- No result was simulated. All counts, walls, losses, act steps, and diagnostics above are derived directly from `records_dev_lpair_001.jsonl` as written incrementally during the pilot, plus the implementation-check output.
- No act meaning was assigned from usage patterns. ACT-0/1/2 remain neutral indices throughout.
- No coverage threshold was invented. No diagnostic threshold was applied.

Scratch files created, all in `~/workspace/dev-lpair-001/`:
- `lpair.py` — scratch implementation.
- `checks.py`, `checks_output.txt` — implementation checks.
- `run_pilot.py`, `run_pilot.log` — sequential runner and log.
- `records_dev_lpair_001.jsonl` — 55 records (11 training, 22 evaluation, 22 diagnostic).
- `weights_L-SHARED_W10_0.npz`, `weights_L-SHARED_W10_1.npz`, `weights_L-SHARED_W10_2.npz`, `weights_L-SHARED_W20_0.npz`, `weights_L-SHARED_W20_1.npz`, `weights_L-SHARED_W20_2.npz`, `weights_L-SVA_W10_0.npz`, `weights_L-SVA_W10_1.npz`, `weights_L-SVA_W10_2.npz`, `weights_L-SVA_W20_0.npz`, `weights_L-SVA_W20_1.npz` — audit weights, completed models only.
- `REPORT-DEV-LPAIR-001-OPTION-B.md` — this report.

End of executor report. Stop after reporting, per instruction.
