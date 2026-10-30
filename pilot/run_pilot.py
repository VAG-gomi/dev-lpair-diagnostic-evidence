#!/usr/bin/env python3
"""DEV-LPAIR-001 Option B pilot runner, sequential, incremental.
Order: L-SHARED W10 seeds 0,1,2, then W20 seeds 0,1,2, then L-SVA same.
For each trained model: training job, then trained eval, then held-out eval.
Records are appended immediately after each job to records_dev_lpair_001.jsonl.
If any job fails / is killed, the script stops (caller must NOT rerun it;
pilot is then INCOMPLETE by instruction). No duplicate guard needed on a
single run, but a duplicate job_id in the file aborts before training.
Weights are saved as npz in scratch for audit only, never official records.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from lpair import make_agent, train_job, eval_job
from world_exp1 import make_world, HELDOUT_OF

SCRATCH = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(SCRATCH, "records_dev_lpair_001.jsonl")
HELDOUT = {"W10": "W10h", "W20": "W20h"}
WORLDS = ["W10", "W20"]
SEEDS = [0, 1, 2]
CONSTRUCTIONS = ["L-SHARED", "L-SVA"]

def write(rec):
    with open(REC, "a") as f:
        f.write(json.dumps(rec) + "\n"); f.flush()

def main():
    seen = set()
    if os.path.exists(REC):
        with open(REC) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line); seen.add(r.get("job_id"))
    for construction in CONSTRUCTIONS:
        for world in WORLDS:
            for seed in SEEDS:
                train_id = "DEV-LPAIR-001:train:%s:%s:%d" % (construction, world, seed)
                if train_id in seen:
                    raise SystemExit("duplicate job %s; aborting, do not rerun" % train_id)
                grid = make_world(world).grid_size
                agent = make_agent(construction, seed, grid*grid+5)
                t0 = time.time()
                stats, _enc = train_job(agent, world, seed, episodes=1000)
                wall = time.time() - t0
                # update counts: shared .updates, sva .updates_sel/.updates_act (equal by construction)
                updates = int(getattr(agent, "updates_sel", getattr(agent, "updates", 0)))
                updates_act = int(getattr(agent, "updates_act", updates))
                act_steps = [0,0,0]; act_eps = [0,0,0]
                for s in stats:
                    for i in range(3):
                        act_steps[i] += s["act_steps"][i]
                        if s["act_eps_used"][i]: act_eps[i] += 1
                rec_train = {"record": "DEV-LPAIR-001", "job_id": train_id, "job_type": "training",
                    "construction": construction, "world": world, "seed": seed,
                    "episodes": len(stats), "wall_s": wall,
                    "reached": sum(1 for s in stats if s["reached"]),
                    "ticks_total": sum(s["ticks"] for s in stats),
                    "ticks_mean": sum(s["ticks"] for s in stats)/len(stats),
                    "crushes_total": sum(s["crushes"] for s in stats),
                    "updates": updates, "updates_act": updates_act,
                    "n_params": int(agent.n_params),
                    "mean_loss_sel": float(np.mean([s["mean_loss_sel"] for s in stats if s["mean_loss_sel"] is not None])) if any(s["mean_loss_sel"] is not None for s in stats) else None,
                    "mean_loss_act": float(np.mean([s["mean_loss_act"] for s in stats if s["mean_loss_act"] is not None])) if any(s["mean_loss_act"] is not None for s in stats) else None,
                    "act_steps": act_steps, "act_episodes_used": act_eps}
                write(rec_train); seen.add(train_id)
                print(train_id, "done wall=%.1f reached=%d/%d updates=%d" % (wall, rec_train["reached"], len(stats), updates), flush=True)
                # Save weights for audit (scratch only)
                if construction == "L-SHARED":
                    np.savez(os.path.join(SCRATCH, "weights_%s_%s_%d.npz" % (construction, world, seed)), **{"net": agent.net.get_params()})
                else:
                    np.savez(os.path.join(SCRATCH, "weights_%s_%s_%d.npz" % (construction, world, seed)), **{"sel": agent.net_sel.get_params(), "act": agent.net_act.get_params()})
                for eval_world, kind in ((world, "trained"), (HELDOUT[world], "held-out")):
                    eval_id = "DEV-LPAIR-001:eval:%s:%s:%d:%s" % (construction, world, seed, eval_world)
                    if eval_id in seen:
                        raise SystemExit("duplicate job %s; aborting" % eval_id)
                    t0 = time.time()
                    episodes, _states, diag = eval_job(agent, eval_world, grid)
                    wall_e = time.time() - t0
                    act_steps_e = [0,0,0]; act_eps_e = [0,0,0]
                    for s in episodes:
                        for i in range(3):
                            act_steps_e[i] += s["act_steps"][i]
                            if s["act_eps_used"][i]: act_eps_e[i] += 1
                    rec_eval = {"record": "DEV-LPAIR-001", "job_id": eval_id, "job_type": "evaluation",
                        "construction": construction, "train_world": world, "eval_world": eval_world,
                        "eval_kind": kind, "seed": seed, "episodes": len(episodes), "wall_s": wall_e,
                        "reached": sum(1 for s in episodes if s["reached"]),
                        "ticks_total": sum(s["ticks"] for s in episodes),
                        "ticks_mean": sum(s["ticks"] for s in episodes)/len(episodes),
                        "crushes_total": sum(s["crushes"] for s in episodes),
                        "act_steps": act_steps_e, "act_episodes_used": act_eps_e}
                    write(rec_eval); seen.add(eval_id)
                    rec_diag = {"record": "DEV-LPAIR-001", "job_id": eval_id + ":diagnostic", "job_type": "diagnostic",
                        "construction": construction, "train_world": world, "eval_world": eval_world,
                        "eval_kind": kind, "seed": seed, "diagnostic": diag}
                    write(rec_diag)
                    print(eval_id, "done wall=%.2f reached=%d/%d" % (wall_e, rec_eval["reached"], len(episodes)), flush=True)
    print("PILOT COMPLETE 36 jobs (12 trainings + 24 evaluations) + 24 diagnostics")

if __name__ == "__main__":
    main()
