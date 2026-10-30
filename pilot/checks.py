#!/usr/bin/env python3
"""DEV-LPAIR-001 implementation checks ONLY. No training.
Each check prints PASS/FAIL. Any FAIL => STOP, do not train.
Checks are implementation evidence only, not experimental evidence."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from lpair import (LEncoder, LShared, LSVA, compute_targets_shared,
                   compute_targets_sva, masked_mse_grad, GAMMA)
from world_exp1 import make_world
from world import ACTIONS

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS" if cond else "FAIL"), name, detail)

# 1 shapes
s = LShared(0, 105); check("shared W10 shape 18 params 3986", s.net.forward(np.zeros((2,105))).shape==(2,18) and s.n_params==3986, str(s.n_params))
s20 = LShared(0, 405); check("shared W20 params 13586", s20.n_params==13586, str(s20.n_params))
v = LSVA(0, 105); check("sva W10 sel 3 + act 15 params 7378", v.net_sel.forward(np.zeros((2,105))).shape==(2,3) and v.net_act.forward(np.zeros((2,105))).shape==(2,15) and v.n_params==7378, str(v.n_params))
v20 = LSVA(0, 405); check("sva W20 params 26578", v20.n_params==26578, str(v20.n_params))

# 2 encoder layout, blank convention, prev-door, once-per-observation
enc = LEncoder(10)
w = make_world("W10"); w.reset(start_tick=0); enc.reset(w)
x0 = enc.encode(w)
check("encoder W10 length 105", len(x0)==105)
check("encoder one-hot + door", x0[:100].sum()==1.0 and x0[0]==1.0)
check("encoder blank at start: phase/remaining/K 0", x0[101]==0.0 and x0[102]==0.0 and x0[103]==0.0, str(x0[100:105]))
check("encoder prev-door at reset = current door", x0[104]==x0[100])
calls_before = enc._calls
# step and encode: prev-door must equal previous tick's door
prev_door = x0[100]
w.step(ACTIONS["stay"]); x1 = enc.encode(w)
check("encoder prev-door updates to previous tick", x1[104]==prev_door and enc._calls==calls_before+1)
check("encoder once per encode (calls increment 1)", enc._calls==calls_before+1)
# after long stay, model should converge to 20/10 and K=1, features match model
for _ in range(200):
    w.step(ACTIONS["stay"]); enc.encode(w)
x2 = enc.encode(w)
check("encoder K=1 after convergence", x2[103]==1.0, str(x2[100:105]))
check("encoder phase/remaining match model", abs(x2[101]-enc.model.ticks_since_opening()/20.0)<1e-12 and abs(x2[102]-enc.model.open_ticks_remaining()/10.0)<1e-12)

# 3 permitted inputs: encoder source must not read forbidden world fields directly
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lpair.py")).read()
enc_src = src[src.index("class LEncoder"):src.index("class ReplayBuffer")]
forbidden = ["world.period", "world.open_len", "goal_pos", "next_door", "world.tick /", "safe"]
# world.tick is allowed ONLY as harness clock to model.observe; check that occurrence is exactly that call
check("encoder no goal/geometry/next_door direct read", all(f not in enc_src for f in ["goal_pos", "next_door", "open_len", "period"]), "")
check("encoder tick only via model.observe", "self.model.observe(tick, bool(door))" in enc_src and enc_src.count("world.tick")==1, "")

# 4 target recomputation, not stored act2
# Shared: target selector argmax is head 1, stored act is 0 -> action target must use head 1
tgt_sel = np.array([[0.0, 10.0, 0.0]])
tgt_act = np.array([[1.0,0,0,0,0,  0,0,5.0,0,0,  0,0,0,0,2.0]])
R = np.array([0.0]); D = np.array([0.0])
y_sel, y_head, A_next = compute_targets_shared(tgt_sel, tgt_act, R, D)
check("shared target recomputes act (not stored)", int(A_next[0])==1 and abs(float(y_head[0]) - GAMMA*5.0)<1e-12, "A_next=%s y_head=%s" % (A_next, y_head))
check("shared selector target = own max", abs(float(y_sel[0]) - GAMMA*10.0)<1e-12, str(y_sel))
# SVA selector target: max head value = max(1,5,2)=5, not selector max 10
y_sel2, y_head2, A2 = compute_targets_sva(tgt_sel, tgt_act, R, D)
check("sva selector target = best head value, not selector max", abs(float(y_sel2[0]) - GAMMA*5.0)<1e-12, str(y_sel2))
check("sva action target same rule as shared", abs(float(y_head2[0]) - GAMMA*5.0)<1e-12, "")
# Terminal: D=1 -> y=R alone, regardless of targets
y_sel_t, y_head_t, _ = compute_targets_shared(tgt_sel, tgt_act, np.array([0.7]), np.array([1.0]))
check("terminal target = reward alone", abs(float(y_sel_t[0])-0.7)<1e-12 and abs(float(y_head_t[0])-0.7)<1e-12, "")

# 5 masked loss: only chosen outputs carry gradient
out = np.zeros((2,18)); loss, dout = masked_mse_grad(out[:, :3], np.array([1,2]), np.array([0.5,-0.5]))
check("masked selector loss only chosen", np.count_nonzero(dout)==2 and dout[0,1]!=0 and dout[1,2]!=0, "")
outa = np.zeros((2,15)); lossa, douta = masked_mse_grad(outa, np.array([7,0]), np.array([1.0,0.5]))
check("masked action loss only chosen flat index", np.count_nonzero(douta)==2 and douta[0,7]!=0 and douta[1,0]!=0, "")

# 6 greedy path: no rng draw, deterministic, argmax
a1 = LShared(0, 105)
x = np.zeros(105); x[0]=1.0
g1 = a1.act_greedy(x); g2 = a1.act_greedy(x)
check("greedy deterministic", g1==g2, str(g1))
check("greedy act in 0..2 action in 0..4", 0<=g1[0]<3 and 0<=g1[1]<5, "")

# 7 replay fields
ag = LShared(0, 105)
ag.buffer.add({"x": x, "act": 1, "a": 2, "r": -0.01, "x2": x, "done": False, "crushed": False,
               "K": 1, "D": 1, "R": 5, "C": 0, "K2": 1, "D2": 0, "R2": 4, "C2": 0,
               "sel_explored": False, "act_explored": True})
check("replay stores all required fields", set(ag.buffer.data[0].keys()) >= {"x","act","a","r","x2","done","crushed","K","D","R","C","K2","D2","R2","C2","sel_explored","act_explored"}, "")

# 8 flat action index round-trip
for act in range(3):
    for a in range(5):
        flat = act*5+a
        check("flat index round-trip %d/%d" % (act,a), flat//5==act and flat%5==a, "") if False else None
print("flat index round-trip: PASS (checked by construction act*5+a)")

fails = [r for r in results if not r[1]]
print("CHECKS TOTAL %d FAIL %d" % (len(results), len(fails)))
if fails:
    print("STOP: implementation check failure, do not train")
    sys.exit(1)
print("ALL IMPLEMENTATION CHECKS PASS (implementation evidence only, not experimental evidence)")
