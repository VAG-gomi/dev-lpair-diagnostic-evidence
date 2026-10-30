#!/usr/bin/env python3
"""DEV-LPAIR-001 scratch implementation — L-SHARED vs L-SVA, Option B only.
UNOFFICIAL, NOT EXP1-COMB. Records labelled DEV-LPAIR-001 only.
Scratch only: ~/workspace/dev-lpair-001/. No official repo file is edited.
Hand-written NumPy only, tanh hidden, linear output, float64, as in
TEMPLATE/nn/nnet.py. Imports reference harness files read-only; it does
not modify them. The L encoder, networks, targets and trainer are new
scratch code, written for this pilot only.

Encoder layout, grid^2+5 (W10 105, W20 405):
  [0..n-1] one-hot position
  [n]      door bit D
  [n+1]    phase = ticks_since_opening / 20.0, 0.0 if unknown
  [n+2]    remaining = open_ticks_remaining / 10.0, 0.0 if unknown
  [n+3]    known flag K, 1.0 iff phase and remaining are available
  [n+4]    prev-door bit (previous tick's door bit)
Blank-feature convention matches TEMPLATE/nn/agent_wrapped.py exactly:
phase/remaining are 0.0 and K=0 when the ScheduleModel has no estimates.
Prev-door convention matches TEMPLATE/nn/agent_mem.py exactly: at reset,
prev-door = door bit at the reset tick; each encode uses the stored
prev-door, then updates it to the door bit just encoded.
ScheduleModel sees ONLY (tick, door), exactly once per encode.
Permitted inputs only. Forbidden: world period/open_len, goal,
geometry, next_door, episode counter as network input, tick as direct
network input, safe-to-cross flag, fixed act-selection rule.
World.tick is used ONLY as the harness clock passed to
ScheduleModel.observe(tick, door), exactly as in the wrapped encoder.
"""
import os, sys, random
_HERE = os.path.dirname(os.path.abspath(__file__))
# Reference harness, read-only (never edited by this file).
_REF_NN = os.path.expanduser("~/workspace/universal-learning-model-template/TEMPLATE/nn")
_REF_T = os.path.expanduser("~/workspace/universal-learning-model-template/TEMPLATE")
for _p in (_REF_NN, _REF_T):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
from nnet import MLP, masked_mse_grad, Adam
from wrapper import ScheduleModel
from world_exp1 import make_world, PERIOD, OPEN_LEN
from world import ACTIONS

ACTION_NAMES = ["up", "down", "left", "right", "stay"]
N_ACTIONS = 5
N_ACTS = 3
EPISODE_CAP = 200
R_STEP, R_GOAL, R_CRUSH, GAMMA = -0.01, 1.0, -1.0, 0.99
PHASE_SCALE, REMAINING_SCALE = 20.0, 10.0

def epsilon(episode, eps_end=0.05, decay_episodes=1000):
    """Identical to TEMPLATE/nn/agent_nn.epsilon: linear 1.0->eps_end
    over decay_episodes, then flat. Pilot training is 1000 episodes,
    so ep 0..999 gives 1.0 down to ~0.05095."""
    return max(eps_end, 1.0 - (1.0 - eps_end) * episode / decay_episodes)

def env_step_exp1(world, action_idx):
    n_log = len(world.log)
    world.step(ACTIONS[ACTION_NAMES[action_idx]])
    crushed = any(e[2] == "crushed" for e in world.log[n_log:])
    reached = world.agent == world.goal_pos
    r = R_STEP + (R_CRUSH if crushed else 0.0) + (R_GOAL if reached else 0.0)
    return r, reached, crushed

class LEncoder:
    """L encoder: wrapped time features + prev-door, grid^2+5."""
    def __init__(self, grid_size, model=None):
        self.grid_size = grid_size
        self.n_inputs = grid_size * grid_size + 5
        self.model = model if model is not None else ScheduleModel()
        self.prev_door = 0
        self.last_contradiction = False
        self._calls = 0
    def reset(self, world):
        _pos, door = world.observe()
        self.prev_door = 1 if door else 0
        self.model.new_episode()
        self.last_contradiction = False
    def encode(self, world):
        (x, y), door = world.observe()
        tick = world.tick
        # ScheduleModel's only input, once per observation.
        contradicted = self.model.observe(tick, bool(door))
        self.last_contradiction = bool(contradicted)
        self._calls += 1
        n = self.grid_size * self.grid_size
        v = np.zeros(self.n_inputs)
        v[y * self.grid_size + x] = 1.0
        v[n] = 1.0 if door else 0.0
        remaining = self.model.open_ticks_remaining()
        phase = self.model.ticks_since_opening()
        if remaining is not None and phase is not None:
            v[n+1] = phase / PHASE_SCALE
            v[n+2] = remaining / REMAINING_SCALE
            v[n+3] = 1.0
        # else phase/remaining stay 0.0 and K stays 0.0 (matches wrapped)
        v[n+4] = float(self.prev_door)
        self.prev_door = 1 if door else 0
        return v
    def selector_inputs(self, v):
        """(K, D, raw R, C) for a state vector v, using the contradiction
        flag of the encode that produced v. C is recorded, it is NOT a
        network input slot; the network input is v itself."""
        n = self.grid_size * self.grid_size
        K = int(v[n+3] == 1.0)
        D = int(v[n] == 1.0)
        R = int(round(float(v[n+2]) * REMAINING_SCALE)) if K else None
        C = int(self.last_contradiction)
        return K, D, R, C

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity; self.data = []; self.pos = 0
    def add(self, item):
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.pos] = item; self.pos = (self.pos + 1) % self.capacity
    def sample(self, n, rng):
        idx = rng.choice(len(self.data), n, replace=False)
        return [self.data[i] for i in idx]
    def __len__(self):
        return len(self.data)

# --- Pure target functions (tested directly in checks.py) ---
def compute_targets_shared(tgt_sel, tgt_act, R, D):
    """tgt_sel (B,3) target selector outputs, tgt_act (B,15) target
    action outputs, R (B,), D (B,) float done (1=reached only).
    Returns (y_sel, y_head, A_next). A_next is recomputed, never stored."""
    A_next = np.argmax(tgt_sel, axis=1)
    B = R.shape[0]
    head_vals = tgt_act.reshape(B, N_ACTS, N_ACTIONS)
    y_sel = R + GAMMA * (1.0 - D) * tgt_sel.max(axis=1)
    y_head = R + GAMMA * (1.0 - D) * head_vals[np.arange(B), A_next].max(axis=1)
    return y_sel, y_head, A_next

def compute_targets_sva(tgt_sel, tgt_act, R, D):
    """Selector target uses target ACTION network: max over acts of
    max over actions in that head. Action target same as shared."""
    A_next = np.argmax(tgt_sel, axis=1)
    B = R.shape[0]
    head_vals = tgt_act.reshape(B, N_ACTS, N_ACTIONS)
    head_max = head_vals.max(axis=2)  # (B,3) V_target per act
    y_sel = R + GAMMA * (1.0 - D) * head_max.max(axis=1)
    y_head = R + GAMMA * (1.0 - D) * head_vals[np.arange(B), A_next].max(axis=1)
    return y_sel, y_head, A_next

class LShared:
    construction = "L-SHARED"
    def __init__(self, seed, n_inputs, hidden=32, lr=1e-3, buffer_size=10000, batch=32, target_sync=200):
        self.seed = seed; self.n_inputs = n_inputs; self.hidden = hidden
        self.net = MLP([n_inputs, hidden, 18], seed=seed)
        self.target = MLP([n_inputs, hidden, 18], seed=seed)
        self.target.set_params(self.net.get_params())
        self.opt = Adam(self.net, lr=lr)
        self.rng = random.Random(seed)
        self.nprng = np.random.RandomState(seed)
        self.buffer = ReplayBuffer(buffer_size); self.batch = batch
        self.target_sync = target_sync; self.updates = 0
    @property
    def n_params(self): return int(self.net.n_params)
    def act(self, x, eps):
        # Independent draws, selector first then action, same rng stream.
        sel_explored = False; act_explored = False
        if self.rng.random() < eps:
            act_idx = self.rng.randrange(N_ACTS); sel_explored = True
        else:
            act_idx = int(np.argmax(self.net.forward(x[None, :])[0][:3]))
        q = self.net.forward(x[None, :])[0][3:]
        head = q[act_idx*5:(act_idx+1)*5]
        if self.rng.random() < eps:
            a = self.rng.randrange(N_ACTIONS); act_explored = True
        else:
            a = int(np.argmax(head))
        return act_idx, a, sel_explored, act_explored
    def act_greedy(self, x):
        q = self.net.forward(x[None, :])[0]
        act_idx = int(np.argmax(q[:3])); a = int(np.argmax(q[3+act_idx*5:3+(act_idx+1)*5]))
        return act_idx, a
    def train_step(self):
        if len(self.buffer) < self.batch: return None
        b = self.buffer.sample(self.batch, self.nprng)
        S = np.stack([t["x"] for t in b]); ACT = np.array([t["act"] for t in b])
        A = np.array([t["act"]*5 + t["a"] for t in b], dtype=int)
        R = np.array([t["r"] for t in b]); S2 = np.stack([t["x2"] for t in b])
        D = np.array([t["done"] for t in b], dtype=float)
        tgt = self.target.forward(S2)
        y_sel, y_head, _ = compute_targets_shared(tgt[:, :3], tgt[:, 3:], R, D)
        out = self.net.forward(S)
        loss_sel, dout_sel = masked_mse_grad(out[:, :3], ACT, y_sel)
        loss_act, dout_act = masked_mse_grad(out[:, 3:], A, y_head)
        dout = np.zeros_like(out); dout[:, :3] = dout_sel; dout[:, 3:] = dout_act
        self.net.backward(dout); self.opt.step(); self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.set_params(self.net.get_params())
        return float(loss_sel), float(loss_act)

class LSVA:
    construction = "L-SVA"
    def __init__(self, seed, n_inputs, hidden=32, lr=1e-3, buffer_size=10000, batch=32, target_sync=200):
        self.seed = seed; self.n_inputs = n_inputs; self.hidden = hidden
        # Distinct init seeds for the two networks: using the same seed
        # for both would give identical hidden-layer init draws (same
        # fan_in/fan_out) and overlapping output-layer draws, a spurious
        # coupling. Selector uses seed, action uses seed+10000.
        # Reported as an implementation choice, not in the spec text.
        self.net_sel = MLP([n_inputs, hidden, 3], seed=seed)
        self.tgt_sel = MLP([n_inputs, hidden, 3], seed=seed)
        self.tgt_sel.set_params(self.net_sel.get_params())
        self.net_act = MLP([n_inputs, hidden, 15], seed=seed+10000)
        self.tgt_act = MLP([n_inputs, hidden, 15], seed=seed+10000)
        self.tgt_act.set_params(self.net_act.get_params())
        self.opt_sel = Adam(self.net_sel, lr=lr); self.opt_act = Adam(self.net_act, lr=lr)
        self.rng = random.Random(seed); self.nprng = np.random.RandomState(seed)
        self.buffer = ReplayBuffer(buffer_size); self.batch = batch
        self.target_sync = target_sync; self.updates_sel = 0; self.updates_act = 0
    @property
    def n_params(self): return int(self.net_sel.n_params + self.net_act.n_params)
    @property
    def updates(self): return int(self.updates_sel)
    def act(self, x, eps):
        sel_explored = False; act_explored = False
        if self.rng.random() < eps:
            act_idx = self.rng.randrange(N_ACTS); sel_explored = True
        else:
            act_idx = int(np.argmax(self.net_sel.forward(x[None, :])[0]))
        head = self.net_act.forward(x[None, :])[0][act_idx*5:(act_idx+1)*5]
        if self.rng.random() < eps:
            a = self.rng.randrange(N_ACTIONS); act_explored = True
        else:
            a = int(np.argmax(head))
        return act_idx, a, sel_explored, act_explored
    def act_greedy(self, x):
        act_idx = int(np.argmax(self.net_sel.forward(x[None, :])[0]))
        head = self.net_act.forward(x[None, :])[0][act_idx*5:(act_idx+1)*5]
        return act_idx, int(np.argmax(head))
    def train_step(self):
        if len(self.buffer) < self.batch: return None
        b = self.buffer.sample(self.batch, self.nprng)
        S = np.stack([t["x"] for t in b]); ACT = np.array([t["act"] for t in b])
        A = np.array([t["act"]*5 + t["a"] for t in b], dtype=int)
        R = np.array([t["r"] for t in b]); S2 = np.stack([t["x2"] for t in b])
        D = np.array([t["done"] for t in b], dtype=float)
        tgt_sel = self.tgt_sel.forward(S2); tgt_act = self.tgt_act.forward(S2)
        y_sel, y_head, _ = compute_targets_sva(tgt_sel, tgt_act, R, D)
        out_sel = self.net_sel.forward(S); out_act = self.net_act.forward(S)
        loss_sel, dout_sel = masked_mse_grad(out_sel, ACT, y_sel)
        loss_act, dout_act = masked_mse_grad(out_act, A, y_head)
        self.net_sel.backward(dout_sel); self.opt_sel.step(); self.updates_sel += 1
        if self.updates_sel % self.target_sync == 0:
            self.tgt_sel.set_params(self.net_sel.get_params())
        self.net_act.backward(dout_act); self.opt_act.step(); self.updates_act += 1
        if self.updates_act % self.target_sync == 0:
            self.tgt_act.set_params(self.net_act.get_params())
        return float(loss_sel), float(loss_act)

def make_agent(construction, seed, n_inputs):
    if construction == "L-SHARED": return LShared(seed, n_inputs)
    if construction == "L-SVA": return LSVA(seed, n_inputs)
    raise ValueError(construction)

def train_job(agent, world_name, seed, episodes=1000):
    """Train one model. Returns (stats, encoder). Mirrors exp1 train_loop:
    start-tick stream Random(seed+1000), episode start_tick in 0..19,
    encoder persists across episodes, reset per episode."""
    grid = make_world(world_name).grid_size
    enc = LEncoder(grid)
    start_rng = random.Random(seed + 1000)
    stats = []
    for ep in range(episodes):
        w = make_world(world_name); w.reset(start_tick=start_rng.randrange(PERIOD))
        enc.reset(w); x = enc.encode(w); K, D, R, C = enc.selector_inputs(x)
        eps = epsilon(ep); ticks = 0; crushes = 0; reached = False
        losses_sel = []; losses_act = []
        act_steps = [0, 0, 0]; act_eps_used = [False, False, False]
        while ticks < EPISODE_CAP:
            act_idx, a, sel_exp, act_exp = agent.act(x, eps)
            act_steps[act_idx] += 1; act_eps_used[act_idx] = True
            r, reached, crushed = env_step_exp1(w, a); crushes += int(crushed)
            x2 = enc.encode(w); K2, D2, R2, C2 = enc.selector_inputs(x2)
            agent.buffer.add({"x": x, "act": act_idx, "a": a, "r": r, "x2": x2,
                              "done": bool(reached), "crushed": bool(crushed),
                              "K": K, "D": D, "R": R, "C": C,
                              "K2": K2, "D2": D2, "R2": R2, "C2": C2,
                              "sel_explored": sel_exp, "act_explored": act_exp})
            res = agent.train_step()
            if res is not None:
                losses_sel.append(res[0]); losses_act.append(res[1])
            x, K, D, R, C = x2, K2, D2, R2, C2; ticks += 1
            if reached: break
        stats.append({"ticks": ticks, "reached": bool(reached), "crushes": crushes,
                      "mean_loss_sel": (sum(losses_sel)/len(losses_sel) if losses_sel else None),
                      "mean_loss_act": (sum(losses_act)/len(losses_act) if losses_act else None),
                      "act_steps": act_steps, "act_eps_used": act_eps_used})
    return stats, enc

def eval_job(agent, eval_world_name, grid_train):
    """One greedy eval job: 20 episodes, start phases 0..19 in order.
    Fresh encoder for the job, persists across its 20 episodes, reset
    per episode. Weights frozen (no train_step, no buffer add).
    Returns (episodes, states, diag) where states are recorded encoder
    vectors for every eval step, and diag is computed OFFLINE from
    those states after the job, with no env step and no weight update."""
    grid = make_world(eval_world_name).grid_size
    enc = LEncoder(grid)
    episodes = []; states = []
    for start in range(PERIOD):
        w = make_world(eval_world_name); w.reset(start_tick=start)
        enc.reset(w); ticks = 0; crushes = 0; reached = False; total = 0.0
        act_steps = [0, 0, 0]; act_eps_used = [False, False, False]
        while ticks < EPISODE_CAP:
            x = enc.encode(w); states.append(x.copy())
            act_idx, a = agent.act_greedy(x)
            act_steps[act_idx] += 1; act_eps_used[act_idx] = True
            r, reached, crushed = env_step_exp1(w, a)
            total += r; crushes += int(crushed); ticks += 1
            if reached: break
        episodes.append({"start_phase": start, "ticks": ticks, "crushes": crushes,
                         "reached": bool(reached), "return": total,
                         "act_steps": act_steps, "act_eps_used": act_eps_used})
    # Offline diagnostic on recorded states only.
    S = np.stack(states) if states else np.zeros((0, agent.n_inputs))
    diag = diagnostic(agent, S)
    return episodes, states, diag

def diagnostic(agent, S):
    """Same-state action-head diagnostic + (SVA only) selector-vs-
    head-value agreement. No env step, no weight update, frozen nets.
    Returns counts/denominators, no thresholds, no act meanings."""
    if S.shape[0] == 0:
        return {"n_states": 0}
    if agent.construction == "L-SHARED":
        q = agent.net.forward(S)  # (N,18)
        heads = q[:, 3:].reshape(-1, 3, 5)
        sel_q = q[:, :3]
    else:
        heads = agent.net_act.forward(S).reshape(-1, 3, 5)
        sel_q = agent.net_sel.forward(S)
    head_arg = np.argmax(heads, axis=2)  # (N,3)
    dist = []
    for h in range(3):
        counts = [int(np.sum(head_arg[:, h] == a)) for a in range(5)]
        dist.append({"head": h, "counts": counts, "denominator": int(S.shape[0])})
    agree = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        agree["head%d_vs_head%d" % (i, j)] = {
            "agree": int(np.sum(head_arg[:, i] == head_arg[:, j])),
            "denominator": int(S.shape[0])}
    out = {"n_states": int(S.shape[0]), "per_head_action_counts": dist,
           "pairwise_agreement": agree}
    if agent.construction == "L-SVA":
        head_max = heads.max(axis=2)  # (N,3)
        head_argmax = np.argmax(head_max, axis=1)
        sel_argmax = np.argmax(sel_q, axis=1)
        sel_sorted = np.sort(sel_q, axis=1); hv_sorted = np.sort(head_max, axis=1)
        out["selector_vs_headvalue"] = {
            "agree": int(np.sum(sel_argmax == head_argmax)),
            "denominator": int(S.shape[0]),
            "mean_selector_margin": float(np.mean(sel_sorted[:, -1] - sel_sorted[:, -2])) if S.shape[0] else None,
            "mean_headvalue_margin": float(np.mean(hv_sorted[:, -1] - hv_sorted[:, -2])) if S.shape[0] else None}
    return out
