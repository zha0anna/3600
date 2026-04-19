"""
Standalone HMM accuracy test.
Bypasses the multiprocess engine - just simulates the rat against our belief tracker
and reports how well we localize the rat over time.

Run from the repo root:
    python 3600-agents/Cat/test_hmm_accuracy.py
    python 3600-agents/Cat/test_hmm_accuracy.py --games 50 --turns 80
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "engine"))
sys.path.insert(0, THIS_DIR)

from game.board import Board
from game.rat import Rat
from game.enums import BOARD_SIZE, Cell
from board_utils import generate_spawns

from rat_belief import RatBeliefHMM
from agent import PlayerAgent


def setup_board() -> Board:
    """Mirrors the corner-block + spawn setup from engine/gameplay.py."""
    board = Board(time_to_play=240.0, build_history=False)
    shapes = [(2, 3), (3, 2), (2, 2)]
    for ox, oy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        w, h = random.choice(shapes)
        for dx in range(w):
            for dy in range(h):
                x = dx if ox == 0 else BOARD_SIZE - 1 - dx
                y = dy if oy == 0 else BOARD_SIZE - 1 - dy
                board.set_cell((x, y), Cell.BLOCKED)
    spawn_a, spawn_b = generate_spawns(board)
    board.player_worker.position = spawn_a
    board.opponent_worker.position = spawn_b
    return board


def load_transition_matrix() -> np.ndarray:
    """Same loader as engine/gameplay.py, but pure NumPy (no JAX dependency)."""
    base = os.path.join(REPO_ROOT, "engine", "transition_matrices")
    pkls = [f for f in os.listdir(base) if f.endswith(".pkl")]
    with open(os.path.join(base, random.choice(pkls)), "rb") as f:
        T = pickle.load(f)
    T = np.asarray(T, dtype=np.float64)
    noise = np.random.uniform(-0.1, 0.1, size=T.shape)
    T = np.maximum(T * (1 + noise), 0)
    row_sum = T.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    return T / row_sum


def simulate_one_game(T: np.ndarray, n_turns: int, search_threshold: float):
    """
    Simulate one game from Player A's perspective, ignoring Player B's existence.
    Each "turn" = (rat moves once, sensor sampled, HMM updates).
    Returns dict of per-turn metrics.
    """
    board = setup_board()
    rat = Rat(T)
    rat.spawn()  # 1000 headstart moves

    hmm = RatBeliefHMM(T)

    top1_hits = 0
    top3_hits = 0
    mean_dist_err = 0.0
    confidences = []
    search_outcomes = []  # list of (confidence, was_correct)
    # EV calibration: for every turn, record (predicted_ev, base_ev_pred, realized_reward)
    # realized_reward = +4 if a search now would hit, -2 if it would miss.
    ev_records = []

    init_time = time.perf_counter()
    update_time_total = 0.0

    for turn in range(n_turns):
        rat.move()
        sensor = rat.sample(board)

        t0 = time.perf_counter()
        hmm.update(board, sensor)
        update_time_total += time.perf_counter() - t0

        true_pos = rat.get_position()
        best_pos, conf, _ = hmm.estimate()
        top_k = hmm.top_k(3)
        top_positions = [pos for pos, _ in top_k]

        if best_pos == true_pos:
            top1_hits += 1
        if true_pos in top_positions:
            top3_hits += 1

        mean_dist_err += abs(best_pos[0] - true_pos[0]) + abs(best_pos[1] - true_pos[1])
        confidences.append(conf)

        # Compute predicted EV for this turn (does not require self).
        predicted_ev = PlayerAgent.calculate_search_ev(None, conf, turn)
        base_ev_pred = conf * 4.0 + (1.0 - conf) * -2.0
        realized = 4 if best_pos == true_pos else -2
        ev_records.append((conf, predicted_ev, base_ev_pred, realized))

        if conf >= search_threshold:
            was_correct = (best_pos == true_pos)
            search_outcomes.append((conf, was_correct))

    return {
        "top1_acc": top1_hits / n_turns,
        "top3_acc": top3_hits / n_turns,
        "mean_dist_err": mean_dist_err / n_turns,
        "mean_conf": float(np.mean(confidences)),
        "max_conf": float(np.max(confidences)),
        "avg_update_us": (update_time_total / n_turns) * 1e6,
        "init_time_ms": (time.perf_counter() - init_time - update_time_total) * 1000,
        "n_searches": len(search_outcomes),
        "search_hit_rate": (sum(1 for _, c in search_outcomes if c) / len(search_outcomes)) if search_outcomes else 0.0,
        "search_ev": (sum(4 if c else -2 for _, c in search_outcomes)) if search_outcomes else 0,
        "ev_records": ev_records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.16, help="Search confidence threshold")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    print(f"Running {args.games} games x {args.turns} turns each, search threshold = {args.threshold}")
    print("-" * 70)

    results = [simulate_one_game(load_transition_matrix(), args.turns, args.threshold) for _ in range(args.games)]

    def avg(key):
        return float(np.mean([r[key] for r in results]))

    total_searches = sum(r["n_searches"] for r in results)
    total_hits = sum(r["n_searches"] * r["search_hit_rate"] for r in results)
    total_ev = sum(r["search_ev"] for r in results)

    print(f"Top-1 accuracy:     {avg('top1_acc')*100:5.1f} %  (HMM argmax == true rat position)")
    print(f"Top-3 accuracy:     {avg('top3_acc')*100:5.1f} %  (true rat in HMM top-3)")
    print(f"Mean L1 error:      {avg('mean_dist_err'):5.2f}    (Manhattan dist HMM->rat)")
    print(f"Mean confidence:    {avg('mean_conf')*100:5.1f} %")
    print(f"Max confidence:     {avg('max_conf')*100:5.1f} %  (averaged across games)")
    print(f"Update speed:       {avg('avg_update_us'):5.1f} us per turn")
    print(f"Init (T^1000) time: {avg('init_time_ms'):5.1f} ms per game")
    print()
    print(f"Searches taken:     {total_searches}  (threshold p>={args.threshold})")
    if total_searches:
        print(f"Search hit rate:    {(total_hits/total_searches)*100:5.1f} %")
        print(f"Total search EV:    {total_ev:+d} pts ({total_ev/total_searches:+.2f} pts/search)")

    # ----- EV calibration -----
    # Aggregate every turn's prediction so we can compare predicted vs actual
    # mean reward, both overall and bucketed by predicted EV.
    all_records = [rec for r in results for rec in r["ev_records"]]
    print()
    print("=" * 70)
    print(f"EV calibration ({len(all_records)} turns sampled)")
    print("=" * 70)

    # Subset: turns where calculate_search_ev says we should search (EV > 0).
    search_set = [(c, p, b, real) for (c, p, b, real) in all_records if p > 0]
    n_search = len(search_set)
    if n_search == 0:
        print("calculate_search_ev never recommended a search (predicted EV always <= 0).")
    else:
        mean_pred_full = float(np.mean([p for (_, p, _, _) in search_set]))
        mean_pred_base = float(np.mean([b for (_, _, b, _) in search_set]))
        mean_actual    = float(np.mean([real for (_, _, _, real) in search_set]))
        hit_rate       = float(np.mean([1.0 if real == 4 else 0.0 for (_, _, _, real) in search_set]))
        mean_conf      = float(np.mean([c for (c, _, _, _) in search_set]))
        # Implied actual EV from the empirical hit rate (sanity check).
        implied_ev = hit_rate * 4.0 + (1.0 - hit_rate) * -2.0
        print(f"Turns where pred EV > 0:          {n_search} / {len(all_records)}")
        print(f"  mean confidence:                {mean_conf*100:5.1f} %")
        print(f"  mean predicted EV (full):       {mean_pred_full:+.3f} pts")
        print(f"  mean predicted EV (base only):  {mean_pred_base:+.3f} pts")
        print(f"  mean ACTUAL reward:             {mean_actual:+.3f} pts")
        print(f"  empirical hit rate:             {hit_rate*100:5.1f} %  (implies EV {implied_ev:+.3f})")
        print(f"  base EV error (pred - actual):  {mean_pred_base - mean_actual:+.3f} pts")
        print(f"  full EV error (pred - actual):  {mean_pred_full - mean_actual:+.3f} pts")

    # Bucket all turns by predicted EV to inspect calibration curve.
    print()
    print(f"{'pred EV bucket':>18}  {'n':>6}  {'mean conf':>10}  {'pred EV':>9}  {'actual':>9}  {'hit %':>7}")
    edges = [-2.01, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        bucket = [(c, p, b, real) for (c, p, b, real) in all_records if lo <= p < hi]
        if not bucket:
            continue
        n = len(bucket)
        mp = float(np.mean([p for (_, p, _, _) in bucket]))
        ma = float(np.mean([real for (_, _, _, real) in bucket]))
        hr = float(np.mean([1.0 if real == 4 else 0.0 for (_, _, _, real) in bucket]))
        mc = float(np.mean([c for (c, _, _, _) in bucket]))
        print(f"  [{lo:+.2f}, {hi:+.2f}) {n:>6}  {mc*100:9.1f}%  {mp:+9.3f}  {ma:+9.3f}  {hr*100:6.1f}%")


if __name__ == "__main__":
    main()
