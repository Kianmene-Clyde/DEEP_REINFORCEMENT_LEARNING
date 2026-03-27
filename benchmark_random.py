#!/usr/bin/env python3
"""
benchmark_random.py - Calcule le nombre de parties/seconde avec un joueur random.

Usage:
    python benchmark_random.py
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environments import LineWorld, GridWorld, TicTacToe, Quarto
from agents import RandomAgent


def benchmark(env, agent, num_games=10000, max_steps=200):
    """Joue num_games parties et retourne les statistiques."""
    wins, draws, losses = 0, 0, 0
    total_steps = 0

    # Warm-up (éviter les effets de cache)
    for _ in range(100):
        s = env.reset()
        for _ in range(max_steps):
            va = env.get_valid_actions(s)
            if len(va) == 0:
                break
            a = agent.select_action(s, va)
            s, r, done, _ = env.step(a)
            if done:
                break

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(num_games):
        s = env.reset()
        ep_reward = 0.0
        steps = 0
        for _ in range(max_steps):
            va = env.get_valid_actions(s)
            if len(va) == 0:
                break
            a = agent.select_action(s, va)
            s, r, done, _ = env.step(a)
            ep_reward += r
            steps += 1
            if done:
                break
        total_steps += steps
        if ep_reward > 0:
            wins += 1
        elif ep_reward < 0:
            losses += 1
        else:
            draws += 1
    elapsed = time.perf_counter() - t0

    games_per_sec = num_games / elapsed
    avg_steps = total_steps / num_games

    return {
        'games_per_sec': games_per_sec,
        'total_time': elapsed,
        'avg_steps': avg_steps,
        'wins': wins, 'draws': draws, 'losses': losses,
    }


def main():
    envs = {
        'LineWorld':  (LineWorld(length=10), 2),
        'GridWorld':  (GridWorld(width=5, height=5), 4),
        'TicTacToe':  (TicTacToe(opponent_type='random'), 9),
        'Quarto':     (Quarto(opponent_type='random'), 16),
    }

    num_games = 10000

    print("=" * 65)
    print("  BENCHMARK : Joueur Random — Nombre de parties / seconde")
    print("=" * 65)
    print()

    for env_name, (env, action_size) in envs.items():
        agent = RandomAgent(action_size)
        res = benchmark(env, agent, num_games=num_games)

        print(f"  {env_name}")
        print(f"    Parties jouées     : {num_games}")
        print(f"    Temps total        : {res['total_time']:.2f} s")
        print(f"    Parties / seconde  : {res['games_per_sec']:,.0f}")
        print(f"    Steps moyens       : {res['avg_steps']:.1f}")
        print(f"    W / D / L          : {res['wins']} / {res['draws']} / {res['losses']}")
        print()

    print("=" * 65)


if __name__ == '__main__':
    main()
