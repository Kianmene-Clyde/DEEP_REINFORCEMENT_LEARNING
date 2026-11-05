"""Local Flask demonstrations backed by validated, versioned run checkpoints."""

import json
import os
import secrets
import time
from pathlib import Path
from collections import OrderedDict
import numpy as np
from flask import Flask, render_template_string, jsonify, request, session
from werkzeug.local import LocalProxy
from src.registry import (
    AGENT_ORDER,
    PURE_PLANNING,
    ALGORITHM_NOTES,
    available_agents_for_env,
    make_environment,
    build_agent,
)
from src.training.evaluator import encode_state
from src.training.checkpoints import load_checkpoint

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config["MODEL_RUN_DIRECTORY"] = os.environ.get("DRL_RUN_DIR")
GUI_AGENT_OPTIONS = ["Human"] + AGENT_ORDER
GUI_AGENTS_BY_ENV = {
    name: ["Human"] + available_agents_for_env(name)
    for name in ["LineWorld", "GridWorld", "TicTacToe", "Quarto"]
}
_games = OrderedDict()


def _current_game():
    identifier = session.get("game_id")
    if identifier is None:
        identifier = secrets.token_hex(16)
        session["game_id"] = identifier
    if identifier not in _games:
        _games[identifier] = {"env": None, "done": True}
    _games.move_to_end(identifier)
    while len(_games) > 64:
        _games.popitem(last=False)
    return _games[identifier]


game_state = LocalProxy(_current_game)
encode_for_agent = encode_state


def create_env(name):
    return make_environment(name, seed=42)


def create_agent(name, env, env_name=None):
    env_name = env_name or game_state["env_name"]
    if name not in GUI_AGENTS_BY_ENV[env_name]:
        raise ValueError("Agent unavailable for this environment")
    if name == "Human":
        return None
    if name in PURE_PLANNING:
        agent = build_agent(
            name,
            env_name,
            env.observation_space,
            env.action_space,
            {"num_rollouts": 8, "num_simulations": 30, "max_rollout_depth": 30},
            seed=42,
        )
        agent._loaded_model_path = None
        agent._model_status = (
            "Baseline aléatoire" if name == "Random" else "Planification avec simulateur, sans entraînement"
        )
    else:
        configured = app.config.get("MODEL_RUN_DIRECTORY")
        manifests = (
            [Path(configured) / "manifest.json"]
            if configured
            else sorted(
                (ROOT / "results" / "runs").glob("*/manifest.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        )
        agent = None
        errors = []
        for path in manifests:
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest.get("protocol_version") != 2 or manifest.get("status") != "complete":
                    continue
                matching = [
                    m for m in manifest["models"] if m["environment"] == env_name and m["agent"] == name
                ]
                for model in matching:
                    checkpoint = (path.parent / model["checkpoint"]).resolve()
                    if not checkpoint.is_relative_to(path.parent.resolve()):
                        raise ValueError("Checkpoint outside run directory")
                    candidate, metadata = load_checkpoint(
                        checkpoint,
                        environment=env_name,
                        observation_size=env.observation_space,
                        action_size=env.action_space,
                    )
                    if metadata["agent"] != name:
                        raise ValueError("Checkpoint agent mismatch")
                    agent = candidate
                    agent._loaded_model_path = str(checkpoint)
                    agent._model_status = (
                        f"{agent.name} · seed {metadata['seed']} · "
                        f"{metadata['episodes_trained']} épisodes · "
                        "vérification courte"
                        if manifest.get("interpretation", "").startswith("smoke")
                        and metadata["episodes_trained"] < 100
                        else f"{agent.name} · seed {metadata['seed']} · {metadata['episodes_trained']} épisodes"
                    )
                    break
                if agent is not None:
                    break
            except (ValueError, OSError, KeyError, TypeError) as exc:
                errors.append(str(exc))
        if agent is None:
            raise FileNotFoundError(
                f"Aucun modèle versionné compatible pour {env_name}/{name}. "
                f'Exécuter : python train_all.py --env {env_name} --agent "{name}"'
            )
    agent.set_training_mode(False)
    if hasattr(agent, "set_env"):
        agent.set_env(env)
    return agent


def render_state_html(env, env_name, state):
    """Render environment state as HTML."""
    if env_name == "LineWorld":
        return render_line_world(env, state)
    elif env_name == "GridWorld":
        return render_grid_world(env, state)
    elif env_name == "TicTacToe":
        return render_tictactoe(env, state)
    elif env_name == "Quarto":
        return render_quarto(env, state)
    return "<p>Environnement Inconnu</p>"


def render_line_world(env, state):
    pos = int(np.argmax(state[: env.length]))
    cells = ""
    for i in range(env.length):
        cls = "agent" if i == pos else ("goal" if i == env.length - 1 else "empty")
        label = "A" if i == pos else ("G" if i == env.length - 1 else "")
        cells += f'<div class="cell {cls}">{label}</div>'
    return f'<div class="line-world">{cells}</div>'


def render_grid_world(env, state):
    rows = ""
    pit_pos = getattr(env, "pit_pos", getattr(env, "trap_pos", None))
    for y in range(env.height):
        cells = ""
        for x in range(env.width):
            is_agent = np.array_equal(env.agent_pos, [x, y])
            is_goal = np.array_equal(env.goal_pos, [x, y])
            is_pit = pit_pos is not None and np.array_equal(pit_pos, [x, y])
            cls = "agent" if is_agent else ("goal" if is_goal else ("pit" if is_pit else "empty"))
            label = "A" if is_agent else ("G" if is_goal else ("X" if is_pit else ""))
            cells += f'<div class="cell {cls}">{label}</div>'
        rows += f'<div class="grid-row">{cells}</div>'
    return f'<div class="grid-world">{rows}</div>'


def render_tictactoe(env, state):
    sym = {0: "", 1: "X", -1: "O"}
    board = env.board.reshape(3, 3)
    rows = ""
    for i in range(3):
        cells = ""
        for j in range(3):
            val = int(board[i, j])
            action_idx = i * 3 + j
            cls = "x-cell" if val == 1 else ("o-cell" if val == -1 else "empty-cell")
            click = (
                f'onclick="humanAction({action_idx})"' if val == 0 and game_state["mode"] == "human" else ""
            )
            cells += f'<div class="ttt-cell {cls}" {click}>{sym[val]}</div>'
        rows += f'<div class="ttt-row">{cells}</div>'
    return f'<div class="tictactoe">{rows}</div>'


def _quarto_piece_html(piece_id: int, env, compact: bool = False) -> str:
    attrs = [(piece_id >> b) & 1 for b in range(4)]
    size_cls = "large" if attrs[0] else "small"
    color_cls = "dark" if attrs[1] else "light"
    shape_cls = "round" if attrs[2] else "square"
    fill_cls = "hollow" if attrs[3] else "solid"
    piece_cls = "mini" if compact else "board"
    symbol = env.piece_symbol(piece_id) if hasattr(env, "piece_symbol") else str(piece_id)
    desc = env.piece_description(piece_id) if hasattr(env, "piece_description") else f"Piece {piece_id}"
    tone_label = "" if compact else ("Claire" if attrs[1] == 0 else "Sombre")
    label = f'<div class="q-piece-id">#{piece_id}</div>' if not compact else ""
    tone_badge = f'<div class="q-piece-tone {color_cls}">{tone_label}</div>' if tone_label else ""
    return (
        f'<div class="q-piece-wrap {piece_cls}" title="Piece #{piece_id} — {desc}">'
        f"{label}"
        f"{tone_badge}"
        f'<div class="q-piece {piece_cls} {shape_cls} {fill_cls} {size_cls} {color_cls}"></div>'
        f'<div class="q-piece-symbol">{symbol}</div>'
        f"</div>"
    )


def render_quarto(env, state):
    board = env.board.reshape(4, 4)
    rows = ""
    for i in range(4):
        cells = ""
        for j in range(4):
            val = int(board[i, j])
            action_idx = i * 4 + j
            if val == -1:
                content = ""
                cls = "empty-cell"
                click = f'onclick="humanAction({action_idx})"' if game_state["mode"] == "human" else ""
            else:
                content = _quarto_piece_html(val, env, compact=False)
                cls = "filled-cell"
                click = ""
            cells += f'<div class="q-cell {cls}" {click}>{content}</div>'
        rows += f'<div class="q-row">{cells}</div>'

    piece_info = ""
    if env.current_piece >= 0:
        desc = env.piece_description(env.current_piece) if hasattr(env, "piece_description") else ""
        piece_info = (
            '<div class="q-current-piece">'
            '<span class="q-current-label">Pièce à poser:</span>'
            f"{_quarto_piece_html(int(env.current_piece), env, compact=False)}"
            f'<span class="q-current-desc">Piece #{int(env.current_piece)} — {desc}</span>'
            "</div>"
            '<p class="q-current-hint">Sélectionner une cellule vide pour positionner la pièce.</p>'
        )

    avail = [int(x) for x in np.where(env.available)[0]]
    avail_html = "".join(_quarto_piece_html(p, env, compact=True) for p in avail)
    available_section = (
        (
            '<div class="q-available-block">'
            '<p class="q-available-title">Pièces disponible:</p>'
            f'<div class="q-available-list">{avail_html}</div>'
            '<p class="q-available-hint">Clique sur une case vide pour placer la pièce</p>'
            "</div>"
        )
        if avail
        else ""
    )

    return f'<div class="quarto">{rows}</div>{piece_info}{available_section}'


# Routes
RULES = {
    "LineWorld": "Atteindre la case 9. +1 au but, -0,1 par déplacement intermédiaire.",
    "GridWorld": "Atteindre (4,4), éviter le piège (0,0). Départ en (1,0).",
    "TicTacToe": "Vous jouez X contre un adversaire aléatoire. Aligner trois symboles.",
    "Quarto": "Placer la pièce imposée. Aligner quatre pièces partageant un attribut. "
    "Le choix de la pièce donnée à l’adversaire est automatique et heuristique.",
}


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, agents_by_env=GUI_AGENTS_BY_ENV)


@app.route("/api/new_game", methods=["POST"])
def new_game():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify(error="Un objet JSON est requis"), 400
    env_name, agent_name = data.get("env", "LineWorld"), data.get("agent", "Random")
    if (
        not isinstance(env_name, str)
        or not isinstance(agent_name, str)
        or env_name not in GUI_AGENTS_BY_ENV
        or agent_name not in GUI_AGENTS_BY_ENV[env_name]
    ):
        return jsonify(error="Environnement ou agent inconnu"), 400
    try:
        env = create_env(env_name)
        agent = create_agent(agent_name, env, env_name)
        state = env.reset(seed=42)
    except (ValueError, FileNotFoundError, OSError) as exc:
        return jsonify(error=str(exc)), 409
    game_state.clear()
    game_state.update(
        env=env,
        env_name=env_name,
        agent=agent,
        agent_name=agent_name,
        state=state,
        done=False,
        reward=0.0,
        total_reward=0.0,
        step=0,
        history=[],
        mode="human" if agent is None else "watch",
    )
    status = "Joueur humain" if agent is None else agent._model_status
    return jsonify(
        board=render_state_html(env, env_name, state),
        info=f"{env_name} · {status}",
        model_status=status,
        algorithm_notes=ALGORITHM_NOTES.get(agent_name, ""),
        rules=RULES[env_name],
        done=False,
        reward=0,
        total_reward=0,
        step=0,
        mode=game_state["mode"],
        valid_actions=env.get_valid_actions(state).tolist(),
    )


def _take_action(human_action=None):
    if game_state.get("env") is None or game_state["done"]:
        return {"error": "Aucune partie active"}, 409
    env = game_state["env"]
    valid = env.get_valid_actions(game_state["state"])
    t0 = time.perf_counter()
    if human_action is None:
        if game_state["agent"] is None:
            return {"error": "Cette partie attend une action humaine"}, 409
        encoded = encode_state(game_state["state"], env, game_state["agent"])
        action = game_state["agent"].select_action(encoded, valid)
    else:
        if game_state["agent"] is not None:
            return {"error": "Cette partie utilise un agent automatique"}, 409
        action = human_action
    elapsed = (time.perf_counter() - t0) * 1000
    if not isinstance(action, (int, np.integer)) or action not in valid:
        return {"error": "Action invalide", "valid_actions": valid.tolist()}, 400
    state, reward, terminated, info = env.step(int(action))
    game_state["step"] += 1
    truncated = not terminated and game_state["step"] >= 200
    done = bool(terminated or truncated)
    outcome = info.get("outcome", "unknown") if terminated else ("timeout" if truncated else None)
    game_state.update(
        state=state, done=done, reward=float(reward), total_reward=game_state["total_reward"] + float(reward)
    )
    game_state["history"].append({"action": int(action), "reward": float(reward), "outcome": outcome})
    return {
        "board": render_state_html(env, game_state["env_name"], state),
        "action": int(action),
        "reward": float(reward),
        "total_reward": float(game_state["total_reward"]),
        "done": done,
        "terminated": bool(terminated),
        "truncated": truncated,
        "outcome": outcome,
        "step": game_state["step"],
        "action_time_ms": round(elapsed, 3),
        "info": json.dumps(info, ensure_ascii=False),
        "mode": game_state["mode"],
        "valid_actions": env.get_valid_actions(state).tolist() if not done else [],
    }, 200


@app.route("/api/step", methods=["POST"])
def step():
    result, status = _take_action()
    return jsonify(result), status


@app.route("/api/human_action", methods=["POST"])
def human_action():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify(error="Un objet JSON est requis"), 400
    action = data.get("action")
    if isinstance(action, bool) or not isinstance(action, int):
        return jsonify(error="Une action entière est requise"), 400
    result, status = _take_action(action)
    return jsonify(result), status


@app.route("/api/auto_play", methods=["POST"])
def auto_play():
    if game_state.get("env") is None or game_state.get("mode") == "human":
        return jsonify(error="Une partie automatique est requise"), 409
    records = []
    while not game_state["done"]:
        result, status = _take_action()
        if status != 200:
            return jsonify(result), status
        records.append(result)
    return jsonify(steps=records)


# HTML Template

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DRL Project - Game GUI</title>
    <style>
        button:focus-visible, select:focus-visible { outline: 3px solid #fff; outline-offset: 3px; }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #1a1a2e; color: #eee;
            min-height: 100vh;
        }
        .container { max-width: 900px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; color: #e94560; margin-bottom: 20px; font-size: 1.8em; }
        
        /* Controls */
        .controls {
            display: flex; gap: 10px; flex-wrap: wrap;
            justify-content: center; margin-bottom: 20px;
            background: #16213e; padding: 15px; border-radius: 10px;
        }
        select, button {
            padding: 8px 16px; border: none; border-radius: 6px;
            font-size: 14px; cursor: pointer;
        }
        select { background: #0f3460; color: #eee; }
        button { background: #e94560; color: white; font-weight: bold; }
        button:hover { background: #c73851; }
        button:disabled { background: #555; cursor: not-allowed; }
        .btn-green { background: #2ecc71; }
        .btn-green:hover { background: #27ae60; }
        .btn-blue { background: #3498db; }
        .btn-blue:hover { background: #2980b9; }
        
        /* Game area */
        .game-area {
            background: #16213e; border-radius: 10px;
            padding: 20px; text-align: center; min-height: 200px;
        }
        .info-bar {
            display: flex; justify-content: space-between;
            background: #0f3460; padding: 10px 15px; border-radius: 8px;
            margin-bottom: 15px; font-size: 14px;
        }
        .info-bar span { color: #a0a0ff; }
        .info-bar strong { color: #e94560; }
        #board { margin: 20px auto; }
        #message { margin-top: 10px; color: #2ecc71; font-size: 16px; min-height: 24px; }
        
        /* LineWorld */
        .line-world { display: flex; gap: 3px; justify-content: center; }
        .line-world .cell {
            width: 50px; height: 50px; border: 2px solid #0f3460;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 18px; border-radius: 6px;
        }
        .cell.agent { background: #e94560; color: white; }
        .cell.goal { background: #2ecc71; color: white; }
        .cell.pit { background: #f39c12; color: white; }
        .cell.empty { background: #16213e; }
        
        /* GridWorld */
        .grid-world { display: inline-block; }
        .grid-row { display: flex; gap: 3px; margin-bottom: 3px; }
        .grid-world .cell {
            width: 60px; height: 60px; border: 2px solid #0f3460;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 18px; border-radius: 6px;
        }
        
        /* TicTacToe */
        .tictactoe { display: inline-block; }
        .ttt-row { display: flex; }
        .ttt-cell {
            width: 80px; height: 80px; border: 2px solid #0f3460;
            display: flex; align-items: center; justify-content: center;
            font-size: 32px; font-weight: bold; cursor: pointer;
            border-radius: 4px; margin: 2px;
        }
        .ttt-cell.x-cell { color: #e94560; background: #1a1a2e; }
        .ttt-cell.o-cell { color: #3498db; background: #1a1a2e; }
        .ttt-cell.empty-cell { background: #16213e; }
        .ttt-cell.empty-cell:hover { background: #1f3a6e; }
        
        /* Quarto */
        .quarto { display: inline-block; }
        .q-row { display: flex; }
        .q-cell {
            width: 105px; height: 105px; border: 2px solid #1d79e8;
            display: flex; align-items: center; justify-content: center;
            font-size: 14px; font-weight: bold; cursor: pointer;
            border-radius: 8px; margin: 4px; background: #11214a;
            transition: background 0.15s ease, transform 0.08s ease;
        }
        .q-cell.filled-cell { background: rgba(150, 200, 255, 0.12); }
        .q-cell.empty-cell { background: #0f1d47; }
        .q-cell.empty-cell:hover { background: #17306d; transform: translateY(-1px); }
        .q-piece-wrap {
            display: flex; align-items: center; justify-content: center;
            flex-direction: column; gap: 6px; width: 100%; height: 100%;
            position: relative;
        }
        .q-piece-wrap.board { padding: 8px; }
        .q-piece-wrap.mini {
            width: 62px; height: 62px; min-width: 62px;
            border: 2px solid #1d79e8; border-radius: 12px; background: #11214a;
            padding: 6px;
        }
        .q-piece-id {
            position: absolute; top: 4px; right: 6px; font-size: 11px; color: #8cbcff;
        }
        .q-piece-tone {
            position: absolute; top: 4px; left: 6px; font-size: 10px; font-weight: 700;
            padding: 2px 6px; border-radius: 999px; letter-spacing: 0.2px;
        }
        .q-piece-tone.light {
            color: #fff7ea; background: rgba(255, 236, 200, 0.18); border: 1px solid rgba(255, 239, 214, 0.35);
        }
        .q-piece-tone.dark {
            color: #d7e5ff; background: rgba(88, 116, 180, 0.22); border: 1px solid rgba(151, 179, 255, 0.35);
        }
        .q-piece-symbol {
            position: absolute; opacity: 0; pointer-events: none; font-size: 1px;
        }
        .q-piece {
            box-sizing: border-box; display: inline-block;
            border-width: 4px; border-style: solid;
        }
        .q-piece.square { border-radius: 0; }
        .q-piece.round { border-radius: 50%; }
        .q-piece.solid.light {
            background: #f5efe3; border-color: #fff8ec;
            box-shadow: 0 0 10px rgba(255, 245, 220, 0.18);
        }
        .q-piece.solid.dark {
            background: #2c3449; border-color: #8fb1ff;
            box-shadow: 0 0 0 2px rgba(12, 18, 34, 0.45) inset, 0 0 8px rgba(90, 130, 255, 0.12);
        }
        .q-piece.hollow.light {
            background: transparent; border-color: #fff1d6;
            box-shadow: 0 0 0 1px rgba(255, 240, 210, 0.12);
        }
        .q-piece.hollow.dark {
            background: transparent; border-color: #8fb1ff;
            box-shadow: 0 0 0 2px rgba(26, 38, 72, 0.45) inset;
        }
        .q-piece.board.small.square { width: 24px; height: 24px; }
        .q-piece.board.small.round { width: 22px; height: 22px; }
        .q-piece.board.large.square { width: 42px; height: 42px; }
        .q-piece.board.large.round { width: 40px; height: 40px; }
        .q-piece.mini.small.square, .q-piece.small.square { width: 18px; height: 18px; }
        .q-piece.mini.small.round, .q-piece.small.round { width: 16px; height: 16px; }
        .q-piece.mini.large.square, .q-piece.large.square { width: 30px; height: 30px; }
        .q-piece.mini.large.round, .q-piece.large.round { width: 28px; height: 28px; }
        .q-current-piece {
            display: flex; align-items: center; justify-content: center; gap: 12px;
            margin-top: 18px; flex-wrap: wrap;
        }
        .q-current-label { font-size: 16px; color: #f5d6bd; font-weight: 700; }
        .q-current-desc { color: #9bbcff; font-size: 14px; }
        .q-current-hint, .q-available-hint { color: #b9a69b; margin-top: 10px; }
        .q-available-block { margin-top: 14px; }
        .q-available-title { color: #b9a69b; margin-bottom: 10px; }
        .q-available-list {
            display: flex; gap: 8px; justify-content: center; flex-wrap: wrap;
            margin-bottom: 10px;
        }
        .small { font-size: 12px; color: #888; margin-top: 5px; }
        
        /* Human keyboard controls visual */
        #humanControls { display: none; margin-top: 15px; }
        .kb-visual { display: flex; flex-direction: column; align-items: center; gap: 4px; }
        .kb-row { display: flex; gap: 4px; }
        .kb-key {
            width: 52px; height: 44px; border: 2px solid #0f3460;
            border-radius: 6px; background: #16213e; color: #eee;
            display: flex; align-items: center; justify-content: center;
            font-size: 20px; user-select: none; transition: all 0.1s;
        }
        .kb-key.active { background: #e94560; border-color: #e94560; transform: scale(0.93); }
        .kb-key.wide { width: 112px; font-size: 14px; }
        .controls-hint { font-size: 12px; color: #888; margin-top: 8px; }

        /* Results log */
        .log { 
            max-height: 200px; overflow-y: auto; margin-top: 15px;
            background: #0f3460; padding: 10px; border-radius: 8px;
            font-family: monospace; font-size: 12px; text-align: left;
        }
    </style>
</head>
<body>
    <div class="container"><p id="modelStatus" role="status">Choisir un environnement et un agent.</p><p id="rules"></p></div>
    <div class="container">
        <h1>🎮 Deep Reinforcement Learning - Game GUI</h1>
        
        <div class="controls">
            <select id="envSelect">
                <option value="LineWorld">LineWorld</option>
                <option value="GridWorld">GridWorld</option>
                <option value="TicTacToe" selected>TicTacToe</option>
                <option value="Quarto">Quarto</option>
            </select>
            <select id="agentSelect"></select>
            <button onclick="newGame()">🔄 New Game</button>
            <button onclick="stepGame()" id="stepBtn" class="btn-blue">⏩ Step</button>
            <button onclick="autoPlay()" id="autoBtn" class="btn-green">▶ Auto Play</button>
        </div>
        
        <div class="game-area">
            <div class="info-bar">
                <span>Step: <strong id="stepCount">0</strong></span>
                <span>Reward: <strong id="lastReward">0</strong></span>
                <span>Total: <strong id="totalReward">0</strong></span>
                <span>Mode: <strong id="gameMode">-</strong></span>
            </div>
            <div id="board"><p style="color:#888;margin-top:60px;">Select environment and agent, then click New Game</p></div>
            <div id="humanControls">
                <div id="kbVisual"></div>
                <p class="controls-hint" id="controlsHint"></p>
            </div>
            <div id="message"></div>
        </div>
        
        <div class="log" id="log"></div>
    </div>
    
    <script>
        let autoPlaying = false;
        let autoTimer = null;
        let currentEnv = '';
        let currentMode = '';
        let gameDone = false;
        
        // Action definitions per environment
        const ENV_ACTIONS = {
            'LineWorld':  [{id: 0, label: '← Gauche', key: 'ArrowLeft'},
                           {id: 1, label: 'Droite →', key: 'ArrowRight'}],
            'GridWorld':  [{id: 0, label: '↑ Haut',   key: 'ArrowUp'},
                           {id: 1, label: '↓ Bas',    key: 'ArrowDown'},
                           {id: 2, label: '← Gauche', key: 'ArrowLeft'},
                           {id: 3, label: 'Droite →', key: 'ArrowRight'}],
            'TicTacToe':  [], // handled by clicking on cells
            'Quarto':     [], // handled by clicking on cells
        };
        
        const KEY_LABELS = {
            'ArrowLeft': '←', 'ArrowRight': '→',
            'ArrowUp': '↑', 'ArrowDown': '↓',
        };

        const AGENTS_BY_ENV = {{ agents_by_env | tojson }};

        function refreshAgentOptions() {
            const env = document.getElementById('envSelect').value;
            const select = document.getElementById('agentSelect');
            const previous = select.value;
            const options = AGENTS_BY_ENV[env] || [];
            select.innerHTML = '';

            for (const agent of options) {
                const option = document.createElement('option');
                option.value = agent;
                option.textContent = agent === 'Human' ? '👤 Human' :
                    ({AlphaZero: 'AlphaZero inspiré', MuZero: 'MuZero prototype', MuZero_Stochastic: 'Planification gaussienne expérimentale'})[agent] || agent;
                if (agent === 'MCTS' && previous === '') {
                    option.selected = true;
                }
                select.appendChild(option);
            }

            if (options.includes(previous)) {
                select.value = previous;
            } else if (options.includes('MCTS')) {
                select.value = 'MCTS';
            } else if (options.length > 0) {
                select.value = options[0];
            }
        }
        
        // Keyboard handler
        document.addEventListener('keydown', function(e) {
            if (currentMode !== 'human' || gameDone) return;
            const actions = ENV_ACTIONS[currentEnv] || [];
            for (const act of actions) {
                if (e.key === act.key) {
                    e.preventDefault();
                    flashKey(e.key);
                    humanAction(act.id);
                    return;
                }
            }
            // TicTacToe: numpad 1-9 (mapped to board positions)
            // Layout: 7 8 9 / 4 5 6 / 1 2 3 → board positions 0-8
            if (currentEnv === 'TicTacToe') {
                const numpadMap = {'7':0,'8':1,'9':2, '4':3,'5':4,'6':5, '1':6,'2':7,'3':8};
                if (numpadMap[e.key] !== undefined) {
                    e.preventDefault();
                    flashKey(e.key);
                    humanAction(numpadMap[e.key]);
                }
            }
            // Quarto: 0-9 and a-f for positions 0-15
            if (currentEnv === 'Quarto') {
                const hexMap = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,
                                '8':8,'9':9,'a':10,'b':11,'c':12,'d':13,'e':14,'f':15};
                if (hexMap[e.key.toLowerCase()] !== undefined) {
                    e.preventDefault();
                    humanAction(hexMap[e.key.toLowerCase()]);
                }
            }
        });
        
        function buildActionButtons() {
            const kbDiv = document.getElementById('kbVisual');
            const hint = document.getElementById('controlsHint');
            const panel = document.getElementById('humanControls');
            kbDiv.innerHTML = '';
            
            if (currentMode !== 'human') {
                panel.style.display = 'none';
                return;
            }
            panel.style.display = 'block';
            
            if (currentEnv === 'LineWorld') {
                kbDiv.innerHTML = '<div class="kb-visual"><div class="kb-row">' +
                    '<div class="kb-key" id="kb-ArrowLeft">←</div>' +
                    '<div class="kb-key" id="kb-ArrowRight">→</div>' +
                    '</div></div>';
                hint.textContent = 'Touches ← → pour se déplacer';
            } else if (currentEnv === 'GridWorld') {
                kbDiv.innerHTML = '<div class="kb-visual">' +
                    '<div class="kb-row"><div class="kb-key" id="kb-ArrowUp">↑</div></div>' +
                    '<div class="kb-row">' +
                    '<div class="kb-key" id="kb-ArrowLeft">←</div>' +
                    '<div class="kb-key" id="kb-ArrowDown">↓</div>' +
                    '<div class="kb-key" id="kb-ArrowRight">→</div>' +
                    '</div></div>';
                hint.textContent = 'Touches ↑ ↓ ← → pour se déplacer';
            } else if (currentEnv === 'TicTacToe') {
                kbDiv.innerHTML = '<div class="kb-visual">' +
                    '<div class="kb-row"><div class="kb-key" id="kb-7">7</div><div class="kb-key" id="kb-8">8</div><div class="kb-key" id="kb-9">9</div></div>' +
                    '<div class="kb-row"><div class="kb-key" id="kb-4">4</div><div class="kb-key" id="kb-5">5</div><div class="kb-key" id="kb-6">6</div></div>' +
                    '<div class="kb-row"><div class="kb-key" id="kb-1">1</div><div class="kb-key" id="kb-2">2</div><div class="kb-key" id="kb-3">3</div></div>' +
                    '</div>';
                hint.textContent = 'Pavé numérique (7=haut-gauche ... 3=bas-droite) ou clic sur une case';
            } else if (currentEnv === 'Quarto') {
                kbDiv.innerHTML = '';
                hint.textContent = 'Clique sur une case vide pour placer la pièce';
            }
        }
        
        // Flash the key visual on press
        function flashKey(keyName) {
            const el = document.getElementById('kb-' + keyName);
            if (!el) return;
            el.classList.add('active');
            setTimeout(() => el.classList.remove('active'), 150);
        }
        
        async function newGame() {
            stopAuto();
            const env = document.getElementById('envSelect').value;
            const agent = document.getElementById('agentSelect').value;
            const res = await fetch('/api/new_game', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({env, agent})
            });
            const data = await res.json();
            if (data.error) {
                document.getElementById('message').textContent = data.error;
                document.getElementById('message').style.color = '#e94560';
                log('ERROR: ' + data.error);
                return;
            }
            currentEnv = env;
            gameDone = false;
            currentMode = data.mode || '';
            updateUI(data);
            buildActionButtons();
            log(data.info || ('New game: ' + env + ' | ' + agent));
        }
        
        async function stepGame() {
            const res = await fetch('/api/step', {method: 'POST'});
            const data = await res.json();
            updateUI(data);
            if (data.error) return;
            if (data.action !== undefined) {
                log('Step ' + data.step + ': action=' + data.action + 
                    ' reward=' + data.reward.toFixed(3) +
                    (data.action_time_ms ? ' (' + data.action_time_ms + 'ms)' : ''));
            }
            if (data.done) {
                log('GAME OVER! Total reward: ' + data.total_reward.toFixed(3));
                stopAuto();
            }
        }
        
        function autoPlay() {
            if (autoPlaying) {
                stopAuto();
                return;
            }
            autoPlaying = true;
            document.getElementById('autoBtn').textContent = '⏸ Pause';
            autoStep();
        }
        
        function autoStep() {
            if (!autoPlaying) return;
            stepGame().then(() => {
                if (autoPlaying) autoTimer = setTimeout(autoStep, 500);
            });
        }
        
        function stopAuto() {
            autoPlaying = false;
            if (autoTimer) clearTimeout(autoTimer);
            document.getElementById('autoBtn').textContent = '▶ Auto Play';
        }
        
        function humanAction(action) {
            fetch('/api/human_action', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action})
            }).then(r => r.json()).then(data => {
                updateUI(data);
                if (data.error) return;
                log('Human action=' + action + ' reward=' + data.reward.toFixed(3));
                if (data.done) {
                    let result = data.outcome || 'Partie terminée';
                    log('GAME OVER! ' + result + ' (total: ' + data.total_reward.toFixed(3) + ')');
                }
            });
        }
        
        function updateUI(data) {
            if (data.error) { document.getElementById('message').textContent = data.error; stopAuto(); return; }
            if (data.model_status) document.getElementById('modelStatus').textContent = data.model_status + (data.algorithm_notes ? ' · ' + data.algorithm_notes : '');
            if (data.rules) document.getElementById('rules').textContent = data.rules;
            if (data.board) document.getElementById('board').innerHTML = data.board;
            if (data.step !== undefined) document.getElementById('stepCount').textContent = data.step;
            if (data.reward !== undefined) document.getElementById('lastReward').textContent = data.reward.toFixed(3);
            if (data.total_reward !== undefined) document.getElementById('totalReward').textContent = data.total_reward.toFixed(3);
            if (data.mode) {
                document.getElementById('gameMode').textContent = data.mode;
                currentMode = data.mode;
            }
            if (data.done) {
                gameDone = true;
                document.getElementById('message').textContent = 
                    ({success:'Objectif atteint',win:'Victoire',failure:'Échec',loss:'Défaite',draw:'Match nul',timeout:'Limite de 200 actions atteinte'})[data.outcome] || 'Partie terminée';
                document.getElementById('message').style.color = 
                    ['success','win'].includes(data.outcome) ? '#2ecc71' : '#f39c12';
            } else {
                document.getElementById('message').textContent = '';
            }
            // Show/hide step button based on mode
            const isHuman = data.mode === 'human';
            document.getElementById('stepBtn').style.display = isHuman ? 'none' : '';
            document.getElementById('autoBtn').style.display = isHuman ? 'none' : '';
        }
        
        function log(msg) {
            const el = document.getElementById('log');
            el.appendChild(document.createTextNode(msg));
            el.appendChild(document.createElement('br'));
            el.scrollTop = el.scrollHeight;
        }

        document.getElementById('envSelect').addEventListener('change', refreshAgentOptions);
        refreshAgentOptions();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Démarrage de la GUI : http://localhost:5000")
    print("Ctrl+C pour arrêter")
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=False)
