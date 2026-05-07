import sys, os, json, copy, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents import RandomAgent, DeepQLearningAgent, TabularQLearningAgent, \
    DDQNWithERAgent, DoubleDeepQLearningAgent, REINFORCEAgent, DDQNWithPERAgent, REINFORCEMeanBaselineAgent, \
    REINFORCECriticBaselineAgent, PPOAgent, MCTSAgent, RandomRolloutAgent, A2CAgent, \
    ExpertApprenticeAgent, AlphaZeroAgent, MuZeroAgent, StochasticMuZeroAgent
from environments import LineWorld, GridWorld, TicTacToe, Quarto

from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)

GUI_AGENT_OPTIONS = [
    'Human',
    'Random',
    'TabularQL',
    'DQN',
    'DoubleDQN',
    'DDQN+ER',
    'DDQN+PER',
    'REINFORCE',
    'REINFORCE+Mean',
    'REINFORCE+Critic',
    'PPO',
    'A2C',
    'RandomRollout',
    'MCTS',
    'ExpertApprentice',
    'AlphaZero',
    'MuZero',
    'MuZero_Stochastic',
]

GUI_AGENTS_BY_ENV = {
    'LineWorld': list(GUI_AGENT_OPTIONS),
    'GridWorld': list(GUI_AGENT_OPTIONS),
    'TicTacToe': [x for x in GUI_AGENT_OPTIONS if x != 'TabularQL'],
    'Quarto': [x for x in GUI_AGENT_OPTIONS if x != 'TabularQL'],
}

# Global game state
game_state = {
    'env': None,
    'env_name': None,
    'agent': None,
    'agent_name': None,
    'state': None,
    'done': False,
    'reward': 0.0,
    'total_reward': 0.0,
    'step': 0,
    'history': [],
    'mode': 'watch',  # 'watch' or 'human'
}


def encode_for_agent(state, env, agent):
    if agent is not None and hasattr(agent, 'uses_tabular') and agent.uses_tabular:
        if hasattr(env, 'state_to_index'):
            return env.state_to_index(state)
        # Fallback: argmax of one-hot
        return int(np.argmax(np.asarray(state)))
    return np.asarray(state, dtype=np.float32).flatten()


def create_env(name):
    envs = {
        'LineWorld': lambda: LineWorld(length=10),
        'GridWorld': lambda: GridWorld(width=5, height=5),
        'TicTacToe': lambda: TicTacToe(opponent_type='random'),
        'Quarto': lambda: Quarto(opponent_type='random'),
    }
    return envs[name]()


def _sanitize_name(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in name)


def _tabular_state_size(env_name, obs_size):
    tabular_state_sizes = {
        'LineWorld': 10,
        'GridWorld': 25,
        'TicTacToe': 3 ** 9,
        'Quarto': obs_size,
    }
    return tabular_state_sizes.get(env_name, obs_size)


def _model_path_candidates(env_name, agent_name):
    base = f'{env_name}_{agent_name}'
    sanitized = _sanitize_name(base)
    names = [
        f'{sanitized}_best',
        f'{base}_best',
        sanitized,
        base,
    ]
    seen = set()
    candidates = []
    for name in names:
        if name not in seen:
            candidates.append(os.path.join('models', name))
            seen.add(name)
    return candidates


def _network_io(net):
    if not hasattr(net, 'layers') or len(net.layers) == 0:
        return None, None
    return int(net.layers[0]['W'].shape[0]), int(net.layers[-1]['W'].shape[1])


def _is_agent_model_compatible(agent, env_name, obs_size, act_size, tab_size):
    if hasattr(agent, 'q_table'):
        return tuple(agent.q_table.shape) == (int(tab_size), int(act_size))

    if hasattr(agent, 'q_net'):
        in_dim, out_dim = _network_io(agent.q_net)
        return in_dim == int(obs_size) and out_dim == int(act_size)

    if hasattr(agent, 'policy_net'):
        in_dim, out_dim = _network_io(agent.policy_net)
        if in_dim != int(obs_size) or out_dim != int(act_size):
            return False
        if hasattr(agent, 'value_net'):
            v_in, v_out = _network_io(agent.value_net)
            return v_in == int(obs_size) and v_out == 1
        return True

    if hasattr(agent, 'actor'):
        a_in, a_out = _network_io(agent.actor)
        if a_in != int(obs_size) or a_out != int(act_size):
            return False
        if hasattr(agent, 'critic'):
            c_in, c_out = _network_io(agent.critic)
            return c_in == int(obs_size) and c_out == 1
        return True

    return True


def create_agent(name, env):
    env_name = game_state['env_name']
    if env_name is None:
        raise ValueError('Environment name must be set before creating the agent')

    if name not in GUI_AGENTS_BY_ENV.get(env_name, []):
        raise ValueError(f'Agent {name} is not supported for {env_name}')

    obs_size = env.observation_space
    if isinstance(obs_size, tuple):
        obs_size = int(np.prod(obs_size))
    act_size = int(env.action_space)
    tab_size = _tabular_state_size(env_name, obs_size)

    constructors = {
        'Random': lambda: RandomAgent(act_size),
        'TabularQL': lambda: TabularQLearningAgent(tab_size, act_size),
        'DQN': lambda: DeepQLearningAgent(obs_size, act_size),
        'DoubleDQN': lambda: DoubleDeepQLearningAgent(obs_size, act_size),
        'DDQN+ER': lambda: DDQNWithERAgent(obs_size, act_size),
        'DDQN+PER': lambda: DDQNWithPERAgent(obs_size, act_size),
        'REINFORCE': lambda: REINFORCEAgent(obs_size, act_size),
        'REINFORCE+Mean': lambda: REINFORCEMeanBaselineAgent(obs_size, act_size),
        'REINFORCE+Critic': lambda: REINFORCECriticBaselineAgent(obs_size, act_size),
        'PPO': lambda: PPOAgent(obs_size, act_size),
        'A2C': lambda: A2CAgent(obs_size, act_size),
        'RandomRollout': lambda: RandomRolloutAgent(act_size, num_rollouts=30),
        'MCTS': lambda: MCTSAgent(act_size, num_simulations=100),
        'ExpertApprentice': lambda: ExpertApprenticeAgent(obs_size, act_size, expert_simulations=30, hidden_layers=[128]),
        'AlphaZero': lambda: AlphaZeroAgent(obs_size, act_size, num_simulations=30, hidden_layers=[128]),
        'MuZero': lambda: MuZeroAgent(obs_size, act_size, num_simulations=20, hidden_layers=[128], latent_dim=64),
        'MuZero_Stochastic': lambda: StochasticMuZeroAgent(obs_size, act_size, num_simulations=20, hidden_layers=[128], latent_dim=64),
        'Human': lambda: None,
    }

    if name == 'Human':
        return None

    def _prepare_agent(agent_obj):
        agent_obj.set_training_mode(False)
        if hasattr(agent_obj, 'set_env'):
            agent_obj.set_env(env)
        agent_obj._loaded_model_path = None
        agent_obj._model_load_error = None
        return agent_obj

    agent = _prepare_agent(constructors[name]())

    last_error = None
    for model_path in _model_path_candidates(env_name, name):
        candidate = _prepare_agent(constructors[name]())
        try:
            candidate.load(model_path)
            if _is_agent_model_compatible(candidate, env_name, obs_size, act_size, tab_size):
                candidate._loaded_model_path = model_path
                print(f'  Loaded model: {model_path}')
                return candidate
            last_error = f'Incompatible model dimensions for {model_path}'
        except Exception as exc:
            last_error = str(exc)

    if last_error:
        agent._model_load_error = last_error
        print(f'  No compatible saved model for {name}, using untrained agent')

    return agent


def render_state_html(env, env_name, state):
    """Render environment state as HTML."""
    if env_name == 'LineWorld':
        return render_line_world(env, state)
    elif env_name == 'GridWorld':
        return render_grid_world(env, state)
    elif env_name == 'TicTacToe':
        return render_tictactoe(env, state)
    elif env_name == 'Quarto':
        return render_quarto(env, state)
    return '<p>Environnement Inconnu</p>'


def render_line_world(env, state):
    pos = int(np.argmax(state[:env.length]))
    cells = ''
    for i in range(env.length):
        cls = 'agent' if i == pos else ('goal' if i == env.length - 1 else 'empty')
        label = 'A' if i == pos else ('G' if i == env.length - 1 else '')
        cells += f'<div class="cell {cls}">{label}</div>'
    return f'<div class="line-world">{cells}</div>'


def render_grid_world(env, state):
    rows = ''
    pit_pos = getattr(env, 'pit_pos', getattr(env, 'trap_pos', None))
    for y in range(env.height):
        cells = ''
        for x in range(env.width):
            is_agent = np.array_equal(env.agent_pos, [x, y])
            is_goal = np.array_equal(env.goal_pos, [x, y])
            is_pit = pit_pos is not None and np.array_equal(pit_pos, [x, y])
            cls = 'agent' if is_agent else ('goal' if is_goal else ('pit' if is_pit else 'empty'))
            label = 'A' if is_agent else ('G' if is_goal else ('X' if is_pit else ''))
            cells += f'<div class="cell {cls}">{label}</div>'
        rows += f'<div class="grid-row">{cells}</div>'
    return f'<div class="grid-world">{rows}</div>'


def render_tictactoe(env, state):
    sym = {0: '', 1: 'X', -1: 'O'}
    board = env.board.reshape(3, 3)
    rows = ''
    for i in range(3):
        cells = ''
        for j in range(3):
            val = int(board[i, j])
            action_idx = i * 3 + j
            cls = 'x-cell' if val == 1 else ('o-cell' if val == -1 else 'empty-cell')
            click = f'onclick="humanAction({action_idx})"' if val == 0 and game_state['mode'] == 'human' else ''
            cells += f'<div class="ttt-cell {cls}" {click}>{sym[val]}</div>'
        rows += f'<div class="ttt-row">{cells}</div>'
    return f'<div class="tictactoe">{rows}</div>'


def _quarto_piece_html(piece_id: int, env, compact: bool = False) -> str:
    attrs = [(piece_id >> b) & 1 for b in range(4)]
    size_cls = 'large' if attrs[0] else 'small'
    color_cls = 'dark' if attrs[1] else 'light'
    shape_cls = 'round' if attrs[2] else 'square'
    fill_cls = 'hollow' if attrs[3] else 'solid'
    piece_cls = 'mini' if compact else 'board'
    symbol = env.piece_symbol(piece_id) if hasattr(env, 'piece_symbol') else str(piece_id)
    desc = env.piece_description(piece_id) if hasattr(env, 'piece_description') else f'Piece {piece_id}'
    tone_label = '' if compact else ('Claire' if attrs[1] == 0 else 'Sombre')
    label = f'<div class="q-piece-id">#{piece_id}</div>' if not compact else ''
    tone_badge = f'<div class="q-piece-tone {color_cls}">{tone_label}</div>' if tone_label else ''
    return (
        f'<div class="q-piece-wrap {piece_cls}" title="Piece #{piece_id} — {desc}">' 
        f'{label}'
        f'{tone_badge}'
        f'<div class="q-piece {piece_cls} {shape_cls} {fill_cls} {size_cls} {color_cls}"></div>'
        f'<div class="q-piece-symbol">{symbol}</div>'
        f'</div>'
    )


def render_quarto(env, state):
    board = env.board.reshape(4, 4)
    rows = ''
    for i in range(4):
        cells = ''
        for j in range(4):
            val = int(board[i, j])
            action_idx = i * 4 + j
            if val == -1:
                content = ''
                cls = 'empty-cell'
                click = f'onclick="humanAction({action_idx})"' if game_state['mode'] == 'human' else ''
            else:
                content = _quarto_piece_html(val, env, compact=False)
                cls = 'filled-cell'
                click = ''
            cells += f'<div class="q-cell {cls}" {click}>{content}</div>'
        rows += f'<div class="q-row">{cells}</div>'

    piece_info = ''
    if env.current_piece >= 0:
        desc = env.piece_description(env.current_piece) if hasattr(env, 'piece_description') else ''
        piece_info = (
            '<div class="q-current-piece">'
            '<span class="q-current-label">Pièce à poser:</span>'
            f'{_quarto_piece_html(int(env.current_piece), env, compact=False)}'
            f'<span class="q-current-desc">Piece #{int(env.current_piece)} — {desc}</span>'
            '</div>'
            '<p class="q-current-hint">Sélectionner une cellule vide pour positionner la pièce.</p>'
        )

    avail = [int(x) for x in np.where(env.available)[0]]
    avail_html = ''.join(_quarto_piece_html(p, env, compact=True) for p in avail)
    available_section = (
        '<div class="q-available-block">'
        '<p class="q-available-title">Pièces disponible:</p>'
        f'<div class="q-available-list">{avail_html}</div>'
        '<p class="q-available-hint">Clique sur une case vide pour placer la pièce</p>'
        '</div>'
    ) if avail else ''

    return f'<div class="quarto">{rows}</div>{piece_info}{available_section}'


# Routes

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/new_game', methods=['POST'])
def new_game():
    data = request.json
    env_name = data.get('env', 'LineWorld')
    agent_name = data.get('agent', 'Random')

    try:
        env = create_env(env_name)
        game_state['env'] = env
        game_state['env_name'] = env_name
        game_state['agent_name'] = agent_name
        game_state['mode'] = 'human' if agent_name == 'Human' else 'watch'
        game_state['agent'] = create_agent(agent_name, env)
        game_state['state'] = env.reset()
        game_state['done'] = False
        game_state['reward'] = 0.0
        game_state['total_reward'] = 0.0
        game_state['step'] = 0
        game_state['history'] = []
    except Exception as exc:
        return jsonify({'error': str(exc)})

    info = f'New game: {env_name} | Agent: {agent_name}'
    if game_state['agent'] is not None and getattr(game_state['agent'], '_loaded_model_path', None):
        info += f' | Model: {game_state["agent"]._loaded_model_path}'

    return jsonify({
        'board': render_state_html(env, env_name, game_state['state']),
        'info': info,
        'done': False,
        'reward': 0,
        'total_reward': 0,
        'step': 0,
        'mode': game_state['mode'],
        'valid_actions': env.get_valid_actions(game_state['state']).tolist(),
    })


@app.route('/api/step', methods=['POST'])
def step():
    """Agent takes one step."""
    if game_state['env'] is None or game_state['done']:
        return jsonify({'error': 'No active game or game is done'})

    env = game_state['env']
    state = game_state['state']
    valid = env.get_valid_actions(state)

    if len(valid) == 0:
        game_state['done'] = True
        return jsonify({
            'board': render_state_html(env, game_state['env_name'], state),
            'done': True, 'reward': 0, 'total_reward': game_state['total_reward'],
            'step': game_state['step'], 'info': 'No valid actions',
        })

    agent = game_state['agent']
    encoded = encode_for_agent(state, env, agent)
    t0 = time.time()
    action = agent.select_action(encoded, valid)
    action_time = (time.time() - t0) * 1000

    next_state, reward, done, info = env.step(action)
    game_state['state'] = next_state
    game_state['done'] = done
    game_state['reward'] = reward
    game_state['total_reward'] += reward
    game_state['step'] += 1
    game_state['history'].append({
        'action': int(action), 'reward': float(reward), 'done': done
    })

    return jsonify({
        'board': render_state_html(env, game_state['env_name'], next_state),
        'action': int(action),
        'reward': float(reward),
        'total_reward': float(game_state['total_reward']),
        'done': done,
        'step': game_state['step'],
        'action_time_ms': round(action_time, 2),
        'info': json.dumps(info) if info else '',
        'valid_actions': env.get_valid_actions(next_state).tolist() if not done else [],
    })


@app.route('/api/human_action', methods=['POST'])
def human_action():
    """Human player takes an action."""
    if game_state['env'] is None or game_state['done']:
        return jsonify({'error': 'No active game or game is done'})

    data = request.json
    action = int(data.get('action', 0))
    env = game_state['env']
    valid = env.get_valid_actions(game_state['state'])

    if action not in valid:
        return jsonify({'error': 'Invalid action', 'valid_actions': valid.tolist()})

    next_state, reward, done, info = env.step(action)
    game_state['state'] = next_state
    game_state['done'] = done
    game_state['reward'] = reward
    game_state['total_reward'] += reward
    game_state['step'] += 1

    return jsonify({
        'board': render_state_html(env, game_state['env_name'], next_state),
        'action': action,
        'reward': float(reward),
        'total_reward': float(game_state['total_reward']),
        'done': done,
        'step': game_state['step'],
        'info': json.dumps(info) if info else '',
        'valid_actions': env.get_valid_actions(next_state).tolist() if not done else [],
    })


@app.route('/api/auto_play', methods=['POST'])
def auto_play():
    """Play a complete game automatically, return all steps."""
    if game_state['env'] is None:
        return jsonify({'error': 'No active game'})

    steps = []
    max_steps = 200
    while not game_state['done'] and game_state['step'] < max_steps:
        env = game_state['env']
        state = game_state['state']
        valid = env.get_valid_actions(state)
        if len(valid) == 0:
            break

        agent = game_state['agent']
        encoded = encode_for_agent(state, env, agent)
        action = agent.select_action(encoded, valid)
        next_state, reward, done, info = env.step(action)

        game_state['state'] = next_state
        game_state['done'] = done
        game_state['total_reward'] += reward
        game_state['step'] += 1

        steps.append({
            'board': render_state_html(env, game_state['env_name'], next_state),
            'action': int(action),
            'reward': float(reward),
            'total_reward': float(game_state['total_reward']),
            'done': done,
            'step': game_state['step'],
        })
        if done:
            break

    return jsonify({'steps': steps})


# HTML Template

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DRL Project - Game GUI</title>
    <style>
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

        const AGENTS_BY_ENV = {
            'LineWorld': ['Human', 'Random', 'TabularQL', 'DQN', 'DoubleDQN', 'DDQN+ER', 'DDQN+PER', 'REINFORCE', 'REINFORCE+Mean', 'REINFORCE+Critic', 'PPO', 'A2C', 'RandomRollout', 'MCTS', 'ExpertApprentice', 'AlphaZero', 'MuZero', 'MuZero_Stochastic'],
            'GridWorld': ['Human', 'Random', 'TabularQL', 'DQN', 'DoubleDQN', 'DDQN+ER', 'DDQN+PER', 'REINFORCE', 'REINFORCE+Mean', 'REINFORCE+Critic', 'PPO', 'A2C', 'RandomRollout', 'MCTS', 'ExpertApprentice', 'AlphaZero', 'MuZero', 'MuZero_Stochastic'],
            'TicTacToe': ['Human', 'Random', 'DQN', 'DoubleDQN', 'DDQN+ER', 'DDQN+PER', 'REINFORCE', 'REINFORCE+Mean', 'REINFORCE+Critic', 'PPO', 'A2C', 'RandomRollout', 'MCTS', 'ExpertApprentice', 'AlphaZero', 'MuZero', 'MuZero_Stochastic'],
            'Quarto': ['Human', 'Random', 'DQN', 'DoubleDQN', 'DDQN+ER', 'DDQN+PER', 'REINFORCE', 'REINFORCE+Mean', 'REINFORCE+Critic', 'PPO', 'A2C', 'RandomRollout', 'MCTS', 'ExpertApprentice', 'AlphaZero', 'MuZero', 'MuZero_Stochastic'],
        };

        function refreshAgentOptions() {
            const env = document.getElementById('envSelect').value;
            const select = document.getElementById('agentSelect');
            const previous = select.value;
            const options = AGENTS_BY_ENV[env] || [];
            select.innerHTML = '';

            for (const agent of options) {
                const option = document.createElement('option');
                option.value = agent;
                option.textContent = agent === 'Human' ? '👤 Human' : agent;
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
            currentEnv = env;
            gameDone = false;
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
            currentMode = data.mode || '';
            updateUI(data);
            buildActionButtons();
            log(data.info || ('New game: ' + env + ' | ' + agent));
        }
        
        async function stepGame() {
            const res = await fetch('/api/step', {method: 'POST'});
            const data = await res.json();
            updateUI(data);
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
                autoTimer = setTimeout(autoStep, 500);
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
                log('Human action=' + action + ' reward=' + data.reward.toFixed(3));
                if (data.done) {
                    let result = data.total_reward > 0 ? '🎉 WIN!' :
                                 data.total_reward < 0 ? '😞 LOSS' : '🤝 DRAW';
                    log('GAME OVER! ' + result + ' (total: ' + data.total_reward.toFixed(3) + ')');
                }
            });
        }
        
        function updateUI(data) {
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
                    data.total_reward > 0 ? '🎉 Victory!' : 
                    data.total_reward < 0 ? '😞 Defeat' : '🤝 Draw / Game Over';
                document.getElementById('message').style.color = 
                    data.total_reward > 0 ? '#2ecc71' : data.total_reward < 0 ? '#e94560' : '#f39c12';
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
            el.innerHTML += msg + '<br>';
            el.scrollTop = el.scrollHeight;
        }

        document.getElementById('envSelect').addEventListener('change', refreshAgentOptions);
        refreshAgentOptions();
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    print('Démarrage de la GUI : http://localhost:5000')
    print('Ctrl+C pour arrêter')
    app.run(debug=False, host='0.0.0.0', port=5000)
