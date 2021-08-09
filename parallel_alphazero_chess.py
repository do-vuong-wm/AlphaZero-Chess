import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import math
import numpy as np
from typing import List
from multiprocessing import Process, Manager, set_start_method
from pettingzoo.classic import chess_v3
from pettingzoo.utils import wrappers

from pettingzoo.classic.chess import chess_utils
import chess
from pettingzoo import AECEnv
from gym import spaces
import warnings
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers

def chess_env():
    env = raw_env()
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):

    metadata = {'render.modes': ['human'], "name": "chess_v3"}

    def __init__(self):
        super().__init__()

        self.board = chess.Board()

        self.agents = ["player_{}".format(i) for i in range(2)]
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)

        self.action_spaces = {name: spaces.Discrete(8 * 8 * 73) for name in self.agents}
        self.observation_spaces = {name: spaces.Dict({
            'observation': spaces.Box(low=0, high=1, shape=(8, 8, 20), dtype=np.bool),
            'action_mask': spaces.Box(low=0, high=1, shape=(4672,), dtype=np.int8)
        }) for name in self.agents}

        self.rewards = None
        self.dones = None
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = None

    def copy(self):
        env = raw_env()
        env.reset()
        env.board = self.board.copy()
        env._agent_selector._current_agent = self._agent_selector._current_agent
        env._agent_selector.selected_agent = self._agent_selector.selected_agent
        env.agent_selection = self.agent_selection

        return env

    def observe(self, agent):
        observation = chess_utils.get_observation(self.board, self.possible_agents.index(agent))
        legal_moves = chess_utils.legal_moves(self.board) if agent == self.agent_selection else []

        action_mask = np.zeros(4672, int)
        for i in legal_moves:
            action_mask[i] = 1

        return {'observation': observation, 'action_mask': action_mask}

    def reset(self):
        self.has_reset = True

        self.agents = self.possible_agents[:]

        self.board = chess.Board()

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

    def set_game_result(self, result_val):
        for i, name in enumerate(self.agents):
            self.dones[name] = True
            result_coef = 1 if i == 0 else -1
            self.rewards[name] = result_val * result_coef
            self.infos[name] = {'legal_moves': []}

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        current_agent = self.agent_selection
        current_index = self.agents.index(current_agent)
        self.agent_selection = self._agent_selector.next()

        chosen_move = chess_utils.action_to_move(self.board, action, current_index)
        assert chosen_move in self.board.legal_moves
        self.board.push(chosen_move)

        next_legal_moves = chess_utils.legal_moves(self.board)

        is_stale_or_checkmate = not any(next_legal_moves)

        if is_stale_or_checkmate or self.board.is_repetition(3) or self.board.can_claim_fifty_moves() or self.board.is_insufficient_material():
            result = self.board.result(claim_draw=True)
            result_val = chess_utils.result_to_int(result)
            self.set_game_result(result_val)

        self._accumulate_rewards()

    def render(self, mode='human'):
        print(self.board)
        return str(self.board)

    def close(self):
        pass

"""# Config"""

class AlphaZeroConfig(object):

    def __init__(self):
        ### Self-Play
        self.num_actors = os.cpu_count() # 5000

        # For the first 30 moves of each game, the temperature is set to τ=1; this 
        # selects moves proportionally to their visit count in MCTS, and ensures a diverse 
        # set of positions are encountered.
        self.num_sampling_moves = 30
        self.max_moves = 512  # for chess and shogi, 722 for Go.
        self.num_simulations = 800 # 100 800

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Training
        self.training_steps = int(700e3)
        self.checkpoint_interval = 2 # int(1e3)
        self.window_size = int(1e5) # int(1e6) # buffer size
        self.batch_size = 32*(self.num_actors-1) # 32 # 256 # 4096

        self.weight_decay = 1e-4 # regularization param
        self.momentum = 0.9
        # Schedule for chess and shogi, Go starts at 2e-2 immediately.
        self.boundaries = [100e3, 300e3, 500e3]
        self.values = [2e-1, 2e-2, 2e-3, 2e-4]

"""# Node"""

class Node(object):

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

"""# Game"""

class Game(object):

    def __init__(self, history=None, env=None):
        if env:
            self.env = env
        else:
            self.env = chess_env()
            self.env.reset()
        self.history = history or []
        self.child_visits = []
        self.num_actions = 4672  # action space size for chess; 11259 for shogi, 362 for Go
        self.observation, self.reward, self.done, self.info = None, None, None, None

    def terminal(self):
        # Game specific termination rules.
        return self.done

    def terminal_value(self, to_play):
        # Game specific value.
        return list(self.env._cumulative_rewards.values())[to_play]

    def legal_actions(self):
        # Game specific calculation of legal actions.
        self.observation, self.reward, self.done, self.info = self.env.last()
        (actions,) = np.where(self.observation['action_mask'] == 1)
        return actions

    def clone(self):
        return Game(list(self.history), self.env.env.env.env.env.copy())

    def apply(self, action, observe=True):
        self.history.append(action)
        self.env.step(action)
        self.observation, self.reward, self.done, self.info = self.env.last(observe)

    def store_search_statistics(self, root): # stores search probabilities pi for states of self-play in game
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index: int):
        # Game specific feature planes.
        # -1 current obs
        # else play new game from beginning with history?
        if state_index == -1:
            self.observation, self.reward, self.done, self.info = self.env.last()
            return self.observation['observation']
        else:
            temp_env = chess_env()
            temp_env.reset()
            for i in range(state_index): # change from range(state_index+1) to range(state_index)
                temp_env.step(self.history[i])
            observation, _, _, _ = temp_env.last()
            return self.observation['observation']

    def make_target(self, state_index: int):
        return (self.terminal_value(state_index % 2), # target value for current player in state
                self.child_visits[state_index]) # target policy

    def to_play(self):
        return len(self.history) % 2

"""# Replay Buffer"""

class ReplayBuffer(object):

    def __init__(self, config: AlphaZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        # Sample uniformly across positions.
        move_sum = float(sum(len(g.history) for g in self.buffer))
        games = np.random.choice(
                self.buffer,
                size=self.batch_size,
                p=[len(g.history) / move_sum for g in self.buffer])
        game_pos = [(g, np.random.randint(len(g.history))) for g in games]
        return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]

"""# Shared Storage
Use for executing asynchronously in parallel.
"""

class SharedStorage(object):

    def __init__(self):
        self._networks = {}

    def latest_network(self):
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            return None # make_uniform_network()  # policy -> uniform, value -> 0.5

    def save_network(self, step: int, network):
        self._networks[step] = network

"""# Entry To Begin Training
AlphaZero training is split into two independent parts: Network training and
self-play data generation.
These two parts only communicate by transferring the latest network checkpoint
from the training to the self-play, and the finished games from the self-play
to the training.
"""

def alphazero(config: AlphaZeroConfig):
    manager = Manager()
    Global = manager.Namespace()
    Lock = manager.Lock()
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)
    
    Global.storage = storage
    Global.replay_buffer = replay_buffer

    p_list = []

    # Actor Processes
    for i in range(config.num_actors-1):
        p = Process(target=run_selfplay, args=(config, Global, Lock, i))
        p_list.append(p)

    for p in p_list:
        p.start()

    # Training Process
    train_network(config, Global)

    for p in p_list:
        p.join()
        p.close()

    return storage.latest_network()

"""# Self-Play"""

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, Global, Lock, pid):
    
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    # tf.config.set_visible_devices([], 'GPU')

    """# Network"""

    class ResnetIdentityBlock(tf.keras.Model):
        def __init__(self, kernel_size, filters):
            super(ResnetIdentityBlock, self).__init__(name='')

            self.conv2a = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='same')
            self.bn2a = tf.keras.layers.BatchNormalization()

            self.conv2b = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='same')
            self.bn2b = tf.keras.layers.BatchNormalization()

        def call(self, input_tensor, training=False):
            x = self.conv2a(input_tensor)
            x = self.bn2a(x, training=training)
            x = tf.nn.relu(x)

            x = self.conv2b(x)
            x = self.bn2b(x, training=training)

            x += input_tensor # layers.add()
            return tf.nn.relu(x)

    class Network(tf.keras.Model):
        def __init__(self):
            super(Network, self).__init__()
            kernel_size = 3
            filters = 64
            self.conv2a = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='same')
            self.bn2a = tf.keras.layers.BatchNormalization()
            # Resnet Blocks
            self.block1 = ResnetIdentityBlock(kernel_size, filters)
            self.block2 = ResnetIdentityBlock(kernel_size, filters)
            # Policy Head
            self.conv2b = tf.keras.layers.Conv2D(2, 1, strides=(1, 1), padding='same')
            self.bn2b = tf.keras.layers.BatchNormalization()
            self.conv2c = tf.keras.layers.Conv2D(73, 1, strides=(1, 1), padding='same')
            self.flatten1 = tf.keras.layers.Flatten()
            # Value Head
            self.conv2d = tf.keras.layers.Conv2D(1, 1, strides=(1, 1), padding='same')
            self.bn2d = tf.keras.layers.BatchNormalization()
            self.flatten2 = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(256)
            self.dense2 = tf.keras.layers.Dense(1)
            
        def call(self, image, training=False):
            x = self.conv2a(image)
            x = self.bn2a(x, training=training)
            x = tf.nn.relu(x)
            x = self.block1(x, training=training)
            x = self.block2(x, training=training)
            
            policy = self.conv2b(x)
            policy = self.bn2b(policy, training=training)
            policy = tf.nn.relu(policy)
            policy = self.conv2c(policy)
            policy = self.flatten1(policy)
            
            value = self.conv2d(x)
            value = self.bn2d(value, training=training)
            value = tf.nn.relu(value)
            value = self.flatten2(value)
            value = self.dense1(value)
            value = tf.nn.relu(value)
            value = self.dense2(value)
            value = tf.nn.tanh(value)
            
            return value, policy

    # Each game is produced by starting at the initial board position, then
    # repeatedly executing a Monte Carlo Tree Search to generate moves until the end
    # of the game is reached.
    def play_game(config: AlphaZeroConfig, network: Network):
        game = Game()
        while not game.terminal() and len(game.history) < config.max_moves:
            if not pid:
                game.env.render()
            action, root = run_mcts(config, game, network) # MCTS runs 800 sims for each position til terminal
            game.apply(action)
            game.store_search_statistics(root) # search probabilities
        return game

    """# Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def run_mcts(config: AlphaZeroConfig, game: Game, network: Network):
        root = Node(0)
        evaluate(root, game, network)
        add_exploration_noise(config, root)

        for _ in range(config.num_simulations):
            node = root
            scratch_game = game.clone()
            search_path = [node]

            while node.expanded() and not scratch_game.terminal(): # special case if reaches terminal state in game
                action, node = select_child(config, node)
                scratch_game.apply(action, observe=False)
                search_path.append(node)
                
            if scratch_game.terminal():
                value = scratch_game.terminal_value(scratch_game.to_play()) # to_play?
            else:
                value = evaluate(node, scratch_game, network)
            backpropagate(search_path, value, scratch_game.to_play())
        return select_action(config, game, root), root

    def select_action(config: AlphaZeroConfig, game: Game, root: Node):
        visit_counts = [(child.visit_count, action)
                        for action, child in root.children.items()]
        if len(game.history) < config.num_sampling_moves:
            _, action = softmax_sample(visit_counts)
        else:
            _, action = max(visit_counts)
        return action

    # Select the child with the highest UCB score.
    def select_child(config: AlphaZeroConfig, node: Node):
        _, action, child = max((ucb_score(config, node, child), action, child)
                                for action, child in node.children.items())
        return action, child

    # The score for a node is based on its value, plus an exploration bonus based on
    # the prior.
    def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
        pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                        config.pb_c_base) + config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = child.value()
        return prior_score + value_score

    # We use the neural network to obtain a value and policy prediction.
    def evaluate(node: Node, game: Game, network: Network):
        image = tf.convert_to_tensor(game.make_image(-1), dtype='float32')
        obs = tf.expand_dims(image, 0)
        value, policy_logits = network(obs, training=False)
        value = tf.squeeze(value, [0])
        policy_logits = tf.squeeze(policy_logits, [0])
        policy_logits = policy_logits.numpy()
        # Expand the node.
        node.to_play = game.to_play()
        legal_actions = game.legal_actions()
        policy = {a: np.exp(policy_logits[a]) for a in legal_actions}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(p / policy_sum)
        return float(value)

    # At the end of a simulation, we propagate the evaluation all the way up the
    # tree to the root.
    def backpropagate(search_path: List[Node], value: float, to_play):
        for node in search_path:
            node.value_sum += value if node.to_play == to_play else -value # was (1 - value) but use -value instead?
            node.visit_count += 1

    # At the start of each search, we add dirichlet noise to the prior of the root
    # to encourage the search to explore new actions.
    def add_exploration_noise(config: AlphaZeroConfig, node: Node):
        actions = node.children.keys()
        noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
        frac = config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def softmax_sample(d):
        temp = 1 # lets you explore more action as temp increases
        move_sum = float(sum(math.pow(visit_count, 1/temp) for visit_count, _ in d))
        index = np.random.choice(
            range(len(d)),
            p=[math.pow(visit_count, 1/temp) / move_sum for visit_count, _ in d])
        return d[index]

    while True:
        weights = Global.storage.latest_network()
        if not weights:
            network = Network()
            network(tf.ones((1, 8, 8, 20)))
        else:
            network = Network()
            network(tf.ones((1, 8, 8, 20)))
            network.set_weights(weights)
        game = play_game(config, network)
        Lock.acquire()
        temp = Global.replay_buffer
        temp.save_game(game) # to sync proxy object
        Global.replay_buffer = temp
        Lock.release()
        print('Process %d: %d %d' % (pid, len(Global.storage._networks), len(Global.replay_buffer.buffer)))

"""# Training"""

def train_network(config: AlphaZeroConfig, Global):

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    # tf.config.set_visible_devices([], 'GPU')

    """# Network"""

    class ResnetIdentityBlock(tf.keras.Model):
        def __init__(self, kernel_size, filters):
            super(ResnetIdentityBlock, self).__init__(name='')

            self.conv2a = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='same')
            self.bn2a = tf.keras.layers.BatchNormalization()

            self.conv2b = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='same')
            self.bn2b = tf.keras.layers.BatchNormalization()

        def call(self, input_tensor, training=False):
            x = self.conv2a(input_tensor)
            x = self.bn2a(x, training=training)
            x = tf.nn.relu(x)

            x = self.conv2b(x)
            x = self.bn2b(x, training=training)

            x += input_tensor # layers.add()
            return tf.nn.relu(x)

    class Network(tf.keras.Model):
        def __init__(self):
            super(Network, self).__init__()
            kernel_size = 3
            filters = 64
            self.conv2a = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='same')
            self.bn2a = tf.keras.layers.BatchNormalization()
            # Resnet Blocks
            self.block1 = ResnetIdentityBlock(kernel_size, filters)
            self.block2 = ResnetIdentityBlock(kernel_size, filters)
            # Policy Head
            self.conv2b = tf.keras.layers.Conv2D(2, 1, strides=(1, 1), padding='same')
            self.bn2b = tf.keras.layers.BatchNormalization()
            self.conv2c = tf.keras.layers.Conv2D(73, 1, strides=(1, 1), padding='same')
            self.flatten1 = tf.keras.layers.Flatten()
            # Value Head
            self.conv2d = tf.keras.layers.Conv2D(1, 1, strides=(1, 1), padding='same')
            self.bn2d = tf.keras.layers.BatchNormalization()
            self.flatten2 = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(256)
            self.dense2 = tf.keras.layers.Dense(1)
            
        def call(self, image, training=False):
            x = self.conv2a(image)
            x = self.bn2a(x, training=training)
            x = tf.nn.relu(x)
            x = self.block1(x, training=training)
            x = self.block2(x, training=training)
            
            policy = self.conv2b(x)
            policy = self.bn2b(policy, training=training)
            policy = tf.nn.relu(policy)
            policy = self.conv2c(policy)
            policy = self.flatten1(policy)
            
            value = self.conv2d(x)
            value = self.bn2d(value, training=training)
            value = tf.nn.relu(value)
            value = self.flatten2(value)
            value = self.dense1(value)
            value = tf.nn.relu(value)
            value = self.dense2(value)
            value = tf.nn.tanh(value)
            
            return value, policy

    def update_weights(optimizer: tf.keras.optimizers.Optimizer, network: Network, batch,
                    weight_decay: float):
        with tf.GradientTape() as tape:
            loss = 0
            for image, (target_value, target_policy) in batch:
                image = tf.convert_to_tensor(image, dtype='float32')
                value, policy_logits = network(tf.expand_dims(image, 0), training=True)
                value = tf.squeeze(value, [0])
                policy_logits = tf.squeeze(policy_logits, [0])
                t = tf.convert_to_tensor(target_policy, dtype='float32')
                v = tf.convert_to_tensor(target_value, dtype='float32')
                loss += (
                    tf.keras.metrics.mean_squared_error(v, value) +
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=policy_logits, labels=t))

            for weights in network.get_weights():
                loss += weight_decay * tf.nn.l2_loss(weights)
        grads = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(grads, network.trainable_variables))

    network = Network()
    network(tf.ones((1, 8, 8, 20)))

    learning_rate_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(config.boundaries, config.values)
    optimizer = tf.keras.optimizers.SGD(learning_rate_schedule, # Maybe other optimizers with no schedule?
                                        config.momentum)
    i = 0
    prev_size = 0
    while True:
        buffer_size = len(Global.replay_buffer.buffer)
        if(not buffer_size % (config.num_actors-1) and buffer_size >= config.num_actors-1 and prev_size != buffer_size):
            print('Next Batch Processing...')
            batch = Global.replay_buffer.sample_batch()
            update_weights(optimizer, network, batch, config.weight_decay)
            if i % config.checkpoint_interval == 0:
                network.save_weights('checkpoints/' + str(i))
            temp = Global.storage
            temp.save_network(i, network.get_weights())
            Global.storage = temp
            prev_size = buffer_size
            i += 1
    temp = Global.storage
    temp.save_network(config.training_steps, network.get_weights())
    Global.storage = temp

if __name__ == '__main__':
    set_start_method('spawn')
    config = AlphaZeroConfig()
    alphazero(config)