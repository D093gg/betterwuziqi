# tests/wuziqi_test.py
# 最初规则制定版, class Game包含所有游戏规则, size和extra_t调整棋盘大小和手数
# 使用pygame可视化, 可忽略

import tensorflow as tf
import numpy as np
from wuziqi_test import Game

EMPTY = 0
BLACK = 1
WHITE = 2
# 2表示白棋，1表示黑棋，0表示空位


'''
class GameEnv:  # 封装已有的Game规则，作为强化学习的环境
    def __init__(self, size=15):
        self.game = Game(size)

    def reset(self):
        self.game.reset()
        return self.get_state()

    def step(self, action):
        row, col = divmod(action, self.game.size)  # 将动作转换为行列坐标
        self.game.make_move(row, col)
        state = self.get_state()
        reward = self.get_reward()
        done = self.game.winner is not None  # 游戏结束条件
        return state, reward, done

    def get_state(self):
        """返回当前棋盘状态的扁平化表示"""
        return np.array(self.game.board).flatten()

    def get_reward(self):
        """根据当前状态返回奖励"""
        if self.game.winner == 1:
            return 1  # 玩家1胜利
        elif self.game.winner == 2:
            return -1  # 玩家2胜利
        else:
            return 0  # 游戏未结束
'''


class Agent:
    def __init__(self, size=9, learning_rate=0.001, discount_factor=0.99, exploration_prob=1.0,
                 exploration_decay=0.995, min_exploration_prob=0.1, model_file=None):
        self.size = size
        self.learning_rate = learning_rate  # learning_rate 学习率控制模型更新的速度
        self.discount_factor = discount_factor  # discount_factor: 折扣因子, 决定未来奖励对当前决策的重要程度
        self.exploration_prob = exploration_prob  # exploration_prob: 初始探索概率, 决定智能体在选择动作时随机选择的概率
        self.exploration_decay = exploration_decay  # exploration_decay: 探索概率的衰减率, 随着训练进行逐渐减少探索
        self.min_exploration_prob = min_exploration_prob  # min_exploration_prob: 最小探索概率, 防止探索概率降得过低
        self.q_table = self.build_model()
        if model_file:
            self.load_model(model_file)  # 如果有已有的就继续上次训练

    def build_model(self):
        """构建 Q-learning 模型"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.size, self.size)))  # 输入为棋盘状态, 接受形状为(size, size)的棋盘状态。
        model.add(tf.keras.layers.Flatten())  # Flatten 层, 将二维棋盘状态展平为一维数组
        model.add(tf.keras.layers.Dense(128, activation='relu'))  # 两层 Dense, 每层 128 神经元, ReLU 激活函数
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(self.size * self.size, activation='linear'))  # 输出层, 输出为每个位置的 Q , size * size
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def get_action(self, state):
        """根据当前状态选择动作"""
        # 获取当前状态下的空位
        available_actions = self.get_available_actions(state)
        if not available_actions:
            return None  # 如果没有可用的动作，返回 None
        if np.random.rand() < self.exploration_prob:
            return np.random.randint(self.size * self.size)  # 随机选择动作
        else:
            q_values = self.q_table.predict(state[np.newaxis, ...])
            return np.argmax(q_values[0])  # 选择 Q 值最高的动作

    def get_available_actions(self, state):
        # 找到所有空位的索引
        empty_positions = np.argwhere(state == EMPTY)
        # 将坐标转换为动作编号
        return [pos[0] * state.shape[1] + pos[1] for pos in empty_positions]

    def train(self, state, action, reward, next_state, done):
        """训练模型"""
        target = reward
        if not done:
            target += self.discount_factor * np.max(self.q_table.predict(next_state[np.newaxis, ...])[0])

        target_f = self.q_table.predict(state[np.newaxis, ...])
        target_f[0][action] = target

        self.q_table.fit(state[np.newaxis, ...], target_f, epochs=1, verbose=0)

        # 概率衰减
        if self.exploration_prob > self.min_exploration_prob:
            self.exploration_prob *= self.exploration_decay

    def reset(self):
        """重置 Agent 状态"""
        self.exploration_prob = 1.0  # 重置探索概率

    def load_model(self, filename):
        """加载已有模型"""
        self.q_table = tf.keras.models.load_model(filename)

    def save_model(self, filename):
        """保存模型"""
        self.q_table.save(filename)


def self_play(agent1, agent2, game, episodes):
    for episode in range(episodes):
        game.reset()
        state = np.array(game._board)  # 获取初始状态，转换为 NumPy 数组
        done = False

        while not done:
            # 轮流让两个智能体下棋
            if game.current_player == BLACK:
                action = agent1.get_action(state)
            else:
                action = agent2.get_action(state)
            if action is None:
                break  # 如果没有可用的动作，结束游戏
            row, col = divmod(action, game.size)
            game.move(row, col)
            # 检查游戏是否结束
            if game.flag:
                done = True
                reward = 1 if game.current_player == BLACK else -1  # 根据当前玩家返回奖励
            else:
                reward = 0  # 继续游戏，奖励为0

            next_state = np.array(game._board)  # 获取下一个状态

            # 训练智能体
            if game.current_player == BLACK:
                agent1.train(state, action, reward, next_state, done)
            else:
                agent2.train(state, action, reward, next_state, done)
            state = next_state  # 更新状态

        if episode % 10 == 0:
            agent1.save_model(f"agent_{episode}.h5")


if __name__ == "__main__":
    game = Game()
    game.extra_t = int(90)
    model_filename = "agent_model_10.h5"
    agent1 = Agent(model_file=model_filename)
    agent2 = Agent(model_file=model_filename)
    self_play(agent1, agent2, game, 100)
