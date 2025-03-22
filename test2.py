import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class Game:
    def __init__(self, size=15):
        self.size = size  # 棋盘大小
        self.board = self.initialize_board()
        self.current_player = 1  # 玩家1用1表示，玩家2用2表示
        self.winner = None  # 用于存储赢家

    def initialize_board(self):
        """初始化棋盘，棋盘为size x size的二维列表，初始值为0表示空位"""
        return [[0 for _ in range(self.size)] for _ in range(self.size)]

    def make_move(self, row, col):
        """执行落子操作，返回是否成功落子"""
        if self.winner is not None:
            return False  # 游戏已经结束，无法落子
        if self.board[row][col] != 0:
            return False  # 该位置已经被占用

        self.board[row][col] = self.current_player  # 在棋盘上标记当前玩家
        if self.check_winner(row, col):
            self.winner = self.current_player  # 如果当前玩家赢了，记录赢家
        self.current_player = 3 - self.current_player  # 切换玩家
        return True  # 落子成功

    def is_valid_move(self, row, col):
        """检查落子是否合法"""
        return self.winner is None and self.board[row][col] == 0

    def check_winner(self, row, col):
        """检查当前玩家是否获胜"""
        # 检查所有方向：水平、垂直、对角线
        return (self.check_direction(row, col, 1, 0) or  # 水平
                self.check_direction(row, col, 0, 1) or  # 垂直
                self.check_direction(row, col, 1, 1) or  # 主对角线
                self.check_direction(row, col, 1, -1))   # 副对角线

    def check_direction(self, row, col, delta_row, delta_col):
        """检查特定方向上是否有五个连续相同的棋子"""
        count = 1  # 当前棋子计数
        # 向一个方向检查
        for step in range(1, 5):
            r, c = row + step * delta_row, col + step * delta_col
            if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == self.current_player:
                count += 1
            else:
                break

        # 向反方向检查
        for step in range(1, 5):
            r, c = row - step * delta_row, col - step * delta_col
            if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == self.current_player:
                count += 1
            else:
                break

        return count >= 5  # 如果计数大于等于5，则获胜

    def reset_game(self):
        """重置游戏"""
        self.board = self.initialize_board()
        self.current_player = 1
        self.winner = None


class GameEnv:  # 封装已有的Game规则，作为强化学习的环境
    def __init__(self, size=15):
        self.game = Game(size)

    def reset(self):
        self.game.reset_game()
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


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def choose_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.action_size)  # 随机选择动作
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # 选择具有最大 Q 值的动作

    def learn(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += 0.95 * np.amax(self.model.predict(next_state)[0])  # 未来奖励
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)


def train_agent(agent, env, num_games):
    for episode in range(num_games):
        state = env.reset()  # 重置环境
        state = state.reshape(1, -1)  # 转换为适合模型的形状
        done = False
        epsilon = max(0.1, 1.0 - episode / 1000)  # 衰减的探索率

        while not done:
            action = agent.choose_action(state, epsilon)  # 选择动作
            row, col = divmod(action, env.game.size)
            if not env.game.is_valid_move(row, col):
                # 重新选择动作，直到找到有效的
                while not env.game.is_valid_move(row, col):
                    action = agent.choose_action(state)
            next_state, reward, done = env.step(action)  # 执行动作
            next_state = next_state.reshape(1, -1)  # 转换为适合模型的形状
            agent.learn(state, action, reward, next_state, done)  # 学习
            state = next_state

        if episode % 100 == 0:
            agent.save_model(f"agent_model_{episode}.h5")


def visualize_game_state(state):
    plt.imshow(state, cmap='gray')  # 假设状态是一个2D数组
    plt.axis('off')
    plt.show()


def play_with_user(agent, env):
    state = env.reset()  # 初始化游戏
    done = False

    while not done:
        # 可视化当前棋盘状态
        print(np.array(env.game.board))  # 打印棋盘
        user_input = input("请输入您的动作（行 列）：")  # 用户输入行列坐标
        row, col = map(int, user_input.split())  # 将输入分割并转换为整数

        # 检查输入的合法性
        if row < 0 or row >= env.game.size or col < 0 or col >= env.game.size:
            print("输入的坐标超出范围，请重新输入。")
            continue

        # 执行用户动作
        next_state, reward, done = env.step(row * env.game.size + col)  # 将行列转换为动作

        # 智能体选择动作
        if not done:  # 只有在游戏未结束时，智能体才会进行选择
            state = next_state.reshape(1, -1)
            agent_action = agent.choose_action(state, epsilon=0)  # 不探索
            next_state, reward, done = env.step(agent_action)

        # 更新状态
        state = next_state

    print("游戏结束！赢家是：", "玩家1" if env.game.winner == 1 else "玩家2")


if __name__ == "__main__":
    size = 15  # 棋盘大小
    env = GameEnv(size)  # 创建环境
    state_size = size * size  # 状态维度
    action_size = size * size  # 动作数量（每个格子一个动作）
    agent = Agent(state_size, action_size)

    mode = input("选择模式（1: 训练, 2: 与用户对弈）：")

    if mode == "1":
        num_episodes = int(input("请输入训练回合数："))
        train_agent(agent, env, num_episodes)
    elif mode == "2":
        agent.load_model("agent_model_0.h5")  # 加载最新模型
        play_with_user(agent, env)
