# tests/wuziqi_test.py
# 最初规则制定版, class Game包含所有游戏规则, size和extra_t调整棋盘大小和手数
# 使用pygame可视化, 可忽略

import pygame

EMPTY = 0
BLACK = 1
WHITE = 2
# 2表示白棋，1表示黑棋，0表示空位

black_color = [0, 0, 0]
white_color = [255, 255, 255]


# 定义棋盘这个类
class Game(object):

    def __init__(self, size=9):
        self.size = size
        self._board = [[EMPTY] * size for _ in range(size)]  # 初始化棋盘
        self._hands = [[0] * size for _ in range(size)]  # 用0初始化手数
        self.reset()
        self.hands = 0
        self.current_player = BLACK
        # 游戏规则参数, extra_t代表棋盘上最多保留的棋子数
        self.extra_t = int(16)  # int()防止出现小数点什么的
        self.flag = False  # 判断游戏是否结束

    # 重置棋盘
    def reset(self):
        self._board = [[EMPTY] * self.size for _ in range(self.size)]
        self._hands = [[0] * self.size for _ in range(self.size)]
        self.hands = 0
        self.current_player = BLACK
        self.flag = False  # 判断游戏是否结束

    # 定义棋盘上的下棋函数，row表示行，col表示列，extra_t是最重要参数
    def move(self, row, col):
        if self._board[row][col] == EMPTY:
            self._board[row][col] = self.current_player
            self.hands += 1
            self._hands[row][col] = self.hands
            # 每一首棋都记录手数，每下若干手就消失第一手，到T就去除T-10
            if self.hands > self.extra_t:
                for r in range(self.size):
                    for c in range(self.size):
                        if self._hands[r][c] == (self.hands - self.extra_t):
                            self._board[r][c] = EMPTY  # 清空该位置
                            break  # 找到后退出循环

            # 检查是否获胜
            if self.check_winner(row, col):
                print(f"Player {self.current_player} wins!")
                self.flag = True
            else:
                self.current_player = 3 - self.current_player  # 切换玩家
            return True
        return False  # 表示落子失败

    # 使用pygame.draw()画出棋盘和落子, 同时标注距离棋子消失的手数
    # Agent对弈时可以忽略此绘制函数
    def draw(self, screen):
        # 给棋盘加一个外框和特殊点位
        for h in range(1, self.size + 1):
            pygame.draw.line(screen, black_color,
                             [40, h * 40], [40 * self.size, h * 40], 1)
            pygame.draw.line(screen, black_color,
                             [h * 40, 40], [h * 40, 40 * self.size], 1)
        pygame.draw.rect(screen, black_color, [36, 36, (40 * self.size - 30), (40 * self.size - 30)], 3)
        pygame.draw.circle(screen, black_color, [40 * ((self.size + 1) / 2), 40 * ((self.size + 1) / 2)], 5, 0)

        # 做2次for循环取得棋盘上所有交叉点的坐标
        for row in range(len(self._board)):
            for col in range(len(self._board[row])):
                # 画出落子
                if self._board[row][col] != EMPTY:
                    color = black_color \
                        if self._board[row][col] == BLACK else white_color
                    pos = [40 * (col + 1), 40 * (row + 1)]
                    # 标上数字的坐标手动微调
                    text_pos = [40 * (col + 1) - 5, 40 * (row + 1) - 8]
                    pygame.draw.circle(screen, color, pos, 18, 0)
                    countdown = int(self.extra_t / 2) - divmod((self.hands - self._hands[row][col]), 2)[0]
                    font = pygame.font.SysFont('bahnschrift', 20)  # 最喜欢的德语字体
                    if divmod(self._hands[row][col], 2)[1] == 0:
                        text = font.render(str(countdown), True, black_color)
                    else:
                        text = font.render(str(countdown), True, white_color)
                    screen.blit(text, tuple(text_pos))

    # 检查当前玩家是否获胜
    def check_winner(self, row, col):
        # 检查所有方向：水平、垂直、对角线
        return (self.check_direction(row, col, 1, 0) or  # 水平
                self.check_direction(row, col, 0, 1) or  # 垂直
                self.check_direction(row, col, 1, 1) or  # 主对角线
                self.check_direction(row, col, 1, -1))  # 副对角线

    # 检查特定方向上是否有五个连续相同的棋子
    def check_direction(self, row, col, delta_row, delta_col):
        count = 1  # 当前棋子计数
        # 向一个方向检查
        for step in range(1, 5):
            r, c = row + step * delta_row, col + step * delta_col
            if 0 <= r < self.size and 0 <= c < self.size and self._board[r][c] == self.current_player:
                count += 1
            else:
                break

        # 向反方向检查
        for step in range(1, 5):
            r, c = row - step * delta_row, col - step * delta_col
            if 0 <= r < self.size and 0 <= c < self.size and self._board[r][c] == self.current_player:
                count += 1
            else:
                break

        return count >= 5  # 如果计数大于等于5，则获胜


def main():
    # 创建棋盘对象
    game = Game()
    # pygame初始化函数，固定写法
    pygame.init()
    pygame.display.set_caption('五子棋')  # 改标题
    # pygame.display.set_mode()表示建立个窗口，左上角为坐标原点，往右为x正向，往下为y轴正向
    screen = pygame.display.set_mode((400, 400))
    # 给窗口填充颜色，颜色用三原色数字列表表示
    screen.fill([205, 186, 150])
    game.draw(screen)  # 给棋盘类发命令，调用draw()函数将棋盘画出来
    pygame.display.flip()  # 刷新窗口显示

    running = True
    # while 主循环的标签，以便跳出循环
    while running:
        # 遍历建立窗口后发生的所有事件
        for event in pygame.event.get():
            # 根据事件的类型，进行判断
            if event.type == pygame.QUIT:
                running = False
            # pygame.MOUSEBUTTONDOWN表示鼠标的键被按下
            elif event.type == pygame.MOUSEBUTTONUP and \
                    event.button == 1:  # button表示鼠标左键
                x, y = event.pos  # 拿到鼠标当前在窗口上的位置坐标
                # 将鼠标的(x, y)窗口坐标，转化换为棋盘上的坐标
                row = round((y - 40) / 40)
                col = round((x - 40) / 40)
                ''' 打印棋盘测试用
                if game.move(row, col):
                    for rows in game._board:
                        for element in rows:
                            print(element, end=' ')
                        print()
                '''
                if game.move(row, col):  # 尝试下棋
                    screen.fill([205, 186, 150])
                    game.draw(screen)
                pygame.display.flip()
                # 调用判断胜负函数
                if game.flag:
                    running = False
    pygame.quit()


if __name__ == '__main__':
    main()
