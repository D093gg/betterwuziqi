import pygame
import numpy as np
from wuziqi_test import Game
from gamebot import Agent

# 游戏参数
model_filename = "agent_100.h5"
EMPTY = 0
BLACK = 1
WHITE = 2
# 2表示白棋，1表示黑棋，0表示空位
black_color = [0, 0, 0]
white_color = [255, 255, 255]
# Game类中draw()需要定义的参数


def play():
    # 创建棋盘对象
    game = Game()
    game.extra_t = int(90)
    # pygame初始化
    pygame.init()
    pygame.display.set_caption('五子棋-人机对弈')  # 改标题
    # pygame.display.set_mode()表示建立个窗口，左上角为坐标原点，往右为x正向，往下为y轴正向
    screen = pygame.display.set_mode((400, 400))
    # 给窗口填充颜色，颜色用三原色数字列表表示
    screen.fill([205, 186, 150])
    game.draw(screen)  # 给棋盘类发命令，调用draw()函数将棋盘画出来
    pygame.display.flip()  # 刷新窗口显示

    agent = Agent(model_file=model_filename)  # 加载训练好的模型
    player = int(input("选择先后手(1 or 2)\n"))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # pygame.MOUSEBUTTONDOWN表示鼠标的键被按下
            # button表示鼠标左键
            if game.current_player == player:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    x, y = event.pos  # 拿到鼠标当前在窗口上的位置坐标
                    # 将鼠标的(x, y)窗口坐标，转化换为棋盘上的坐标
                    row = round((y - 40) / 40)
                    col = round((x - 40) / 40)
                    if game.move(row, col):  # 尝试下棋
                        screen.fill([205, 186, 150])
                        game.draw(screen)
                    pygame.display.flip()
                    # 判断胜负
                    if game.flag:
                        print("Player wins！")
                        running = False
            else:
                # 智能体落子
                action = agent.get_action(np.array(game._board))
                row, col = divmod(action, game.size)
                game.move(row, col)
                screen.fill([205, 186, 150])
                game.draw(screen)
                pygame.display.flip()
                if game.flag:
                    print("Game bot wins！")
                    running = False

    pygame.quit()


if __name__ == "__main__":
    play()
