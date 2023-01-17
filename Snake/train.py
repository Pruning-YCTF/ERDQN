import argparse

from project_files.agent import GameAIAgent
from project_files.game import Game
from project_files.plotting import plot


def train_snake(file_for_saving, show_plots):
    plot_scores, plot_mean_scores, total_score, record = [], [], 0, 0
    # 记录score， mean_scores, ttal_score, 最好记录
    agent = GameAIAgent() # 实例化一个agent对象
    game = Game(Train=True)         # 实例化一个游戏对象

    while True:
        # get the old state
        old_state = agent.get_state(game) # 根据当前游戏界面获得当前的state

        move = agent.trainer.make_action(old_state) # 根据当前state做出action
        agent.trainer.global_step += 1

        # perform move and get new state
        reward, game_over, score = game.step(move) # 执行action后，获取本次执行所得到的reward, game_over, score

        # get new state
        new_state = agent.get_state(game) # 在获取新的state

        agent.train_short_memory(old_state, move, reward, new_state, game_over) # 拿这一组数据训练一次
        agent.remember(old_state, move, reward, new_state, game_over) # 把这个四元组保存在memory里

        if game_over: # 游戏结束
            # train long memory, plotting
            game.reset() # 重新设置游戏
            agent.trainer.n_iterations += 1 # 迭代次数+1
            agent.train_long_memory() # 拿获得的多组数据进行训练

            if score > record: # 如果分数比最高纪录还好
                record = score # 更新一下最高纪录
                agent.model.save(file_for_saving) # 保存一下模型

            total_score += score # 每玩一次总分数加上当前分数

            mean_score = total_score / agent.trainer.n_iterations # 计算平均分

            if show_plots:  # 展示图像
                plot_scores.append(score)
                plot_mean_scores.append(mean_score)

                plot(plot_scores, plot_mean_scores)

            print(f'Game: {agent.trainer.n_iterations}, Score: {score}, Mean Score: {mean_score}, Record: {record}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train your snake')
    parser.add_argument('-f', '--filename', type=str, help='Path to the file where to save model after training',
                        required=False, default='./model/model.pth')
    parser.add_argument('-s', '--short_form', action='store_false',
                        help='Dont show plotting of scores and mean score while training')

    args = parser.parse_args()

    train_snake(args.filename, args.short_form)
