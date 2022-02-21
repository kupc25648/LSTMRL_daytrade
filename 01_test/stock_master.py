'''
==================================================================
Master stock file for training DDPG agent to trade stock using Ichimoku data
==================================================================
'''

#====================================================================
#Import Part
#インポート部
#====================================================================
from ichimoku import *
from stock_GEN import *
from stock_ENV import *
from stock_RL import *

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import gc
import os
import random
import pandas as pd
import datetime
random.seed(3407)
np.random.seed(3407)

tensor_seed = 3407
tf.random.set_seed(tensor_seed)
#====================================================================
#Parameter Part
#パラメータ部
#====================================================================
# stock GEN ------------------------------------------
#data_dir = 'individual_stocks_5yr/AAL_data.csv'
len_data = 1000

init_cap = 10000 # USD
# stock ENV ------------------------------------------
objective = 'Capital'
time_window = 60
#end_step = len_data-time_window-26 #will be decide by len_data source instead 20210607
stop_gain_multiplier = 1
stop_loss_multiplier = 1
# stock RL -------------------------------------------√
lr = 0.0001 # neural network agent learning rate
ep = 0.05 # initial epsilon value / 初期イプシロン値
epd = 0.9998 # epsilon decay value / イプシロン減衰値
gamma = 0.99 # reward discount factor
# how neural network is built(neuron in each layer)
a_nn = [512,256,256,256] # Actor ニューラルネットワーク
c_nn = [512,256,256,256] # Critic ニューラルネットワーク
max_mem = 100000 # maximum length of replay buffer / 再生バッファの最大長
num_ob = [14,10] #nueral network input(state) (related to game_type) / ニューラルネットワーク入力(状態)（game_typeに関連
num_action = 5 #[class Game:num_ob , class Game0:6 , class Game1:6 , class Game2:3] neural network output(action) / ニューラルネットワーク出力（行動）
# Parameters for exploration in continuous action space / 連続アクション空間での探索のためのパラメーター
theta = [
        0.1,0.1,0.1,0.1,0.1,0.1,0.1
        ] #len(theta) == num_action # multiplier
mu = [
    0.2,0.2,0.2,0.2,0.2,0.2,0.2
    ] #len(mu) == num_action #means
sigma = [
        0.2,0.2,0.2,0.1,0.1,0.2,0.2
        ] #len(sigma) == num_action # noise
train_period = 0 #(0/1) 0=練習なし　, 1=練習あり

#----------------------------------
# How many game to train 研究室
# トレーニングするゲームの数
#----------------------------------
num_episodes = 100
#----------------------------------
# Load reinforcement learning model 研究室
# 強化学習モデルの読み込み
#----------------------------------
base_num = 13000
#----------------------------------
# How many game to save neural network 研究室
# ニューラルネットワークを保存するゲームの数
#----------------------------------
save_iter = 1000

#====================================================================
# Function Part
# 機能部
#====================================================================

#----------------------------------
# Plot function: Function to plot a graph of objective-step in each game
# プロット関数：各ゲームの目的ステップのグラフをプロットする関数
#----------------------------------
log_capital = []
game_reward = [0]
now = str(datetime.datetime.now().strftime("_%Y-%m-%d_%H_%M_%S"))
def plotforgame(objective):
    objective = objective
    name_of_file = 'Game_{}_Initial_{}_Final_{}_Reward_{}.png'.format(
        counter+1,
        log_capital[0],
        log_capital[-1],
        round(game_reward[0],4))
    save_path = 'Log_{}-Step'.format(objective)+now+'/'
    if not os.path.exists('Log_{}-Step'.format(objective)+now):
        os.makedirs('Log_{}-Step'.format(objective)+now)
    name = os.path.join(save_path, name_of_file)
    plt.ylabel('{}'.format(objective))
    plt.xlabel("day")
    plt.plot(log_capital)
    plt.savefig(name)
    plt.close("all")

#----------------------------------
# Reinforcement Learning function: Function to run reinforcement learning loop for 1 game
# 強化学習機能：1ゲームの強化学習ループを実行する機能
#----------------------------------
def run(game,objective,train_period=1):
    objective = objective
    env1_test.reset()
    intcounter = 0
    keep = 0
    for_out  = []
    for_outz = []
    for_out_ob = []
    my_step = 1
    int_r = 0
    temp_mem = [None,None,None,None,None]
    r_execute = 0
    while env1_test.over != 1:


        try:
            sub_state = []
            sub_action =[]
            sub_nextstate = []
            sub_reward = []
            sub_done =[]


            if intcounter == 0:
                log_capital.append(round(env1_test.game.dojo.int_capital,2))
                intcounter = 1

            #----------------------------------
            # Reinforcement Learning Loop for 1 step in a game (研究室)
            # ゲームの1ステップの強化学習ループ
            s = env1_test.game._game_get_state(env1_test.game.game_step)

            sub_state.append(s)


            s[0].reshape((reinforcement_learning.num_state[0],reinforcement_learning.window,1))

            #s.reshape((reinforcement_learning.num_state,reinforcement_learning.window,1))

            s1 = np.asarray(s[0]).astype(np.float32)

            s[1].reshape((reinforcement_learning.num_state[1],1))

            #s.reshape((reinforcement_learning.num_state,reinforcement_learning.window,1))

            s2 = np.asarray(s[1]).astype(np.float32)
            # debug
            '''
            print(env1_test.game.dojo.hand_long)
            print(env1_test.game.dojo.long_token)
            print(env1_test.game.dojo.hand_short)
            print(env1_test.game.dojo.short_token)
            print(env1_test.game.dojo.int_capital)
            print(env1_test.game.dojo.capital)
            '''
            #print(env1_test.game.game_step)
            #print(env1_test.game.dojo.hand_long)
            #print(env1_test.game.dojo.hand_short)


            action = reinforcement_learning.act([s1,s2])
            #print(action[0])

            sub_action.append(action)
            #print(action)
            n_s,remember_token1,remember_token2 = env1_test.game._game_get_next_state(action)
            sub_nextstate.append(n_s)


            r,d,remember_token3 = env1_test.game._game_get_reward()
            sub_reward.append(r-int_r)
            int_r = r
            sub_done.append(d)
            # ----------------------------------------------------
            # Memory manipulated scheme
            '''
            if keep==0:
                if (remember_token1==1):
                    temp_mem = [sub_state[-1],sub_action[-1],0,None,0]
                    keep = 1
                    r_execute = (r-int_r)
                else:
                    pass
            elif keep==1:

                if (remember_token3 == 1):
                    temp_mem[2] = r-r_execute#sub_reward[-1]
                    temp_mem[3] = sub_nextstate[-1]
                    temp_mem[4] = sub_done[-1]
                    keep = 0
                else:
                    pass

            if type(temp_mem[3]) != type(None):
                reinforcement_learning.remember(temp_mem[0],temp_mem[1],temp_mem[2],temp_mem[3],temp_mem[4])
                temp_mem = [None,None,None,None,None]
            else:
                pass
            '''
            # ----------------------------------------------------
            # Memory not  manipulated scheme


            reinforcement_learning.remember(
                sub_state[-1],sub_action[-1],0,sub_nextstate[-1],0)
            reinforcement_learning.temprp[-1][2] = sub_reward[-1]
            reinforcement_learning.temprp[-1][4] = sub_done[-1]


            # ----------------------------------------------------
            #print(len(reinforcement_learning.temprp))
            #print(reinforcement_learning.temprp[-1][3])
            env1_test.game.reward_counter = sub_reward[-1]
            #----------------------------------
            # train the neural network agent / ニューラルネットワークエージェントの 練習
            if train_period==1:
                reinforcement_learning.train()
                # update the neural network agent / ニューラルネットワークエージェントを更新
                reinforcement_learning.update() # use for Double-DQN, actor critic and DDPG / Double-DQN、俳優評論家、DDPGに使用

            #----------------------------------
            # Add data for the graph / グラフのデータを追加する
            log_capital.append(round(env1_test.game.dojo.capital+env1_test.game.dojo.stock_capital,2))
            #game_reward[0] = env1_test.game.reward_counter
            game_reward[0] = r
            #----------------------------------
            # Print out result on the console / 結果をコンソールに出力する

            if env1_test.game.dojo.hand_position[-1][0] != 0:
                hand_stock = round(env1_test.game.dojo.hand_position[-1][0],6)
            elif env1_test.game.dojo.hand_position[-1][1] != 0:
                hand_stock = round(env1_test.game.dojo.hand_position[-1][1],6)
            else:
                hand_stock = None
            try:
                print('Step {} {} int {} {} now {}|{},{} (stockpricenow {}) Reward {}'.format(
                    env1_test.game.game_step,
                    objective,
                    round(env1_test.game.dojo.int_capital,6),
                    objective,
                    round(env1_test.game.dojo.capital,6),
                    hand_stock,
                    env1_test.game.dojo.hand_position[-1][4],
                    round(env1_test.game.dojo.data[10][env1_test.game.game_step+time_window-1],6),
                    round(env1_test.game.reward_counter,6)))
            except:
                env1_test.game.done_counter = 1

            #----------------------------------
            # Game move to nextstep and check if the end condition is met / ゲームは次のステップに移動し、終了条件が満たされているかどうかを確認します
            env1_test.game.step()
            env1_test.check_over()
        except:
            env1_test.game.done_counter = 1
            env1_test.game.step()
            env1_test.check_over()



    env1_test.reset()
    reinforcement_learning.check_mem()

# ----------------------------------
# Save and Restore Neural Network Agent
# ニューラルネットワークエージェントの保存と復元
# ----------------------------------
if base_num!= 0:
    base_name_of_Actor_pickle = "Actor_pickle.h5"
    base_name_of_Critic_pickle = "Critic_pickle.h5"
    base_pickle_path = '{}pickle_base/'.format(base_num)
    base_Actor_picklename = os.path.join(base_pickle_path, base_name_of_Actor_pickle)
    base_Critic_picklename = os.path.join(base_pickle_path, base_name_of_Critic_pickle)
    memory_name = "kioku_no_uta_vocaloid.txt"
    mem = os.path.join(base_pickle_path, memory_name)


# ----------------------------------
# Main program
# メインプログラム
# ----------------------------------

reinforcement_learning = DDPG_LSTM_head(lr,ep,epd,gamma,a_nn,c_nn,max_mem,num_ob,num_action,mu,theta,sigma,time_window) # Make an agent / エージェントを作る
if base_num!= 0: # Load model / モデルをロード
    try:
        reinforcement_learning.actor_model = load_model(base_Actor_picklename)
        reinforcement_learning.critic_model = load_model(base_Critic_picklename)
        reinforcement_learning.target_actor_model = load_model(base_Actor_picklename)
        reinforcement_learning.critic_target_model = load_model(base_Critic_picklename)
        print("Load model success!")
    except:
        print("No model file to restore")
counter = 0
while counter < num_episodes:

    err = None
    while err == None:


        try:
            data_rand = random.choice(os.listdir('max_100'))
            if str(data_rand)[-6:]=='us.txt':
                #data_dir = 'individual_stocks_5yr/'+str(data_rand)
                data_dir = 'max_100/'+str(data_rand)
                data = ichimoku(data_dir,len_data+time_window+26)
                dojo = Trade(data,init_cap) # define model / モデルを定義する
                err = 1
            else:
                err = None
        except:
            pass

    print('------------------------------')
    print(data_dir)
    print('Epsilon {}--------------------'.format(round(reinforcement_learning.ep,6)))
    game = Game6(dojo,time_window,stop_gain_multiplier,stop_loss_multiplier) # choose game /ゲームを選択 (研究室)
    env1_test = ENV(game) # Put the game in ENV / ENVにゲームを置く
    print('Episode{}'.format(counter+1)) # Print the Number of current game in console / 現在のゲームの数をコンソールに出力する
    run(counter+1,objective,train_period) # run reinforcement learning loop / 強化学習ループを実行する
    plotforgame(objective) # plot the graph after game end / ゲーム終了後にグラフをプロットする
    log_capital = [] # reset graph data / グラフデータをリセットする
    game_reward = [0] # reset reward / 報酬をリセット
    counter += 1 # game counter += 1
    gc.collect() #release unreferenced memory
    # reducing noise
    '''
    e = 0.99
    sigma = [sigma[0]*e,sigma[1]*e,sigma[2]*e,sigma[3]*e,sigma[4]*e,sigma[5]*e,sigma[6]*e,sigma[7]*e,sigma[8]*e,sigma[9]*e]

    reinforcement_learning.noise = []
    reinforcement_learning.create_noise(mu, theta, sigma)
    for i in range(len(reinforcement_learning.noise)):
        reinforcement_learning.noise[i].dt *= e
    '''


    if counter%save_iter == 0: # save neural network weigth every defined interval / ＃定義された間隔ごとにニューラルネットワークの重みを保存
        #reinforcement_learning.lr /= 10
        name_of_Actor_pickle = "Actor_pickle.h5"
        name_of_Critic_pickle = "Critic_pickle.h5"
        pickle_path = '{}pickle_base/'.format(counter)
        if not os.path.exists('{}pickle_base/'.format(counter)):
            os.makedirs('{}pickle_base/'.format(counter))
            Actor_picklename = os.path.join(pickle_path, name_of_Actor_pickle)
            Critic_picklename = os.path.join(pickle_path, name_of_Critic_pickle)
        Actor_picklename = os.path.join(pickle_path, name_of_Actor_pickle)
        Critic_picklename = os.path.join(pickle_path, name_of_Critic_pickle)
        reinforcement_learning.actor_model.save(Actor_picklename)
        reinforcement_learning.critic_model.save(Critic_picklename)
        # ------------------
        # memory
        # ------------------
        '''
        memory_name = "kioku_no_uta_vocaloid.txt"
        memory_name_picklename = os.path.join(pickle_path, memory_name)
        memory = open(memory_name_picklename, "w+")
        for i in range(len(reinforcement_learning.temprp)):
            memory.write("{},{},{},{},{} ".format(reinforcement_learning.temprp[i][0].tolist(),reinforcement_learning.temprp[i][1].tolist(),reinforcement_learning.temprp[i][2],reinforcement_learning.temprp[i][3].tolist(),reinforcement_learning.temprp[i][4]))
            memory.write("\n")
        memory.close()
        '''
        print("saved")

