
'''
==================================================================
Stock environemnt for training Trader LSTM file
This file contains environment for train agent using reinforcement learning
    ENV    : contains Game class
    Game1 : Maximizing capital : DDPG with LSTM head 3 action short long nothing

==================================================================
'''
import numpy as np
import random

random.seed(3407)
np.random.seed(3407)


'''
======================================================
Helper functions
======================================================
'''
def state_data(data,t,time_window,hand,cap,int_cap,now,finalstep,fee_c,p_max,p_min):

    price_high = data[1][t-time_window+1:t+1]#['Close'][t-time_window:t]
    price_low = data[2][t-time_window+1:t+1]#['volume'][t-time_window:t]
    price_close = data[3][t-time_window+1:t+1]#['Close'][t-time_window:t]
    volume_data = data[4][t-time_window+1:t+1]#['volume'][t-time_window:t]
    kijun_sen = data[5][t-time_window+1:t+1]#['kijun_sen'][t-time_window:t]
    tenkan_sen = data[6][t-time_window+1:t+1]#['tenkan_sen'][t-time_window:t]
    chikou_span = data[7][t-time_window+1:t+1]#['chikou_span'][t-time_window:t]
    senkou_span_A = data[8][t-time_window+1:t+1]#['senkou_span_A'][t-time_window:t]
    senkou_span_B = data[9][t-time_window+1:t+1]#['senkou_span_B'][t-time_window:t]
    epsilon = 1e-10
    # normalized only within window

    price_high = (price_high-min(price_high))/(max(price_high)-min(price_high)+epsilon)
    price_low = (price_low-min(price_low))/(max(price_low)-min(price_low)+epsilon)
    price_close = (price_close-min(price_close))/(max(price_close)-min(price_close)+epsilon)
    volume_data = (volume_data-min(volume_data))/(max(volume_data)-min(volume_data)+epsilon)
    kijun_sen = (kijun_sen-min(kijun_sen))/(max(kijun_sen)-min(kijun_sen)+epsilon)
    tenkan_sen = (tenkan_sen-min(tenkan_sen))/(max(tenkan_sen)-min(tenkan_sen)+epsilon)
    chikou_span = (chikou_span-min(chikou_span))/(max(chikou_span)-min(chikou_span)+epsilon)
    senkou_span_A = (senkou_span_A-min(senkou_span_A))/(max(senkou_span_A)-min(senkou_span_A)+epsilon)
    senkou_span_B = (senkou_span_B-min(senkou_span_B))/(max(senkou_span_B)-min(senkou_span_B)+epsilon)

    # Normalized data using history time window
    # indicators state
    # P_C price lower than the cloud(senkou_span_A,senkou_span_B) (0/1)
    # P_K price lower than kijunsen (0/1)
    # C_T chikou_span lower than tenkan_sen (0/1)
    P_C = data[11][t-time_window+1:t+1]
    P_K = data[12][t-time_window+1:t+1]
    C_T = data[13][t-time_window+1:t+1]
    norm_close = data[14][t-time_window+1:t+1]

    A_o_B = data[15][t-time_window+1:t+1]
    T_K = data[16][t-time_window+1:t+1]

    diff_1 = data[17][t-time_window+1:t+1]
    diff_5 = data[18][t-time_window+1:t+1]
    diff_10 = data[19][t-time_window+1:t+1]


    # Not LSTM -------------------------------------------------------------------

    # Normalized data using history time window
    # indicators state
    # P_C price lower than the cloud(senkou_span_A,senkou_span_B) (0/1)
    # P_K price lower than kijunsen (0/1)
    # C_T chikou_span lower than tenkan_sen (0/1)
    point_P_C = data[11][t]
    point_P_K = data[12][t]
    point_C_T = data[13][t]
    point_norm_close = data[14][t]

    point_A_o_B = data[15][t]
    point_T_K = data[16][t]

    diff_1_hand = data[17][t]
    diff_5_hand = data[18][t]
    diff_10_hand = data[19][t]
    # ---------------------------------------------------------------------------

    hand1 = 0
    hand2 = 0

    if (hand[0] != 0) and (hand[1] == 0):
        hand1 = (hand[0]-p_min)/(p_max-p_min+epsilon)
        hand2 = 0
    elif (hand[0] == 0) and (hand[1] != 0):
        hand1 = 0
        hand2 = (hand[1]-p_min)/(p_max-p_min+epsilon)

    hand3 = (hand[2]-p_min)/(p_max-p_min+epsilon)
    hand4 = (hand[3]-p_min)/(p_max-p_min+epsilon)
    hand5 = hand[4]/100


    cap = cap/(int_cap) # end at cap >= 1.2*int_cap
    price_per_cap =  data[14][t]/cap

    good_long_sell  = -1
    #bad_long_sell   = 0
    good_short_sell = -1
    #bad_short_sell  = 0
    if (hand[0] != 0) and (hand[1] == 0):
        if data[10][t] > hand[0]:
            good_long_sell = (data[10][t]-hand[0])/hand[0]

    elif (hand[0] == 0) and (hand[1] != 0):
        if data[10][t] < hand[1]:
            good_short_sell = (hand[1]-data[10][t])/hand[1]

    now_is_higher = 0
    if data[10][t] > data[10][t-1]:
        now_is_higher = 1

    timer = now/finalstep
    f_count = fee_c/finalstep

    return [price_high,price_low,price_close,volume_data,kijun_sen,tenkan_sen,chikou_span,senkou_span_A,senkou_span_B,P_C,P_K,C_T,A_o_B,T_K],norm_close,[hand1,hand2,hand3,hand4,now_is_higher]

'''
======================================================
CLASS PART
======================================================
'''
# Environment class, contain Game class. Do not change / 環境クラス、ゲームクラスを含みます。 変えないで
class ENV:
    def __init__(self,game):
        self.name = 'STOCK_ENV'
        self.game = game
        self.over = 0
        #=======================
        #State Action Reward Next_State Done
        #=======================
        self.state = self.game.state
        self.action = self.game.action #get action from rl or data file
        self.reward = self.game.reward
        self.next_state = self.game.next_state
        self.done = self.game.done
        #=======================
        #Output Action
        #=======================
        self.output = [] #output = [St,at,rt,St+1,Done]

    def check_over(self):
        if self.game.done_counter == 1:
            self.over = 1
        else:
            pass

    def reset(self):
        self.over = 0
        self.game.reset()
        self.state = self.game.state
        self.action = self.game.action
        self.reward = self.game.reward
        self.next_state = self.game.next_state
        self.done = self.game.done
        self.output = []

    def gen_output(self):
        '''
        Output_list
        1. replay buffer = [St,at,rt,St+1,Done]
        2. other format(.txt)
        3. render
        '''
        # reset output to empty list
        for i in range(self.num_agents):
            x = []
            # output = [St,at,rt,St+1,Done] one replay buffer
            x.append(self.state[-1])
            x.append(self.action[-1])
            x.append(self.reward[-1])
            x.append(self.next_state[-1])
            x.append(self.done[-1])
            self.output.append(x)

    def save_output(self):
        # save replaybufferfile as txt, csv
        pass



#=============================================================================
# GAME 6
class Game6:
    def __init__(self,dojo,time_window,stop_gain_multiplier,stop_loss_multiplier,tell_action=False):
        self.name = 'GAME 6'
        self.description = 'DDPG with LSTM head 3 outputs(short-long-sell)'
        self.objective = 'Maximizing capital using penalty '
        self.tell_action =tell_action
        self.time_window = time_window # how many time step agent can view (maximum 100 for now)
        self.dojo = dojo

        self.game_step = 1
        self.end_step = 60#len(self.dojo.data[9])-self.time_window-26 # when will the game end
        # reduce dojo data
        if len(self.dojo.data[10])>=1000:
            ran = random.randrange(0,len(self.dojo.data[10])-self.time_window-26-260-150)
            #self.dojo.data = self.dojo.data[:][ran:ran+self.end_step+1]
            new_data = []
            for i in range(len(self.dojo.data)):
                new_data.append(self.dojo.data[i][ran:ran+self.end_step+150])
                #self.dojo.data[i] = self.dojo.data[i][ran:ran+self.end_step+1]
                #print(self.dojo.data[i][ran:ran+self.end_step+1])
            self.dojo.data = new_data
        else:
            pass

        self.fee = 10 #usd
        self.fee_token = 0
        self.stop_gain_multiplier = stop_gain_multiplier
        self.stop_loss_multiplier = stop_loss_multiplier
        self.prev_capital = self.dojo.int_capital
        self.fee_count = 0

        self.short_point = 10000
        self.long_point = 0

        self.cap_now = self.dojo.int_capital
        self.cap_next = 0

        self.best_cap = self.dojo.int_capital
        self.gain_ratio = 0.001
        self.cut_long = 0
        self.cut_short = 0

        self.remember_cut = 0

        self.has_hand = 0
        #=======================
        #State Action'
        # for MADDPG
        #=======================
        self.state = []
        self.action = [] #get action from rl or data file
        self.reward = 0
        self.reward_counter = 0
        self.next_state = []
        self.done = []

        #=======================
        # Game rules
        #=======================
        self.done_counter = 0

        print('-------------------------------------------------------')
        print(self.description)
        print(self.objective)
        print('GAME WILL BE ENDED AFTER {} STEP'.format(self.end_step))
        print('-------------------------------------------------------')

    def _game_get_state(self,step):

        state_dat,norm_close,state_hand = state_data(self.dojo.data,step+self.time_window,self.time_window,self.dojo.hand_position[-1],self.dojo.capital+self.dojo.stock_capital,self.dojo.int_capital,self.game_step,self.end_step,self.fee_count,self.dojo.price_max,self.dojo.price_min)


        state_dat = np.array(state_dat).transpose() # at time t=0

        #state_hand.append(self.dojo.hand_time/self.end_step)
        bad_handtime = 0
        if self.dojo.hand_time >= 5:
            bad_handtime = 1
        bad_blanktime = 0
        if self.dojo.blank_time >= 5:
            bad_blanktime = 1
        state_hand.append(bad_handtime)
        state_hand.append(bad_blanktime)
        state_hand.append(self.dojo.hand_time/self.end_step)
        state_hand.append(self.dojo.blank_time/self.end_step)

        #self.dojo.check_margin(self.dojo.data,self.game_step+self.time_window)
        margin_input = (self.dojo.capital+self.dojo.stock_capital)/(2*self.dojo.int_capital)

        state_hand.append(margin_input)

        state_hand = np.array(state_hand)

        state = [state_dat,state_hand]
        #print('-------------------------------------')
        #print('state {}'.format(norm_close))

        return state

    def _game_get_next_state(self,action):

        #print(num)
        #print(num)
        #print(self.game_step)
        #print(self.dojo.hand_position)
        # action[0][0],action[0][1] : short-long (sigmoid)
        # action[0][2] : (short-long)volume (sigmoid)
        # action[0][3] : sell (sigmoid)

        for i in range(len(action[0])):
            if action[0][i] > 1:
                action[0][i] = 1
            elif action[0][i] < 0:
                action[0][i] = 0

        num = [action[0][0],action[0][1],action[0][2],action[0][3],action[0][4]]
        #self.cut_long = action[0][3]
        #self.cut_short = action[0][4]
        #short_long = [action[0][0],action[0][1],action[0][3],action[0][4]]
        decision_position = np.argmax(num)
        #sell_short_long = [action[0][2],action[0][3]]
        #decision_sell = action[0][2]
        # volume
        '''
        try:
            if action[0][2]>=1:
                volume = int(100)
            elif action[0][2]>0:
                volume = int(action[0][2]*100)
            else:
                volume = int(10)

        except:
            volume = int(10)

        if volume <= 0:
            volume = int(10)
        '''

        #volume = 50


        #print(stop_gain_t)
        #print(stop_loss_t)
        #position_stop_gain = max([stop_gain_t,0.0001])*self.dojo.data[9][self.game_step] # percentage of current proce
        #position_stop_loss = max([stop_loss_t,0.0001])*self.dojo.data[9][self.game_step] # percentage of current proce

        position_stop_gain = 1.05 # percentage of current proce
        position_stop_loss = 1.01 # percentage of current proce

        #print(position_stop_gain)
        #print(position_stop_loss)
        # next_state = f(action)
        # Interprete action
        remember_token_1 = 0
        remember_token_2 = 0

        if self.dojo.hand_time != 0:
            self.dojo.hand_time += 1
        if self.dojo.blank_time != 0:
            self.dojo.blank_time += 1

        if decision_position ==0:
            if ((self.dojo.hand_position[-1][0] == 0) and (self.dojo.hand_position[-1][1] == 0)):
                #volume = min([int(abs(action[0][4])*75),75])
                volume = int((self.dojo.capital)/self.dojo.data[10][self.game_step+self.time_window-0])
                self.gain_ratio = action[0][0]
                self.cut_short = action[0][0]/2
                self.cut_long = action[0][0]/2
                # long
                self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,1,position_stop_gain,position_stop_loss,volume)
                self.has_hand = 1
                self.prev_capital = self.dojo.capital+self.dojo.stock_capital
                #self.fee_token = 1
                remember_token_1 = 1
                self.dojo.blank_time = 0
                self.dojo.hand_time = 1

            else:
                self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,99,0,0,0)
                remember_token_2 = 0

        elif decision_position ==1:
            if ((self.dojo.hand_position[-1][0] == 0) and (self.dojo.hand_position[-1][1] == 0)):
                #volume = min([int(abs(action[0][4])*75),75])
                volume = int((self.dojo.capital)/self.dojo.data[10][self.game_step+self.time_window-0])
                self.gain_ratio = action[0][1]
                self.cut_short = action[0][1]/2
                self.cut_long = action[0][1]/2
                # SHORT
                self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,0,position_stop_gain,position_stop_loss,volume)
                self.has_hand = 1
                self.prev_capital = self.dojo.capital+self.dojo.stock_capital
                #self.fee_token = 1
                remember_token_1 = 1
                self.dojo.blank_time = 0
                self.dojo.hand_time = 1
            else:
                self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,99,0,0,0)
                remember_token_2 = 0

        elif decision_position ==2:

            if self.game_step>=3:
                if (self.dojo.hand_position[self.game_step-1][0] != 0) and (self.dojo.hand_position[self.game_step-1][1] == 0):
                    #print('-------------------------------------')
                    #print('Sell Hand position {}'.format(self.dojo.hand_position[self.game_step-2]))
                    #print('Current price {}'.format(self.dojo.data[9][self.game_step+self.time_window-1]))
                    #print('Time {}'.format(self.game_step+self.time_window-1))
                    # Has long position, ready to sell long position

                    # current price = self.dojo.data[9][self.game_step]
                    # position price = self.dojo.hand_position[self.game_step-1][0]
                    # profit threshold = self.dojo.hand_position[self.game_step-1][2]
                    vol = self.dojo.hand_position[self.game_step-1][-1]
                    s = self.dojo.sell_position(self.dojo.data,self.game_step+self.time_window-1,1)
                    self.has_hand = 0
                    print('long sell at price {}'.format(s))
                    self.fee_token = 1
                    self.fee_count += 1
                    #self.reward -= 10*0.15/self.end_step
                    self.short_point = 10000
                    self.long_point = 0
                    self.remember_cut = 1

                    self.dojo.blank_time = 1
                    self.dojo.hand_time = 0

                    self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,99,0,0,0)
                    remember_token_2 = 0

                else:
                    self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,99,0,0,0)
                    remember_token_2 = 0
            else:
                self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,99,0,0,0)
                remember_token_2 = 0


        elif decision_position ==3:
            if self.game_step>=3:
                if (self.dojo.hand_position[self.game_step-1][0] == 0) and (self.dojo.hand_position[self.game_step-1][1] != 0):
                    #print('-------------------------------------')
                    #print('Sell Hand position {}'.format(self.dojo.hand_position[self.game_step-2]))
                    #print('Current price {}'.format(self.dojo.data[9][self.game_step+self.time_window-1]))
                    #print('Time {}'.format(self.game_step+self.time_window-1))
                    # Has long position, ready to sell long position

                    # current price = self.dojo.data[9][self.game_step]
                    # position price = self.dojo.hand_position[self.game_step-1][1]
                    # profit threshold = self.dojo.hand_position[self.game_step-1][2]

                    # 2. if current price reach cut loss threshold

                    vol = self.dojo.hand_position[self.game_step-1][-1]
                    s = self.dojo.sell_position(self.dojo.data,self.game_step+self.time_window-1,0)
                    self.has_hand = 0
                    print('short sell at price {}'.format(s))

                    self.fee_token = 1
                    self.fee_count += 1
                    #self.reward -= 10*0.15/self.end_step
                    self.short_point = 10000
                    self.long_point = 0
                    self.remember_cut = 1

                    self.dojo.blank_time = 1
                    self.dojo.hand_time = 0

                    self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,99,0,0,0)
                    remember_token_2 = 0

                else:
                    self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,99,0,0,0)
                    remember_token_2 = 0
            else:
                self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,99,0,0,0)
                remember_token_2 = 0

        else:
            self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,99,0,0,0)
            remember_token_2 = 0



        '''
        # BASE ------------------------------------------------------------------------------------------------
        if ((self.dojo.hand_position[-1][0] == 0) and (self.dojo.hand_position[-1][1] == 0)):
            if decision_position ==0:
                #volume = min([int(abs(action[0][4])*75),75])
                volume = int((self.dojo.capital*0.8)/self.dojo.data[10][self.game_step+self.time_window-0])
                self.gain_ratio = 0.001*action[0][0]
                # long
                self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,1,position_stop_gain,position_stop_loss,volume)
                #self.prev_capital = self.dojo.capital+self.dojo.stock_capital
                #self.fee_token = 1
                remember_token_1 = 1
            elif decision_position ==1:
                #volume = min([int(abs(action[0][4])*75),75])
                volume = int((self.dojo.capital*0.8)/self.dojo.data[10][self.game_step+self.time_window-0])
                self.gain_ratio = 0.001*action[0][1]
                # SHORT
                self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,0,position_stop_gain,position_stop_loss,volume)
                #self.prev_capital = self.dojo.capital+self.dojo.stock_capital
                #self.fee_token = 1
                remember_token_1 = 1
            elif decision_position ==2:
                self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,99,0,0,0)
                remember_token_1 = 0

            else:
                self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,99,0,0,0)
                remember_token_2 = 0
        else:
            self.dojo.buy_position(self.dojo.data,self.game_step+self.time_window-0,99,0,0,0)
            remember_token_2 = 0
        # -----------------------------------------------------------------------------------------------------
        '''

        self.game_step += 1

        state_dat,norm_close,state_hand = state_data(self.dojo.data,self.game_step+self.time_window,self.time_window,self.dojo.hand_position[-1],self.dojo.capital+self.dojo.stock_capital,self.dojo.int_capital,self.game_step,self.end_step,self.fee_count,self.dojo.price_max,self.dojo.price_min)

        state_dat = np.array(state_dat).transpose() # at time t=0

        #state_hand.append(self.dojo.hand_time/self.end_step)
        bad_handtime = 0
        if self.dojo.hand_time >= 5:
            bad_handtime = 1
        bad_blanktime = 0
        if self.dojo.blank_time >= 5:
            bad_blanktime = 1

        state_hand.append(bad_handtime)
        state_hand.append(bad_blanktime)
        state_hand.append(self.dojo.hand_time/self.end_step)
        state_hand.append(self.dojo.blank_time/self.end_step)

        #self.dojo.check_margin(self.dojo.data,self.game_step+self.time_window)
        margin_input = (self.dojo.capital+self.dojo.stock_capital)/(2*self.dojo.int_capital)

        state_hand.append(margin_input)

        state_hand = np.array(state_hand) # at time t=0

        next_state = [state_dat,state_hand]
        #next_state,norm_close = np.array(state_data(self.dojo.data,self.game_step+self.time_window,self.time_window)).transpose()


        #print('-------------------------------------')
        #print('Game step {}'.format(self.game_step))
        #print('Time window {}'.format(self.time_window))
        #print('Current price {}'.format(self.dojo.data[9][self.game_step+self.time_window-1]))
        #print('Hand position {}'.format(self.dojo.hand_position[-1]))
        #print('next state {}'.format(norm_close))

        return next_state,remember_token_1,remember_token_2

    # Function to calculate reward for each agent / 各エージェントの報酬を計算する機能
    def _game_get_reward(self):
        #print(self.game_step)
        #print(len(self.dojo.hand_position))
        #print(len(self.dojo.data[9]))
        '''
        20210706
        problem is at self.dojo.data[9][self.game_step+self.time_window-1]
        sometime len(self.dojo.data[9]) is == self.game_step+self.time_window-1
        the problem is Is it because of data or my code?

        '''
        good_job = 0
        bad_job = 0
        vol = 0



        reward_action = 0


        ratio = 0.005


        if self.game_step == self.end_step: # Check if game is end / ゲームが終了したかどうかを確認する
        #print(self.dojo.hand_position[self.game_step-1])
            if (self.dojo.hand_position[self.game_step-1][0] != 0) and (self.dojo.hand_position[self.game_step-1][1] == 0):
                #vol = self.dojo.hand_position[self.game_step-1][-1]
                s = self.dojo.sell_position(self.dojo.data,self.game_step+self.time_window-1,1)
                self.has_hand = 0
                print('end sell long at price {}, clear port -------------------------- '.format(s))
                self.fee_token = 1
                self.fee_count += 1
                self.short_point = 10000
                self.long_point = 0
                self.remember_cut = 1


            elif (self.dojo.hand_position[self.game_step-1][0] == 0) and (self.dojo.hand_position[self.game_step-1][1] != 0):
                #vol = self.dojo.hand_position[self.game_step-1][-1]
                s = self.dojo.sell_position(self.dojo.data,self.game_step+self.time_window-1,0)
                self.has_hand = 0
                print('end sell short at price {}, clear port -------------------------- '.format(s))
                self.fee_token = 1
                self.fee_count += 1
                self.short_point = 10000
                self.long_point = 0
                self.remember_cut = 1

            self.done_counter = 1

        #self.dojo.check_margin(self.dojo.data,self.game_step+self.time_window-1)


        if self.remember_cut ==1:
            self.dojo.capital -= 2*self.fee
            self.cap_next = self.dojo.capital
            #print('debug-------------------')
            #print(self.cap_next)
            #print(self.cap_now)
            self.reward += self.cap_next-self.cap_now
            self.remember_cut = 0
            self.cap_now = self.cap_next

        self.dojo.check_margin(self.dojo.data,self.game_step+self.time_window-2)
        # -----------------------------------------------------------------------------


        #self.reward += self.dojo.capital-self.prev_capital

        '''
        if self.dojo.capital+self.dojo.stock_capital > self.dojo.int_capital*1.0075:
            self.reward += 3
        elif self.dojo.capital+self.dojo.stock_capital > self.dojo.int_capital*1.005:
            self.reward += 2
        elif self.dojo.capital+self.dojo.stock_capital > self.dojo.int_capital:
            self.reward += 1
        elif self.dojo.capital+self.dojo.stock_capital > self.dojo.int_capital*0.995:
            self.reward -= 1
        elif self.dojo.capital+self.dojo.stock_capital > self.dojo.int_capital*0.99:
            self.reward -= 2
        else:
            self.reward -= 3
        '''

        # Margin 1st
        if self.dojo.margin_capital > 0:
            self.reward += 2
        elif self.dojo.margin_capital <= 0:
            self.reward -= 1
        '''
        # Stock 1st
        if self.dojo.capital+self.dojo.stock_capital > self.dojo.int_capital*1.005:
            self.reward += 4
        elif self.dojo.capital+self.dojo.stock_capital <= self.dojo.int_capital*1.005:
            self.reward -= 4
        else:
            self.reward -= 2
        '''
        # maintaining 2nd
        if self.dojo.capital+self.dojo.stock_capital <= self.prev_capital:
            self.reward -= 3
        elif self.dojo.capital+self.dojo.stock_capital > self.prev_capital:
            self.reward += 1
        # penalty for static 2nd
        if self.dojo.hand_time >= 5:
            self.reward -= 0.5
        if self.dojo.blank_time >= 5:
            self.reward -= 0.5

        #self.reward += ((self.dojo.capital+self.dojo.stock_capital)-self.dojo.int_capital)



        #self.dojo.check_margin(self.dojo.data,self.game_step+self.time_window-2)
        '''
        if remember_cut==1:

        '''
        '''
        factor = 2
        if remember_cut==1:
            if (self.dojo.capital+self.dojo.stock_capital-self.prev_capital) <= 0:
                #self.reward -= factor*1#*100*(self.dojo.capital+self.dojo.stock_capital-self.dojo.int_capital)/self.dojo.int_capital
                self.reward += 1000*(self.dojo.capital+self.dojo.stock_capital-self.prev_capital)/self.prev_capital
            elif (self.dojo.capital+self.dojo.stock_capital-self.prev_capital) > 0:
                self.reward += 1000*(self.dojo.capital+self.dojo.stock_capital-self.prev_capital)/self.prev_capital
                #self.reward += 1#100*(self.dojo.capital+self.dojo.stock_capital-self.dojo.int_capital)/self.dojo.int_capital
            else:
                pass
        '''


        #self.reward -= 0.001
        #self.reward += (self.dojo.capital+self.dojo.stock_capital-self.dojo.int_capital)/self.dojo.int_capital
        '''
        # End if performance drop
        if (self.dojo.capital+self.dojo.stock_capital < self.prev_capital):
            #self.reward -= 1
            self.done_counter = 1
            print('Keep Improving ---------------------------------------------')
        '''

        # ===========================================================
        # TEST SCHMEME 1
        '''
        over_condition= 0
        # Adjust only in the test
        if self.dojo.capital+self.dojo.stock_capital >= self.dojo.int_capital*1.005:
            self.reward += 10
            self.done_counter = 1
            over_condition = 1
            print('Good Job -------------------------------------------------')

        # Add only in the test
        elif self.dojo.capital+self.dojo.stock_capital <= self.dojo.int_capital*0.995:
            #self.reward -= abs(self.reward)*((self.end_step-self.game_step)/self.end_step)
            self.reward -= 10
            self.done_counter = 1
            over_condition = 1
            #print('Bad  Job -------------------------------------------------')

        elif self.dojo.capital+self.dojo.stock_capital < self.prev_capital:
            if self.prev_capital >= self.dojo.int_capital*1.00:
                self.done_counter = 1
                over_condition = 1
                print('OK Job -------------------------------------------------')
        '''
        # ===========================================================
        # TEST SCHMEME 2

        over_condition= 0
        # Adjust only in the test
        if self.dojo.capital+self.dojo.stock_capital >= self.dojo.int_capital*1.005:
            self.reward += 10
            self.done_counter = 1
            over_condition = 1
            print('Good Job -------------------------------------------------')

        # Add only in the test
        elif self.dojo.capital+self.dojo.stock_capital <= self.dojo.int_capital*0.995:
            #self.reward -= abs(self.reward)*((self.end_step-self.game_step)/self.end_step)
            self.reward -= 10
            self.done_counter = 1
            over_condition = 1
            #print('Bad  Job -------------------------------------------------')

        elif self.dojo.capital+self.dojo.stock_capital < self.prev_capital:
            if self.prev_capital >= self.dojo.int_capital*1.00:
                self.done_counter = 1
                over_condition = 1
                print('OK Job -------------------------------------------------')

        if self.dojo.capital+self.dojo.stock_capital+self.dojo.margin_capital <= self.dojo.int_capital*0.99:
            #self.reward -= abs(self.reward)*((self.end_step-self.game_step)/self.end_step)
            self.reward -= 10
            self.done_counter = 1
            over_condition = 1
            print('Bad  Job -------------------------------------------------')



        # ===========================================================
        if over_condition == 1: # Check if game is end / ゲームが終了したかどうかを確認する
        #print(self.dojo.hand_position[self.game_step-1])
            if (self.dojo.hand_position[self.game_step-1][0] != 0) and (self.dojo.hand_position[self.game_step-1][1] == 0):
                #vol = self.dojo.hand_position[self.game_step-1][-1]
                s = self.dojo.sell_position(self.dojo.data,self.game_step+self.time_window-1,1)
                self.has_hand = 0
                print('sell long at price {}, clear port -------------------------- '.format(s))
                self.fee_token = 1
                self.fee_count += 1
                self.short_point = 10000
                self.long_point = 0
                self.remember_cut = 1


            elif (self.dojo.hand_position[self.game_step-1][0] == 0) and (self.dojo.hand_position[self.game_step-1][1] != 0):
                #vol = self.dojo.hand_position[self.game_step-1][-1]
                s = self.dojo.sell_position(self.dojo.data,self.game_step+self.time_window-1,0)
                self.has_hand = 0
                print('sell short at price {}, clear port -------------------------- '.format(s))
                self.fee_token = 1
                self.fee_count += 1
                self.short_point = 10000
                self.long_point = 0
                self.remember_cut = 1

        #print('previous step {}'.format( self.prev_capital))
        if self.dojo.capital+self.dojo.stock_capital >= self.dojo.int_capital:
            self.prev_capital = self.dojo.capital+self.dojo.stock_capital

        #print('now {}'.format(self.dojo.capital+self.dojo.stock_capital))


        return round(self.reward,2),self.done_counter,self.remember_cut

    def reset(self):
        #=======================
        #State Action'
        # for MADDPG
        #=======================
        self.state = []
        self.action = [] #get action from rl or data file
        self.reward = 0
        self.reward_counter = 0
        self.next_state = []
        self.done = []
        self.prev_capital = self.dojo.int_capital
        self.game_step = 1
        self.gain_ratio = 0.001

        self.short_point = 10000
        self.long_point = 0
        #=======================
        # Game rules
        #=======================
        self.done_counter = 0
        self.fee_token = 0
        self.fee_count  = 0

        self.cap_now = self.dojo.int_capital
        self.cap_next = 0
        self.best_cap = self.dojo.int_capital
        self.cut_long = 0
        self.cut_short = 0

        self.has_hand = 0
        self.remember_cut = 0
    def step(self):
        self.state = self.next_state
        self.action = []
        #self.reward = 0
        self.next_state = []
        self.done = []

        #self.game_step += 1


