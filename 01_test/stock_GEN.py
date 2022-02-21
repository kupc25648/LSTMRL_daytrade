'''
==================================================================
Stock generation for training Trader LSTM file
==================================================================
'''
import numpy as np
np.random.seed(3407)



from sklearn.preprocessing import MinMaxScaler as min_max_scaler


class Trade:
    def __init__(self,data,capital):
        self.data = np.array(data).transpose()

        self.hand_long = [None]
        self.long_token = 0 #0/1 0:Free, 1:Full(cannot buy)
        self.hand_short = [None]
        self.short_token = 0 #0/1 0:Free, 1:Full(cannot buy)
        self.stop_gain = 0 # percentage of Pt ( maximum =100%*multiplier)
        self.stop_loss = 0 # percentage of Pt ( maximum =100%*multiplier)
        self.hand_position = [[0,0,0,0,0]]
        self.position_token = 0
        self.int_capital = capital
        self.capital = capital
        self.stock_capital = 0
        self.margin_capital = 0
        self.hand_time = 0
        self.blank_time = 0
        self.price_max = 0
        self.price_min = 0

        self.price_mean = 0
        self.price_std = 0

        self.norm_data()

    def norm_data(self):
        #self.data[3] = (self.data[3]-self.data[3].min())/(self.data[3].max()-self.data[3].min())
        #self.data[4] = (self.data[4]-self.data[4].min())/(self.data[4].max()-self.data[4].min())
        #self.data[5] = (self.data[5]-self.data[5].min())/(self.data[5].max()-self.data[5].min())
        #self.data[6] = (self.data[6]-self.data[6].min())/(self.data[6].max()-self.data[6].min())
        #self.data[7] = (self.data[7]-self.data[7].min())/(self.data[7].max()-self.data[7].min())
        #self.data[8] = (self.data[8]-self.data[8].min())/(self.data[8].max()-self.data[8].min())
        #self.data[9] = (self.data[9]-self.data[9].min())/(self.data[9].max()-self.data[9].min())
        #small = 1e-10

        #print(self.data[15].min())
        #print(self.data[15].max())
        #print('--------')
        #print(self.data[15])
        # self.data[15] = (self.data[15]-self.data[15].min())/(self.data[15].max()-self.data[15].min())
        #self.data[16] = (self.data[16]-self.data[16].min())/(self.data[16].max()-self.data[16].min())
        #self.data[17] = (self.data[17]-self.data[17].min())/(self.data[17].max()-self.data[17].min())

        self.price_max = self.data[10].max()
        self.price_min = self.data[10].min()

        #self.data[1:,] = min_max_scaler.fit_transform(self.data[1:,])
        '''
        for i in range(len(self.data)-1):
            for j in range(len(self.data[1+i])):
                self.data[1+i:][j] = (self.data[1+i][j]-self.data[1+i].min(axis=1)[i]+1e-10)/(self.data[1+i].max(axis=1)[i]-self.data[1+i].min(axis=1)[i]+1e-10)
        '''
        #self.data[1:] = (self.data[1:]-self.data[1:].min(axis=1))/(self.data[1:].max(axis=1)-self.data[1:].min(axis=1))
    def norm_data_zero_mean(self):
        #self.data[3] = (self.data[3]-self.data[3].mean())/self.data[3].std()
        #self.data[4] = (self.data[4]-self.data[4].mean())/self.data[4].std()
        #self.data[5] = (self.data[5]-self.data[5].mean())/self.data[5].std()
        #self.data[6] = (self.data[6]-self.data[6].mean())/self.data[6].std()
        #self.data[7] = (self.data[7]-self.data[7].mean())/self.data[7].std()
        #self.data[8] = (self.data[8]-self.data[8].mean())/self.data[8].std()
        #self.data[9] = (self.data[9]-self.data[9].mean())/self.data[9].std()
        #small = 1e-10

        #print(self.data[15].min())
        #print(self.data[15].max())
        #print('--------')
        #print(self.data[15])
        # self.data[15] = (self.data[15]-self.data[15].min())/(self.data[15].max()-self.data[15].min())
        #self.data[16] = (self.data[16]-self.data[16].min())/(self.data[16].max()-self.data[16].min())
        #self.data[17] = (self.data[17]-self.data[17].min())/(self.data[17].max()-self.data[17].min())

        self.price_max = self.data[10].max()
        self.price_min = self.data[10].min()

        self.price_mean = self.data[10].mean()
        self.price_std = self.data[10].std()

        #self.data[1:,] = min_max_scaler.fit_transform(self.data[1:,])
        '''
        for i in range(len(self.data)-1):
            for j in range(len(self.data[1+i])):
                self.data[1+i:][j] = (self.data[1+i][j]-self.data[1+i].min(axis=1)[i]+1e-10)/(self.data[1+i].max(axis=1)[i]-self.data[1+i].min(axis=1)[i]+1e-10)
        '''
        #self.data[1:] = (self.data[1:]-self.data[1:].min(axis=1))/(self.data[1:].max(axis=1)-self.data[1:].min(axis=1))

    def check_margin(self,data,t):
        self.margin_capital = 0
        margin = 0
        # there is hand long
        if (self.hand_position[-1][0]!=0):
            margin = (data[10][t]-self.hand_position[-1][0])*self.hand_position[-1][4]
        # there is hand short
        elif (self.hand_position[-1][1]!=0):
            margin = (self.hand_position[-1][0]-data[10][t])*self.hand_position[-1][4]
        else:
            pass
        self.margin_capital = margin



    def buy_position(self,data,t,long_pos,stop_gain_percentage,stop_loss_percentage,volume):
        # stop_gain = percentage of positioning price
        # stop_loss = percentage of positioning price


        if self.position_token==0:
            if long_pos==1:
                # buy long (expecting price to increase)
                # 1. Check if self.hand_position[t-1] is avialble

                if (self.hand_position[-1][0] == 0) and (self.hand_position[-1][1] == 0):
                    # if avialble
                    self.capital -= data[10][t]*volume # reduce capital
                    self.position_token = 1 # what is this?
                    stop_gain = stop_gain_percentage*data[10][t]
                    stop_loss = stop_loss_percentage*data[10][t]
                    self.hand_position.append([data[10][t],0,stop_gain,stop_loss,volume]) #[Long,Short]
                    self.stock_capital = data[10][t]*volume
                    self.hand_time = 0
                    self.blank_time = 0
                    print('buy long position at price {}'.format(data[10][t]))

                else:
                    # if not avialble
                    self.hand_position.append(self.hand_position[-1]) # do nothing

            elif long_pos==0:
                # buy short (expecting price to decrease)
                # 1. Check if self.hand_position[t-1] is avialble
                if (self.hand_position[-1][0] == 0) and (self.hand_position[-1][1] == 0):
                    # if avialble
                    self.capital -= data[10][t]*volume # reduce capital
                    self.position_token = 1 # what is this?
                    stop_gain = stop_gain_percentage*data[10][t]
                    stop_loss = stop_loss_percentage*data[10][t]
                    self.hand_position.append([0,data[10][t],stop_gain,stop_loss,volume]) #[Long,Short]
                    self.stock_capital = data[10][t]*volume
                    self.hand_time = 0
                    self.blank_time = 0
                    print('buy short position at price {}'.format(data[10][t]))

                else:
                    # if not avialble
                    self.hand_position.append(self.hand_position[-1]) # do nothing
                    self.hand_time += 1

            else:
                # do nothing

                self.hand_position.append([0,0,0,0,0])
                self.blank_time += 1
                #pass
        else:
            # do nothing
            self.hand_position.append(self.hand_position[-1])
            self.blank_time += 1



    def sell_position(self,data,t,long_pos):
        # sell long position (expected price to increase)
        #print('--------------------------------------')
        #print(self.hand_position[-1])
        if long_pos==1:
            # profit = pricenow - hand_position[-1][0]
            '''
            print('debug')
            print(data[9][t])
            print(self.hand_position[t-1])
            print(self.hand_position[t-1][0])
            print(t)
            '''
            self.capital += self.hand_position[-1][0]*self.hand_position[-1][4]
            self.capital += (data[10][t]-self.hand_position[-1][0])*self.hand_position[-1][4]


            #print('debug')
            #print(data[9][t])
            #print(self.hand_position[t-1][0])
            #print(data[9][t] - self.hand_position[t-1][0])
            # hand_position is now avialble
            self.stock_capital = 0
            self.hand_position[-1]=[0,0,0,0,0] # reset
            #self.hand_position.append([None,None,None,None])
            self.position_token = 0
            #self.hand_time  = 0
            self.blank_time = 0
            return data[10][t]
        # sell short position (expected price to decrease)
        elif long_pos==0:
            # profit = hand_position[-1][1] - pricenow
            self.capital += self.hand_position[-1][1]*self.hand_position[-1][4]
            self.capital += (self.hand_position[-1][1] - data[10][t])*self.hand_position[-1][4]
            #print('debug')
            #print(data[9][t])
            #print(self.hand_position[t-1][1])
            #print(data[9][t] - self.hand_position[t-1][1])
            # hand_position is now avialble
            self.stock_capital = 0
            self.hand_position[-1]=[0,0,0,0,0] # reset
            #self.hand_position.append([None,None,None,None])
            self.position_token = 0
            #self.hand_time  = 0
            self.blank_time = 0
            return data[10][t]
        #print(self.hand_position[-1])
        #print('--------------------------------------')



