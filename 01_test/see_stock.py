import pandas as pd
import matplotlib.pyplot as plt

# Get data
'''
data = pd.read_csv('GOOG.csv')
df = pd.DataFrame(data)
print(df.head())
df = df[['date','high','low']]
print(df['date'][1])
'''

def gen_data(dir,length=None):
    # get data from csv file
    data = pd.read_csv(dir)
    df = pd.DataFrame(data)
    df = df[['date','high','low','close','volume']] #use only ['date','high','low''close'] columns : close = price at aclose =reference point**
    if length!=None:
        df = df[:length]
    return df

def kijun_sen(data,period=26):
    # return kijun_sen
    # kijun_sen = (highest high + lowest low)/2 of the last 26 period
    kijun_sen = []
    for i in range(len(data['date'])):
        high_period = []
        low_period = []
        for j in range(period):
            try:
                high_period.append(data['high'][i-(period-j-1)])
                low_period.append(data['low'][i-(period-j-1)])
            except:
                high_period.append(0)
                low_period.append(0)
        kijun = (max(high_period) + min(low_period))/2
        kijun_sen.append(kijun)
    data["kijun_sen"] = kijun_sen
    return data

def tenkan_sen(data,period=9):
    # return tenkan_sen
    # tenkan_sen = (highest high + lowest low)/2 of the last 9 period
    tenkan_sen = []
    for i in range(len(data['date'])):
        high_period = []
        low_period = []
        for j in range(period):
            try:
                high_period.append(data['high'][i-(period-j-1)])
                low_period.append(data['low'][i-(period-j-1)])
            except:
                high_period.append(0)
                low_period.append(0)
        tenkan = (max(high_period) + min(low_period))/2
        tenkan_sen.append(tenkan)
    data["tenkan_sen"] = tenkan_sen
    return data

def chikou_span(data,period=26):
    # return chikou_span with period of 26
    # chikou_span = price of today then set it to be today-26 period
    chikou_span = []
    for i in range(len(data['date'])):
        try:
            chikou_span.append(data['close'][i-(period-1)])
        except:
            chikou_span.append(0)
    data["chikou_span"] = chikou_span
    return data

def senkou_span(data,period_a=26,period_b=52):
    # return ichimoku_cloud
    # senkou_span_A = (tenkan_sen + kijun_sen)/2 **And plot data into the future 26 period
    # senkou_span_B = (highest high + lowest low)/2 of the last 52 period **And plot data into the future 26 period

    senkou_span_A = []
    senkou_span_B = []
    Price_not_norm = []
    for i in range(period_a):
        senkou_span_A.append(0)

    for i in range(len(data['date'])):
        try:
            high_period = []
            low_period = []
            senkou_span_A.append((data['tenkan_sen'][i]+data['kijun_sen'][i])/2)
            for j in range(period_b):
                try:
                    high_period.append(data['high'][i-(period_b-j-1)])
                    low_period.append(data['low'][i-(period_b-j-1)])
                except:
                    high_period.append(0)
                    low_period.append(0)
            senkou_span_B.append((max(high_period)+min(low_period))/2)
        except:
            senkou_span_A.append(0)
            senkou_span_B.append(0)
        Price_not_norm.append(data['close'][i])
    #print(len(data['date']))
    #print(len(senkou_span_A))
    #print(len(senkou_span_B))
    data["senkou_span_A"] = senkou_span_A[:-period_a]
    data["senkou_span_B"] = senkou_span_B#[:-period_a]
    data["price_not_norm"] = Price_not_norm

    return data

def indicator(data):
    # used for create indicator vector
    A_B_list = []
    P_C_list = []
    P_K_list = []
    C_T_list = []
    T_K_list = []
    datanorm = []

    # A_o_B : senkou_span_A is over senkou_span_B (0/1)
    # T_K tenkan_sen lower than kijunsen (0/1)s
    # P_C price lower than the cloud(senkou_span_A,senkou_span_B) (0/1)
    # P_K price lower than kijunsen (0/1)s
    # C_T chikou_span lower than tenkan_sen (0/1)
    for i in range(len(data['date'])):
        # enkou_span_A is over senkou_span_B (0/1)
        if (data["senkou_span_A"][i] > data["senkou_span_A"][i]):
            A_B_list.append(1)
        else:
            A_B_list.append(0)
        # P_C price lower than the cloud(senkou_span_A,senkou_span_B) (0/1)
        if (data["price_not_norm"][i] < data["senkou_span_A"][i]) and (data["price_not_norm"][i] < data["senkou_span_B"][i]):
            P_C_list.append(1)
        else:
            P_C_list.append(0)
        # P_K price lower than kijunsen (0/1)
        if (data["price_not_norm"][i] < data["kijun_sen"][i]):
            P_K_list.append(1)
        else:
            P_K_list.append(0)
        # C_T chikou_span lower than tenkan_sen (0/1)
        if (data["chikou_span"][i] < data["tenkan_sen"][i]):
            C_T_list.append(1)
        else:
            C_T_list.append(0)
        # T_K tenkan_sen lower than kijunsen (0/1)s
        if (data["tenkan_sen"][i] < data["kijun_sen"][i]):
            T_K_list.append(1)
        else:
            T_K_list.append(0)

        datanorm.append(data["close"][i]/1000)

    data["P_C"] = P_C_list
    data["P_K"] = P_K_list
    data["C_T"] = C_T_list
    data["norm_close"] = datanorm
    data["A_o_B"] = A_B_list
    data["T_K"] = T_K_list

    return data

def differential(data):
    # calculate differentiation of price/time at time t
    # Given n = time window
    # eq = (Price@t - Price@t-n)/n
    # already normalized price
    diff_1 = [0]     # n=1
    diff_5 = [0,0,0,0,0]     # n=5
    diff_10 = [0,0,0,0,0,0,0,0,0,0]    # n=10
    for i in range(len(data['date'])-1):
        diff_1.append(((data['close'][i+1]-data['close'][26:].min())/(data['close'][26:].max() - data['close'][26:].min())-(data['close'][i]-data['close'][26:].min())/(data['close'][26:].max() - data['close'][26:].min()))/1)

    for i in range(len(data['date'])-5):
        diff_5.append(((data['close'][i+5]-data['close'][26:].min())/(data['close'][26:].max() - data['close'][26:].min())-(data['close'][i]-data['close'][26:].min())/(data['close'][26:].max() - data['close'][26:].min()))/5)
    for i in range(len(data['date'])-10):
        diff_10.append(((data['close'][i+10]-data['close'][26:].min())/(data['close'][26:].max() - data['close'][26:].min())-(data['close'][i]-data['close'][26:].min())/(data['close'][26:].max() - data['close'][26:].min()))/10)

    data["diff_1"] = diff_1
    data["diff_5"] = diff_5
    data["diff_10"] = diff_10

    return data





def ichimoku(data,length=None,kijun_period=26,tenkan_period=26,chikou_period=26,senkou_a_period=26,senkou_b_period=26):
    data = gen_data(data,length)
    # some preprocess
    #
    data = kijun_sen(data,kijun_period)
    data = tenkan_sen(data,tenkan_period)
    data = chikou_span(data,chikou_period)
    data = senkou_span(data,senkou_a_period,senkou_b_period)
    data = indicator(data)
    data = differential(data)

    # preprocess to remove empty and Nan value row from dataframe
    # --------------------------------------------------------------------------------
    nan_value = float("NaN")
    data.replace("", nan_value, inplace=True)
    data.dropna(subset = ["date"], inplace=True)
    data.dropna(subset = ["high"], inplace=True)
    data.dropna(subset = ["low"], inplace=True)
    data.dropna(subset = ["close"], inplace=True)
    data.dropna(subset = ["volume"], inplace=True)
    data.dropna(subset = ["kijun_sen"], inplace=True)
    data.dropna(subset = ["tenkan_sen"], inplace=True)
    data.dropna(subset = ["chikou_span"], inplace=True)
    data.dropna(subset = ["senkou_span_A"], inplace=True)
    data.dropna(subset = ["senkou_span_B"], inplace=True)
    data.dropna(subset = ["price_not_norm"], inplace=True)
    data.dropna(subset = ["P_C"], inplace=True)
    data.dropna(subset = ["P_K"], inplace=True)
    data.dropna(subset = ["C_T"], inplace=True)
    data.dropna(subset = ["norm_close"], inplace=True)
    data.dropna(subset = ["diff_1"], inplace=True)
    data.dropna(subset = ["diff_5"], inplace=True)
    data.dropna(subset = ["diff_10"], inplace=True)
    # --------------------------------------------------------------------------------


    return data[26:]#.to_numpy()
    # some visualization
    #

# Try making functions to auto detect some buy and sell signal
#       if/else buying bot

# Try making RL+LSTM function to make a policy to buy and sell
#       RL with LSTM agent
#       why LSTM

import os
for i in range(len(os.listdir('max_150'))):
    if str(os.listdir('max_150')[i])[-8:]=='data.csv':
        data_dir = 'max_150/'+str(os.listdir('max_150')[i])
        data = ichimoku(data_dir,1000+45+26)

#directory = 'individual_stocks_5yr/ABC_data.csv'
#data = ichimoku(directory,1000)


        ax = plt.gca()
        data.plot(kind='line',x='date',y='close',ax=ax,label=str(os.listdir('max_150')[i])) # close
        #data.plot(kind='line',x='date',y='kijun_sen',ax=ax) # close
        #data.plot(kind='line',x='date',y='tenkan_sen',ax=ax) # close
        #data.plot(kind='line',x='date',y='chikou_span',ax=ax) # close
        #data.plot(kind='line',x='date',y='senkou_span_A',ax=ax) # close
        #data.plot(kind='line',x='date',y='senkou_span_B',ax=ax) # close
        print('done {}'.format(i))
plt.show()

'''

import os
data_list = os.listdir('individual_stocks_5yr')
err_list = []
for i in range(len(data_list)):
    print(str(data_list[i])[-8:])
    print(str(data_list[i])[-8:]=='data.csv')
    try:
        data_dir = 'individual_stocks_5yr/'+str(data_list[i])
        data = ichimoku(data_dir,1000)
        print(data.isnull().values.any())
        if data.isnull().values.any():
            print(str(data_list[i]))
            err_list.append(str(data_list[i]))
    except:
        print(str(data_list[i]))
        err_list.append(str(data_list[i]))

print(err_list)

'''
