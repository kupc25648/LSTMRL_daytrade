'''
==================================================================
This file export result into csv file
このファイルはCSVファイルに結果をエクスポートします
=======50===========================================================
'''
#--------------------------------
# Import part / インポート部
#--------------------------------
import os, sys
import re
import math
import numpy as np
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt

#--------------------------------
# Parameters /  パラメーターb
#------------z--------------------
path = ['Log_Capital-Step_2022-01-25_20_12_16'] # folder path / folder path
see = None # for debug [0:Game No ,1:Initial value ,2:Final value ,3:Reward] / デバッグ用[0：ゲーム番号、1：初期値、2：最終値、3：報酬]
name = 'result' # filename of the output / 出力のファイル名

#--------------------------------
# Function to export game-initial-final-reward to csv file / game-initial-final-rewardをcsvファイルにエクスポートする関数
#--------------------------------
def cal_success(path,see=None):
    dirs = os.listdir( path )
    # Lists to contain data in each category / 各カテゴリのデータを含むリスト
    rewardlist = []
    for_table = []
    result_game =[] # Game number / ゲーム番号
    result_init =[] # Initial value / 初期値
    result_final =[] # Final value / Final value
    result_reward =[] # Reward / 報酬
    # Read each filename and put data into lists / 各ファイル名を読み取り、データをリストに入れます
    for file in dirs:
        try:
            for_table.append([
                int(file.split('_')[1])
                ,float(file.split('_')[3])
                ,float(file.split('_')[5])
                ,float(file.split('_')[7][:-4])])
            result_game.append(int(file.split('_')[1]))
            result_init.append(float(file.split('_')[3]))
            result_final.append(float(file.split('_')[5]))
            result_reward.append(float(file.split('_')[7][:-4]))
            if for_table[-1][2] < for_table[-1][1]:
                for_table[-1].append(1) # success
            else:
                for_table[-1].append(0) # fail
        except:
            pass

    # Export and Save .cvs file /.csvファイルのエクスポートと保存
    result = {
    'Game':result_game,
    'Initial':result_init,
    'Final':result_final,
    'Reward':result_reward
    }
    dataframe = pd.DataFrame(result)
    dataframe.to_csv('{}.csv'.format(name))

    # Count success rate / 成功率を数える
    count_suc = 0
    sum_reward = 0
    sum_suc = []
    reward = []
    e_change = []
    avg_e_change = 0
    for i in range(len(for_table)):
        if for_table[i][-1] == 1:
            count_suc += 1
            sum_suc.append(count_suc)
            e_change.append(round(((for_table[i][1]-for_table[i][2])/for_table[i][1]),2))
            avg_e_change += e_change[-1]

        else:
            sum_suc.append(count_suc)
        sum_reward += (for_table[i][-2])
        reward.append(for_table[i][-2])
    # Print Result / 結果を印刷
    if see!=None:
        for i in range(len(for_table)):
            print(for_table[i][see])
    # Print Summary / 概要を印刷
    '''
    print('-----------------------------------------')
    print('Game {}'.format(path))
    print('Success rate             {} %'.format((count_suc*100/len(for_table))))
    print('Avg. Objective change    {} %'.format(round(avg_e_change,2)))
    print('Max Objective change     {} %'.format(max(e_change)*100))
    print('Min Objective change     {} %'.format(min(e_change)*100))
    print('Avg Reward               {} '.format(sum_reward/len(for_table)))
    print('Max Reward               {} '.format(max(reward)))
    print('Min Reward               {} '.format(min(reward)))
    print('-----------------------------------------')
    '''

#--------------------------------
# Main function / 主な機能
#--------------------------------
for i in range(len(path)):
    cal_success(path[i],see)
