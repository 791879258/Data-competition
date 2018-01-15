# -*- coding: utf-8 -*-

import os
os.chdir('H:/hbc_competition/hbc-master')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time

### define some query functions
def getTagsFromOrderHistoryByUserid(df_ordhistory, userid, countryset, cityset):
    df_ordhistory_of_userid = df_ordhistory[df_ordhistory['userid'] == userid]
    sum_all = len(df_ordhistory_of_userid)   #买过旅游服务 # 购的次数
    sum0 = np.sum(df_ordhistory_of_userid['orderType'] == 0)    # 购买过普通旅游服务的次数
    sum1 = np.sum(df_ordhistory_of_userid['orderType'] == 1)    # 购买过精品旅游服务的次数
    ratio1_0 = (sum1+1.0) / (sum0+1.0)                  # 精品/普通(比例)
    time_last_1 = 0#np.min(df_ordhistory['orderTime'])      # 最后一次订单的时间
    time_last_1_year = 0
    time_last_1_month = 0
    time_last_2 = 0
    time_last_2_year = 0
    time_last_2_month = 0
    time_last_3 = 0
    time_last_3_year = 0
    time_last_3_month = 0
    sum_cont1 = 0
    sum_cont2 = 0
    sum_cont3 = 0
    sum_cont4 = 0
    sum_cont5 = 0
    sum_countries = pd.DataFrame()
    sum_cities = pd.DataFrame()
    for k in range(len(countryset)):
        sum_countries[countryset[k]] = [0]
    for k in range(len(cityset)):
        sum_cities[cityset[k]] = [0]
    if len(df_ordhistory_of_userid) > 0:
        time_last_1 = np.max(df_ordhistory_of_userid['orderTime'])
        time_local = time.localtime(time_last_1)
        time_last_1_year = time_local.tm_year
        time_last_1_month = time_local.tm_mon
        sum_cont1 = np.sum(df_ordhistory_of_userid['continent'] == '亚洲')
        sum_cont2 = np.sum(df_ordhistory_of_userid['continent'] == '欧洲')
        sum_cont3 = np.sum(df_ordhistory_of_userid['continent'] == '北美洲')
        sum_cont4 = np.sum(df_ordhistory_of_userid['continent'] == '大洋洲')
        sum_cont5 = np.sum(df_ordhistory_of_userid['continent'] == '非洲')
        for k in range(len(countryset)):
            sum_countries[countryset[k]] = [np.sum(df_ordhistory_of_userid['country'] == countryset[k])]
        for k in range(len(cityset)):
            sum_cities[cityset[k]] = [np.sum(df_ordhistory_of_userid['city'] == cityset[k])]
    if len(df_ordhistory_of_userid) > 1:
        time_last_2 = df_ordhistory_of_userid.sort_values(by=['orderTime'])['orderTime'].iat[-2]
        time_local = time.localtime(time_last_2)
        time_last_2_year = time_local.tm_year
        time_last_2_month = time_local.tm_mon
    if len(df_ordhistory_of_userid) > 2:
        time_last_3 = df_ordhistory_of_userid.sort_values(by=['orderTime'])['orderTime'].iat[-3]
        time_local = time.localtime(time_last_3)
        time_last_3_year = time_local.tm_year
        time_last_3_month = time_local.tm_mon
    return sum_all, sum0, sum1, ratio1_0, time_last_1, time_last_1_year, time_last_1_month, time_last_2, time_last_2_year, time_last_2_month, time_last_3, time_last_3_year, time_last_3_month, sum_cont1, sum_cont2, sum_cont3, sum_cont4, sum_cont5, sum_countries, sum_cities

def getActionTimeSpan(df_action_of_userid, actiontypeA, actiontypeB, timethred=100):
    timespan_list = []
    i = 0
    while i < (len(df_action_of_userid)-1):
        if df_action_of_userid['actionType'].iat[i] == actiontypeA:
            timeA = df_action_of_userid['actionTime'].iat[i]
            for j in range(i+1, len(df_action_of_userid)):
                if df_action_of_userid['actionType'].iat[j] == actiontypeA:
                    timeA = df_action_of_userid['actionTime'].iat[j]
                if df_action_of_userid['actionType'].iat[j] == actiontypeB:
                    timeB = df_action_of_userid['actionTime'].iat[j]
                    timespan_list.append(timeB-timeA)
                    i = j
                    break
        i+=1
    return np.sum(np.array(timespan_list) <= timethred) / (np.sum(np.array(timespan_list)) + 1.0)

def get2ActionTimeSpanLast(df_action_of_userid, actiontypeA, actiontypeB):
    timespan_list = []
    i = 0
    while i < (len(df_action_of_userid)-1):
        if df_action_of_userid['actionType'].iat[i] == actiontypeA:
            timeA = df_action_of_userid['actionTime'].iat[i]
            for j in range(i+1, len(df_action_of_userid)):
                if df_action_of_userid['actionType'].iat[j] == actiontypeA:
                    timeA = df_action_of_userid['actionTime'].iat[j]
                    continue
                if df_action_of_userid['actionType'].iat[j] == actiontypeB:
                    timeB = df_action_of_userid['actionTime'].iat[j]
                    timespan_list.append(timeB-timeA)
                    i = j
                    break
        i+=1
    if len(timespan_list) > 0:
        return timespan_list[-1]
    else:
        return -1

def calc_seqentialratio(df_action_of_userid):
    pos_5 = -1
    result = 0
    df_len = len(df_action_of_userid)
    for i in range(0, df_len):
        if df_action_of_userid['actionType'].iat[i] == 5:
            pos_5 = i
    if pos_5 != -1:
        result += 1
        if pos_5+1 < df_len:
            if df_action_of_userid['actionType'].iat[pos_5+1] == 6:
                result += 1
                if pos_5+2 < df_len:
                    if df_action_of_userid['actionType'].iat[pos_5+2] == 7:
                        result += 1
                        if pos_5+3 < df_len:
                            if df_action_of_userid['actionType'].iat[pos_5+3] == 8:
                                result += 1
    return result

def getTagsFromActionByUserid(df_action, userid):
    df_action_of_userid = df_action[df_action['userid'] == userid]
    sum_action = len(df_action_of_userid)  # 一个用户action的总次数
    actiontime_last_1_year = -1
    actiontime_last_1_month = -1
    action_last_1 = 0   # 倒数第1次actionType
    action_last_2 = 0   # 倒数第2次actionType
    action_last_3 = 0   # 倒数第3次actionType
    action_last_4 = 0   # 倒数第4次actionType
    action_last_5 = 0   # 倒数第5次actionType
    action_last_6 = 0   # 倒数第6次actionType
    action_last_7 = 0   # 倒数第7次actionType
    action_last_8 = 0   # 倒数第8次actionType
    action_last_9 = 0   # 倒数第9次actionType
    action_last_10 = 0  # 倒数第10次actionType
    action_last_11 = 0  # 倒数第11次actionType
    action_last_12 = 0  # 倒数第12次actionType
    action_last_13 = 0  # 倒数第13次actionType
    action_last_14 = 0  # 倒数第14次actionType
    action_last_15 = 0  # 倒数第15次actionType
    action_last_16 = 0  # 倒数第16次actionType
    action_last_17 = 0  # 倒数第17次actionType
    action_last_18 = 0  # 倒数第18次actionType
    action_last_19 = 0  # 倒数第19次actionType
    action_last_20 = 0  # 倒数第20次actionType
    #actiontime_mean = np.mean(df_action['actionTime'])
    actiontime_last_1 = 0   # 倒数第1次actionTime
    actiontime_last_2 = 0   # 倒数第2次actionTime
    actiontime_last_3 = 0   # 倒数第3次actionTime
    actiontime_last_4 = 0   # 倒数第4次actionTime
    actiontime_last_5 = 0   # 倒数第5次actionTime
    actiontime_last_6 = 0   # 倒数第6次actionTime
    actiontime_last_7 = 0   # 倒数第7次actionTime
    actiontime_last_8 = 0   # 倒数第8次actionTime
    actiontime_last_9 = 0   # 倒数第9次actionTime
    actiontime_last_10 = 0  # 倒数第10次actionTime
    actiontime_last_11 = 0  # 倒数第11次actionTime
    actiontime_last_12 = 0  # 倒数第12次actionTime
    actiontime_last_13 = 0  # 倒数第13次actionTime
    actiontime_last_14 = 0  # 倒数第14次actionTime
    actiontime_last_15 = 0  # 倒数第15次actionTime
    actiontime_last_16 = 0  # 倒数第16次actionTime
    actiontime_last_17 = 0  # 倒数第17次actionTime
    actiontime_last_18 = 0  # 倒数第18次actionTime
    actiontime_last_19 = 0  # 倒数第19次actionTime
    actiontime_last_20 = 0  # 倒数第20次actionTime
    actiontypeprop_1 = 0  # actionType1占比
    actiontypeprop_2 = 0  # actionType2占比
    actiontypeprop_3 = 0  # actionType3占比
    actiontypeprop_4 = 0  # actionType4占比
    actiontypeprop_5 = 0  # actionType5占比
    actiontypeprop_6 = 0  # actionType6占比
    actiontypeprop_7 = 0  # actionType7占比
    actiontypeprop_8 = 0  # actionType8占比
    actiontypeprop_9 = 0  # actionType9占比
    timespanthred = 100
    actiontimespancount_1_5 = 0  # actionType1-5的时间差小于timespanthred的数量
    actiontimespancount_5_6 = 0  # actionType5-6的时间差小于timespanthred的数量
    actiontimespancount_6_7 = 0  # actionType6-7的时间差小于timespanthred的数量
    actiontimespancount_7_8 = 0  # actionType7-8的时间差小于timespanthred的数量
    actiontimespancount_8_9 = 0  # actionType8-9的时间差小于timespanthred的数量
    actionratio_24_59 = 1.0      # actionType2-4与5-9之间的比值
    actiontype_lasttime_1 = 0    # actionType1最后一次出现的时间
    actiontype_lasttime_5 = 0    # actionType5最后一次出现的时间
    actiontype_lasttime_6 = 0    # actionType6最后一次出现的时间
    actiontype_lasttime_7 = 0    # actionType7最后一次出现的时间
    actiontype_lasttime_8 = 0    # actionType8最后一次出现的时间
    actiontype_lasttime_9 = 0    # actionType9最后一次出现的时间
    actiontype_lasttime_24 = 0   # actionType2-4最后一次出现的时间
    actiontimespanlast_1_5 = 0   # 最后一次actionType1与5之间的间隔
    actiontimespanlast_5_6 = 0   # 最后一次actionType5与6之间的间隔
    actiontimespanlast_6_7 = 0   # 最后一次actionType6与7之间的间隔
    actiontimespanlast_7_8 = 0   # 最后一次actionType7与8之间的间隔
    actiontimespanlast_5_7 = 0   # 最后一次actionType5与7之间的间隔
    actiontimespanlast_5_8 = 0   # 最后一次actionType5与8之间的间隔
    action59seqentialratio = 0      # actionType5-9的连续程度
    actiontypeproplast20_1 = 0  # 最后20个action中，actionType1占比
    actiontypeproplast20_2 = 0  # 最后20个action中，actionType2占比
    actiontypeproplast20_3 = 0  # 最后20个action中，actionType3占比
    actiontypeproplast20_4 = 0  # 最后20个action中，actionType4占比
    actiontypeproplast20_5 = 0  # 最后20个action中，actionType5占比
    actiontypeproplast20_6 = 0  # 最后20个action中，actionType6占比
    actiontypeproplast20_7 = 0  # 最后20个action中，actionType7占比
    actiontypeproplast20_8 = 0  # 最后20个action中，actionType8占比
    actiontypeproplast20_9 = 0  # 最后20个action中，actionType9占比
    actiontime_1 = 0            # 第一个actionTime（用户第一次使用app的时间）
    if sum_action >= 1:
        actiontime_1 = df_action_of_userid['actionTime'].iat[0]
        action_last_1 = df_action_of_userid['actionType'].iat[-1]
        actiontime_last_1 = df_action_of_userid['actionTime'].iat[-1]
        time_local = time.localtime(actiontime_last_1)
        actiontime_last_1_year = time_local.tm_year
        actiontime_last_1_month = time_local.tm_mon
        if len(df_action_of_userid[df_action_of_userid['actionType'] == 1]) > 0:
            actiontype_lasttime_1 = df_action_of_userid[df_action_of_userid['actionType'] == 1].iloc[-1]['actionTime']
        if len(df_action_of_userid[df_action_of_userid['actionType'] == 5]) > 0:
            actiontype_lasttime_5 = df_action_of_userid[df_action_of_userid['actionType'] == 5].iloc[-1]['actionTime']
        if len(df_action_of_userid[df_action_of_userid['actionType'] == 6]) > 0:
            actiontype_lasttime_6 = df_action_of_userid[df_action_of_userid['actionType'] == 6].iloc[-1]['actionTime']
        if len(df_action_of_userid[df_action_of_userid['actionType'] == 7]) > 0:
            actiontype_lasttime_7 = df_action_of_userid[df_action_of_userid['actionType'] == 7].iloc[-1]['actionTime']
        if len(df_action_of_userid[df_action_of_userid['actionType'] == 8]) > 0:
            actiontype_lasttime_8 = df_action_of_userid[df_action_of_userid['actionType'] == 8].iloc[-1]['actionTime']
        if len(df_action_of_userid[df_action_of_userid['actionType'] == 9]) > 0:
            actiontype_lasttime_9 = df_action_of_userid[df_action_of_userid['actionType'] == 9].iloc[-1]['actionTime']
        if len(df_action_of_userid[(df_action_of_userid['actionType'] >= 2) & (df_action_of_userid['actionType'] <= 4)]) > 0:    
            actiontype_lasttime_24 = df_action_of_userid[(df_action_of_userid['actionType'] >= 2) & (df_action_of_userid['actionType'] <= 4)].iloc[-1]['actionTime']
    if sum_action >= 2:
        action_last_2 = df_action_of_userid['actionType'].iat[-2]
        actiontime_last_2 = df_action_of_userid['actionTime'].iat[-2]
    if sum_action >= 3:
        action_last_3 = df_action_of_userid['actionType'].iat[-3]
        actiontime_last_3 = df_action_of_userid['actionTime'].iat[-3]
        actiontimespanlast_1_5 = get2ActionTimeSpanLast(df_action_of_userid, 1, 5)
        actiontimespanlast_5_6 = get2ActionTimeSpanLast(df_action_of_userid, 5, 6)
        actiontimespanlast_6_7 = get2ActionTimeSpanLast(df_action_of_userid, 6, 7)
        actiontimespanlast_7_8 = get2ActionTimeSpanLast(df_action_of_userid, 7, 8)
        actiontimespanlast_5_7 = get2ActionTimeSpanLast(df_action_of_userid, 5, 7)
        actiontimespanlast_5_8 = get2ActionTimeSpanLast(df_action_of_userid, 5, 8)
        action59seqentialratio = calc_seqentialratio(df_action_of_userid)
    if sum_action >= 4:
        action_last_4 = df_action_of_userid['actionType'].iat[-4]
        actiontime_last_4 = df_action_of_userid['actionTime'].iat[-4]
    if sum_action >= 5:
        action_last_5 = df_action_of_userid['actionType'].iat[-5]
        actiontime_last_5 = df_action_of_userid['actionTime'].iat[-5]
    if sum_action >= 6:
        action_last_6 = df_action_of_userid['actionType'].iat[-6]
        actiontime_last_6 = df_action_of_userid['actionTime'].iat[-6]
    if sum_action >= 7:
        action_last_7 = df_action_of_userid['actionType'].iat[-7]
        actiontime_last_7 = df_action_of_userid['actionTime'].iat[-7]
    if sum_action >= 8:
        action_last_8 = df_action_of_userid['actionType'].iat[-8]
        actiontime_last_8 = df_action_of_userid['actionTime'].iat[-8]
    if sum_action >= 9:
        action_last_9 = df_action_of_userid['actionType'].iat[-9]
        actiontime_last_9 = df_action_of_userid['actionTime'].iat[-9]
    if sum_action >= 10:
        action_last_10 = df_action_of_userid['actionType'].iat[-10]
        actiontime_last_10 = df_action_of_userid['actionTime'].iat[-10]
    if sum_action >= 11:
        action_last_11 = df_action_of_userid['actionType'].iat[-11]
        actiontime_last_11 = df_action_of_userid['actionTime'].iat[-11]
    if sum_action >= 12:
        action_last_12 = df_action_of_userid['actionType'].iat[-12]
        actiontime_last_12 = df_action_of_userid['actionTime'].iat[-12]
    if sum_action >= 13:
        action_last_13 = df_action_of_userid['actionType'].iat[-13]
        actiontime_last_13 = df_action_of_userid['actionTime'].iat[-13]
    if sum_action >= 14:
        action_last_14 = df_action_of_userid['actionType'].iat[-14]
        actiontime_last_14 = df_action_of_userid['actionTime'].iat[-14]
    if sum_action >= 15:
        action_last_15 = df_action_of_userid['actionType'].iat[-15]
        actiontime_last_15 = df_action_of_userid['actionTime'].iat[-15]
    if sum_action >= 16:
        action_last_16 = df_action_of_userid['actionType'].iat[-16]
        actiontime_last_16 = df_action_of_userid['actionTime'].iat[-16]
    if sum_action >= 17:
        action_last_17 = df_action_of_userid['actionType'].iat[-17]
        actiontime_last_17 = df_action_of_userid['actionTime'].iat[-17]
    if sum_action >= 18:
        action_last_18 = df_action_of_userid['actionType'].iat[-18]
        actiontime_last_18 = df_action_of_userid['actionTime'].iat[-18]
    if sum_action >= 19:
        action_last_19 = df_action_of_userid['actionType'].iat[-19]
        actiontime_last_19 = df_action_of_userid['actionTime'].iat[-19]
    if sum_action >= 20:
        action_last_20 = df_action_of_userid['actionType'].iat[-20]
        actiontime_last_20 = df_action_of_userid['actionTime'].iat[-20]
        actiontypeproplast20_1 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==1)
        actiontypeproplast20_2 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==2)
        actiontypeproplast20_3 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==3)
        actiontypeproplast20_4 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==4)
        actiontypeproplast20_5 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==5)
        actiontypeproplast20_6 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==6)
        actiontypeproplast20_7 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==7)
        actiontypeproplast20_8 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==8)
        actiontypeproplast20_9 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==9)
    actiontypeprop_1 = np.sum(df_action_of_userid['actionType']==1) / (sum_action+1.0)
    actiontypeprop_2 = np.sum(df_action_of_userid['actionType']==2) / (sum_action+1.0)
    actiontypeprop_3 = np.sum(df_action_of_userid['actionType']==3) / (sum_action+1.0)
    actiontypeprop_4 = np.sum(df_action_of_userid['actionType']==4) / (sum_action+1.0)
    actiontypeprop_5 = np.sum(df_action_of_userid['actionType']==5) / (sum_action+1.0)
    actiontypeprop_6 = np.sum(df_action_of_userid['actionType']==6) / (sum_action+1.0)
    actiontypeprop_7 = np.sum(df_action_of_userid['actionType']==7) / (sum_action+1.0)
    actiontypeprop_8 = np.sum(df_action_of_userid['actionType']==8) / (sum_action+1.0)
    actiontypeprop_9 = np.sum(df_action_of_userid['actionType']==9) / (sum_action+1.0)
    actiontimespancount_1_5 = getActionTimeSpan(df_action_of_userid, 1, 5, timespanthred)
    actiontimespancount_5_6 = getActionTimeSpan(df_action_of_userid, 5, 6, timespanthred)
    actiontimespancount_6_7 = getActionTimeSpan(df_action_of_userid, 6, 7, timespanthred)
    actiontimespancount_7_8 = getActionTimeSpan(df_action_of_userid, 5, 8, timespanthred)
    actiontimespancount_8_9 = getActionTimeSpan(df_action_of_userid, 4, 9, timespanthred)
    sum_action_24 = np.sum((df_action_of_userid['actionType'] >= 2) & (df_action_of_userid['actionType'] <= 4))
    sum_action_59 = np.sum((df_action_of_userid['actionType'] >= 5) & (df_action_of_userid['actionType'] <= 9))
    actionratio_24_59 = (sum_action_24 + 1.0) / (sum_action_59 + 1.0)
    return sum_action, actiontime_last_1_year, actiontime_last_1_month, action_last_1, action_last_2, action_last_3, action_last_4, action_last_5, action_last_6, action_last_7, action_last_8, action_last_9, action_last_10, action_last_11, action_last_12, action_last_13, action_last_14, action_last_15, action_last_16, action_last_17, action_last_18, action_last_19, action_last_20, actiontime_last_1, actiontime_last_2, actiontime_last_3, actiontime_last_4, actiontime_last_5, actiontime_last_6, actiontime_last_7, actiontime_last_8, actiontime_last_9, actiontime_last_10, actiontime_last_11, actiontime_last_12, actiontime_last_13, actiontime_last_14, actiontime_last_15, actiontime_last_16, actiontime_last_17, actiontime_last_18, actiontime_last_19, actiontime_last_20, actiontypeprop_1, actiontypeprop_2, actiontypeprop_3, actiontypeprop_4, actiontypeprop_5, actiontypeprop_6, actiontypeprop_7, actiontypeprop_8, actiontypeprop_9, actiontimespancount_1_5, actiontimespancount_5_6, actiontimespancount_6_7, actiontimespancount_7_8, actiontimespancount_8_9, actionratio_24_59, actiontype_lasttime_1, actiontype_lasttime_5, actiontype_lasttime_6, actiontype_lasttime_7, actiontype_lasttime_8, actiontype_lasttime_9, actiontype_lasttime_24, actiontimespanlast_1_5, actiontimespanlast_5_6, actiontimespanlast_6_7, actiontimespanlast_7_8, actiontimespanlast_5_7, actiontimespanlast_5_8, action59seqentialratio, actiontypeproplast20_1, actiontypeproplast20_2, actiontypeproplast20_3, actiontypeproplast20_4, actiontypeproplast20_5, actiontypeproplast20_6, actiontypeproplast20_7, actiontypeproplast20_8, actiontypeproplast20_9, actiontime_1

def getTagsFromCommentByUserid(df_comm, userid):
    df_comm_of_userid = df_comm[df_comm['userid'] == userid]
    sum_comm = len(df_comm_of_userid)                    # 一个用户comment的总次数
    rating_ave = 5.0
    rating_min = 5.0
    rating_last = 5.0
    if sum_comm > 0:
        rating_ave = np.mean(df_comm_of_userid['rating'])
        rating_min = np.min(df_comm_of_userid['rating'])
        rating_last = df_comm_of_userid['rating'].iat[-1]
    return rating_ave, rating_min, rating_last

def getTagsFromProfileByUserid(df_profile, userid, provinceset):
    df_profile_of_userid = df_profile[df_profile['userid'] == userid]
    hasgender = 1   # 是否有男女信息
    ismale = 0      # 是否是男
    isfemale = 0    # 是否是女
    hasage = 1
    is60 = 0
    is70 = 0
    is80 = 0
    is90 = 0
    is00 = 0
    province_df = pd.DataFrame()
    for k in range(len(provinceset)):
        province_df[provinceset[k]] = [0]
    for k in range(len(provinceset)):
        if str(df_profile_of_userid['province'].iat[0]) == provinceset[k]:
            province_df[provinceset[k]] = 1
        else:
            province_df[provinceset[k]] = 0
    if df_profile_of_userid['gender'].iat[0] == '男':
        ismale = 1
    elif df_profile_of_userid['gender'].iat[0] == '女':
        isfemale = 1
    else:
        hasgender = 0
    if df_profile_of_userid['age'].iat[0] == '60后':
        is60 = 1
    elif df_profile_of_userid['age'].iat[0] == '70后':
        is70 = 1
    elif df_profile_of_userid['age'].iat[0] == '80后':
        is80 = 1
    elif df_profile_of_userid['age'].iat[0] == '90后':
        is90 = 1
    elif df_profile_of_userid['age'].iat[0] == '00后':
        is00 = 1
    else:
        hasage = 0
    return hasgender, ismale, isfemale, hasage, is60, is70, is80, is90, is00, province_df

################################################################################################
### load data
train_or_test = 'train'
#train_or_test = 'test'
df_profile = pd.read_csv('./data_train_test/userProfile_'+train_or_test+'.csv')
df_ordhistory = pd.read_csv('./data_train_test/orderHistory_'+train_or_test+'.csv')
df_action = pd.read_csv('./data_train_test/action_'+train_or_test+'.csv')
df_comm = pd.read_csv('./data_train_test/userComment_'+train_or_test+'.csv')
df_ordfuture = pd.read_csv('./data_train_test/orderFuture_'+train_or_test+'.csv')
usercount = len(df_profile)
df = pd.DataFrame(df_profile['userid'])
df['futureOrderType'] = df_ordfuture['orderType']
print("load data complite!")
    print("start get tags from orderhistory----------------------------")
    ### extract tags of user
    # get tags from orderhistory
    df_ordhistory_train = pd.read_csv('./data_train_test/orderHistory_train.csv')
    countryset = np.array(list(set(df_ordhistory_train['country'])))
    cityset = np.array(list(set(df_ordhistory_train['city'])))
    sum_all_list = np.zeros(usercount)
    sum0_list = np.zeros(usercount)
    sum1_list = np.zeros(usercount)
    ratio1_0_list = np.zeros(usercount)
    time_last_1_list = np.zeros(usercount)
    time_last_1_year_list = np.zeros(usercount)
    time_last_1_month_list = np.zeros(usercount)
    time_last_2_list = np.zeros(usercount)
    time_last_2_year_list = np.zeros(usercount)
    time_last_2_month_list = np.zeros(usercount)
    time_last_3_list = np.zeros(usercount)
    time_last_3_year_list = np.zeros(usercount)
    time_last_3_month_list = np.zeros(usercount)
    sum_cont1_list = np.zeros(usercount)
    sum_cont2_list = np.zeros(usercount)
    sum_cont3_list = np.zeros(usercount)
    sum_cont4_list = np.zeros(usercount)
    sum_cont5_list = np.zeros(usercount)
    sum_countries_all_dflist = pd.DataFrame()
    sum_cities_all_dflist = pd.DataFrame()
    for i in range(usercount):
        sum_all, sum0, sum1, ratio1_0, time_last_1, time_last_1_year, time_last_1_month, time_last_2, time_last_2_year, time_last_2_month, time_last_3, time_last_3_year, time_last_3_month, sum_cont1, sum_cont2, sum_cont3, sum_cont4, sum_cont5, sum_countries, sum_cities = getTagsFromOrderHistoryByUserid(df_ordhistory, df_profile['userid'][i], countryset, cityset)
        sum_all_list[i] = sum_all
        sum0_list[i] = sum0
        sum1_list[i] = sum1
        ratio1_0_list[i] = ratio1_0
        time_last_1_list[i] = time_last_1
        time_last_1_year_list[i] = time_last_1_year
        time_last_1_month_list[i] = time_last_1_month
        time_last_2_list[i] = time_last_2
        time_last_2_year_list[i] = time_last_2_year
        time_last_2_month_list[i] = time_last_2_month
        time_last_3_list[i] = time_last_3
        time_last_3_year_list[i] = time_last_3_year
        time_last_3_month_list[i] = time_last_3_month
        sum_cont1_list[i] = sum_cont1
        sum_cont2_list[i] = sum_cont2
        sum_cont3_list[i] = sum_cont3
        sum_cont4_list[i] = sum_cont4
        sum_cont5_list[i] = sum_cont5
        sum_countries_all_dflist = sum_countries_all_dflist.append(sum_countries)
        sum_cities_all_dflist = sum_cities_all_dflist.append(sum_cities)
    df['histord_sum_all'] = sum_all_list
    df['histord_sum_0'] = sum0_list
    df['histord_sum_1'] = sum1_list
    df['histord_ratio1_0'] = ratio1_0_list
    df['histord_time_last_1'] = time_last_1_list
    df['histord_time_last_1_year'] = time_last_1_year_list
    df['histord_time_last_1_month'] = time_last_1_month_list
    df['histord_time_last_2'] = time_last_2_list
    df['histord_time_last_2_year'] = time_last_2_year_list
    df['histord_time_last_2_month'] = time_last_2_month_list
    df['histord_time_last_3'] = time_last_3_list
    df['histord_time_last_3_year'] = time_last_3_year_list
    df['histord_time_last_3_month'] = time_last_3_month_list
    df['histord_sum_cont1'] = sum_cont1_list
    df['histord_sum_cont2'] = sum_cont2_list
    df['histord_sum_cont3'] = sum_cont3_list
    df['histord_sum_cont4'] = sum_cont4_list
    df['histord_sum_cont5'] = sum_cont5_list
    sum_countries_all_dflist['id'] = range(0, usercount)
    sum_countries_all_dflist = sum_countries_all_dflist.set_index('id')
    df = pd.concat([df, sum_countries_all_dflist], axis=1)
    sum_cities_all_dflist['id'] = range(0, usercount)
    sum_cities_all_dflist = sum_cities_all_dflist.set_index('id')
    df = pd.concat([df, sum_cities_all_dflist], axis=1)
    print("get tags from orderhistory complite!")
print("start get tags from action----------------------------")
# get tags from action
sum_action_list = np.zeros(usercount)
actiontime_last_1_year_list = np.zeros(usercount)
actiontime_last_1_month_list = np.zeros(usercount)
action_last_1_list = np.zeros(usercount)
action_last_2_list = np.zeros(usercount)
action_last_3_list = np.zeros(usercount)
action_last_4_list = np.zeros(usercount)
action_last_5_list = np.zeros(usercount)
action_last_6_list = np.zeros(usercount)
action_last_7_list = np.zeros(usercount)
action_last_8_list = np.zeros(usercount)
action_last_9_list = np.zeros(usercount)
action_last_10_list = np.zeros(usercount)
action_last_11_list = np.zeros(usercount)
action_last_12_list = np.zeros(usercount)
action_last_13_list = np.zeros(usercount)
action_last_14_list = np.zeros(usercount)
action_last_15_list = np.zeros(usercount)
action_last_16_list = np.zeros(usercount)
action_last_17_list = np.zeros(usercount)
action_last_18_list = np.zeros(usercount)
action_last_19_list = np.zeros(usercount)
action_last_20_list = np.zeros(usercount)
actiontime_last_1_list = np.zeros(usercount)
actiontime_last_2_list = np.zeros(usercount)
actiontime_last_3_list = np.zeros(usercount)
actiontime_last_4_list = np.zeros(usercount)
actiontime_last_5_list = np.zeros(usercount)
actiontime_last_6_list = np.zeros(usercount)
actiontime_last_7_list = np.zeros(usercount)
actiontime_last_8_list = np.zeros(usercount)
actiontime_last_9_list = np.zeros(usercount)
actiontime_last_10_list = np.zeros(usercount)
actiontime_last_11_list = np.zeros(usercount)
actiontime_last_12_list = np.zeros(usercount)
actiontime_last_13_list = np.zeros(usercount)
actiontime_last_14_list = np.zeros(usercount)
actiontime_last_15_list = np.zeros(usercount)
actiontime_last_16_list = np.zeros(usercount)
actiontime_last_17_list = np.zeros(usercount)
actiontime_last_18_list = np.zeros(usercount)
actiontime_last_19_list = np.zeros(usercount)
actiontime_last_20_list = np.zeros(usercount)
actiontypeprop_1_list = np.zeros(usercount)
actiontypeprop_2_list = np.zeros(usercount)
actiontypeprop_3_list = np.zeros(usercount)
actiontypeprop_4_list = np.zeros(usercount)
actiontypeprop_5_list = np.zeros(usercount)
actiontypeprop_6_list = np.zeros(usercount)
actiontypeprop_7_list = np.zeros(usercount)
actiontypeprop_8_list = np.zeros(usercount)
actiontypeprop_9_list = np.zeros(usercount)
actiontimespancount_1_5_list = np.zeros(usercount)
actiontimespancount_5_6_list = np.zeros(usercount)
actiontimespancount_6_7_list = np.zeros(usercount)
actiontimespancount_7_8_list = np.zeros(usercount)
actiontimespancount_8_9_list = np.zeros(usercount)
actionratio_24_59_list = np.zeros(usercount)
actiontype_lasttime_1_list = np.zeros(usercount)
actiontype_lasttime_5_list = np.zeros(usercount)
actiontype_lasttime_6_list = np.zeros(usercount)
actiontype_lasttime_7_list = np.zeros(usercount)
actiontype_lasttime_8_list = np.zeros(usercount)
actiontype_lasttime_9_list = np.zeros(usercount)
actiontype_lasttime_24_list = np.zeros(usercount)
actiontimespanlast_1_5_list = np.zeros(usercount)
actiontimespanlast_5_6_list = np.zeros(usercount)
actiontimespanlast_6_7_list = np.zeros(usercount)
actiontimespanlast_7_8_list = np.zeros(usercount)
actiontimespanlast_5_7_list = np.zeros(usercount)
actiontimespanlast_5_8_list = np.zeros(usercount)
action59seqentialratio_list = np.zeros(usercount)
actiontypeproplast20_1_list = np.zeros(usercount)
actiontypeproplast20_2_list = np.zeros(usercount)
actiontypeproplast20_3_list = np.zeros(usercount)
actiontypeproplast20_4_list = np.zeros(usercount)
actiontypeproplast20_5_list = np.zeros(usercount)
actiontypeproplast20_6_list = np.zeros(usercount)
actiontypeproplast20_7_list = np.zeros(usercount)
actiontypeproplast20_8_list = np.zeros(usercount)
actiontypeproplast20_9_list = np.zeros(usercount)
actiontime_1_list = np.zeros(usercount)
for i in range(usercount):
    sum_action, actiontime_last_1_year, actiontime_last_1_month, action_last_1, action_last_2, action_last_3, action_last_4, action_last_5, action_last_6, action_last_7, action_last_8, action_last_9, action_last_10, action_last_11, action_last_12, action_last_13, action_last_14, action_last_15, action_last_16, action_last_17, action_last_18, action_last_19, action_last_20, actiontime_last_1, actiontime_last_2, actiontime_last_3, actiontime_last_4, actiontime_last_5, actiontime_last_6, actiontime_last_7, actiontime_last_8, actiontime_last_9, actiontime_last_10, actiontime_last_11, actiontime_last_12, actiontime_last_13, actiontime_last_14, actiontime_last_15, actiontime_last_16, actiontime_last_17, actiontime_last_18, actiontime_last_19, actiontime_last_20, actiontypeprop_1, actiontypeprop_2, actiontypeprop_3, actiontypeprop_4, actiontypeprop_5, actiontypeprop_6, actiontypeprop_7, actiontypeprop_8, actiontypeprop_9, actiontimespancount_1_5, actiontimespancount_5_6, actiontimespancount_6_7, actiontimespancount_7_8, actiontimespancount_8_9, actionratio_24_59, actiontype_lasttime_1, actiontype_lasttime_5, actiontype_lasttime_6, actiontype_lasttime_7, actiontype_lasttime_8, actiontype_lasttime_9, actiontype_lasttime_24, actiontimespanlast_1_5, actiontimespanlast_5_6, actiontimespanlast_6_7, actiontimespanlast_7_8, actiontimespanlast_5_7, actiontimespanlast_5_8, action59seqentialratio, actiontypeproplast20_1, actiontypeproplast20_2, actiontypeproplast20_3, actiontypeproplast20_4, actiontypeproplast20_5, actiontypeproplast20_6, actiontypeproplast20_7, actiontypeproplast20_8, actiontypeproplast20_9, actiontime_1 = getTagsFromActionByUserid(df_action, df_profile['userid'][i])
    sum_action_list[i] = sum_action
    actiontime_last_1_year_list[i] = actiontime_last_1_year
    actiontime_last_1_month_list[i] = actiontime_last_1_month
    action_last_1_list[i] = action_last_1
    action_last_2_list[i] = action_last_2
    action_last_3_list[i] = action_last_3
    action_last_4_list[i] = action_last_4
    action_last_5_list[i] = action_last_5
    action_last_6_list[i] = action_last_6
    action_last_7_list[i] = action_last_7
    action_last_8_list[i] = action_last_8
    action_last_9_list[i] = action_last_9
    action_last_10_list[i] = action_last_10
    action_last_11_list[i] = action_last_11
    action_last_12_list[i] = action_last_12
    action_last_13_list[i] = action_last_13
    action_last_14_list[i] = action_last_14
    action_last_15_list[i] = action_last_15
    action_last_16_list[i] = action_last_16
    action_last_17_list[i] = action_last_17
    action_last_18_list[i] = action_last_18
    action_last_19_list[i] = action_last_19
    action_last_20_list[i] = action_last_20
    actiontime_last_1_list[i] = actiontime_last_1
    actiontime_last_2_list[i] = actiontime_last_2
    actiontime_last_3_list[i] = actiontime_last_3
    actiontime_last_4_list[i] = actiontime_last_4
    actiontime_last_5_list[i] = actiontime_last_5
    actiontime_last_6_list[i] = actiontime_last_6
    actiontime_last_7_list[i] = actiontime_last_7
    actiontime_last_8_list[i] = actiontime_last_8
    actiontime_last_9_list[i] = actiontime_last_9
    actiontime_last_10_list[i] = actiontime_last_10
    actiontime_last_11_list[i] = actiontime_last_11
    actiontime_last_12_list[i] = actiontime_last_12
    actiontime_last_13_list[i] = actiontime_last_13
    actiontime_last_14_list[i] = actiontime_last_14
    actiontime_last_15_list[i] = actiontime_last_15
    actiontime_last_16_list[i] = actiontime_last_16
    actiontime_last_17_list[i] = actiontime_last_17
    actiontime_last_18_list[i] = actiontime_last_18
    actiontime_last_19_list[i] = actiontime_last_19
    actiontime_last_20_list[i] = actiontime_last_20
    actiontypeprop_1_list[i] = actiontypeprop_1
    actiontypeprop_2_list[i] = actiontypeprop_2
    actiontypeprop_3_list[i] = actiontypeprop_3
    actiontypeprop_4_list[i] = actiontypeprop_4
    actiontypeprop_5_list[i] = actiontypeprop_5
    actiontypeprop_6_list[i] = actiontypeprop_6
    actiontypeprop_7_list[i] = actiontypeprop_7
    actiontypeprop_8_list[i] = actiontypeprop_8
    actiontypeprop_9_list[i] = actiontypeprop_9
    actiontimespancount_1_5_list[i] = actiontimespancount_1_5
    actiontimespancount_5_6_list[i] = actiontimespancount_5_6
    actiontimespancount_6_7_list[i] = actiontimespancount_6_7
    actiontimespancount_7_8_list[i] = actiontimespancount_7_8
    actiontimespancount_8_9_list[i] = actiontimespancount_8_9
    actionratio_24_59_list[i] = actionratio_24_59
    actiontype_lasttime_1_list[i] = actiontype_lasttime_1
    actiontype_lasttime_5_list[i] = actiontype_lasttime_5
    actiontype_lasttime_6_list[i] = actiontype_lasttime_6
    actiontype_lasttime_7_list[i] = actiontype_lasttime_7
    actiontype_lasttime_8_list[i] = actiontype_lasttime_8
    actiontype_lasttime_9_list[i] = actiontype_lasttime_9
    actiontype_lasttime_24_list[i] = actiontype_lasttime_24
    actiontimespanlast_1_5_list[i] = actiontimespanlast_1_5
    actiontimespanlast_5_6_list[i] = actiontimespanlast_5_6
    actiontimespanlast_6_7_list[i] = actiontimespanlast_6_7
    actiontimespanlast_7_8_list[i] = actiontimespanlast_7_8
    actiontimespanlast_5_7_list[i] = actiontimespanlast_5_7
    actiontimespanlast_5_8_list[i] = actiontimespanlast_5_8
    action59seqentialratio_list[i] = action59seqentialratio
    actiontypeproplast20_1_list[i] = actiontypeproplast20_1
    actiontypeproplast20_2_list[i] = actiontypeproplast20_2
    actiontypeproplast20_3_list[i] = actiontypeproplast20_3
    actiontypeproplast20_4_list[i] = actiontypeproplast20_4
    actiontypeproplast20_5_list[i] = actiontypeproplast20_5
    actiontypeproplast20_6_list[i] = actiontypeproplast20_6
    actiontypeproplast20_7_list[i] = actiontypeproplast20_7
    actiontypeproplast20_8_list[i] = actiontypeproplast20_8
    actiontypeproplast20_9_list[i] = actiontypeproplast20_9
    actiontime_1_list[i] = actiontime_1
df['action_sum'] = sum_action_list
df['actiontime_last_1_year'] = actiontime_last_1_year_list
df['actiontime_last_1_month'] = actiontime_last_1_month_list
df['actiontype_last_1'] = action_last_1_list
df['actiontype_last_2'] = action_last_2_list
df['actiontype_last_3'] = action_last_3_list
df['actiontype_last_4'] = action_last_4_list
df['actiontype_last_5'] = action_last_5_list
df['actiontype_last_6'] = action_last_6_list
df['actiontype_last_7'] = action_last_7_list
df['actiontype_last_8'] = action_last_8_list
df['actiontype_last_9'] = action_last_9_list
df['actiontype_last_10'] = action_last_10_list
df['actiontype_last_11'] = action_last_11_list
df['actiontype_last_12'] = action_last_12_list
df['actiontype_last_13'] = action_last_13_list
df['actiontype_last_14'] = action_last_14_list
df['actiontype_last_15'] = action_last_15_list
df['actiontype_last_16'] = action_last_16_list
df['actiontype_last_17'] = action_last_17_list
df['actiontype_last_18'] = action_last_18_list
df['actiontype_last_19'] = action_last_19_list
df['actiontype_last_20'] = action_last_20_list
df['actiontime_last_1'] = actiontime_last_1_list
df['actiontime_last_2'] = actiontime_last_2_list
df['actiontime_last_3'] = actiontime_last_3_list
df['actiontime_last_4'] = actiontime_last_4_list
df['actiontime_last_5'] = actiontime_last_5_list
df['actiontime_last_6'] = actiontime_last_6_list
df['actiontime_last_7'] = actiontime_last_7_list
df['actiontime_last_8'] = actiontime_last_8_list
df['actiontime_last_9'] = actiontime_last_9_list
df['actiontime_last_10'] = actiontime_last_10_list
df['actiontime_last_11'] = actiontime_last_11_list
df['actiontime_last_12'] = actiontime_last_12_list
df['actiontime_last_13'] = actiontime_last_13_list
df['actiontime_last_14'] = actiontime_last_14_list
df['actiontime_last_15'] = actiontime_last_15_list
df['actiontime_last_16'] = actiontime_last_16_list
df['actiontime_last_17'] = actiontime_last_17_list
df['actiontime_last_18'] = actiontime_last_18_list
df['actiontime_last_19'] = actiontime_last_19_list
df['actiontime_last_20'] = actiontime_last_20_list
df['actiontypeprop_1'] = actiontypeprop_1_list
df['actiontypeprop_2'] = actiontypeprop_2_list
df['actiontypeprop_3'] = actiontypeprop_3_list
df['actiontypeprop_4'] = actiontypeprop_4_list
df['actiontypeprop_5'] = actiontypeprop_5_list
df['actiontypeprop_6'] = actiontypeprop_6_list
df['actiontypeprop_7'] = actiontypeprop_7_list
df['actiontypeprop_8'] = actiontypeprop_8_list
df['actiontypeprop_9'] = actiontypeprop_9_list
df['actiontimespancount_1_5'] = actiontimespancount_1_5_list
df['actiontimespancount_5_6'] = actiontimespancount_5_6_list
df['actiontimespancount_6_7'] = actiontimespancount_6_7_list
df['actiontimespancount_7_8'] = actiontimespancount_7_8_list
df['actiontimespancount_8_9'] = actiontimespancount_8_9_list
df['actionratio_24_59'] = actionratio_24_59_list
df['actiontype_lasttime_1'] = actiontype_lasttime_1_list
df['actiontype_lasttime_5'] = actiontype_lasttime_5_list
df['actiontype_lasttime_6'] = actiontype_lasttime_6_list
df['actiontype_lasttime_7'] = actiontype_lasttime_7_list
df['actiontype_lasttime_8'] = actiontype_lasttime_8_list
df['actiontype_lasttime_9'] = actiontype_lasttime_9_list
df['actiontype_lasttime_24'] = actiontype_lasttime_24_list
df['actiontimespanlast_1_5'] = actiontimespanlast_1_5_list
df['actiontimespanlast_5_6'] = actiontimespanlast_5_6_list
df['actiontimespanlast_6_7'] = actiontimespanlast_6_7_list
df['actiontimespanlast_7_8'] = actiontimespanlast_7_8_list
df['actiontimespanlast_5_7'] = actiontimespanlast_5_7_list
df['actiontimespanlast_5_8'] = actiontimespanlast_5_8_list
df['action59seqentialratio'] = action59seqentialratio_list
df['actiontypeproplast20_1'] = actiontypeproplast20_1_list
df['actiontypeproplast20_2'] = actiontypeproplast20_2_list
df['actiontypeproplast20_3'] = actiontypeproplast20_3_list
df['actiontypeproplast20_4'] = actiontypeproplast20_4_list
df['actiontypeproplast20_5'] = actiontypeproplast20_5_list
df['actiontypeproplast20_6'] = actiontypeproplast20_6_list
df['actiontypeproplast20_7'] = actiontypeproplast20_7_list
df['actiontypeproplast20_8'] = actiontypeproplast20_8_list
df['actiontypeproplast20_9'] = actiontypeproplast20_9_list
df['actiontime_1'] = actiontime_1_list
print("get tags from action complite!")
print("start get tags from comment----------------------------")
# get tags from comment
rating_ave_list = np.zeros(usercount)
rating_min_list = np.zeros(usercount)
rating_last_list = np.zeros(usercount)
for i in range(usercount):
    rating_ave, rating_min, rating_last = getTagsFromCommentByUserid(df_comm, df_profile['userid'][i])
    rating_ave_list[i] = rating_ave
    rating_min_list[i] = rating_min
    rating_last_list[i] = rating_last
df['rating_mean'] = rating_ave_list
df['rating_min'] = rating_min_list
df['rating_last'] = rating_last_list
print("get tags from comment complite!")
print("start get tags from profile----------------------------")
# get tags from profile
df_userprofile_train = pd.read_csv('./data_train_test/userProfile_train.csv')
provinceset = np.array(list(set(df_userprofile_train['province'])))
hasgender_list = np.zeros(usercount)
ismale_list = np.zeros(usercount)
isfemale_list = np.zeros(usercount)
hasage_list = np.zeros(usercount)
is60_list = np.zeros(usercount)
is70_list = np.zeros(usercount)
is80_list = np.zeros(usercount)
is90_list = np.zeros(usercount)
is00_list = np.zeros(usercount)
province_all_dflist = pd.DataFrame()
for i in range(usercount):
    hasgender, ismale, isfemale, hasage, is60, is70, is80, is90, is00, province_df = getTagsFromProfileByUserid(df_profile, df_profile['userid'][i], provinceset)
    hasgender_list[i] = hasgender
    ismale_list[i] = ismale
    isfemale_list[i] = isfemale
    hasage_list[i] = hasage
    is60_list[i] = is60
    is70_list[i] = is70
    is80_list[i] = is80
    is90_list[i] = is90
    is00_list[i] = is00
    province_all_dflist = province_all_dflist.append(province_df)
df['gender_exist'] = hasgender_list
df['gender_male'] = ismale_list
df['gender_female'] = isfemale_list
df['age_exist'] = hasage_list
df['age_60'] = is60_list
df['age_70'] = is70_list
df['age_80'] = is80_list
df['age_90'] = is90_list
df['age_00'] = is00_list
province_all_dflist['id'] = range(0, usercount)
province_all_dflist = province_all_dflist.set_index('id')
province_all_dflist.rename(columns={'nan': 'hasprovince'}, inplace=True)
df = pd.concat([df, province_all_dflist], axis=1)
print("get tags from profile complite!")
print("start get other information between different tables----------------------------")
# get other information between different tables
#df = pd.read_csv('./result/data_train.csv', encoding='gb2312')
#df = pd.read_csv('./result/data_test.csv', encoding='gb2312')
df['timespan_action_lastord'] = df['actiontime_last_1'] - df['histord_time_last_1']
df['timespan_action1tolast'] = df['actiontime_last_1'] - df['actiontype_lasttime_1']
df['timespan_action5tolast'] = df['actiontime_last_1'] - df['actiontype_lasttime_5']
df['timespan_action6tolast'] = df['actiontime_last_1'] - df['actiontype_lasttime_6']
df['timespan_action7tolast'] = df['actiontime_last_1'] - df['actiontype_lasttime_7']
df['timespan_action8tolast'] = df['actiontime_last_1'] - df['actiontype_lasttime_8']
df['timespan_action9tolast'] = df['actiontime_last_1'] - df['actiontype_lasttime_9']
df['timespan_action24tolast'] = df['actiontime_last_1'] -  df['actiontype_lasttime_24']
df['timespan_lastord_1_2'] = df['histord_time_last_1'] - df['histord_time_last_2']
df['timespan_lastord_2_3'] = df['histord_time_last_2'] - df['histord_time_last_3']
df['timespan_last_1'] = (df['actiontime_last_1'] - df['actiontime_last_2'])
df['timespan_last_2'] = (df['actiontime_last_2'] - df['actiontime_last_3'])
df['timespan_last_3'] = (df['actiontime_last_3'] - df['actiontime_last_4'])
df['timespan_last_4'] = (df['actiontime_last_4'] - df['actiontime_last_5'])
df['timespan_last_5'] = (df['actiontime_last_5'] - df['actiontime_last_6'])
df['timespan_last_6'] = (df['actiontime_last_6'] - df['actiontime_last_7'])
df['timespan_last_7'] = (df['actiontime_last_7'] - df['actiontime_last_8'])
df['timespan_last_8'] = (df['actiontime_last_8'] - df['actiontime_last_9'])
df['timespan_last_9'] = (df['actiontime_last_9'] - df['actiontime_last_10'])
df['timespanmean_last_3'] = np.mean(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3']], axis=1)
df['timespanmin_last_3'] = np.min(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3']], axis=1)
df['timespanmax_last_3'] = np.max(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3']], axis=1)
df['timespanstd_last_3'] = np.std(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3']], axis=1)
df['timespanmean_last_4'] = np.mean(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4']], axis=1)
df['timespanmin_last_4'] = np.min(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4']], axis=1)
df['timespanmax_last_4'] = np.max(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4']], axis=1)
df['timespanstd_last_4'] = np.std(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4']], axis=1)
df['timespanmean_last_5'] = np.mean(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5']], axis=1)
df['timespanmin_last_5'] = np.min(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5']], axis=1)
df['timespanmax_last_5'] = np.max(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5']], axis=1)
df['timespanstd_last_5'] = np.std(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5']], axis=1)
df['timespanmean_last_6'] = np.mean(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6']], axis=1)
df['timespanmin_last_6'] = np.min(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6']], axis=1)
df['timespanmax_last_6'] = np.max(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6']], axis=1)
df['timespanstd_last_6'] = np.std(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6']], axis=1)
df['timespanmean_last_7'] = np.mean(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6', 'timespan_last_7']], axis=1)
df['timespanmin_last_7'] = np.min(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6', 'timespan_last_7']], axis=1)
df['timespanmax_last_7'] = np.max(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6', 'timespan_last_7']], axis=1)
df['timespanstd_last_7'] = np.std(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6', 'timespan_last_7']], axis=1)
df['timespanmean_last_8'] = np.mean(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6', 'timespan_last_7', 'timespan_last_8']], axis=1)
df['timespanmin_last_8'] = np.min(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6', 'timespan_last_7', 'timespan_last_8']], axis=1)
df['timespanmax_last_8'] = np.max(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6', 'timespan_last_7', 'timespan_last_8']], axis=1)
df['timespanstd_last_8'] = np.std(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6', 'timespan_last_7', 'timespan_last_8']], axis=1)
df['timespanmean_last_9'] = np.mean(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6', 'timespan_last_7', 'timespan_last_8', 'timespan_last_9']], axis=1)
df['timespanmin_last_9'] = np.min(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6', 'timespan_last_7', 'timespan_last_8', 'timespan_last_9']], axis=1)
df['timespanmax_last_9'] = np.max(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6', 'timespan_last_7', 'timespan_last_8', 'timespan_last_9']], axis=1)
df['timespanstd_last_9'] = np.std(df[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5', 'timespan_last_6', 'timespan_last_7', 'timespan_last_8', 'timespan_last_9']], axis=1)
df['timespan_total'] = df['actiontime_last_1'] - df['actiontime_1']
print("get other information between different tables complite!")
print("start myfeat ----------------------------")
def timediff(timediff):
    days = (timediff/86400)
      #计算小时数
    remain = timediff%86400
    hours = (remain/3600)
      #计算分钟数
    remain = remain%3600
    mins = (remain/60)
      #计算秒数
    secs = remain%60
    res = secs/86400 + mins/1440 + hours + days*24
    return res


def simple_feat(userP,orderf,userC,action):
    df = pd.merge(userP,orderf,how="left",on="userid")
    df = pd.merge(df,userC,how="left",on="userid")
    count_1 = action[action['actionType'] == 1].groupby(["userid"],as_index=False).actionType.agg({'count_1':np.sum})
    count_234 = action[(action['actionType'] == 2) | (action['actionType'] == 3) | (action['actionType'] == 4)].groupby(["userid"],as_index=False).actionType.count()
    count_234.rename(columns={'actionType':'count_234'}, inplace = True)
    count_5 = action[(action['actionType'] == 5)].groupby(["userid"],as_index=False).actionType.count()
    count_6 = action[(action['actionType'] == 6)].groupby(["userid"],as_index=False).actionType.count()
    count_7 = action[(action['actionType'] == 7)].groupby(["userid"],as_index=False).actionType.count()
    count_8 = action[(action['actionType'] == 8)].groupby(["userid"],as_index=False).actionType.count()
    count_9 = action[(action['actionType'] == 9)].groupby(["userid"],as_index=False).actionType.count()
    count_5.rename(columns={'actionType':'count_5'}, inplace = True)
    count_6.rename(columns={'actionType':'count_6'}, inplace = True)
    count_7.rename(columns={'actionType':'count_7'}, inplace = True)
    count_8.rename(columns={'actionType':'count_8'}, inplace = True)
    count_9.rename(columns={'actionType':'count_9'}, inplace = True)
    count = pd.merge(count_1,count_234,how="left",on="userid")
    count = pd.merge(count,count_5,how="left",on="userid")
    count = pd.merge(count,count_6,how="left",on="userid")
    count = pd.merge(count,count_7,how="left",on="userid")
    count = pd.merge(count,count_8,how="left",on="userid")
    count = pd.merge(count,count_9,how="left",on="userid")
    df = pd.merge(df,count,how="left",on="userid")
    df1 = df.drop(["tags","commentsKeyWords"],axis=1)

    one_hot_gebder = pd.get_dummies(df1['gender'],prefix='gender_onehot')
    one_hot_province = pd.get_dummies(df1['province'],prefix='province_onehot')
    one_hot_age = pd.get_dummies(df1['age'],prefix='age_onehot')
    df1 = pd.concat([df1,one_hot_gebder,one_hot_province,one_hot_age],axis=1)
    return df1

def action_count_feat(df1):
    df1['count_all'] = df1[['count_1',
                            'count_234',
                            'count_5',
                            'count_5', 'count_6', 'count_7', 'count_8', 'count_9']].apply(lambda x: x.sum(), axis=1)
    df1["count_1_rate"] = df1["count_1"] / df1["count_all"]
    df1["count_234_rate"] = df1["count_234"] / df1["count_all"]
    df1["count_5_rate"] = df1["count_5"] / df1["count_all"]
    df1["count_6_rate"] = df1["count_6"] / df1["count_all"]
    df1["count_7_rate"] = df1["count_7"] / df1["count_all"]
    df1["count_8_rate"] = df1["count_8"] / df1["count_all"]
    df1["count_9_rate"] = df1["count_9"] / df1["count_all"]
    df1["count_234_5_rate"] = df1["count_234"] / df1["count_5"]
    df1["count_234_9_rate"] = df1["count_234"] / df1["count_9"]
    return df1

def action_time_feat(action):
    action_1_time = action[action["actionType"] == 1].drop_duplicates(['userid'], 'first')[['userid', 'actionTime']]
    action_1_time.columns = ['userid', '1_first_time']
    action_234_time = \
    action[(action['actionType'] == 2) | (action['actionType'] == 3) | (action['actionType'] == 4)].drop_duplicates(
        ['userid'], 'first')[['userid', 'actionTime']]
    action_234_time.columns = ['userid', '234_first_time']
    action_5_time = action[action["actionType"] == 5].drop_duplicates(['userid'], 'first')[['userid', 'actionTime']]
    action_5_time.columns = ['userid', '5_first_time']
    action_6_time = action[action["actionType"] == 6].drop_duplicates(['userid'], 'first')[['userid', 'actionTime']]
    action_6_time.columns = ['userid', '6_first_time']
    action_7_time = action[action["actionType"] == 6].drop_duplicates(['userid'], 'first')[['userid', 'actionTime']]
    action_7_time.columns = ['userid', '7_first_time']
    action_8_time = action[action["actionType"] == 6].drop_duplicates(['userid'], 'first')[['userid', 'actionTime']]
    action_8_time.columns = ['userid', '8_first_time']
    action_9_time = action[action["actionType"] == 6].drop_duplicates(['userid'], 'first')[['userid', 'actionTime']]
    action_9_time.columns = ['userid', '9_first_time']

    action_time = pd.merge(action_1_time, action_234_time, on=["userid"], how="left")
    action_time = pd.merge(action_time, action_5_time, on=["userid"], how="left")
    action_time = pd.merge(action_time, action_6_time, on=["userid"], how="left")
    action_time = pd.merge(action_time, action_7_time, on=["userid"], how="left")
    action_time = pd.merge(action_time, action_8_time, on=["userid"], how="left")
    action_time = pd.merge(action_time, action_9_time, on=["userid"], how="left")

    action_time["1_5_diff_time"] = action_time["5_first_time"] - action_time["1_first_time"]
    action_time["234_5_diff_time"] = action_time["5_first_time"] - action_time["234_first_time"]
    action_time["5_6_diff_time"] = action_time["6_first_time"] - action_time["5_first_time"]
    action_time["6_7_diff_time"] = action_time["7_first_time"] - action_time["6_first_time"]
    action_time["7_8_diff_time"] = action_time["8_first_time"] - action_time["7_first_time"]
    action_time["8_9_diff_time"] = action_time["9_first_time"] - action_time["8_first_time"]
    action_time["234_9_diff_time"] = action_time["9_first_time"] - action_time["234_first_time"]
    action_time["5_9_diff_time"] = action_time["9_first_time"] - action_time["5_first_time"]

    # action_time["1_5_diff_time"] = action_time["1_5_diff_time"].apply(lambda x: x % 86400 % 3600 // 60)
    # action_time["234_5_diff_time"] = action_time["234_5_diff_time"].apply(lambda x: x % 86400 % 3600 // 60)
    # action_time["5_6_diff_time"] = action_time["5_6_diff_time"].apply(lambda x: x % 86400 % 3600 // 60)
    # action_time["6_7_diff_time"] = action_time["6_7_diff_time"].apply(lambda x: x % 86400 % 3600 // 60)
    # action_time["7_8_diff_time"] = action_time["7_8_diff_time"].apply(lambda x: x % 86400 % 3600 // 60)
    # action_time["8_9_diff_time"] = action_time["8_9_diff_time"].apply(lambda x: x % 86400 % 3600 // 60)
    # action_time["234_9_diff_time"] = action_time["234_9_diff_time"].apply(lambda x: x % 86400 % 3600 // 60)
    # action_time["5_9_diff_time"] = action_time["5_9_diff_time"].apply(lambda x: x % 86400 % 3600 // 60)
    action_time["1_5_diff_time"] = action_time["1_5_diff_time"].apply(timediff)
    action_time["234_5_diff_time"] = action_time["234_5_diff_time"].apply(timediff)
    action_time["5_6_diff_time"] = action_time["5_6_diff_time"].apply(timediff)
    action_time["6_7_diff_time"] = action_time["6_7_diff_time"].apply(timediff)
    action_time["7_8_diff_time"] = action_time["7_8_diff_time"].apply(timediff)
    action_time["8_9_diff_time"] = action_time["8_9_diff_time"].apply(timediff)
    action_time["234_9_diff_time"] = action_time["234_9_diff_time"].apply(timediff)
    action_time["5_9_diff_time"] = action_time["5_9_diff_time"].apply(timediff)


    action_time = action_time[
        ["userid", "234_5_diff_time", "5_6_diff_time", "6_7_diff_time", "7_8_diff_time", "8_9_diff_time",
         "234_9_diff_time", "5_9_diff_time", "1_5_diff_time"]]
    return action_time


df1 = simple_feat(df_profile,df_ordfuture,df_comm,df_action)
df1 = action_count_feat(df1)
action_time = action_time_feat(action)
df1 = pd.merge(df1,action_time,on="userid",how="left")
print("myfeat complite!")
pd.merge(df,df1,on="userid",how="left")
pd.DataFrame(df).to_csv('./result/data_train.csv', header=True, index=False)
#pd.DataFrame(df).to_csv('./result/data_test.csv', header=True, index=False)
print("complite!")
