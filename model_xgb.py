# -*- coding: utf-8 -*-

import os
os.chdir('E:/develop/projects/hbc')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, ensemble, metrics, grid_search, model_selection, decomposition, linear_model
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from matplotlib.pyplot import rcParams
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.datasets import load_svmlight_file
plt.rcParams['font.sans-serif']=['SimHei']
rcParams['figure.figsize'] = 80, 10
cross_val_score()
# load training&test set
df_train = pd.read_csv('./result/data_train.csv', encoding='gb2312')
df_test = pd.read_csv('./result/data_test.csv', encoding='gb2312')
idcol = 'userid'
target = 'futureOrderType'
drop_tags = [idcol, target]
drop_tags = [idcol, target,
             'actiontype_last_8', 'actiontype_last_9', 'actiontype_last_10', 'actiontype_last_11', 'actiontype_last_12', 'actiontype_last_13', 'actiontype_last_14', 'actiontype_last_15', 'actiontype_last_16', 'actiontype_last_17', 'actiontype_last_18', 'actiontype_last_19', 'actiontype_last_20',
             'actiontime_last_8', 'actiontime_last_9', 'actiontime_last_10', 'actiontime_last_11', 'actiontime_last_12', 'actiontime_last_13', 'actiontime_last_14', 'actiontime_last_15', 'actiontime_last_16', 'actiontime_last_17', 'actiontime_last_18', 'actiontime_last_19', 'actiontime_last_20']

drop_tags = [idcol, target,
             #'actiontypeprop_1', 'actiontypeprop_2', 'actiontypeprop_3', 'actiontypeprop_4', 'actiontypeprop_5', 'actiontypeprop_6', 'actiontypeprop_7', 'actiontypeprop_8', 'actiontypeprop_9',
             'timespanmean_last_4', 'timespanmean_last_7', 'timespanmean_last_8', 'timespanmean_last_9',
             'actiontype_last_8', 'actiontype_last_9', 'actiontype_last_10', 'actiontype_last_11', 'actiontype_last_12', 'actiontype_last_13', 'actiontype_last_14', 'actiontype_last_15', 'actiontype_last_16', 'actiontype_last_17', 'actiontype_last_18', 'actiontype_last_19', 'actiontype_last_20',
             'actiontime_last_8', 'actiontime_last_9', 'actiontime_last_10', 'actiontime_last_11', 'actiontime_last_12', 'actiontime_last_13', 'actiontime_last_14', 'actiontime_last_15', 'actiontime_last_16', 'actiontime_last_17', 'actiontime_last_18', 'actiontime_last_19', 'actiontime_last_20',
             'histord_time_last_2', 'histord_time_last_2_year', 'histord_time_last_2_month', 'histord_time_last_3', 'histord_time_last_3_year', 'histord_time_last_3_month', '新竹', '布里斯班', '丹佛', '成田市', '莫斯科', '维多利亚', '上海', '雷克雅未克', '开普敦', '京都', '墨西哥城', '牛津', '阿姆斯特丹', '都柏林', '东京', '珀斯', '华盛顿', '卡萨布兰卡', '亚特兰大', '巴尔的摩', '法兰克福', '富山市', '惠灵顿', '萨尔茨堡', '富国岛', '戛纳', '仰光', '名古屋', '台南', '堪培拉', '冲绳--那霸', '新加坡.1', '马赛', '达拉斯', '垦丁', '北海道--札幌', '马尼拉', '温哥华', '澳门', '苏梅岛', '波尔多', '斯德哥尔摩', '温莎', '巴伦西亚', '渥太华', '里昂', '迈阿密', '卢塞恩', '西雅图', '奥兰多', '岐阜县', '汉密尔顿', '波士顿', '尼斯', '那不勒斯', '拉科鲁尼亚', '多哈', '华沙', '云顶高原', '横滨', '圣地亚哥', '考文垂', '福冈', '霍巴特', '台中', '墨尔本', '冲绳市', '毛里求斯.1', '洛杉矶', '蒙特利尔', '曼彻斯特', '嘉义', '大叻', '伊尔库茨克', '楠迪', '阿里山', '布拉格', '沙巴--亚庇', '日内瓦', '蒂卡波湖', '里斯本', '巴黎', '釜山', '洞爷湖', '静冈县', '普吉岛', '台北', '夏威夷欧胡岛（檀香山）', '科茨沃尔德', '科隆', '夏威夷大岛', '开罗', '凯恩斯', '圣保罗', '威尼斯', '旧金山', '奈良', '赫尔辛基', '哥本哈根', '阿布扎比', '御殿场市', '万象', '哈尔施塔特', '富良野', '阿德莱德', '香港', '岘港', '金泽', '高雄', '巴厘岛', '谢菲尔德', '罗马', '哥德堡', '济州岛', '北海道--登别', '马拉加', '华欣', '塞班岛', '塞维利亚', '北海道--旭川', '吉隆坡', '新山', '奥克兰', '马德里', '斯克兰顿（宾夕法尼亚州）', '槟城', '格拉纳达', '伦敦', '黄金海岸', '皇后镇', '轻井泽', '暹粒', '广岛', '大西洋城', '汉堡', '神户', '河内', '布法罗', '福森', '巴塞罗那', '富士河口湖', '宜兰', '多伦多', '夏威夷茂宜岛', '胡志明市', '纽约', '星野度假村', '熊本市', '约克', '布达佩斯', '奥斯陆', '盐湖城', '杜塞尔多夫', '阿维尼翁', '爱丁堡', '千叶市', '维也纳', '雅典', '大阪', '圣彼得堡', '清迈', '巴斯', '芭堤雅', '芝加哥', '富士山', '贝尔法斯特', '费城', '迪拜', '基督城', '新北', '加德满都', '底特律', '柏林', '箱根', '金边', '花莲', '米兰', '曼谷', '利瓦绿洲', '甲米', '佛罗伦萨', '因特拉肯', '慕尼黑', '波尔图', '约翰内斯堡', '苏黎世', '北海道--小樽', '休斯敦', '拉斯维加斯', '布鲁塞尔', '千叶', '阿尔勒', '摩纳哥', '长崎', '悉尼', '芽庄', '卡尔加里', '兰卡威', '首尔', '雅加达', '宿务', '北海道--函馆', '伊斯坦布尔', '伯明翰', '南投', 'action_sum', 'rating_min', 'rating_last', 'gender_exist', 'gender_male', 'gender_female'
             ]
x_tags = [x for x in df_train.columns if x not in drop_tags]
y_tag = target

X_train = np.array(df_train[x_tags])
y_train = np.array(df_train[y_tag])

X_test = np.array(df_test[x_tags])


def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds)
        print('best n_estimators:', cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    # fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='auc')
    # predict train set
    dtrain_pred = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:,1]
    # print model report
    print('\nModel Report:')
    print("AUC Score (Train): %.5g" % metrics.roc_auc_score(y_train, dtrain_predprob))
    print("Accuracy: %.5g" % metrics.accuracy_score(y_train, dtrain_pred))
    # plot feature importance
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')

# prediction model
# params tuning
# find n_estimators for a high learning rate
np.random.seed(272)
model_xgb1 = XGBClassifier(
        learning_rate=0.03,
        n_estimators=3000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=8,
        scale_pos_weight=1,
        seed=272
        )
modelfit(model_xgb1, X_train, y_train)
#model_xgb1.fit(X_train, y_train)
print('score_AUC:', round(metrics.roc_auc_score(y_train, model_xgb1.predict_proba(X_train)[:,1]), 5))
print('score_precision:', round(metrics.accuracy_score(y_train, model_xgb1.predict(X_train)), 5))
scores_cross = model_selection.cross_val_score(model_xgb1, X_train, y_train, cv=5, scoring='roc_auc')
print('score_cross:', round(np.mean(scores_cross), 5), 'std:', round(np.std(scores_cross), 5))


# grid search on max_depth and min_child_weight
param_test1 = {
        'max_depth': [3,5,7,9],
        'min_child_weight': [1,3,5]
        }
gsearch1 = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=424, max_depth=5,
                                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=314),
        param_grid=param_test1, scoring='roc_auc', iid=False, cv=5
        )
gsearch1.fit(X_train, y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

param_test2 = {
        'max_depth':[6,7,8],
        'min_child_weight':[4,5,6]
}
gsearch2 = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=424, max_depth=7,
                                min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=314),
        param_grid=param_test2, scoring='roc_auc', iid=False, cv=5
        )
gsearch2.fit(X_train, y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

param_test2b = {
        'min_child_weight':[6,8,10]
}
gsearch2b = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=424, max_depth=7,
                                min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=314),
        param_grid=param_test2b, scoring='roc_auc', iid=False, cv=5
        )
gsearch2b.fit(X_train, y_train)
gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_

# tune gamma
param_test3 = {
        'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=424, max_depth=7,
                                min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=314),
        param_grid=param_test3, scoring='roc_auc', iid=False, cv=5
        )
gsearch3.fit(X_train, y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

model_xgb2 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=7,
        min_child_weight=6,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=314
        )
modelfit(model_xgb2, X_train, y_train)

# tune subsample and colsample_bytree
param_test4 = {
        'subsample': [i/10.0 for i in range(6,10)],
        'colsample_bytree': [i/10.0 for i in range(6,10)]
}
gsearch4 = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=240, max_depth=7,
                                min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=314),
        param_grid=param_test4, scoring='roc_auc', iid=False, cv=5
        )
gsearch4.fit(X_train, y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

param_test5 = {
        'subsample': [0.85,0.9],
        'colsample_bytree': [0.85,0.9]
}
gsearch5 = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=240, max_depth=7,
                                min_child_weight=6, gamma=0.1, subsample=0.9, colsample_bytree=0.9,
                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=314),
        param_grid=param_test5, scoring='roc_auc', iid=False, cv=5
        )
gsearch5.fit(X_train, y_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

# tuning Regularization Parameters
param_test6 = {
        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=240, max_depth=7,
                                min_child_weight=6, gamma=0.1, subsample=0.85, colsample_bytree=0.9,
                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=314),
        param_grid=param_test6, scoring='roc_auc', iid=False, cv=5
        )
gsearch6.fit(X_train, y_train)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

param_test7 = {
        'reg_alpha':[0.3, 0.5, 0.7]
}
gsearch7 = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=240, max_depth=7,
                                min_child_weight=6, gamma=0.1, subsample=0.85, colsample_bytree=0.9,
                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=314),
        param_grid=param_test7, scoring='roc_auc', iid=False, cv=5
        )
gsearch7.fit(X_train, y_train)
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_

model_xgb3 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=7,
        min_child_weight=6,
        gamma=0.1,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=314
        )
modelfit(model_xgb3, X_train, y_train)

# reducing Learning Rate
model_xgb4 = XGBClassifier(
        learning_rate=0.01,
        n_estimators=5000,
        max_depth=7,
        min_child_weight=6,
        gamma=0.1,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=314
        )
modelfit(model_xgb4, X_train, y_train)

### final model
model_xgb_final = model_xgb1
#model_xgb_final.fit(X_train, y_train)
y_train_pred = model_xgb_final.predict(X_train)
y_train_predprob = model_xgb_final.predict_proba(X_train)[:,1]
importances = model_xgb_final.feature_importances_
df_featImp = pd.DataFrame({'tags': x_tags, 'importance': importances})
df_featImp_sorted = df_featImp.sort_values(by=['importance'], ascending=False)
df_featImp_sorted.plot(x='tags', y='importance', kind='bar')
df_featImp.to_csv('./feature/feat_2017-12-29-4c.csv')
df_featImp_sorted.to_csv('./feature/feat_2017-12-29-4c.csv')
#featcount = 151
#x_tags = df_featImp_sorted[:featcount]['tags'].tolist()


print('score_AUC:', round(metrics.roc_auc_score(y_train, y_train_predprob), 5))
print('score_precision:', round(metrics.accuracy_score(y_train, y_train_pred), 5))
scores_cross = model_selection.cross_val_score(model_xgb_final, X_train, y_train, cv=5, scoring='roc_auc')
print('score_cross:', round(np.mean(scores_cross), 5), 'std:', round(np.std(scores_cross), 5))

# write out prediction result
y_test_pred = model_xgb_final.predict_proba(X_test)[:,1]

#y_test_pred_super = np.zeros(len(y_test_pred))
#alpha = 0.5
#for i in range(len(y_test_pred)):
#    if y_test_pred[i] < 0.5:
#        y_test_pred_super[i] = y_test_pred[i] - alpha * (y_test_pred[i] - 0)
#    elif y_test_pred[i] > 0.5:
#        y_test_pred_super[i] = y_test_pred[i] + alpha * (1.0 - y_test_pred[i])

df_profile = pd.read_csv('./data_train_test/userProfile_test.csv')
restable = pd.DataFrame(np.concatenate((np.array(df_profile['userid']).reshape((-1,1)), y_test_pred.reshape((-1,1))), axis=1))
restable.loc[:,0] = restable.loc[:,0].astype(np.int64)
pd.DataFrame(restable).to_csv("./result/orderFuture_test-20171229-4c.csv", header=['userid','orderType'], index=False)
