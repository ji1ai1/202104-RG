import datetime
import numpy
import pandas
import random
import sklearn
import lightgbm

求職者表 = pandas.read_csv("trainset/person.csv", header=0, names=["求職者編號", "性別", "工作年限", "最高學歴", "應聘者專業", "年齡", "最近工作崗位", "最近所在行業", "当前工作所在地", "語言能力", "專業特長"])
求職者表.性別 = (求職者表.性別 == "女").astype("float")
求職者表.最高學歴 = 求職者表.最高學歴.map({"其它": 0, "中专": 1, "高中（职高、技校）": 2, "大专": 3, "大学本科": 4, "硕士研究生": 5, "博士研究生": 6, "博士后": 7})
求職者表.應聘者專業 = 求職者表.應聘者專業.astype("category")
意向表 = pandas.read_csv("trainset/person_cv.csv", header=0, names=["求職者編號", "自薦信", "崗位類別", "工作地點", "所在行業", "可到職天數", "其他説明"])
意向表["自薦信字數"] = 意向表.自薦信.str.len()
工作經歴表 = pandas.read_csv("trainset/person_job_hist.csv", header=0, names=["求職者編號", "崗位類別", "單位所在地", "單位所屬行業", "主要業績"])
工作經歴表["主要業績字數"] = 工作經歴表.主要業績.str.len()
專業證書表 = pandas.read_csv("trainset/person_pro_cert.csv", header=0, names=["求職者編號", "專業證書名稱", "備註"])
項目經驗表 = pandas.read_csv("trainset/person_project.csv", header=0, names=["求職者編號", "項目名稱", "項目説明", "職責説明", "關鍵技術"])
崗位表 = pandas.read_csv("trainset/recruit.csv", header=0, names=["崗位編號", "招聘對象代碼", "招聘對象", "招聘職位", "對應聘者的專業要求", "崗位最低學歴", "崗位工作地點", "崗位工作年限", "具體要求"])
崗位表.招聘對象代碼 = 崗位表.招聘對象代碼.fillna(-1).astype("category")
崗位表.招聘對象 = 崗位表.招聘對象.astype("category")
崗位表.崗位最低學歴 = 崗位表.崗位最低學歴.map({"其它": 0, "中专": 1, "高中（职高、技校）": 2, "大专": 3, "大学本科": 4, "硕士研究生": 5, "博士研究生": 6, "博士后": 7})
崗位表.崗位工作年限 = 崗位表.崗位工作年限.map({"不限": -1, "应届毕业生": 0, "0至1年": 0, "1至2年": 1, "3至5年": 3, "5年以上": 5})
崗位表["具體要求字數"] = 崗位表.具體要求.str.len()

工作經歴資料表 = 工作經歴表.groupby("求職者編號").aggregate({"崗位類別": "count", "主要業績字數": ["mean", "sum"]}).reset_index()
工作經歴資料表.columns = ["求職者編號", "工作經歴數", "平均主要業績字數", "總主要業績字數"]
項目經驗資料表 = 項目經驗表.groupby("求職者編號").aggregate({"項目名稱": "count"}).reset_index()
項目經驗資料表.columns = ["求職者編號", "項目經驗數"]

訓練表 = pandas.read_csv("trainset/recruit_folder.csv", header=0, names=["崗位編號", "求職者編號", "標籤"])
測試表 = pandas.read_csv("testset/recruit_folder.csv", header=0, names=["崗位編號", "求職者編號", "標籤"])

測訓表 = pandas.concat([測試表, 訓練表], ignore_index=True)
求職者資料表 = 測訓表.groupby("求職者編號").aggregate({"崗位編號": "count"}).reset_index()
求職者資料表.columns = ["求職者編號", "求職者數"]
崗位資料表 = 測訓表.groupby("崗位編號").aggregate({"求職者編號": "count"}).reset_index()
崗位資料表.columns = ["崗位編號", "崗位數"]


def 取得資料表(某表, 某特征表):
	某特征求職者資料表 = 某特征表.groupby("求職者編號").aggregate({"標籤": "mean"}).reset_index()
	某特征求職者資料表.columns = ["求職者編號", "求職者平均標籤"]
	某特征崗位資料表 = 某特征表.groupby("崗位編號").aggregate({"標籤": "mean"}).reset_index()
	某特征崗位資料表.columns = ["崗位編號", "崗位平均標籤"]
	
	某表 = 某表.merge(求職者表, on="求職者編號", how="left")
	某表 = 某表.merge(意向表, on="求職者編號", how="left")
	某表 = 某表.merge(崗位表, on="崗位編號", how="left")
	某表 = 某表.merge(項目經驗資料表, on="求職者編號", how="left")
	某表 = 某表.merge(工作經歴資料表, on="求職者編號", how="left")
	某表 = 某表.merge(求職者資料表, on="求職者編號", how="left")
	某表 = 某表.merge(崗位資料表, on="崗位編號", how="left")
	某表 = 某表.merge(某特征求職者資料表, on="求職者編號", how="left")
	某表 = 某表.merge(某特征崗位資料表, on="崗位編號", how="left")
	某表["工作地點符合否"] = (某表.工作地點 == 某表.崗位工作地點).astype("float")
	
	某資料表 = 某表.loc[:, ["崗位編號", "求職者編號", "標籤"
		, "性別", "工作年限", "最高學歴", "應聘者專業", "年齡", "自薦信字數", "可到職天數"
		, "項目經驗數"
		, "工作經歴數", "平均主要業績字數", "總主要業績字數"
		, "招聘對象代碼", "招聘對象", "崗位最低學歴", "崗位工作年限", "具體要求字數", "工作地點符合否"
		, "求職者數", "崗位數"
		, "求職者平均標籤", "崗位平均標籤"
	]]
	
	某資料表 = 某資料表.loc[:, ["崗位編號", "求職者編號", "標籤"] + [子 for 子 in 某資料表.columns if 子 not in ["崗位編號", "求職者編號", "標籤"]]]
	
	return 某資料表


折數 = 4
訓練資料表 = None
for 甲 in range(折數):
	甲標籤表 = 訓練表[訓練表.index % 折數 == 甲].reset_index(drop=True)
	甲特征表 = 訓練表[訓練表.index % 折數 != 甲].reset_index(drop=True)
	
	甲資料表 = 取得資料表(甲標籤表, 甲特征表)
	訓練資料表 = pandas.concat([訓練資料表, 甲資料表], ignore_index=True)

輕模型 = lightgbm.train(train_set=lightgbm.Dataset(訓練資料表.iloc[:, 3:], label=訓練資料表.標籤)
	, num_boost_round=500, params={"objective": "binary", "learning_rate": 0.03, "max_depth": 6, "num_leaves": 32, "verbose": -1, "bagging_fraction": 0.8, "feature_fraction": 0.8})

測試資料表 = 取得資料表(測試表, 訓練表)
預測表 = 測試資料表.loc[:, ["崗位編號", "求職者編號"]]
預測表["預測打分"] = 輕模型.predict(測試資料表.iloc[:, 3:])
預測表 = 預測表.sort_values("預測打分", ascending=False, ignore_index=True)
預測表["預測"] = 0
預測表.loc[:int(0.15 * len(預測表)), ["預測"]] = 1


提交表 = 預測表.loc[:, ["崗位編號", "求職者編號", "預測"]]
提交表.columns = ["RECRUIT_ID", "PERSON_ID", "LABEL"]
提交表.to_csv("result.csv", index=False)
