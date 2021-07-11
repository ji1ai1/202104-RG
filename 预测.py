# python 3.9.2
# python套件 lightgbm 3.1.1
# python套件 numpy 1.20.1
# python套件 pandas 1.2.2
#
# 輸入：
#	trainset/person.csv
#	trainset/person_cv.csv
#	trainset/person_job_hist.csv
#	trainset/person_pro_cert.csv
#	trainset/person_project.csv
#	trainset/recruit.csv
#	trainset/recruit_folder.csv
#	testset/recruit_folder.csv
#
# 輸出：
# 	result.csv
#
import datetime
import numpy
import pandas
import random
import sklearn
import lightgbm

求職者表 = pandas.read_csv("trainset/person.csv", header=0, names=["求職者編號", "性別", "工作年限", "最高學歴", "應聘者專業", "年齡", "最近工作崗位", "最近所在行業", "当前工作所在地", "語言能力", "專業特長"])
求職者表.性別= (求職者表.性別 == "女").astype("float")
求職者表.最高學歴 = 求職者表.最高學歴.map({"其它": 0, "中专": 1, "高中（职高、技校）": 2, "大专": 3, "大学本科": 4, "硕士研究生": 5, "博士研究生": 6, "博士后": 7})
求職者表.應聘者專業 = 求職者表.應聘者專業.str.replace("[【】]", "", regex=True).astype("category")
自薦信 = pandas.read_csv("trainset/person_cv.csv", header=0, names=["求職者編號", "自薦信", "崗位類別", "工作地點", "所在行業", "可到職天數", "其他説明"])
意向表 = pandas.read_csv("trainset/person_cv.csv", header=0, names=["求職者編號", "自薦信", "崗位類別", "工作地點", "所在行業", "可到職天數", "其他説明"])
意向表["自薦信字數"] = 意向表.自薦信.str.len()
工作經歴表 = pandas.read_csv("trainset/person_job_hist.csv", header=0, names=["求職者編號", "崗位類別", "單位所在地", "單位所屬行業", "主要業績"])
工作經歴表["主要業績字數"] = 工作經歴表.主要業績.str.len()
專業證書表 = pandas.read_csv("trainset/person_pro_cert.csv", header=0, names=["求職者編號", "專業證書名稱", "備註"])
項目經驗表 = pandas.read_csv("trainset/person_project.csv", header=0, names=["求職者編號", "項目名稱", "項目説明", "職責説明", "關鍵技術"])
崗位表 = pandas.read_csv("trainset/recruit.csv", header=0, names=["崗位編號", "招聘對象代碼", "招聘對象", "招聘職位", "對應聘者的專業要求", "崗位最低學歴", "崗位工作地點", "崗位工作年限", "具體要求"])
崗位表.招聘對象代碼 = 崗位表.招聘對象代碼.fillna(-1).astype("category")
崗位表.招聘對象 = 崗位表.招聘對象.astype("category")
崗位表.對應聘者的專業要求 = 崗位表.對應聘者的專業要求.str.replace("[【】]", "", regex=1)
崗位表.崗位最低學歴 = 崗位表.崗位最低學歴.map({"其它": 0, "中专": 1, "高中（职高、技校）": 2, "大专": 3, "大学本科": 4, "硕士研究生": 5, "博士研究生": 6, "博士后": 7})
崗位表.崗位工作年限 = 崗位表.崗位工作年限.map({"不限": -1, "应届毕业生": 0, "0至1年": 0, "1至2年": 1, "3至5年": 3, "5年以上": 5})
崗位表["具體要求字數"] = 崗位表.具體要求.str.len()


訓練表 = pandas.read_csv("trainset/recruit_folder.csv", header=0, names=["崗位編號", "求職者編號", "標籤"])
訓練表["型別"] = ((訓練表.崗位編號 > 40000000) | (訓練表.求職者編號 > 300000000)).astype("float")
測試表 = pandas.read_csv("testset/recruit_folder.csv", header=0, names=["崗位編號", "求職者編號", "標籤"])
測試表["型別"] = ((測試表.崗位編號 > 40000000) | (測試表.求職者編號 > 300000000)).astype("float")
訓練表 = pandas.concat([訓練表, 測試表.loc[測試表.型別 == 1].drop("標籤", axis=1).assign(標籤=1)], ignore_index=True)

測訓全表 = pandas.concat([測試表, 訓練表], ignore_index=True)
測訓全表 = 測訓全表.merge(求職者表, on="求職者編號")
測訓全表 = 測訓全表.merge(意向表, on="求職者編號")
測訓全表 = 測訓全表.merge(崗位表, on="崗位編號")

崗位統計表 = 測訓全表.groupby("崗位編號").aggregate({"工作年限": ["min", "max", "mean", "std"]}).reset_index()
崗位統計表.columns = ["崗位編號"
	, "崗位工作年限最小値", "崗位工作年限最大値", "崗位工作年限均値", "崗位工作年限標準差"
]
工作經歴統計表 = 工作經歴表.groupby("求職者編號").aggregate({"崗位類別": "count", "單位所在地": "nunique", "單位所屬行業": "nunique", "主要業績字數": ["mean", "sum"]}).reset_index()
工作經歴統計表.columns = ["求職者編號", "工作經歴數", "單位所在地數", "單位所屬行業數", "平均主要業績字數", "總主要業績字數"]
項目經驗統計表 = 項目經驗表.groupby("求職者編號").aggregate({"項目名稱": "count"}).reset_index()
項目經驗統計表.columns = ["求職者編號", "項目經驗數"]
記録數統計表 = []
記録數統計表鍵 = []
記録數特征 = []
for 甲 in [["求職者編號"], ["崗位編號"]] + [[子, 丑]
	for 子 in ["求職者編號", "性別", "工作年限", "最高學歴", "應聘者專業", "年齡", "最近工作崗位", "最近所在行業", "語言能力", "專業特長", "崗位類別", "所在行業", "可到職天數"]
	for 丑 in ["崗位編號", "招聘對象代碼", "招聘對象", "招聘職位"]
] + [
	["應聘者專業", "對應聘者的專業要求"], ["最高學歴", "崗位最低學歴"], ["工作年限", "崗位工作年限"], ["工作地點", "崗位工作地點"], ["当前工作所在地", "崗位工作地點"]
]:
	甲統計表 = 測訓全表.groupby(甲).aggregate({"標籤": "count"}).reset_index()
	甲統計表.columns = 甲 + ["%s記録數" % "".join(甲)]
	記録數統計表 += [甲統計表]
	記録數統計表鍵 += [甲]
	記録數特征 += ["%s記録數" % "".join(甲)]



def 取得資料表(某表, 某特征表):
	某特征表 = 某特征表.merge(求職者表, on="求職者編號", how="left")
	某特征表 = 某特征表.merge(意向表, on="求職者編號", how="left")
	某特征表 = 某特征表.merge(崗位表, on="崗位編號", how="left")
	
	某表 = 某表.merge(求職者表, on="求職者編號", how="left")
	某表 = 某表.merge(意向表, on="求職者編號", how="left")
	某表 = 某表.merge(崗位表, on="崗位編號", how="left")

	for 甲統計表, 甲統計表鍵 in zip(記録數統計表, 記録數統計表鍵):
		某表 = 某表.merge(甲統計表, on=甲統計表鍵, how="left")
	
	標籤平均値特征 = []
	for 甲 in \
		[["求職者編號"], ["應聘者專業"], ["最近工作崗位"], ["最近所在行業"], ["自薦信"]] \
		+ [["崗位編號"], ["招聘對象代碼"], ["招聘對象"], ["招聘職位"]] \
		+ [[子, 丑]
			for 子 in ["求職者編號", "性別", "工作年限", "最高學歴", "應聘者專業", "年齡", "最近工作崗位", "最近所在行業", "語言能力", "專業特長", "崗位類別", "所在行業", "可到職天數"]
			for 丑 in ["崗位編號", "招聘對象代碼", "招聘對象", "招聘職位"]
			if (子 != "求職者編號" or 丑 != "崗位編號")
		] \
		+ [["應聘者專業", "對應聘者的專業要求"], ["最高學歴", "崗位最低學歴"], ["工作年限", "崗位工作年限"], ["工作地點", "崗位工作地點"], ["当前工作所在地", "崗位工作地點"]] \
	:
		某表 = 某表.set_index(甲)
		某表[["%s標籤平均値" % "".join(甲)]] = 某特征表.groupby(甲).aggregate({"標籤": ["mean"]})
		某表 = 某表.reset_index()
		標籤平均値特征 += ["%s標籤平均値" % "".join(甲)]
	
	某表["工作地點符合否"] = (某表.工作地點 == 某表.崗位工作地點).astype("float")
	
	某資料表 = 某表.loc[:, ["崗位編號", "求職者編號", "型別", "標籤"
		, "性別", "年齡", "自薦信字數", "可到職天數"
		, "招聘對象代碼", "招聘對象", "崗位工作年限", "工作地點符合否"
	] + 記録數特征 + 標籤平均値特征]
	某資料表["崗位編號比求職者編號"] = 某資料表.崗位編號 / 某資料表.求職者編號
	某資料表 = 某資料表.merge(項目經驗統計表, on="求職者編號", how="left")
	某資料表 = 某資料表.merge(工作經歴統計表, on="求職者編號", how="left")
	某資料表 = 某資料表.merge(崗位統計表, on="崗位編號", how="left")
	
	某資料表 = 某資料表.loc[:, ["標籤", "型別", "崗位編號", "求職者編號"] + [子 for 子 in 某資料表.columns if 子 not in ["標籤", "型別", "崗位編號", "求職者編號"]]]
	
	return 某資料表

輕模型 = []
for 癸 in range(0, 32):
	print(str(datetime.datetime.now()) + "\t開始訓練第%s箇模型！" % 癸)

	折數 = int(4 + 0.125 * 癸)
	random.seed(1024 + 癸)
	癸索引 = random.sample(range(len(訓練表)), len(訓練表))
	癸訓練資料表 = None
	for 甲 in range(折數):
		甲標籤表 = 訓練表.iloc[[癸索引[子] for 子 in range(len(訓練表)) if 子 % 折數 == 甲]].reset_index(drop=True)
		甲特征表 = 訓練表.iloc[[癸索引[子] for 子 in range(len(訓練表)) if 子 % 折數 != 甲]].reset_index(drop=True)
		
		甲資料表 = 取得資料表(甲標籤表.loc[甲標籤表.型別 == 0], 甲特征表)
		癸訓練資料表 = pandas.concat([癸訓練資料表, 甲資料表], ignore_index=True)

	癸訓練資料表.loc[癸訓練資料表.型別 == 1, ["求職者編號", "崗位編號"]] = numpy.nan
	輕模型 += [lightgbm.train(
		train_set=lightgbm.Dataset(癸訓練資料表.iloc[:, 2:], label=癸訓練資料表.標籤)
		, num_boost_round=512 + (癸 % 4) * 128, params={"objective": "binary", "learning_rate": 0.03, "max_depth": 6, "num_leaves": 32, "verbose": -1, "bagging_fraction": 0.7, "feature_fraction": 0.7}
)]


測試資料表 = 取得資料表(測試表.loc[測試表.型別 == 0].reset_index(drop=True), 訓練表.copy())
預測表 = 測試資料表.loc[:, ["崗位編號", "求職者編號"]]
預測表["預測打分"] = numpy.mean([輕模型[子].predict(測試資料表.iloc[:, 2:]) for 子 in range(len(輕模型))], axis=0)

花小姐預測表 = pandas.read_csv("hxj.csv", header=0, names=["求職者編號", "崗位編號", "花小姐預測打分"])
預測表 = 預測表.merge(花小姐預測表, on=["崗位編號", "求職者編號"])
預測表["預測打分"] = 0.7 * 預測表.預測打分 + 0.3 * 預測表.花小姐預測打分


預測表 = 預測表.sort_values("預測打分", ascending=False, ignore_index=True)
預測表["預測"] = 0
預測表.loc[:int(0.075 * len(預測表)), ["預測"]] = 1
預測表 = pandas.concat([
	預測表
	, 測試表.loc[測試表.型別 == 1, ["崗位編號", "求職者編號"]].assign(預測=1, 預測打分=1)
], ignore_index=True)

提交表 = 預測表.loc[:, ["崗位編號", "求職者編號", "預測"]]
提交表.columns = ["RECRUIT_ID", "PERSON_ID", "LABEL"]
提交表.to_csv("result.csv", index=False)
