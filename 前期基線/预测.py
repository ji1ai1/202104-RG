# python 3.9.2
# python包 lightgbm 3.1.1
# python包 numpy 1.20.1
# python包 pandas 1.2.2
#
# 输入：
#	trainset/person.csv
#	trainset/person_cv.csv
#	trainset/person_job_hist.csv
#	trainset/person_pro_cert.csv
#	trainset/person_project.csv
#	trainset/recruit.csv
#	trainset/recruit_folder.csv
#	testset/recruit_folder.csv
#
# 输出：
# 	result.csv
#
# 0.8516
#
import numpy
import pandas
import random
import sklearn
import lightgbm

求职者表 = pandas.read_csv("trainset/person.csv", header=0, names=["求职者编号", "性別", "工作年限", "最高学历", "应聘者专业", "年龄", "最近工作岗位", "最近所在行业", "当前工作所在地", "语言能力", "专业特长"])
求职者表.性別 = (求职者表.性別 == "女").astype("float")
求职者表.最高学历 = 求职者表.最高学历.map({"其它": 0, "中专": 1, "高中（职高、技校）": 2, "大专": 3, "大学本科": 4, "硕士研究生": 5, "博士研究生": 6, "博士后": 7})
求职者表.应聘者专业 = 求职者表.应聘者专业.astype("category")
意向表 = pandas.read_csv("trainset/person_cv.csv", header=0, names=["求职者编号", "自荐信", "岗位类别", "工作地点", "所在行业", "可到职天数", "其他说明"])
意向表["自荐信字数"] = 意向表.自荐信.str.len()
工作经历表 = pandas.read_csv("trainset/person_job_hist.csv", header=0, names=["求职者编号", "岗位类别", "单位所在地", "单位所属行业", "主要业绩"])
工作经历表["主要业绩字数"] = 工作经历表.主要业绩.str.len()
专业證書表 = pandas.read_csv("trainset/person_pro_cert.csv", header=0, names=["求职者编号", "专业證書名称", "備註"])
项目经验表 = pandas.read_csv("trainset/person_project.csv", header=0, names=["求职者编号", "项目名称", "项目说明", "职责说明", "关键技术"])
岗位表 = pandas.read_csv("trainset/recruit.csv", header=0, names=["岗位编号", "招聘对象代码", "招聘对象", "招聘职位", "对应聘者的专业要求", "岗位最低学历", "岗位工作地点", "岗位工作年限", "具体要求"])
岗位表.招聘对象代码 = 岗位表.招聘对象代码.fillna(-1).astype("category")
岗位表.招聘对象 = 岗位表.招聘对象.astype("category")
岗位表.岗位最低学历 = 岗位表.岗位最低学历.map({"其它": 0, "中专": 1, "高中（职高、技校）": 2, "大专": 3, "大学本科": 4, "硕士研究生": 5, "博士研究生": 6, "博士后": 7})
岗位表.岗位工作年限 = 岗位表.岗位工作年限.map({"不限": -1, "应届毕业生": 0, "0至1年": 0, "1至2年": 1, "3至5年": 3, "5年以上": 5})
岗位表["具体要求字数"] = 岗位表.具体要求.str.len()

工作经历数据表 = 工作经历表.groupby("求职者编号").aggregate({"岗位类别": "count", "主要业绩字数": ["mean", "sum"]}).reset_index()
工作经历数据表.columns = ["求职者编号", "工作经历数", "平均主要业绩字数", "总主要业绩字数"]
项目经验資料表 = 项目经验表.groupby("求职者编号").aggregate({"项目名称": "count"}).reset_index()
项目经验資料表.columns = ["求职者编号", "项目经验数"]

训练表 = pandas.read_csv("trainset/recruit_folder.csv", header=0, names=["岗位编号", "求职者编号", "标签"])
测试表 = pandas.read_csv("testset/recruit_folder.csv", header=0, names=["岗位编号", "求职者编号", "标签"])

测訓表 = pandas.concat([测试表, 训练表], ignore_index=True)
求职者数据表 = 测訓表.groupby("求职者编号").aggregate({"岗位编号": "count"}).reset_index()
求职者数据表.columns = ["求职者编号", "求职者数"]
岗位数据表 = 测訓表.groupby("岗位编号").aggregate({"求职者编号": "count"}).reset_index()
岗位数据表.columns = ["岗位编号", "岗位数"]


def 取得数据表(某表, 某特征表):
	某特征求职者数据表 = 某特征表.groupby("求职者编号").aggregate({"标签": "mean"}).reset_index()
	某特征求职者数据表.columns = ["求职者编号", "求职者平均标签"]
	某特征岗位数据表 = 某特征表.groupby("岗位编号").aggregate({"标签": "mean"}).reset_index()
	某特征岗位数据表.columns = ["岗位编号", "岗位平均标签"]
	
	某表 = 某表.merge(求职者表, on="求职者编号", how="left")
	某表 = 某表.merge(意向表, on="求职者编号", how="left")
	某表 = 某表.merge(岗位表, on="岗位编号", how="left")
	某表 = 某表.merge(项目经验資料表, on="求职者编号", how="left")
	某表 = 某表.merge(工作经历数据表, on="求职者编号", how="left")
	某表 = 某表.merge(求职者数据表, on="求职者编号", how="left")
	某表 = 某表.merge(岗位数据表, on="岗位编号", how="left")
	某表 = 某表.merge(某特征求职者数据表, on="求职者编号", how="left")
	某表 = 某表.merge(某特征岗位数据表, on="岗位编号", how="left")
	某表["工作地点符合否"] = (某表.工作地点 == 某表.岗位工作地点).astype("float")
	
	某資料表 = 某表.loc[:, ["岗位编号", "求职者编号", "标签"
		 , "性別", "工作年限", "最高学历", "应聘者专业", "年龄", "自荐信字数", "可到职天数"
		 , "项目经验数"
		 , "工作经历数", "平均主要业绩字数", "总主要业绩字数"
		 , "招聘对象代码", "招聘对象", "岗位最低学历", "岗位工作年限", "具体要求字数", "工作地点符合否"
		 , "求职者数", "岗位数"
		 , "求职者平均标签", "岗位平均标签"
	]]
	
	某資料表 = 某資料表.loc[:, ["岗位编号", "求职者编号", "标签"] + [子 for 子 in 某資料表.columns if 子 not in ["岗位编号", "求职者编号", "标签"]]]
	
	return 某資料表


折数 = 4
训练数据表 = None
for 甲 in range(折数):
	甲标签表 = 训练表[训练表.index % 折数 == 甲].reset_index(drop=True)
	甲特征表 = 训练表[训练表.index % 折数 != 甲].reset_index(drop=True)
	
	甲数据表 = 取得数据表(甲标签表, 甲特征表)
	训练数据表 = pandas.concat([训练数据表, 甲数据表], ignore_index=True)

轻模型 = lightgbm.train(train_set=lightgbm.Dataset(训练数据表.iloc[:, 3:], label=训练数据表.标签)
	, num_boost_round=500, params={"objective": "binary", "learning_rate": 0.03, "max_depth": 6, "num_leaves": 32, "verbose": -1, "bagging_fraction": 0.8, "feature_fraction": 0.8})

测试数据表 = 取得数据表(测试表, 训练表)
预测表 = 测试数据表.loc[:, ["岗位编号", "求职者编号"]]
预测表["预测打分"] = 轻模型.predict(测试数据表.iloc[:, 3:])
预测表 = 预测表.sort_values("预测打分", ascending=False, ignore_index=True)
预测表["预测"] = 0
预测表.loc[:int(0.15 * len(预测表)), ["预测"]] = 1


提交表 = 预测表.loc[:, ["岗位编号", "求职者编号", "预测"]]
提交表.columns = ["RECRUIT_ID", "PERSON_ID", "LABEL"]
提交表.to_csv("result.csv", index=False)
