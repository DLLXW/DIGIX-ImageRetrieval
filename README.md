# 华为DIGIX图像检索
华为DIGIX图像检索

- 首先下载原始数据,然后需要运行utils/splitTrainVal.py将原始数据分割为训练和验证两部分.
- modeling/models.py下面的LandmarkNet类目录下定义了模型(res50,101,pnasnet三个模型)
- 在config/config.py下面设置好数据路径,优化器,loss类型,pooling类型等训练参数
- 运行train_arcLoss.py可以进行训练
- 
