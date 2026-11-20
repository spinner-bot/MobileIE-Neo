# DCT分支双零初始化：静态条件推导

2026-11-20，Codex  (sqs_A100)。依据固定530c692…的author-source/model/utils.py:60–124、323–379，以及model/__init__.py和config/uie.yaml。没有执行Torch、反向传播、训练或加载checkpoint。

令D为固定DW-DCT算子，P为无bias的PW线性权重，g为gamma；该分支为y=tanh(g)·P·D(x)。源码g=0（81行），P=0（97行）。故在此点，dy/dP=tanh(0)D(x)=0；dy/dg=sech²(0)PD(x)=0。输入及上游损失梯度有限时，两可学因子都无一阶梯度。独立DW固定且不能改变该事实；后续concat/conv_out不使这两个零梯度复活，DCT对应融合列对输入零也无该通路贡献。

默认config原网、pretrained:false、need_slim:false，main标准Adam/weight_decay0，没有显式对这两者扰动或非零重初始化。因此在fresh默认初始化、常规有限数值梯度与无外部非零state下，这一分支会持续不贡献信号。“gamma零所以逐步有益激活”的论文解释不足以覆盖该双零实现。即使PW/g无信号，其他普通卷积/注意力仍可正常学习，不能说整个网络不能训练或所有性能无效。

另有DCTAddIntoConv3x3的coeff/gamma也双零，但本默认MBRConv3实际调用DCTLinearBranch；不能用未调用类证明实际路径。全部可得源码未发现对活跃分支后续重置，已归档训练/导入路径；若非默认权重加载或历史源码不同，上述默认训练推断不适用。

这不是对作者历史实验的运行证明。公开slim checkpoint不含训练分支的可核历史，作者log无源码/权重哈希；没有据此宣称论文DCT消融伪造。解锁需要实际对应的训练配置/原始checkpoint/源码版本及分支状态/梯度证据，届时按另阶段协议核查，不修改本固定归档。当前不擅自修作者实现、改变论文对照或执行Gate3。
