目录结构
=======
**DB4AI**\
|-- operators &nbsp;&nbsp;**//自定义算子的样例**\
|-- pic &nbsp;&nbsp;**//md文件图片资源**\
|-- test &nbsp;&nbsp;**//自定义算子和算法的测试文件**\
|-- Analyze_expression.py &nbsp;&nbsp;**//解析类SQL语言中的函数和表达式** \
|-- DB4AI_OPs.zip &nbsp;&nbsp;**//OpenGauss内部算子代码实现和文档**\
|-- Digraph.py &nbsp;&nbsp;**//计算图**\
|-- Executor.py &nbsp;&nbsp;**//执行器，负责执行计算图并向数据库动态发送算子执行命令（debug中）**\
|-- FirstLevelLanguageParser.py &nbsp;&nbsp;**//任务型查询语言解析器**\
|-- gdbc.py &nbsp;&nbsp;**//数据库连接组件**\
|-- Nodes.py &nbsp;&nbsp;**//Python端的算子，包含前向和反向传播函数以及自动微分，执行时将对应命令发送到数据库在数据库内计算（debug中）**\
|-- requirements.py &nbsp;&nbsp;**//Python环境**\
|-- SecondLevelLanguageParser.py &nbsp;&nbsp;**//描述型查询语言解析器（有待和演示系统整合）**\
|-- UserOperatorInfo.py &nbsp;&nbsp;**//用户自定义算子计算图存储**\
|-- UserOperatorName.py &nbsp;&nbsp;**//用户自定义算子列表**\
|-- utils.py &nbsp;&nbsp;**//工具类**\

系统设计
=======
<font size=3>
为了灵活地编写运行于数据库内部的机器学习/深度学习程序，本系统采用两级类SQL查询语言描述并执行机器学习任务，将查询语言描述的计算过程解析为有向循环计算图，该计算图中包含数据流和控制流信息。

执行器按规则执行计算图，将算子的执行指令发送至数据库内部进行数据处理和计算。同时执行器负责在python端维护与数据库中张量一一对应的张量对象（不存储数据），用来构成动态自动微分图，并支持基于引用计数的自动内存回收
____________________
系统支持的特性:
* 广播
* 自动微分
* 中间结果垃圾回收

支持的机器学习任务范围：
* 基于线性代数计算的传统统计学习算法
* 低维深度学习算法
* 其他依赖于线性代数计算和自动微分的计算任务


</font>

![](D:\PycharmProjects\DB4AI\pic\SQL-like.png)

剩余工作
========
<font size=3>

* 广播操作改为数据库内算子（doing...）
* 自动微分梯度值存在错误（debug....）
* 性能调优
* 演示系统整合

</font>

