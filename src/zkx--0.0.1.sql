
-------------------------------------------------------------------
--            openGaussOPs.c中相应函数的注册sql部分                 
--                 宋明阳所负责的部分 需求V2.0
--
--              使用时请将其整合至对应源文件部分
--               oriented：~/postgres/OPs/opengaussOPs.sql
--               located in:~/soft/openGauss-server/contrib/zkx/zkx.cpp V3.0
-----------------------------------------------------------------


------------just testing--------------------------
 --select qp4ai_abs('a','a1');
 --select * from a;
------------------------------------------------------------
-- qp4ai_abs(input_table_name TEXT, output_table_name TEXT)
-- input_table_name ???????????
-- output_table_name ???????????
----------------------------------------------------------
CREATE OR REPLACE FUNCTION
qp4ai_abs(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_absP20FunctionCallInfoData' -- ?????????????·?? ??????????.so?е??????
LANGUAGE C STRICT;
------------the end------------------------------


---------------------------------------------------------
-- select qp4ai_sub('a1','a2','res');
-- select qp4ai_sub('mtrx','mtrx2','res');
-- select * from smy_output;
------------------------------------------------------------
-- qp4ai_sub(input_table1_name TEXT,input_table2_name TEXT,output_table_name TEXT)
-- input_table1_name 
-- input_table2_name 
-- output_table_name
-------------------------------------------
CREATE OR REPLACE FUNCTION
qp4ai_sub(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_subP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

------------------------------------------------------------
-- select qp4ai_sum('a1',1,'res');
-- select qp4ai_sum('mtrx',0,'res');
-- select * from smy_output;
------------------------------------------------------------
-- qp4ai_sum(input_table_name TEXT,ndim INT,output_table_name TEXT)
-- input_table_name 要做求和运算的矩阵表名
-- ndim 要求和的方向（维度）
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码  0：正常 -1:输入矩阵不存在 -2：ndim不为1或0
-- 效果 若ndim = 0，则将输入矩阵按列求和，若ndim = 1则将输入矩阵按行求和，结果保存在输出矩阵中
------------------------------------------------------------

CREATE OR REPLACE FUNCTION
qp4ai_sum(input_table_name TEXT, ndim INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_sumP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

------------------------------------------------------------
-- select qp4ai_sqrt('a2','res');
-- select qp4ai_sqrt('mtrx','res');
-- select * from smy_output;
------------------------------------------------------------
-- qp4ai_sqrt(input_table_name TEXT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求平方根，形成输出矩阵表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
qp4ai_sqrt(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z10qp4ai_sqrtP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;



------------------------------------------------------------
--select qp4ai_sort('mtrx',1,'res');
--select qp4ai_sort('mtrx2',0,'res');
------------------------------------------------------------
-- qp4ai_sort(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- dim 要进行排序的维度，0为按列排序，1为按行排序
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表按维度进行排序，结果保存在输出表中
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
qp4ai_sort(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z10qp4ai_sqrtP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;


------------------------------------------------------------
--select qp4ai_softmax('mtrx',0,'res');
--select * from smy_output;
------------------------------------------------------------
-- qp4ai_softmax(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- dim 要进行softmax运算的维度，0为按列计算（按列归一化），1为按行计算（按行归一化）
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵进行softmax运算
------------------------------------------------------------

CREATE OR REPLACE FUNCTION
qp4ai_softmax(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z13qp4ai_softmaxP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;


------------------------------------------------------------
-- select qp4ai_tensordot('ft1','ft2',1,'res');
-- select qp4ai_tensordot('mtrx2','mtrx3',1,'res');
------------------------------------------------------------
-- qp4ai_tensordot(input_table_name1 TEXT,input_table_name2 TEXT, dim INT, output_table_name TEXT)
-- input_table_name1 输入的矩阵1表名
-- input_table_name2 输入的矩阵2表名
-- dim 要进行tensordot运算的维度，这部分的意义详细请参阅pytorch官方文档（只能是1或2！）
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的2个矩阵进行tensordot运算
------------------------------------------------------------

CREATE OR REPLACE FUNCTION
qp4ai_tensordot(input_table_name1 TEXT,input_table_name2 TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z15qp4ai_tensordotP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;


------------------------------------------------------------
--select qp4ai_matmul('mtrx2','mtrx3','res');
------------------------------------------------------------
-- qp4ai_matmul(input_table_name1 TEXT,input_table_name2 TEXT, output_table_name TEXT)
-- input_table_name1 输入的矩阵1表名
-- input_table_name2 输入的矩阵2表名
-- output_table_name 输出的矩阵表名
-- 要求 qp4ai_tensordot 已经注册
-- 返回 执行状态码
-- 效果 将输入的2个矩阵进行matmul运算,得到两个矩阵的乘积
------------------------------------------------------------

CREATE OR REPLACE FUNCTION
qp4ai_matmul(input_table_name1 TEXT,input_table_name2 TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z12qp4ai_matmulP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--复用tensordot的内部实现--基本一致--

------------------------------------------------------------
--select qp4ai_dot('ft1','ft2','res');
------------------------------------------------------------
-- qp4ai_dot(input_table_name1 TEXT,input_table_name2 TEXT, output_table_name TEXT)
-- input_table_name1 输入的矩阵（向量）1表名
-- input_table_name2 输入的矩阵（向量）2表名
-- output_table_name 输出的矩阵（向量）表名
-- 要求 qp4ai_tensordot 已经注册
-- 返回 执行状态码
-- 效果 将输入的2个向量进行内积运算
------------------------------------------------------------

CREATE OR REPLACE FUNCTION
qp4ai_dot(input_table_name1 TEXT,input_table_name2 TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_dotP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--复用tensordot的内部实现--基本一致--

------------------------------------------------------------
--select qp4ai_argsort('mtrx',1,'res');
------------------------------------------------------------
-- qp4ai_argsort(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- dim 要进行排序的维度，0为按列排序，1为按行排序
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入矩阵表按维度进行排序，将排序的顺序信息保存在输出表中
------------------------------------------------------------

CREATE OR REPLACE FUNCTION
qp4ai_argsort(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z13qp4ai_argsortP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;


------------------------------------------------------------
-- select qp4ai_acc('mtrx','mtrx2',0,'res');
-- select qp4ai_acc('ft1','ft2',0,'res');
------------------------------------------------------------
-- qp4ai_acc(input_table1_name TEXT,input_table2_name TEXT, normalize INT, output_table_name TEXT)
-- input_table1_name 矩阵1表名
-- input_table2_name 矩阵2表名
-- normalize 是否进行标准化标记
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 比较两矩阵差异，返回相同的个数（或比例），即计算训练得分
------------------------------------------------------------

CREATE OR REPLACE FUNCTION
qp4ai_acc(input_table1_name TEXT, input_table2_name TEXT, normalize INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_accP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

------------------------------------------------------------
--select qp4ai_reverse('mtrx2',0,'res');
--select qp4ai_reverse('mtrx',1,'res');
--select qp4ai_reverse('mtrx',2,'res');
------------------------------------------------------------
-- qp4ai_reverse(input_table_name TEXT,ndim INT,output_table_name TEXT)
-- input_table_name 要做翻转运算的矩阵表名
-- ndim 要翻转的方向（维度）
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码  0：正常 -1:输入矩阵不存在 -6：ndim不为1或0
-- 效果 若ndim = 0，则将输入矩阵按列翻转，若ndim = 1则将输入矩阵按行翻转，若ndim=2 则将输入矩阵全部翻转（二维）
-- 注意 不同于pytorch的调用方式
------------------------------------------------------------

CREATE OR REPLACE FUNCTION
qp4ai_reverse(input_table_name TEXT, ndim INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z13qp4ai_reverseP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

------------------------------------------------------------
-- select qp4ai_select('a2');
-- select qp4ai_select('mtrx');
-- select qp4ai_repeat('a2',2,3,'res');
-- select qp4ai_repeat('mtrx',2,3,'res');
-- select * from smy_output;
------------------------------------------------------------
-- db4ai_repeat(input_table_name TEXT,dim1 INT, dim2 INT,output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- dim1 行上重复的次数
-- dim2 列上重复的次数
-- output_table_name 输出的向量表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表重复，形成输出矩阵表，返回状态码
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
qp4ai_repeat(input_table_name TEXT,dim1 INT, dim2 INT,output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z12qp4ai_repeatP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
------------------------------------------------------------
-- select qp4ai_random(3,4,2,0.8,1.0,0.5,'res');
-- select qp4ai_random(3,4,1,1,5,0,'res');
-- select * from smy_output;
------------------------------------------------------------
-- db4ai_random(dim1 INT, dim2 INT, _distribution TEXT, _args TEXT, output_table_name TEXT)
-- dim1 行数
-- dim2 列数
-- _distribution 分布方式: Normal(0) Uniform(1) Bernoulli(2)
-- ——args 更多参数，和_distribution参数有关：
    -- Supported parameters:
        -- Normal: mu, sigma
        -- Uniform: min, max
        -- Bernoulli: lower, upper, prob
-- output_table_name 输出的表名
-- 返回 执行状态码
-- 效果 创建一个dim1行dim2列的名为output_table_name的随机矩阵表
-- 实例 select qp4ai_random(2,10,0,0.8,0.2,0,'res');
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
qp4ai_random(dim1 INT, dim2 INT, _distribution INT, arg1 float8, arg2 float8, arg3 float8, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z12qp4ai_randomP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

-------------------Nov 25 recoding the ops-----------------------
---------------------------------------------------------
-- select qp4ai_pow('a1',2,'res');
-- select qp4ai_pow('mtrx',2,'res');
-- select * from smy_output;
------------------------------------------------------------
-- db4ai_pow(input_table_name TEXT,pow_exp float8,output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- pow_exp 为每个元素进行指数运算的指数值
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求相应的指数，形成输出矩阵表
-------------------------------------------
CREATE OR REPLACE FUNCTION
qp4ai_pow(input_table_name TEXT,pow_exp FLOAT8,output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_powP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

---------------------------------------------------------
-- select qp4ai_select('a2');
-- select qp4ai_argmin('a2',0,'res');
-- select qp4ai_argmin('mtrx',0,'res');
-- select * from smy_output;
------------------------------------------------------------
-- db4ai_argmin(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- dim 处理的维度
-- output_table_name 输出的表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都找最小值的索引，形成输出表
-- 注意 输入的dim是0或者1，0列1行
-- 测试
    -- SELECT db4ai_argmin('a',1,'argmin');
    -- SELECT * FROM argmin;
-------------------------------------------
CREATE OR REPLACE FUNCTION
qp4ai_argmin(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z12qp4ai_argminP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_argmax(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- dim 处理的维度
-- output_table_name 输出的表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都找最大值的索引，形成输出表
-- 注意 输入的dim是0或者1，0列1行
-- 测试
-- select qp4ai_select('a2');
-- select qp4ai_select('mtrx');
-- select qp4ai_argmax('a2',1,'res');
-- select qp4ai_argmax('mtrx',1,'res');
-- select * from smy_output;
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
qp4ai_argmax(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z12qp4ai_argmaxP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
------------------------------------------------------------
-- select qp4ai_select('a2');
-- select qp4ai_select('mtrx');
-- select qp4ai_full('a2',1.1,'res');
-- select qp4ai_full('mtrx',1.1,'res');
-- select * from smy_output;
------------------------------------------------------------
-- db4ai_full(input_table_name TEXT,full_value float8,output_table_name TEXT)
-- input_table_name 输入的表名
-- full_value 填充值
-- output_table_name 输出的表名
-- 返回 执行状态码
-- 效果 将输入表内的值全部改成full_value并生成输出表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
qp4ai_full(input_table_name TEXT, full_value float8,output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z10qp4ai_fullP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
------------------------------------------------------------
-- select qp4ai_select('a2');
-- select qp4ai_select('mtrx');
-- select qp4ai_log('a2','res');
-- select qp4ai_log('mtrx','res');
-- select * from smy_output;
------------------------------------------------------------
-- db4ai_log(input_table_name TEXT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求对数，形成输出矩阵表。注意负数会变成nan。
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
qp4ai_log(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_logP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
------------------------------------------------------------
-- select qp4ai_select('a2');
-- select qp4ai_select('mtrx');
-- select qp4ai_max('a2',1,'res');
-- select qp4ai_max('mtrx',1,'res');
-- select * from smy_output;
------------------------------------------------------------
-- qp4ai_max(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表1名
-- dim 聚合最小值的维度 1为按列 2为按行
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表按选择的维度寻找最大值，形成数值表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
qp4ai_max(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_maxP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

------------------------------------------------------------
-- select qp4ai_select('a2');
-- select qp4ai_select('mtrx');
-- select qp4ai_mean('a2',2,'res');
-- select qp4ai_mean('mtrx',2,'res');
-- select * from smy_output;
------------------------------------------------------------
-- qp4ai_mean(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表1名
-- dim 聚合最小值的维度 0为按列 or 1为按行 or 2为求整个矩阵所有元素的均值
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表按选择的维度计算均值，形成数值表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
qp4ai_mean(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z10qp4ai_meanP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
------------------------------------------------------------
-- select qp4ai_select('a2');
-- select qp4ai_select('mtrx');
-- select qp4ai_min('a2',1,'res');
-- select qp4ai_min('mtrx',1,'res');
-- select * from smy_output;
------------------------------------------------------------
-- qp4ai_min(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表1名
-- dim 聚合最小值的维度 1为按列 2为按行
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表按选择的维度寻找最小值，形成数值表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
qp4ai_min(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_minP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
------------------------------------------------------------
-- select qp4ai_select('a2');
-- select qp4ai_select('mtrx');
-- select qp4ai_shape('a2','res');
-- select qp4ai_shape('mtrx','res');
-- select * from smy_output;
------------------------------------------------------------
-- db4ai_shape(input_table_name TEXT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表求行数和列数，形成输出矩阵表，返回状态码
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
qp4ai_shape(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z11qp4ai_shapeP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
------------------------------------------------------------
-- select qp4ai_select('a2');
-- select qp4ai_select('mtrx');
-- select qp4ai_slice('a2','res');
-- select qp4ai_slice('mtrx',0,1,0,0,'res');
-- select * from smy_output;
------------------------------------------------------------
-- db4ai_slice(input_table_name TEXT, dim1_start INT, dim1_end INT, dim2_start INT, dim2_end INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- dim1_start 起始行
-- dim1_end 终止行
-- dim2_start 起始列
-- dim2_end 终止列
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表切片，保存到输出表中。
-- 测试 select db4ai_slice('a', 2,2,3,3,'slice');
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
qp4ai_slice(input_table_name TEXT, dim1_start INT, dim1_end INT, dim2_start INT, dim2_end INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z11qp4ai_sliceP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
------------------------------------------------------------
-- select qp4ai_select('a2');
-- select qp4ai_select('mtrx');
-- select qp4ai_trace('a2','res');
-- select qp4ai_trace('mtrx','res');
-- select * from smy_output;
------------------------------------------------------------
-- db4ai_trace(input_table_name TEXT, output_table_name TEXT)
-- input_table_name 输入的表名
-- output_table_name 输出的表名
-- 返回 执行状态码 如果不是方阵则返回-3。
-- 效果 将输入的表名对应的方阵求trace后返回，和其它张量操作不同。
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
qp4ai_trace(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z11qp4ai_traceP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

------------------------------------------------------------
-- select qp4ai_select('a2');
-- select qp4ai_select('mtrx');
-- select qp4ai_pow_table('a2',1.0,'res');
-- select qp4ai_pow_table('mtrx',1.0,'res');
-- select * from smy_output;
------------------------------------------------------------
-- db4ai_pow_table(input_table_name TEXT,pow_buttom float8,output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- pow_exp 为每个元素进行指数运算的指数值
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求相应的指数，形成输出矩阵表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
qp4ai_pow_table(input_table_name TEXT,pow_buttom TEXT,output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z15qp4ai_pow_tableP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

------------------------------------------------------------
-- select qp4ai_f1('ft1','ft2','res');
------------------------------------------------------------
-- qp4ai_f1(y_true_table_name TEXT,y_pred_table_name TEXT, output_table_name TEXT)
-- input_table_name2 输入的真值(ground truth)表名
-- input_table_name2 输入的预测值(predict value)表名
-- output_table_name 输出的矩阵表名(其中只有一个元素，即为f1的值)
-- 返回 执行状态码
-- 效果 将输入的预测表和真值表进行precision-recall运算，返回求取的F1函数
------------------------------------------------------------

CREATE OR REPLACE FUNCTION
qp4ai_f1(input_table_name1 TEXT,input_table_name2 TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z8qp4ai_f1P20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;


CREATE OR REPLACE FUNCTION
qp4ai_ones(row_num INT,col_num INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z10qp4ai_onesP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

--@brief to get matrix count in matrixMap
--select qp4ai_printCount();
--count will just show up in the return value
CREATE OR REPLACE FUNCTION
qp4ai_printCount()
RETURNS INT
AS 'zkx','_Z16outer_printCountP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

--@brief to clear the matrixMap
--usage:
--select erase_map();
CREATE OR REPLACE FUNCTION
qp4ai_erase_map()
RETURNS INT
AS 'zkx','_Z15qp4ai_erase_mapP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

--@brief to initial the matrixMap with some test cases
--usage:
--select qp4ai_init_map();
--CREATE OR REPLACE FUNCTION
--qp4ai_init_map()
--RETURNS INT
--AS 'zkx','_Z14qp4ai_init_mapP20FunctionCallInfoData' 
--LANGUAGE C STRICT;

--@brief to print the size of matrixMap
--usage:
--select qp4ai_matrixMap_size();
--size will just show up in the return value
CREATE OR REPLACE FUNCTION
qp4ai_matrixMap_size()
RETURNS INT
AS 'zkx','_Z20qp4ai_matrixMap_sizeP20FunctionCallInfoData' 
LANGUAGE C STRICT;


CREATE OR REPLACE FUNCTION
qp4ai_erase_element(intput_table_name TEXT)
RETURNS INT
AS 'zkx','_Z19qp4ai_erase_elementP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_select(output_table_name TEXT)
RETURNS TEXT
AS 'zkx','_Z12qp4ai_selectP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_add(input_table_name1 TEXT,input_table_name2 TEXT,output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_addP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_div(input_table_name1 TEXT,input_table_name2 TEXT,output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_divP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_mul(input_table_name1 TEXT,input_table_name2 TEXT,output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_mulP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_exp(input_table_name TEXT,output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_expP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_abs(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_absP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_zeros(row_num INT,col_num INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z11qp4ai_zerosP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_reshape(input_table_name TEXT, row_num INT,col_num INT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z13qp4ai_reshapeP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_shuffle(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z13qp4ai_shuffleP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_load(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z10qp4ai_loadP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_back_sub(output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z14qp4ai_back_subP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;


CREATE OR REPLACE FUNCTION
qp4ai_back_div(input_table_name1 TEXT, input_table_name2 TEXT, output_table_name1 TEXT,output_table_name2 TEXT)
RETURNS INT
AS 'zkx','_Z14qp4ai_back_divP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_back_negative(output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z19qp4ai_back_negativeP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_back_log(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z14qp4ai_back_logP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_back_mul(input_table_name1 TEXT, input_table_name2 TEXT, output_table_name1 TEXT,output_table_name2 TEXT)
RETURNS INT
AS 'zkx','_Z14qp4ai_back_mulP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_back_pow(input_table_name1 TEXT, input_table_name2 TEXT, output_table_name1 TEXT,output_table_name2 TEXT)
RETURNS INT
AS 'zkx','_Z14qp4ai_back_powP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_back_sqrt(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z15qp4ai_back_sqrtP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_back_matmul(input_table_name1 TEXT, input_table_name2 TEXT, output_table_name1 TEXT,output_table_name2 TEXT, gradput TEXT)
RETURNS INT
AS 'zkx','_Z17qp4ai_back_matmulP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_back_mean(input_table_name TEXT, output_table_name TEXT, axis INT)
RETURNS INT
AS 'zkx','_Z15qp4ai_back_meanP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_print_matrix(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z18qp4ai_print_matrixP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_val(val float8,output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z9qp4ai_valP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_assignment(input_table_name TEXT, output_table_name TEXT) -- a=b, 第一个参数为b, 第二个参数为a
RETURNS INT
AS 'zkx','_Z16qp4ai_assignmentP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_if_tensor_exists(input_table_name TEXT)
RETURNS BOOL
AS 'zkx','_Z22qp4ai_if_tensor_existsP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_op_broadcast(op INT, input_table_name1 TEXT,input_table_name2 TEXT,output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z18qp4ai_op_broadcastP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_update_data(data_arr float8[], output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z17qp4ai_update_dataP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_negative(output_table_name TEXT, input_table1_name TEXT)
RETURNS INT
AS 'zkx','_Z14qp4ai_negativeP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_back_softmax(input_table1_name TEXT, output_table_name TEXT,  grad_output TEXT)
RETURNS INT
AS 'zkx','_Z18qp4ai_back_softmaxP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_relu(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z10qp4ai_reluP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_back_relu(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z15qp4ai_back_reluP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_leakyrelu(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z15qp4ai_leakyreluP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
qp4ai_back_leakyrelu(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'zkx','_Z20qp4ai_back_leakyreluP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;