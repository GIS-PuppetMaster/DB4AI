------------------------------------------------------------
------------------------------------------------------------
-- db4ai_abs(input_table_name TEXT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求绝对值，形成输出矩阵表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_abs(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_absP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_abs(float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_absP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_acc(input_table1_name TEXT,input_table2_name TEXT, normalize INT, output_table_name TEXT)
-- input_table1_name 矩阵1表名
-- input_table2_name 矩阵2表名
-- normalize 是否进行标准化标记
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 比较两矩阵差异，返回相同的个数（或比例），即计算训练得分
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_acc(input_table1_name TEXT, input_table2_name TEXT, normalize INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_accP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_acc(float8[],float8[],INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_accP20FunctionCallInfoData'
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
    --SELECT db4ai_argmax('a',0,'argmax');
    --SELECT * FROM argmax;
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_argmax(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z18outer_db4ai_argmaxP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_argmax(INT,INT,INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z18inner_db4ai_argmaxP20FunctionCallInfoData'
LANGUAGE C STRICT;

------------------------------------------------------------
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
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_argmin(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z18outer_db4ai_argminP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_argmin(INT,INT,INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z18inner_db4ai_argminP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_argsort(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- dim 要进行排序的维度，0为按列排序，1为按行排序
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入矩阵表按维度进行排序，将排序的顺序信息保存在输出表中
-- 测试
    -- SELECT db4ai_argsort('a',1,'argsort');
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_argsort(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z19outer_db4ai_argsortP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_argsort(float8[],INT,INT,INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z19inner_db4ai_argsortP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_add(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT)
-- input_table1_name 输入的矩阵表1名
-- input_table2_name 输入的矩阵表2名
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求和，形成输出矩阵表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_add(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT) --调用时表名不用加什么引号，传个字符串即可
RETURNS INTEGER
AS 'db4ai_funcs','_Z15outer_db4ai_addP20FunctionCallInfoData'
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_add(float8[], float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_addP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_div(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT)
-- input_table1_name 输入的矩阵表1名
-- input_table2_name 输入的矩阵表2名
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求除，形成输出矩阵表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_div(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT) --调用时表名不用加什么引号，传个字符串即可
RETURNS INTEGER
AS 'db4ai_funcs','_Z15outer_db4ai_divP20FunctionCallInfoData'
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_div(float8[], float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_divP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_dot(input_table_name1 TEXT,input_table_name2 TEXT, output_table_name TEXT)
-- input_table_name1 输入的矩阵（向量）1表名
-- input_table_name2 输入的矩阵（向量）2表名
-- output_table_name 输出的矩阵（向量）表名
-- 要求 db4ai_inner_tensordot 已经注册
-- 返回 执行状态码
-- 效果 将输入的2个向量进行内积运算
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_dot(input_table_name1 TEXT,input_table_name2 TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_dotP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--复用tensordot的内部实现--基本一致--
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_exp(input_table_name TEXT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求e为底的指数，形成输出矩阵表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_exp(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_expP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_exp(float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_expP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
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
db4ai_full(input_table_name TEXT, full_value float8,output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','outer_db4ai_full' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_full(float8,float8[])
RETURNS float8[]
AS 'db4ai_funcs','inner_db4ai_full'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_log(input_table_name TEXT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求对数，形成输出矩阵表。注意负数会变成nan。
-- 测试
    -- SELECT db4ai_log('a','log');
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_log(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_logP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_log(float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_logP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_matmul(input_table_name1 TEXT,input_table_name2 TEXT, output_table_name TEXT)
-- input_table_name1 输入的矩阵1表名
-- input_table_name2 输入的矩阵2表名
-- output_table_name 输出的矩阵表名
-- 要求 db4ai_inner_tensordot 已经注册
-- 返回 执行状态码
-- 效果 将输入的2个矩阵进行matmul运算,得到两个矩阵的乘积
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_matmul(input_table_name1 TEXT,input_table_name2 TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z18outer_db4ai_matmulP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--复用tensordot的内部实现--基本一致--

------------------------------------------------------------
------------------------------------------------------------
-- db4ai_max(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表1名
-- dim 聚合最小值的维度 1为按列 2为按行
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表按选择的维度寻找最大值，形成数值表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_max(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_maxP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_max(INT,INT,INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_maxP20FunctionCallInfoData'
LANGUAGE C STRICT;

------------------------------------------------------------
------------------------------------------------------------
-- db4ai_mean(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表1名
-- dim 聚合最小值的维度 0为按列 or 1为按行
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表按选择的维度计算均值，形成数值表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_mean(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z16outer_db4ai_meanP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_mean(INT,INT,INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z16inner_db4ai_meanP20FunctionCallInfoData'
LANGUAGE C STRICT;

------------------------------------------------------------
------------------------------------------------------------
-- db4ai_min(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表1名
-- dim 聚合最小值的维度 1为按列 2为按行
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表按选择的维度寻找最小值，形成数值表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_min(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_minP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_min(INT,INT,INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_minP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_mul(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT)
-- input_table1_name 输入的矩阵表1名
-- input_table2_name 输入的矩阵表2名
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求乘积，形成输出矩阵表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_mul(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT) --调用时表名不用加什么引号，传个字符串即可
RETURNS INTEGER
AS 'db4ai_funcs','_Z15outer_db4ai_mulP20FunctionCallInfoData'
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_mul(float8[], float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_mulP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_ones(rows INT, cols INT, output_table_name TEXT)
-- rows cols 矩阵的行数 列数
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 按照参数的规格生成一个全0的矩阵
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_ones(rows INT, cols INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z16outer_db4ai_onesP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_ones(INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z16inner_db4ai_onesP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_pow(input_table_name TEXT,pow_exp float8,output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- pow_exp 为每个元素进行指数运算的指数值
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求相应的指数，形成输出矩阵表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_pow(input_table_name TEXT,pow_exp FLOAT8,output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_powP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_pow(float8[],FLOAT)
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_powP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
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
-- 实例 select db4ai_random(10,10,0,0.8,0.2,0,'random');
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_random(dim1 INT, dim2 INT, _distribution INT, arg1 FLOAT, arg2 FLOAT, arg3 FLOAT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z18outer_db4ai_randomP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_random(INT,INT,float8,float8,float8)
RETURNS float8[]
AS 'db4ai_funcs','_Z18inner_db4ai_randomP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_repeat(input_table_name TEXT,dim1 INT, dim2 INT,output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- dim1 行上重复的次数
-- dim2 列上重复的次数
-- output_table_name 输出的向量表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表重复，形成输出矩阵表，返回状态码
-- 测试
    -- SELECT db4ai_repeat('a',3,2,'repeat');
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_repeat(input_table_name TEXT,dim1 INT, dim2 INT,output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z18outer_db4ai_repeatP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_repeat(INT,INT,INT,INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z18inner_db4ai_repeatP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_reshape(input_table_name TEXT,dim1 INT, dim2 INT,output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- dim1 新的行数
-- dim2 新的列数 要求dim1*dim2 = old_dim1 * old_dim2 
-- output_table_name 输出的向量表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表重组，形成输出矩阵表，返回状态码
-- 测试 select db4ai_reshape('ones', 2, 6, 'reshape');
------------------------------------------------------------
--[客户端接口]--
--[执行函数]--
CREATE OR REPLACE FUNCTION
db4ai_reshape(input_table_name TEXT,dim1 INT, dim2 INT,output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z19outer_db4ai_reshapeP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_reverse(input_table_name TEXT,ndim INT,output_table_name TEXT)
-- input_table_name 要做翻转运算的矩阵表名
-- ndim 要翻转的方向（维度）
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码  0：正常 -1:输入矩阵不存在 -6：ndim不为1或0
-- 效果 若ndim = 0，则将输入矩阵按列翻转，若ndim = 1则将输入矩阵按行翻转，若ndim=2 则将输入矩阵全部翻转（二维）
-- 注意 不同于pytorch的调用方式
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_reverse(input_table_name TEXT, ndim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z19outer_db4ai_reverseP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_reverse(float8[],INT,INT,INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z19inner_db4ai_reverseP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_shape(input_table_name TEXT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表求行数和列数，形成输出矩阵表，返回状态码
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_shape(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z17outer_db4ai_shapeP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
------------------------------------------------------------
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
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_slice(input_table_name TEXT, dim1_start INT, dim1_end INT, dim2_start INT, dim2_end INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z17outer_db4ai_sliceP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_slice(INT,INT,INT,INT,INT,INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z17inner_db4ai_sliceP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_softmax(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- dim 要进行softmax运算的维度，0为按列计算（按列归一化），1为按行计算（按行归一化）
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵进行softmax运算
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_softmax(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z19outer_db4ai_softmaxP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_softmax(float8[],INT,INT,INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z19inner_db4ai_softmaxP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_sort(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- dim 要进行排序的维度，0为按列排序，1为按行排序
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求平方根，形成输出矩阵表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_sort(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z16outer_db4ai_sortP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sort(float8[],INT,INT,INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z16inner_db4ai_sortP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_sqrt(input_table_name TEXT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求平方根，形成输出矩阵表
-- 注意 负数会处理为Nan
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_sqrt(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z16outer_db4ai_sqrtP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sqrt(float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z16inner_db4ai_sqrtP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_sub(input_table1_name TEXT,input_table2_name TEXT,output_table_name TEXT)
-- input_table1_name 被减数矩阵表名
-- input_table2_name 减数矩阵表名
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将被减数矩阵与减数矩阵按位做差的差值矩阵保存在输出矩阵表中
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_sub(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_subP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sub(float8[],float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_subP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_sum(input_table_name TEXT,ndim INT,output_table_name TEXT)
-- input_table_name 要做求和运算的矩阵表名
-- ndim 要求和的方向（维度）
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码  0：正常 -1:输入矩阵不存在 -2：ndim不为1或0
-- 效果 若ndim = 0，则将输入矩阵按列求和，若ndim = 1则将输入矩阵按行求和，结果保存在输出矩阵中
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_sum(input_table_name TEXT, ndim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_sumP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sum(float8[],INT,INT,INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_sumP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_tensordot(input_table_name1 TEXT,input_table_name2 TEXT, dim INT, output_table_name TEXT)
-- input_table_name1 输入的矩阵1表名
-- input_table_name2 输入的矩阵2表名
-- dim 要进行tensordot运算的维度，这部分的意义详细请参阅pytorch官方文档（只能是0或1！）
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的2个矩阵进行tensordot运算
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_tensordot(input_table_name1 TEXT,input_table_name2 TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z21outer_db4ai_tensordotP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_tensordot(INT,INT,INT,INT,INT,INT,float8[],float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z21inner_db4ai_tensordotP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_trace(input_table_name TEXT, output_table_name TEXT)
-- input_table_name 输入的表名
-- output_table_name 输出的表名
-- 返回 执行状态码 如果不是方阵则返回-3。
-- 效果 将输入的表名对应的方阵求trace后返回，和其它张量操作不同。
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_trace(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z17outer_db4ai_traceP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_trace(INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z17inner_db4ai_traceP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_zeros(rows INT, cols INT, output_table_name TEXT)
-- rows cols 矩阵的行数 列数
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 按照参数的规格生成一个全0的矩阵
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_zeros(rows INT, cols INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z17outer_db4ai_zerosP20FunctionCallInfoData' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_zeros(INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z17inner_db4ai_zerosP20FunctionCallInfoData'
LANGUAGE C STRICT;