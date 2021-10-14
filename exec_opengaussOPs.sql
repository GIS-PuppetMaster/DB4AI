-------------------------------------------------------------------
--            openGaussOPs.c中相应函数的注册sql部分                 
--                 宋明阳所负责的部分 需求V2.0
--
--              使用时请将其整合至对应源文件部分
--               位置：~/postgres/OPs/opengaussOPs.sql
--------------------------------------------------------------------



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
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_sub' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sub(float8[],float8[])
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_sub'
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
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_sum' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sum(float8[],INT,INT,INT)
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_sum'
LANGUAGE C STRICT;


------------------------------------------------------------
------------------------------------------------------------
-- db4ai_sqrt(input_table_name TEXT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求平方根，形成输出矩阵表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_sqrt(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_sqrt' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sqrt(float8[])
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_sqrt'
LANGUAGE C STRICT;

------------------------------------------------------------
------------------------------------------------------------
-- db4ai_sort(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- dim 要进行排序的维度，0为按列排序，1为按行排序
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表按维度进行排序，结果保存在输出表中
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_sort(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_sort' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sort(float8[],INT,INT,INT)
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_sort'
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
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_softmax' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_softmax(float8[],INT,INT,INT)
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_softmax'
LANGUAGE C STRICT;

------------------------------------------------------------
------------------------------------------------------------
-- db4ai_tensordot(input_table_name1 TEXT,input_table_name2 TEXT, dim INT, output_table_name TEXT)
-- input_table_name1 输入的矩阵1表名
-- input_table_name2 输入的矩阵2表名
-- dim 要进行tensordot运算的维度，这部分的意义详细请参阅pytorch官方文档（只能是1或2！）
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的2个矩阵进行tensordot运算
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_tensordot(input_table_name1 TEXT,input_table_name2 TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_tensordot' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_tensordot(INT,INT,INT,INT,INT,INT,float8[],float8[])
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_tensordot'
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
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_matmul' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--复用tensordot的内部实现--基本一致--

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
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_dot' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--复用tensordot的内部实现--基本一致--

------------------------------------------------------------
------------------------------------------------------------
-- db4ai_argsort(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- dim 要进行排序的维度，0为按列排序，1为按行排序
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入矩阵表按维度进行排序，将排序的顺序信息保存在输出表中
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_argsort(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_argsort' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_argsort(float8[],INT,INT,INT)
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_argsort'
LANGUAGE C STRICT;

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
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_acc' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_acc(float8[],float8[],INT)
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_acc'
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
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_reverse' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_reverse(float8[],INT,INT,INT)
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_reverse'
LANGUAGE C STRICT;