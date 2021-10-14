------------------------------------------------------------
--            openGaussOPs.c中相应函数的注册sql部分
--                    胡飞所负责的部分
------------------------------------------------------------


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
AS '/home/postgres/code/db4ai_funcs','outer_db4ai_shape' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
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
AS '/home/postgres/code/db4ai_funcs','outer_db4ai_pow' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_pow(float8[],FLOAT)
RETURNS float8[]
AS '/home/postgres/code/db4ai_funcs','inner_db4ai_pow'
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
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_repeat(input_table_name TEXT,dim1 INT, dim2 INT,output_table_name TEXT)
RETURNS INT
AS '/home/postgres/code/db4ai_funcs','outer_db4ai_repeat' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_repeat(INT,INT,INT,INT,float8[])
RETURNS float8[]
AS '/home/postgres/code/db4ai_funcs','inner_db4ai_repeat'
LANGUAGE C STRICT;

------------------------------------------------------------
------------------------------------------------------------
-- db4ai_log(input_table_name TEXT, output_table_name TEXT)
-- input_table_name 输入的矩阵表名
-- output_table_name 输出的矩阵表名
-- 返回 执行状态码
-- 效果 将输入的矩阵表的每个元素都求对数，形成输出矩阵表
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_log(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS '/home/postgres/code/db4ai_funcs','outer_db4ai_log' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_log(float8[])
RETURNS float8[]
AS '/home/postgres/code/db4ai_funcs','inner_db4ai_log'
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
AS '/home/postgres/code/db4ai_funcs','outer_db4ai_full' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_full(float8,float8[])
RETURNS float8[]
AS '/home/postgres/code/db4ai_funcs','inner_db4ai_full'
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
AS '/home/postgres/code/db4ai_funcs','outer_db4ai_trace' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_trace(INT,float8[])
RETURNS float8[]
AS '/home/postgres/code/db4ai_funcs','inner_db4ai_trace'
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
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_argmin(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS '/home/postgres/code/db4ai_funcs','outer_db4ai_argmin' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_argmin(INT,INT,INT,float8[])
RETURNS float8[]
AS '/home/postgres/code/db4ai_funcs','inner_db4ai_argmin'
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
------------------------------------------------------------
--[客户端接口]--
CREATE OR REPLACE FUNCTION
db4ai_argmax(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS '/home/postgres/code/db4ai_funcs','outer_db4ai_argmax' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_argmax(INT,INT,INT,float8[])
RETURNS float8[]
AS '/home/postgres/code/db4ai_funcs','inner_db4ai_argmax'
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
AS '/home/postgres/code/db4ai_funcs','outer_db4ai_min' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_min(INT,INT,INT,float8[])
RETURNS float8[]
AS '/home/postgres/code/db4ai_funcs','inner_db4ai_min'
LANGUAGE C STRICT;

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
AS '/home/postgres/code/db4ai_funcs','outer_db4ai_max' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_max(INT,INT,INT,float8[])
RETURNS float8[]
AS '/home/postgres/code/db4ai_funcs','inner_db4ai_max'
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
AS '/home/postgres/code/db4ai_funcs','outer_db4ai_mean' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_mean(INT,INT,INT,float8[])
RETURNS float8[]
AS '/home/postgres/code/db4ai_funcs','inner_db4ai_mean'
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
AS '/home/postgres/code/db4ai_funcs','outer_db4ai_slice' -- 前者手动改为绝对路径 后者手动改成.so中的符号名
LANGUAGE C STRICT;
--[执行函数]--
CREATE OR REPLACE FUNCTION
inner_db4ai_slice(INT,INT,INT,INT,INT,INT,float8[])
RETURNS float8[]
AS '/home/postgres/code/db4ai_funcs','inner_db4ai_slice'
LANGUAGE C STRICT;
