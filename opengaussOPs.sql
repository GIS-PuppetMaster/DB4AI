------------------------------------------------------------
------------------------------------------------------------
-- db4ai_abs(input_table_name TEXT, output_table_name TEXT)
-- input_table_name ����ľ������
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ�����ÿ��Ԫ�ض������ֵ���γ���������
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_abs(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_absP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_abs(float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_absP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_acc(input_table1_name TEXT,input_table2_name TEXT, normalize INT, output_table_name TEXT)
-- input_table1_name ����1����
-- input_table2_name ����2����
-- normalize �Ƿ���б�׼�����
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� �Ƚ���������죬������ͬ�ĸ��������������������ѵ���÷�
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_acc(input_table1_name TEXT, input_table2_name TEXT, normalize INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_accP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_acc(float8[],float8[],INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_accP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_argmax(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name ����ľ������
-- dim �����ά��
-- output_table_name ����ı���
-- ���� ִ��״̬��
-- Ч�� ������ľ�����ÿ��Ԫ�ض������ֵ���������γ������
-- ע�� �����dim��0����1��0��1��
-- ���� 
    --SELECT db4ai_argmax('a',0,'argmax');
    --SELECT * FROM argmax;
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_argmax(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z18outer_db4ai_argmaxP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_argmax(INT,INT,INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z18inner_db4ai_argmaxP20FunctionCallInfoData'
LANGUAGE C STRICT;

------------------------------------------------------------
------------------------------------------------------------
-- db4ai_argmin(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name ����ľ������
-- dim �����ά��
-- output_table_name ����ı���
-- ���� ִ��״̬��
-- Ч�� ������ľ�����ÿ��Ԫ�ض�����Сֵ���������γ������
-- ע�� �����dim��0����1��0��1��
-- ���� 
    -- SELECT db4ai_argmin('a',1,'argmin');
    -- SELECT * FROM argmin;
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_argmin(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z18outer_db4ai_argminP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_argmin(INT,INT,INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z18inner_db4ai_argminP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_argsort(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name ����ľ������
-- dim Ҫ���������ά�ȣ�0Ϊ��������1Ϊ��������
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ����������ά�Ƚ������򣬽������˳����Ϣ�������������
-- ����
    -- SELECT db4ai_argsort('a',1,'argsort');
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_argsort(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z19outer_db4ai_argsortP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_argsort(float8[],INT,INT,INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z19inner_db4ai_argsortP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_add(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT)
-- input_table1_name ����ľ����1��
-- input_table2_name ����ľ����2��
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ�����ÿ��Ԫ�ض���ͣ��γ���������
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_add(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT) --����ʱ�������ü�ʲô���ţ������ַ�������
RETURNS INTEGER
AS 'db4ai_funcs','_Z15outer_db4ai_addP20FunctionCallInfoData'
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_add(float8[], float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_addP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_div(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT)
-- input_table1_name ����ľ����1��
-- input_table2_name ����ľ����2��
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ�����ÿ��Ԫ�ض�������γ���������
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_div(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT) --����ʱ�������ü�ʲô���ţ������ַ�������
RETURNS INTEGER
AS 'db4ai_funcs','_Z15outer_db4ai_divP20FunctionCallInfoData'
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_div(float8[], float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_divP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_dot(input_table_name1 TEXT,input_table_name2 TEXT, output_table_name TEXT)
-- input_table_name1 ����ľ���������1����
-- input_table_name2 ����ľ���������2����
-- output_table_name ����ľ�������������
-- Ҫ�� db4ai_inner_tensordot �Ѿ�ע��
-- ���� ִ��״̬��
-- Ч�� �������2�����������ڻ�����
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_dot(input_table_name1 TEXT,input_table_name2 TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_dotP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--����tensordot���ڲ�ʵ��--����һ��--
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_exp(input_table_name TEXT, output_table_name TEXT)
-- input_table_name ����ľ������
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ�����ÿ��Ԫ�ض���eΪ�׵�ָ�����γ���������
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_exp(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_expP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_exp(float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_expP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_full(input_table_name TEXT,full_value float8,output_table_name TEXT)
-- input_table_name ����ı���
-- full_value ���ֵ
-- output_table_name ����ı���
-- ���� ִ��״̬��
-- Ч�� ��������ڵ�ֵȫ���ĳ�full_value�����������
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_full(input_table_name TEXT, full_value float8,output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','outer_db4ai_full' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_full(float8,float8[])
RETURNS float8[]
AS 'db4ai_funcs','inner_db4ai_full'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_log(input_table_name TEXT, output_table_name TEXT)
-- input_table_name ����ľ������
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ�����ÿ��Ԫ�ض���������γ���������ע�⸺������nan��
-- ����
    -- SELECT db4ai_log('a','log');
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_log(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_logP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_log(float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_logP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_matmul(input_table_name1 TEXT,input_table_name2 TEXT, output_table_name TEXT)
-- input_table_name1 ����ľ���1����
-- input_table_name2 ����ľ���2����
-- output_table_name ����ľ������
-- Ҫ�� db4ai_inner_tensordot �Ѿ�ע��
-- ���� ִ��״̬��
-- Ч�� �������2���������matmul����,�õ���������ĳ˻�
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_matmul(input_table_name1 TEXT,input_table_name2 TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z18outer_db4ai_matmulP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--����tensordot���ڲ�ʵ��--����һ��--

------------------------------------------------------------
------------------------------------------------------------
-- db4ai_max(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name ����ľ����1��
-- dim �ۺ���Сֵ��ά�� 1Ϊ���� 2Ϊ����
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ����ѡ���ά��Ѱ�����ֵ���γ���ֵ��
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_max(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_maxP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_max(INT,INT,INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_maxP20FunctionCallInfoData'
LANGUAGE C STRICT;

------------------------------------------------------------
------------------------------------------------------------
-- db4ai_mean(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name ����ľ����1��
-- dim �ۺ���Сֵ��ά�� 0Ϊ���� or 1Ϊ����
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ����ѡ���ά�ȼ����ֵ���γ���ֵ��
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_mean(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z16outer_db4ai_meanP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_mean(INT,INT,INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z16inner_db4ai_meanP20FunctionCallInfoData'
LANGUAGE C STRICT;

------------------------------------------------------------
------------------------------------------------------------
-- db4ai_min(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name ����ľ����1��
-- dim �ۺ���Сֵ��ά�� 1Ϊ���� 2Ϊ����
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ����ѡ���ά��Ѱ����Сֵ���γ���ֵ��
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_min(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_minP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_min(INT,INT,INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_minP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_mul(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT)
-- input_table1_name ����ľ����1��
-- input_table2_name ����ľ����2��
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ�����ÿ��Ԫ�ض���˻����γ���������
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_mul(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT) --����ʱ�������ü�ʲô���ţ������ַ�������
RETURNS INTEGER
AS 'db4ai_funcs','_Z15outer_db4ai_mulP20FunctionCallInfoData'
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_mul(float8[], float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_mulP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_ones(rows INT, cols INT, output_table_name TEXT)
-- rows cols ��������� ����
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ���ղ����Ĺ������һ��ȫ0�ľ���
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_ones(rows INT, cols INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z16outer_db4ai_onesP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_ones(INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z16inner_db4ai_onesP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_pow(input_table_name TEXT,pow_exp float8,output_table_name TEXT)
-- input_table_name ����ľ������
-- pow_exp Ϊÿ��Ԫ�ؽ���ָ�������ָ��ֵ
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ�����ÿ��Ԫ�ض�����Ӧ��ָ�����γ���������
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_pow(input_table_name TEXT,pow_exp FLOAT8,output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_powP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_pow(float8[],FLOAT)
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_powP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_random(dim1 INT, dim2 INT, _distribution TEXT, _args TEXT, output_table_name TEXT)
-- dim1 ����
-- dim2 ����
-- _distribution �ֲ���ʽ: Normal(0) Uniform(1) Bernoulli(2)
-- ����args �����������_distribution�����йأ�
    -- Supported parameters:
        -- Normal: mu, sigma
        -- Uniform: min, max
        -- Bernoulli: lower, upper, prob
-- output_table_name ����ı���
-- ���� ִ��״̬��
-- Ч�� ����һ��dim1��dim2�е���Ϊoutput_table_name����������
-- ʵ�� select db4ai_random(10,10,0,0.8,0.2,0,'random');
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_random(dim1 INT, dim2 INT, _distribution INT, arg1 FLOAT, arg2 FLOAT, arg3 FLOAT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z18outer_db4ai_randomP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_random(INT,INT,float8,float8,float8)
RETURNS float8[]
AS 'db4ai_funcs','_Z18inner_db4ai_randomP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_repeat(input_table_name TEXT,dim1 INT, dim2 INT,output_table_name TEXT)
-- input_table_name ����ľ������
-- dim1 �����ظ��Ĵ���
-- dim2 �����ظ��Ĵ���
-- output_table_name �������������
-- ���� ִ��״̬��
-- Ч�� ������ľ�����ظ����γ�������������״̬��
-- ����
    -- SELECT db4ai_repeat('a',3,2,'repeat');
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_repeat(input_table_name TEXT,dim1 INT, dim2 INT,output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z18outer_db4ai_repeatP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_repeat(INT,INT,INT,INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z18inner_db4ai_repeatP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_reshape(input_table_name TEXT,dim1 INT, dim2 INT,output_table_name TEXT)
-- input_table_name ����ľ������
-- dim1 �µ�����
-- dim2 �µ����� Ҫ��dim1*dim2 = old_dim1 * old_dim2 
-- output_table_name �������������
-- ���� ִ��״̬��
-- Ч�� ������ľ�������飬�γ�������������״̬��
-- ���� select db4ai_reshape('ones', 2, 6, 'reshape');
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
db4ai_reshape(input_table_name TEXT,dim1 INT, dim2 INT,output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z19outer_db4ai_reshapeP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_reverse(input_table_name TEXT,ndim INT,output_table_name TEXT)
-- input_table_name Ҫ����ת����ľ������
-- ndim Ҫ��ת�ķ���ά�ȣ�
-- output_table_name ����ľ������
-- ���� ִ��״̬��  0������ -1:������󲻴��� -6��ndim��Ϊ1��0
-- Ч�� ��ndim = 0������������з�ת����ndim = 1����������з�ת����ndim=2 ���������ȫ����ת����ά��
-- ע�� ��ͬ��pytorch�ĵ��÷�ʽ
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_reverse(input_table_name TEXT, ndim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z19outer_db4ai_reverseP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_reverse(float8[],INT,INT,INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z19inner_db4ai_reverseP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_shape(input_table_name TEXT, output_table_name TEXT)
-- input_table_name ����ľ������
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ�������������������γ�������������״̬��
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_shape(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z17outer_db4ai_shapeP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_slice(input_table_name TEXT, dim1_start INT, dim1_end INT, dim2_start INT, dim2_end INT, output_table_name TEXT)
-- input_table_name ����ľ������
-- dim1_start ��ʼ��
-- dim1_end ��ֹ��
-- dim2_start ��ʼ��
-- dim2_end ��ֹ��
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ������Ƭ�����浽������С�
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_slice(input_table_name TEXT, dim1_start INT, dim1_end INT, dim2_start INT, dim2_end INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z17outer_db4ai_sliceP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_slice(INT,INT,INT,INT,INT,INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z17inner_db4ai_sliceP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_softmax(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name ����ľ������
-- dim Ҫ����softmax�����ά�ȣ�0Ϊ���м��㣨���й�һ������1Ϊ���м��㣨���й�һ����
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ������softmax����
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_softmax(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z19outer_db4ai_softmaxP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_softmax(float8[],INT,INT,INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z19inner_db4ai_softmaxP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_sort(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name ����ľ������
-- dim Ҫ���������ά�ȣ�0Ϊ��������1Ϊ��������
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ�����ÿ��Ԫ�ض���ƽ�������γ���������
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_sort(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z16outer_db4ai_sortP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sort(float8[],INT,INT,INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z16inner_db4ai_sortP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_sqrt(input_table_name TEXT, output_table_name TEXT)
-- input_table_name ����ľ������
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ�����ÿ��Ԫ�ض���ƽ�������γ���������
-- ע�� �����ᴦ��ΪNan
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_sqrt(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z16outer_db4ai_sqrtP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sqrt(float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z16inner_db4ai_sqrtP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_sub(input_table1_name TEXT,input_table2_name TEXT,output_table_name TEXT)
-- input_table1_name �������������
-- input_table2_name �����������
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ���������������������λ����Ĳ�ֵ���󱣴�������������
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_sub(input_table1_name TEXT, input_table2_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_subP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sub(float8[],float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_subP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_sum(input_table_name TEXT,ndim INT,output_table_name TEXT)
-- input_table_name Ҫ���������ľ������
-- ndim Ҫ��͵ķ���ά�ȣ�
-- output_table_name ����ľ������
-- ���� ִ��״̬��  0������ -1:������󲻴��� -2��ndim��Ϊ1��0
-- Ч�� ��ndim = 0���������������ͣ���ndim = 1�������������ͣ�������������������
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_sum(input_table_name TEXT, ndim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z15outer_db4ai_sumP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sum(float8[],INT,INT,INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z15inner_db4ai_sumP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_tensordot(input_table_name1 TEXT,input_table_name2 TEXT, dim INT, output_table_name TEXT)
-- input_table_name1 ����ľ���1����
-- input_table_name2 ����ľ���2����
-- dim Ҫ����tensordot�����ά�ȣ��ⲿ�ֵ�������ϸ�����pytorch�ٷ��ĵ���ֻ����0��1����
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� �������2���������tensordot����
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_tensordot(input_table_name1 TEXT,input_table_name2 TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z21outer_db4ai_tensordotP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_tensordot(INT,INT,INT,INT,INT,INT,float8[],float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z21inner_db4ai_tensordotP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_trace(input_table_name TEXT, output_table_name TEXT)
-- input_table_name ����ı���
-- output_table_name ����ı���
-- ���� ִ��״̬�� ������Ƿ����򷵻�-3��
-- Ч�� ������ı�����Ӧ�ķ�����trace�󷵻أ�����������������ͬ��
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_trace(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z17outer_db4ai_traceP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_trace(INT,float8[])
RETURNS float8[]
AS 'db4ai_funcs','_Z17inner_db4ai_traceP20FunctionCallInfoData'
LANGUAGE C STRICT;
------------------------------------------------------------
------------------------------------------------------------
-- db4ai_zeros(rows INT, cols INT, output_table_name TEXT)
-- rows cols ��������� ����
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ���ղ����Ĺ������һ��ȫ0�ľ���
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_zeros(rows INT, cols INT, output_table_name TEXT)
RETURNS INT
AS 'db4ai_funcs','_Z17outer_db4ai_zerosP20FunctionCallInfoData' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_zeros(INT)
RETURNS float8[]
AS 'db4ai_funcs','_Z17inner_db4ai_zerosP20FunctionCallInfoData'
LANGUAGE C STRICT;