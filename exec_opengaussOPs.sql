-------------------------------------------------------------------
--            openGaussOPs.c����Ӧ������ע��sql����                 
--                 ������������Ĳ��� ����V2.0
--
--              ʹ��ʱ�뽫����������ӦԴ�ļ�����
--               λ�ã�~/postgres/OPs/opengaussOPs.sql
--------------------------------------------------------------------



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
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_sub' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sub(float8[],float8[])
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_sub'
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
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_sum' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sum(float8[],INT,INT,INT)
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_sum'
LANGUAGE C STRICT;


------------------------------------------------------------
------------------------------------------------------------
-- db4ai_sqrt(input_table_name TEXT, output_table_name TEXT)
-- input_table_name ����ľ������
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ�����ÿ��Ԫ�ض���ƽ�������γ���������
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_sqrt(input_table_name TEXT, output_table_name TEXT)
RETURNS INT
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_sqrt' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sqrt(float8[])
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_sqrt'
LANGUAGE C STRICT;

------------------------------------------------------------
------------------------------------------------------------
-- db4ai_sort(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name ����ľ������
-- dim Ҫ���������ά�ȣ�0Ϊ��������1Ϊ��������
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ������ľ����ά�Ƚ������򣬽���������������
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_sort(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_sort' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_sort(float8[],INT,INT,INT)
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_sort'
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
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_softmax' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_softmax(float8[],INT,INT,INT)
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_softmax'
LANGUAGE C STRICT;

------------------------------------------------------------
------------------------------------------------------------
-- db4ai_tensordot(input_table_name1 TEXT,input_table_name2 TEXT, dim INT, output_table_name TEXT)
-- input_table_name1 ����ľ���1����
-- input_table_name2 ����ľ���2����
-- dim Ҫ����tensordot�����ά�ȣ��ⲿ�ֵ�������ϸ�����pytorch�ٷ��ĵ���ֻ����1��2����
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� �������2���������tensordot����
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_tensordot(input_table_name1 TEXT,input_table_name2 TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_tensordot' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_tensordot(INT,INT,INT,INT,INT,INT,float8[],float8[])
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_tensordot'
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
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_matmul' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--����tensordot���ڲ�ʵ��--����һ��--

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
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_dot' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--����tensordot���ڲ�ʵ��--����һ��--

------------------------------------------------------------
------------------------------------------------------------
-- db4ai_argsort(input_table_name TEXT, dim INT, output_table_name TEXT)
-- input_table_name ����ľ������
-- dim Ҫ���������ά�ȣ�0Ϊ��������1Ϊ��������
-- output_table_name ����ľ������
-- ���� ִ��״̬��
-- Ч�� ����������ά�Ƚ������򣬽������˳����Ϣ�������������
------------------------------------------------------------
--[�ͻ��˽ӿ�]--
CREATE OR REPLACE FUNCTION
db4ai_argsort(input_table_name TEXT, dim INT, output_table_name TEXT)
RETURNS INT
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_argsort' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_argsort(float8[],INT,INT,INT)
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_argsort'
LANGUAGE C STRICT;

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
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_acc' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_acc(float8[],float8[],INT)
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_acc'
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
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','outer_db4ai_reverse' -- ǰ���ֶ���Ϊ����·�� �����ֶ��ĳ�.so�еķ�����
LANGUAGE C STRICT;
--[ִ�к���]--
CREATE OR REPLACE FUNCTION
inner_db4ai_reverse(float8[],INT,INT,INT)
RETURNS float8[]
AS '/home/ubunov/codes/test/smyV2.0/exec_opengaussOPs','inner_db4ai_reverse'
LANGUAGE C STRICT;