#include "postgres.h" // ������ÿ������postgres������C�ļ���

#include "fmgr.h" // ����PG_GETARG_XXX �Լ� PG_RETURN_XXX
#include "access/hash.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/array.h" // �����ṩArrayType*
#include "executor/spi.h" // ���ڵ���SPI

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#define MAX_SQL_LEN 1024
/*
 * Taken from the intarray contrib header
 */
#define ARRPTR(x)  ( (double *) ARR_DATA_PTR(x) )
#define ARRNELEMS(x)  ArrayGetNItems( ARR_NDIM(x), ARR_DIMS(x))

PG_MODULE_MAGIC; // ��PostgreSQL�ڼ���ʱ��������

// ����ִ�е��е���sql��ѯ֮�󡿴Ӳ�ѯ����л�ȡ���֣����ڻ�ȡ�������������Ƿ�ת��
// USAGE: int32 val = get_int32_from_qresult();
#define get_int32_from_qresult() atoi(SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1))

// ����ִ�е��е���sql��ѯ֮�󡿴Ӳ�ѯ����л�ȡ�ַ���������
// USAGE: char* get_string_from_qresult();
#define get_string_from_qresult() SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1)

// ���״̬�룺
// 0 -> ����
// -1 -> ���������
// -2 -> ����������ͬ
// -3 -> ����������ͬ
// -4 -> �������������������ֲ���������
// -5 -> �µ��������µ������ĳ˻�������ԭ��Ԫ�ظ���
// -6 -> ��ʾ��������Ĳ���dim��Ϊ0����-1

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ��������Ԫ��ȡ����ֵ�Ľ����ŵ�������С�
// ������������� �������
// ���أ�0���� -1���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_absP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_absP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_abs(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // ��������
    SPI_connect();  //���룺��������
    
    ///////////////////////////////////////////////////////////////////////// 
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_abs(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT rows, cols, trans, inner_db4ai_abs(data) as data into %s from %s;",
        output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_abs(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = ARRNELEMS(arr_raw);                          // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(arr[i]<0?-arr[i]:arr[i]);
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ����������������ͬ��Ԫ�صĸ�����normalize=0������ͬԪ�ظ�����ռ�ı�����normalize = 1��
// �������������1 �������2 ��׼��normalize �������
// ���أ�0���� -1��������� -2������ͬ -3 ������ͬ
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_accP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_accP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_acc(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    int32 normalize = PG_GETARG_INT32(2);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(3));
    // ��������
    SPI_connect();  //���룺��������
    ///////////////////////////////////////////////////////////////////////// 
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists1 = get_int32_from_qresult();
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists2 = get_int32_from_qresult();
    if(!(if_input_table_exists1&&if_input_table_exists2)){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ȷ����1�ͱ�2������ͬ������
    // ��ȡ��1������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();

    // �����ͬ�򱨴�
    if(table1_rows!=table2_rows){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-2); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ȷ����1�ͱ�2������ͬ������
    // ��ȡ��1������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select cols from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();

    // �����ͬ�򱨴�
    if(table1_cols!=table2_cols){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-3); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT %d as rows, %d as cols, t1.trans as trans, inner_db4ai_acc(t1.data, t2.data,%d) as data into %s from %s as t1, %s as t2;"
        ,1,1,normalize, output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_acc(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8����X2��
    ArrayType* arr_raw1 = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType* arr_raw2 = PG_GETARG_ARRAYTYPE_P(1);
    int32 norm = PG_GETARG_INT32(2);
    float8* arr1 = (float8 *) ARR_DATA_PTR(arr_raw1);
    float8* arr2 = (float8 *) ARR_DATA_PTR(arr_raw2);
    int size = ARRNELEMS(arr_raw1);                          // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    float8 acc = 0;
    for(int i=0; i<size; i++){
        acc+=(arr1[i]==arr2[i]?1:0);//������ͬ�ĸ���
    }
    if(norm == 1){
        acc = acc/size;
    }
    ans_arr_back[0] = Float8GetDatum(acc);
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, 1, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ����������ӵĽ����ŵ�������С�
// �������������1 �������2 �������
// ���أ�0���� -1���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_addP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_addP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_add(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������

    /////////////////////////////////////////////////////////////////////////
    // �����1�����ڣ�������ӡ
    char sql_table1_exists[MAX_SQL_LEN];
    sprintf(sql_table1_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table1_exists, 0);
    int32 if_input_table1_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table1_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // �����2�����ڣ�������ӡ
    char sql_table2_exists[MAX_SQL_LEN];
    sprintf(sql_table2_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table2_exists, 0);
    int32 if_input_table2_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table2_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ȷ����1�ͱ�2������ͬ������
    // ��ȡ��1������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();

    // �����ͬ�򱨴�
    if(table1_rows!=table2_rows){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-2); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ȷ����1�ͱ�2������ͬ������
    // ��ȡ��1������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select cols from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();

    // �����ͬ�򱨴�
    if(table1_cols!=table2_cols){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-3); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////


    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////


    // ����INNER����: SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_add(t1.data, t2.data) as data into output_table_name from input_table_name1 as t1, input_table_name2 as t2;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_add(t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        , output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_add(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8����X2��
    ArrayType* arr_raw1 = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType* arr_raw2 = PG_GETARG_ARRAYTYPE_P(1);
    float8* arr1 = (float8 *) ARR_DATA_PTR(arr_raw1);
    float8* arr2 = (float8 *) ARR_DATA_PTR(arr_raw2);
    int size = ARRNELEMS(arr_raw1);                          // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(arr1[i]+arr2[i]);
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ�������ľ�����ÿ��Ԫ�ض������ֵ���������γ������
// ���������� ���� ���ֵ
// ���أ�0����,���dim����0��1����-6
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z18outer_db4ai_argmaxP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z18inner_db4ai_argmaxP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_argmax(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������

    /////////////////////////////////////////////////////////////////////////
    // ������ֲ��������������Ϊ������Ŀ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    // ��ȡ����������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // ��ȡ����������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ������dim�Ƿ���ȷ
    if(dim != 0 && dim != 1){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-6); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    int32 table2_cols = 0;
    if(dim == 0){
        table2_cols = table1_cols;
    }
    else if(dim == 1){
        table2_cols = table1_rows;
    }
    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT 1 as rows,  %d as cols, trans, inner_db4ai_argmax(%d,%d,%d,data) as data into %s from %s;",
        table2_cols,table1_rows,table1_cols,dim,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_argmax(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    int32 dim = PG_GETARG_INT32(2);
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(3);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size;
    if(dim == 1){
        size = rows;
    }else{
        size = cols;
    }
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    if(dim == 1){
        for(int i=0; i<rows;i++){
            float8 max,argmax;
            for(int j=0; j<cols-1; j++){
                if(j == 0){max = arr[i*cols+j]; argmax = (float8)j;}
                if(max < arr[i*cols+j+1]){max = arr[i*cols+j+1]; argmax = (float8)(j+1);}
            }
            ans_arr_back[i] = Float8GetDatum(argmax);
        }
    }
    else if(dim == 0){
        for(int i=0; i<cols;i++){
            float8 max,argmax;
            for(int j=0; j<rows-1; j++){
                if(j == 0){max = arr[j*cols+i]; argmax = (float8)j;}
                if(max < arr[(j+1)*cols+i]){max = arr[(j+1)*cols+i]; argmax = (float8) (j+1);}
            }
            ans_arr_back[i] = Float8GetDatum(argmax);
        }
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ�������ľ�����ÿ��Ԫ�ض�����Сֵ���������γ������
// ���������� ���� ���ֵ
// ���أ�0����,���dim����0��1����-6
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z18outer_db4ai_argminP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z18inner_db4ai_argminP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_argmin(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������

    /////////////////////////////////////////////////////////////////////////
    // ������ֲ��������������Ϊ������Ŀ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    // ��ȡ����������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // ��ȡ����������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ������dim�Ƿ���ȷ
    if(dim != 0 && dim != 1){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-6); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    int32 table2_cols = 0;
    if(dim == 0){
        table2_cols = table1_cols;
    }
    else if(dim == 1){
        table2_cols = table1_rows;
    }
    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT 1 as rows,  %d as cols, trans, inner_db4ai_argmin(%d,%d,%d,data) as data into %s from %s;",
        table2_cols,table1_rows,table1_cols,dim,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_argmin(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    int32 dim = PG_GETARG_INT32(2);
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(3);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size;
    if(dim == 1){
        size = rows;
    }else{
        size = cols;
    }
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    if(dim == 1){
        for(int i=0; i<rows;i++){
            float8 min,argmin;
            for(int j=0; j<cols-1; j++){
                if(j == 0){min = arr[i*cols+j]; argmin = (float8)j;}
                if(min > arr[i*cols+j+1]){min = arr[i*cols+j+1]; argmin = (float8)(j+1);}
            }
            ans_arr_back[i] = Float8GetDatum(argmin);
        }
    }
    else if(dim == 0){
        for(int i=0; i<cols;i++){
            float8 min,argmin;
            for(int j=0; j<rows-1; j++){
                if(j == 0){min = arr[j*cols+i]; argmin = (float8)j;}
                if(min > arr[(j+1)*cols+i]){min = arr[(j+1)*cols+i]; argmin = (float8) (j+1);}
            }
            ans_arr_back[i] = Float8GetDatum(argmin);
        }
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ�������������˳����Ϣ������������С�
// ������������� ����ά�� �������
// ���أ�0���� -1��������� -6ά����Ϊ0��1
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z19outer_db4ai_argsortP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z19inner_db4ai_argsortP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_argsort(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������
    ///////////////////////////////////////////////////////////////////////// 
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    //�������Ƿ�Ϸ������Ϸ�����
    if(dim != (int32)0 && dim != (int32)1){
        // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        SPI_finish();  // ���룺�ر�����
        PG_RETURN_INT32(-6); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_argsort(t1.data,t1.rows,t1.cols,%d) as data into %s from %s as t1;"
        ,dim, output_table_name, input_table_name);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_argsort(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    // ��ȡ���� int32����������������ά��ֵ
    int32 rows = PG_GETARG_INT32(1);
    int32 cols = PG_GETARG_INT32(2);
    int32 dim = PG_GETARG_INT32(3);
    int size = ARRNELEMS(arr_raw);                          // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    int32* argset = (int32*)malloc(size*sizeof(int32));
    // ��Ҫ�߼�����
    //���ö��幤�ߺ�����û�����ţ�ʹ��ѡ������
    if(dim == 0){//��������
        for(int i=0;i<size;i++){
            argset[i]=i/cols;
        }
        for(int k = 0;k<cols;k++){
            for(int i =0;i<rows-1;i++){
                int pos = i;
                for(int j=i;j<rows;j++){
                    if(arr[j*cols+k]<arr[pos*cols+k]){
                        pos = j;
                    }
                }
                if(pos != i){
                    float8 temp = arr[pos*cols+k];
                    arr[pos*cols+k] = arr[i*cols+k];
                    arr[i*cols+k] = temp;

                    int32 argtmp = argset[pos*cols+k];//����λ��һ���䶯
                    argset[pos*cols+k] = argset[i*cols+k];
                    argset[i*cols+k] = argtmp;
                }
            }
            for(int i=0;i<rows;i++) arr[argset[i*cols+k]*cols+k] = i;//����λ������
        }
    }
    else{//��������
        for(int i=0;i<size;i++){
            argset[i]=i%cols;
        }
        for(int k = 0;k<rows;k++){
            for(int i =0;i<cols-1;i++){
                int pos = i;
                for(int j=i;j<cols;j++){
                    if(arr[k*cols+j]<arr[k*cols+pos])
                        pos = j;
                }
                if(pos != i){
                    float8 temp = arr[k*cols+pos];
                    arr[k*cols+pos] = arr[k*cols+i];
                    arr[k*cols+i] = temp;

                    int32 argtmp = argset[k*cols+pos];
                    argset[k*cols+pos] = argset[k*cols+i];
                    argset[k*cols+i] = argtmp;
                }
            }
            for(int i=0;i<cols;i++) arr[k*cols+argset[k*cols+i]] = i;
        }
    }
    free(argset);
    //����������
    for(int i=0;i<size;i++) ans_arr_back[i] = Float8GetDatum(arr[i]);
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ����������ӵĽ����ŵ�������С�
// �������������1 �������2 �������
// ���أ�0���� -1���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_divP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_divP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_div(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������

    /////////////////////////////////////////////////////////////////////////
    // �����1�����ڣ�������ӡ
    char sql_table1_exists[MAX_SQL_LEN];
    sprintf(sql_table1_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table1_exists, 0);
    int32 if_input_table1_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table1_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // �����2�����ڣ�������ӡ
    char sql_table2_exists[MAX_SQL_LEN];
    sprintf(sql_table2_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table2_exists, 0);
    int32 if_input_table2_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table2_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ȷ����1�ͱ�2������ͬ������
    // ��ȡ��1������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();

    // �����ͬ�򱨴�
    if(table1_rows!=table2_rows){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-2); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ȷ����1�ͱ�2������ͬ������
    // ��ȡ��1������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select cols from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();

    // �����ͬ�򱨴�
    if(table1_cols!=table2_cols){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-3); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////


    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////


    // ����INNER����: SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_div(t1.data, t2.data) as data into output_table_name from input_table_name1 as t1, input_table_name2 as t2;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_div(t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        , output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_div(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8����X2��
    ArrayType* arr_raw1 = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType* arr_raw2 = PG_GETARG_ARRAYTYPE_P(1);
    float8* arr1 = (float8 *) ARR_DATA_PTR(arr_raw1);
    float8* arr2 = (float8 *) ARR_DATA_PTR(arr_raw2);
    int size = ARRNELEMS(arr_raw1);                          // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(arr1[i]/arr2[i]);
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ������������������ڻ��������������������
// �������������1 �������2 �������
// ���أ�0���� -2������������ͬ -3 ������������ͬ -10 ������Ϊ1ά����
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_dotP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_dot(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������
    ///////////////////////////////////////////////////////////////////////// 
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists1 = get_int32_from_qresult();
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists2 = get_int32_from_qresult();
    if(!(if_input_table_exists1&&if_input_table_exists2)){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    // �����ڻ�֮ǰ�ļ�鲽��
    // ��ȡ��1������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();
    // ��ȡ��1������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select cols from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();
    if(table1_cols != table2_cols){//����=�������������
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-3); // ���ؽ��״̬��
    }
    else if(table1_rows != table2_rows){
         SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
         PG_RETURN_INT32(-2); // ���ؽ��״̬��
    }
    if(table1_rows !=1&&table1_cols!=1){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-10); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT %d as rows, %d as cols, t1.trans as trans, inner_db4ai_tensordot(t1.cols,t2.rows,t2.cols,%d,%d,%d,t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        ,1,1,2, table1_rows,table1_cols, output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ��������Ԫ��ȡeΪ��ָ��ֵ�Ľ����ŵ�������С�
// ������������� �������
// ���أ�0���� -1���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_expP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_expP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_exp(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // ��������
    SPI_connect();  //���룺��������
    
    ///////////////////////////////////////////////////////////////////////// 
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_exp(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT rows, cols, trans, inner_db4ai_exp(data) as data into %s from %s;",
        output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_exp(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = ARRNELEMS(arr_raw);                          // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(exp(arr[i]));
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ�����һ�����������ͬ��С����Ϊoutput_table_name��Ԫ�ؾ�Ϊfull_value�ľ����
// ���������� ���ֵ ���
// ���أ�0����
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z16outer_db4ai_fullP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z16inner_db4ai_fullP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_full(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    float8 full_value = PG_GETARG_FLOAT8(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // ����INNER����:
    // SELECT <rows> as rows, <cols> as cols, 0 as trans, inner_db4ai_full(<rows>, <cols>) as data into <output_table_name>;
    char sql[MAX_SQL_LEN];
    sprintf(sql,"SELECT rows, cols,trans, inner_db4ai_full(%f,data) as data into %s from %s;",
        full_value, output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_full(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    float8 full_value = PG_GETARG_FLOAT8(0);
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(1);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = ARRNELEMS(arr_raw);
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(full_value);
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ�������ľ�����ÿ��Ԫ�ض���������γ���������
// ������������� �������
// ���أ�0���� -1���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_logP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_logP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_log(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // ��������
    SPI_connect();  //���룺��������

    /////////////////////////////////////////////////////////////////////////
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_log(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT rows, cols, trans, inner_db4ai_log(data) as data into %s from %s;",
        output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_log(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = ARRNELEMS(arr_raw);                          // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(log(arr[i]));
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ�������ľ����ѡ���ά��Ѱ����Сֵ���γ���ֵ��
// ���������� ���� ���ֵ
// ���أ�0����,���dim����0��1����-6
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_maxP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_maxP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_max(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������

    /////////////////////////////////////////////////////////////////////////
    // ������ֲ��������������Ϊ������Ŀ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    // ��ȡ����������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // ��ȡ����������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ������dim�Ƿ���ȷ
    if(dim != 0 && dim != 1){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-6); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    int32 table2_cols = 0;
    if(dim == 0){
        table2_cols = table1_cols;
    }
    else if(dim == 1){
        table2_cols = table1_rows;
    }
    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT 1 as rows,  %d as cols, trans, inner_db4ai_max(%d,%d,%d,data) as data into %s from %s;",
        table2_cols,table1_rows,table1_cols,dim,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_max(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    int32 dim = PG_GETARG_INT32(2);
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(3);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size;
    if(dim == 1){
        size = rows;
    }else{
        size = cols;
    }
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    if(dim == 1){
        for(int i=0; i<rows;i++){
            float8 max,argmax;
            for(int j=0; j<cols-1; j++){
                if(j == 0){max = arr[i*cols+j]; argmax = (float8)j;}
                if(max < arr[i*cols+j+1]){max = arr[i*cols+j+1]; argmax = (float8)(j+1);}
            }
            ans_arr_back[i] = Float8GetDatum(max);
        }
    }
    else if(dim == 0){
        for(int i=0; i<cols;i++){
            float8 max,argmax;
            for(int j=0; j<rows-1; j++){
                if(j == 0){max = arr[j*cols+i]; argmax = (float8)j;}
                if(max < arr[(j+1)*cols+i]){max = arr[(j+1)*cols+i]; argmax = (float8) (j+1);}
            }
            ans_arr_back[i] = Float8GetDatum(max);
        }
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ�������ľ����ѡ���ά�ȼ����ֵ���γ���ֵ��
// ���������� ���� ���ֵ
// ���أ�0����,���dim����0��1����-6
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z16outer_db4ai_meanP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z16inner_db4ai_meanP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_mean(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������

    /////////////////////////////////////////////////////////////////////////
    // ������ֲ��������������Ϊ������Ŀ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    // ��ȡ����������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // ��ȡ����������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ������dim�Ƿ���ȷ
    if(dim != 0 && dim != 1){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-6); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    int32 table2_cols = 0;
    if(dim == 0){
        table2_cols = table1_cols;
    }
    else if(dim == 1){
        table2_cols = table1_rows;
    }
    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT 1 as rows,  %d as cols, trans, inner_db4ai_mean(%d,%d,%d,data) as data into %s from %s;",
        table2_cols,table1_rows,table1_cols,dim,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_mean(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    int32 dim = PG_GETARG_INT32(2);
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(3);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size;
    if(dim == 1){
        size = rows;
    }else{
        size = cols;
    }
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    if(dim == 1){
        for(int i=0; i<rows;i++){
            float8 sum=0;
            float8 mean=0;
            for(int j=0; j<cols; j++){
                sum += arr[i*cols+j];
            }
            mean = sum / cols;
            ans_arr_back[i] = Float8GetDatum(mean);
        }
    }
    else if(dim == 0){
        for(int i=0; i<cols;i++){
            float8 sum=0;
            float8 mean=0;
            for(int j=0; j<rows; j++){
                sum += arr[j*cols+i];
            }
            mean = sum / rows;
            ans_arr_back[i] = Float8GetDatum(mean);
        }
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ�������ľ����ѡ���ά��Ѱ����Сֵ���γ���ֵ��
// ���������� ���� ���ֵ
// ���أ�0����,���dim����0��1����-6
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_minP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_minP20FunctionCallInfoData);
// �ⲿ����in
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_min(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������

    /////////////////////////////////////////////////////////////////////////
    // ������ֲ��������������Ϊ������Ŀ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    // ��ȡ����������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // ��ȡ����������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ������dim�Ƿ���ȷ
    if(dim != 0 && dim != 1){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-6); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    int32 table2_cols = 0;
    if(dim == 0){
        table2_cols = table1_cols;
    }
    else if(dim == 1){
        table2_cols = table1_rows;
    }
    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT 1 as rows,  %d as cols, trans, inner_db4ai_min(%d,%d,%d,data) as data into %s from %s;",
        table2_cols,table1_rows,table1_cols,dim,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_min(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    int32 dim = PG_GETARG_INT32(2);
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(3);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size;
    if(dim == 1){
        size = rows;
    }else{
        size = cols;
    }
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    if(dim == 1){
        for(int i=0; i<rows;i++){
            float8 min,argmin;
            for(int j=0; j<cols-1; j++){
                if(j == 0){min = arr[i*cols+j]; argmin = (float8)j;}
                if(min > arr[i*cols+j+1]){min = arr[i*cols+j+1]; argmin = (float8)(j+1);}
            }
            ans_arr_back[i] = Float8GetDatum(min);
        }
    }
    else if(dim == 0){
        for(int i=0; i<cols;i++){
            float8 min,argmin;
            for(int j=0; j<rows-1; j++){
                if(j == 0){min = arr[j*cols+i]; argmin = (float8)j;}
                if(min > arr[(j+1)*cols+i]){min = arr[(j+1)*cols+i]; argmin = (float8) (j+1);}
            }
            ans_arr_back[i] = Float8GetDatum(min);
        }
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ���������������˷��������������������
// �������������1 �������2 �������
// ���أ�0���� -1��������� -9�������ʱǰһ��������������ں�һ���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z18outer_db4ai_matmulP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_matmul(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������
    ///////////////////////////////////////////////////////////////////////// 
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists1 = get_int32_from_qresult();
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists2 = get_int32_from_qresult();
    if(!(if_input_table_exists1&&if_input_table_exists2)){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    //����˷�֮ǰ�ļ�鲽��
    // ��ȡ��1������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();
    // ��ȡ��1������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select cols from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();
    int32 outRows = 0;
    int32 outCols = 0;
    if(table1_cols == table2_rows){//����=�������������
        outCols = table2_cols;
        outRows = table1_rows;
    }
    else{
         SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
         PG_RETURN_INT32(-9); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT %d as rows, %d as cols, t1.trans as trans, inner_db4ai_tensordot(t1.cols,t2.rows,t2.cols,%d,%d,%d,t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        ,outRows,outCols,1, outRows,outCols, output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
}
// �˺����������������ӵ��ڲ��������������ʵ���Լ����ڲ�������

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ����������˵Ľ����ŵ�������С�
// �������������1 �������2 �������
// ���أ�0���� -1���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_mulP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_mulP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_mul(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������

    /////////////////////////////////////////////////////////////////////////
    // �����1�����ڣ�������ӡ
    char sql_table1_exists[MAX_SQL_LEN];
    sprintf(sql_table1_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table1_exists, 0);
    int32 if_input_table1_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table1_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // �����2�����ڣ�������ӡ
    char sql_table2_exists[MAX_SQL_LEN];
    sprintf(sql_table2_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table2_exists, 0);
    int32 if_input_table2_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table2_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ȷ����1�ͱ�2������ͬ������
    // ��ȡ��1������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();

    // �����ͬ�򱨴�
    if(table1_rows!=table2_rows){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-2); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ȷ����1�ͱ�2������ͬ������
    // ��ȡ��1������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select cols from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();

    // �����ͬ�򱨴�
    if(table1_cols!=table2_cols){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-3); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////


    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////


    // ����INNER����: SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_mul(t1.data, t2.data) as data into output_table_name from input_table_name1 as t1, input_table_name2 as t2;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_mul(t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        , output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_mul(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8����X2��
    ArrayType* arr_raw1 = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType* arr_raw2 = PG_GETARG_ARRAYTYPE_P(1);
    float8* arr1 = (float8 *) ARR_DATA_PTR(arr_raw1);
    float8* arr2 = (float8 *) ARR_DATA_PTR(arr_raw2);
    int size = ARRNELEMS(arr_raw1);                          // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(arr1[i]*arr2[i]);
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ���������������֣�����ȫ0�������
// ���������� ���� �������
// ���أ�0���� -4���ֲ���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z16outer_db4ai_onesP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z16inner_db4ai_onesP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_ones(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������
    
    /////////////////////////////////////////////////////////////////////////
    // ������ֲ��������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    if(rows<=0 || cols<=0){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-4); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // ����INNER����: 
    // SELECT <rows> as rows, <cols> as cols, 0 as trans, inner_db4ai_ones(<rows>, <cols>) as data into <output_table_name>;
    char sql[MAX_SQL_LEN];
    sprintf(sql,"SELECT %d as rows, %d as cols, 0 as trans, inner_db4ai_ones(%d) as data into %s;",
        rows, cols, rows * cols, output_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_ones(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    int32 size = PG_GETARG_INT32(0);
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(1.0);
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ���������ÿ��Ԫ������Ӧ��ָ�����������ŵ�������С�
// ������������� Ϊÿ��Ԫ�ؽ���ָ�������ָ��ֵ �������
// ���أ�0���� -1���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_powP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_powP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_pow(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    float8 pow_exp = PG_GETARG_FLOAT8(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������

    /////////////////////////////////////////////////////////////////////////
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT rows, cols, trans, inner_db4ai_pow(data,%f) as data into %s from %s;",
        pow_exp,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_pow(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    float8 pow_exp = PG_GETARG_FLOAT8(1);
    int size = ARRNELEMS(arr_raw);                          // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(pow(arr[i],pow_exp));
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}
/////////////////////////////////////////////////////////////////////////
// ���ܣ�������ľ�������µ��к��С����顱���γ�������������״̬��
// ������������� ������ ������ ������� ��Ҫ���µ�����*�µ���������ԭ����Ԫ�ظ�����
// ���أ�0���� -4����������
/////////////////////////////////////////////////////////////////////////
// �����ǵĴ洢��ʽ�У�����ֻ��Ҫ�ı��к��������ɣ�������û��ʵ�ʸĶ�
// ����ע��
PG_FUNCTION_INFO_V1(_Z18outer_db4ai_repeatP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z18inner_db4ai_repeatP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_repeat(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim1 = PG_GETARG_INT32(1);
    int32 dim2 = PG_GETARG_INT32(2);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(3));
    // ��������
    SPI_connect();  //���룺��������

    /////////////////////////////////////////////////////////////////////////
    // ������ֲ��������������Ϊ��������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    if(dim1<=0 || dim2<=0){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-4); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t.rows*%d as rows, t.cols*%d as cols, trans, inner_db4ai_repeat(rows,cols,%d,%d,data) as data into %s from %s as t;",
        dim1,dim2,dim1,dim2,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
}

// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_repeat(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    int32 dim1 = PG_GETARG_INT32(2);
    int32 dim2 = PG_GETARG_INT32(3);
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(4);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = ARRNELEMS(arr_raw);                                  // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    size = size * dim2 * dim1;
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    for(int i=0; i<dim1; i++){
        for(int j=0; j<dim2; j++){
            for(int k=0; k<rows; k++){
                for(int x=0; x<cols;x++){
                    ans_arr_back[(i*dim2)*rows*cols+k*cols*dim2+j*cols+x] = Float8GetDatum(arr[k*cols+x]);
                }
            }
        }
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}
/////////////////////////////////////////////////////////////////////////
// ���ܣ�������ľ�������µ��к��С����顱���γ�������������״̬��
// ������������� ������ ������ ������� ��Ҫ���µ�����*�µ���������ԭ����Ԫ�ظ�����
// ���أ�0���� -4����������
/////////////////////////////////////////////////////////////////////////
// �����ǵĴ洢��ʽ�У�����ֻ��Ҫ�ı��к��������ɣ�������û��ʵ�ʸĶ�
// ����ע��
PG_FUNCTION_INFO_V1(_Z19outer_db4ai_reshapeP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_reshape(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 rows = PG_GETARG_INT32(1);
    int32 cols = PG_GETARG_INT32(2);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(3));
    // ��������
    SPI_connect();  //���룺��������
    
    /////////////////////////////////////////////////////////////////////////
    // ������ֲ��������������Ϊ��������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    if(rows<=0 || cols<=0){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-4); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ������ֲ��������������Ϊ������Ŀ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    // ��ȡ����������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // ��ȡ����������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    // �鿴�Ƿ�������
    if((table1_rows*table1_cols)!=(rows*cols)){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-5); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // ����INNER����: 
    // SELECT <rows> as rows, <cols> as cols, 0 as trans, inner_db4ai_reshape(<rows>, <cols>) as data into <output_table_name>;
    char sql[MAX_SQL_LEN];
    sprintf(sql,"SELECT %d as rows, %d as cols, 0 as trans, data as data into %s from %s;",
        rows, cols, output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �����������ڲ�����

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ�����������з�ת���������==0�����з�ת���������==1����ȫ����ת���������=3�����������������С�
// ע�⣺��torch.flip��������
// ������������� �������dim��Ҫ��ֻ��Ϊ0��1��2 �������
// ���أ�0���� -1��������� -6���������Ϊ0��1��2
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z19outer_db4ai_reverseP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z19inner_db4ai_reverseP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_reverse(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);//��ȷ������Ĳ���Ӧ������о��ǰ�װ�������
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    
    // ��������
    SPI_connect();  
    ///////////////////////////////////////////////////////////////////////// 
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    //�������Ƿ�Ϸ������Ϸ�����
    if(dim != (int32)0 && dim != (int32)1 && dim != (int32)2){
        // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        SPI_finish();  // ���룺�ر�����
        PG_RETURN_INT32(-6); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows as rows, t1.cols as cols, t1.trans as trans, inner_db4ai_reverse(t1.data,t1.rows,t1.cols,%d) as data into %s from %s as t1;"
            , dim, output_table_name, input_table_name);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_reverse(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8����X2��
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    int32 rows = PG_GETARG_INT32(1);
    int32 cols = PG_GETARG_INT32(2);
    int32 ndim = PG_GETARG_INT32(3);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = ARRNELEMS(arr_raw);      
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum)); //������
       //�������߼�����
     if(ndim == (int32)0){//���з�ת
         // ��palloc��̬�����ڴ�
        float8* temp_acc_arr = (float8*) malloc(cols * sizeof(float8));//��ʱ���飬���ڷ�ת����
        for(int i=0;i<rows/2;i++){
            for(int j=0;j<cols;j++){
                temp_acc_arr[j]=arr[i*cols+j]; //��������
            }   
            int mid = rows - i - 1;
            for(int j=0;j<cols;j++){
                arr[i*cols+j] = arr[mid*cols + j];
            }   
            for(int j=0;j<cols;j++){
                arr[mid*cols+j] = temp_acc_arr[j];
            }
        }
        free(temp_acc_arr);
    }
    else if(ndim == 1){//���з�ת
        for(int i=0;i<rows;i++){
            float8 temp;
            for(int j=0;j<cols/2;j++){//��ʼ�ۼ�
                int mid = cols-j-1;
                temp =  arr[i*cols+j];
                arr[i*cols+j] =  arr[i*cols+mid];
                arr[i*cols+mid] = temp;
            }   
        }
    }
    else{
        float8 temp;
        for(int i=0;i<size/2;i++){
            int mid = size-i-1;
            temp = arr[i];
            arr[i] = arr[mid];
            arr[mid] = temp;
        }
    }
    for(int i=0;i<size;i++){
        ans_arr_back[i] = Float8GetDatum(arr[i]);//���������������
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ���������������������ŵ�������С�
// ������������� �������
// ���أ�0���� -1���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z17outer_db4ai_shapeP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_shape(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name =text_to_cstring(PG_GETARG_TEXT_PP(1));

    // ��������
    SPI_connect();  //���룺��������

     /////////////////////////////////////////////////////////////////////////
    // ������ֲ��������������Ϊ������Ŀ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    // ��ȡ����������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // ��ȡ����������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_shape(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT 1 as rows, 2 as cols, 0 as trans, '{%d,%d}' as data into %s from %s;",
        table1_rows,table1_cols,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �����������ڲ�����

/////////////////////////////////////////////////////////////////////////
// ���ܣ�������ľ������Ƭ�����浽������С�
// ������������� ��ʼ�� ��ʼ�� ��ֹ�� ��ֹ�� �������
// ���أ�0���� -4����������
/////////////////////////////////////////////////////////////////////////
// �����ǵĴ洢��ʽ�У�����ֻ��Ҫ�ı��к��������ɣ�������û��ʵ�ʸĶ�
// ����ע��
PG_FUNCTION_INFO_V1(_Z17outer_db4ai_sliceP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z17inner_db4ai_sliceP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_slice(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim1_start = PG_GETARG_INT32(1);
    int32 dim1_end = PG_GETARG_INT32(2);
    int32 dim2_start = PG_GETARG_INT32(3);
    int32 dim2_end = PG_GETARG_INT32(4);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(5));
    // ��������
    SPI_connect();  //���룺��������

    /////////////////////////////////////////////////////////////////////////
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ������ֲ��������������Ϊ������Ŀ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    // ��ȡ����������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // ��ȡ����������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ������dim�Ƿ���ȷ
    if(!(dim1_start>=0 && dim1_start<dim1_end && dim1_end<table1_rows &&
        dim2_start>=0 && dim2_start<dim2_end && dim2_end <table1_cols)){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-12); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT %d as rows, %d as cols, trans, inner_db4ai_slice(rows,cols,%d,%d,%d,%d,data) as data into %s from %s;",
        dim1_end - dim1_start + 1,dim2_end - dim2_start + 1,dim1_start,dim1_end,dim2_start,dim2_end,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
}

// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_slice(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    int32 dim1_start = PG_GETARG_INT32(2);
    int32 dim1_end = PG_GETARG_INT32(3);
    int32 dim2_start = PG_GETARG_INT32(4);
    int32 dim2_end = PG_GETARG_INT32(5);
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(6);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int rows2 = dim1_end - dim1_start + 1;
    int cols2 = dim2_end - dim2_start + 1;
    int size = rows2 * cols2;                                 // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    for(int i = 0; i < rows2; i++){
        for(int j = 0; j < cols2; j++){
            ans_arr_back[i*cols2+j] = Float8GetDatum(arr[(dim1_start+i)*cols + dim2_start+j]);
        }
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ�����������softmax�������������������С�
// ������������� �������
// ���أ�0���� -1��������� -6ά����Ϊ0��1
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z19outer_db4ai_softmaxP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z19inner_db4ai_softmaxP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_softmax(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������
    ///////////////////////////////////////////////////////////////////////// 
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    //�������Ƿ�Ϸ������Ϸ�����
    if(dim != (int32)0 && dim != (int32)1){
        // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        SPI_finish();  // ���룺�ر�����
        PG_RETURN_INT32(-6); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_softmax(t1.data,t1.rows,t1.cols,%d) as data into %s from %s as t1;"
        ,dim, output_table_name, input_table_name);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_softmax(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    // ��ȡ���� int32����������������ά��ֵ
    int32 rows = PG_GETARG_INT32(1);
    int32 cols = PG_GETARG_INT32(2);
    int32 dim = PG_GETARG_INT32(3);
    int size = ARRNELEMS(arr_raw);                          // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    //����exp����
    for(int i=0;i<size;i++) arr[i] = exp(arr[i]);
    if(dim == 0){//���й�һ
        float8* byDiv =  (float8*)malloc(cols * sizeof(float8));//������ʹ�������
        for(int i=0;i<cols;i++){
            byDiv[i] = 0;
            for(int j=0;j<rows;j++){
                byDiv[i] +=  arr[j*cols+i];
            }
        }
        for(int i=0;i<size;i++) arr[i] /= byDiv[i%cols]; 
        free(byDiv);
    }
    else{//���й�һ
        float8* byDiv =  (float8*)malloc(rows * sizeof(float8));
        for(int i=0;i<rows;i++){
            byDiv[i] = 0;
            for(int j=0;j<cols;j++){
                byDiv[i] +=  arr[i*cols+j];
            }
        }
        for(int i=0;i<size;i++) arr[i] /= byDiv[i/cols]; 
        free(byDiv);
    }
    //����������
    for(int i=0;i<size;i++) ans_arr_back[i] = Float8GetDatum(arr[i]);
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}


////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ��������������������������С�
// ������������� �������
// ���أ�0���� -1��������� -6ά����Ϊ0��1
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z16outer_db4ai_sortP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z16inner_db4ai_sortP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_sort(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������
    ///////////////////////////////////////////////////////////////////////// 
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    //�������Ƿ�Ϸ������Ϸ�����
    if(dim != (int32)0 && dim != (int32)1){
        // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        SPI_finish();  // ���룺�ر�����
        PG_RETURN_INT32(-6); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_sort(t1.data,t1.rows,t1.cols,%d) as data into %s from %s as t1;"
        ,dim, output_table_name, input_table_name);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_sort(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    // ��ȡ���� int32����������������ά��ֵ
    int32 rows = PG_GETARG_INT32(1);
    int32 cols = PG_GETARG_INT32(2);
    int32 dim = PG_GETARG_INT32(3);
    int size = ARRNELEMS(arr_raw);                          // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    //���ö��幤�ߺ�����û�����ţ�ʹ��ѡ������
    if(dim == 0){//��������
        for(int k = 0;k<cols;k++){
            for(int i =0;i<rows-1;i++){
                int pos = i;
                for(int j=i;j<rows;j++){
                    if(arr[j*cols+k]<arr[pos*cols+k]){
                        pos = j;
                    }
                }
                if(pos != i){
                    float8 temp = arr[pos*cols+k];
                    arr[pos*cols+k] = arr[i*cols+k];
                    arr[i*cols+k] = temp;
                }
            }
        }
    }
    else{//��������
        for(int k = 0;k<rows;k++){
            for(int i =0;i<cols-1;i++){
                int pos = i;
                for(int j=i;j<cols;j++){
                    if(arr[k*cols+j]<arr[k*cols+pos])
                        pos = j;
                }
                if(pos != i){
                    float8 temp = arr[k*cols+pos];
                    arr[k*cols+pos] = arr[k*cols+i];
                    arr[k*cols+i] = temp;
                }
            }
        }
    }
    //����������
    for(int i=0;i<size;i++) ans_arr_back[i] = Float8GetDatum(arr[i]);
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ��������λȡƽ�������������������С�
// ������������� �������
// ���أ�0���� -1���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z16outer_db4ai_sqrtP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z16inner_db4ai_sqrtP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_sqrt(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // ��������
    SPI_connect();  //���룺��������
    ///////////////////////////////////////////////////////////////////////// 
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_sqrt(t1.data) as data into %s from %s as t1;"
        , output_table_name, input_table_name);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_sqrt(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = ARRNELEMS(arr_raw);                          // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(sqrt(arr[i]));
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ������������ǰ����1���������2����˳��Զ�ӦԪ��������������������С�
// �������������1 �������2 �������
// ���أ�0���� -1���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_subP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_subP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_sub(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������
    ///////////////////////////////////////////////////////////////////////// 
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists1 = get_int32_from_qresult();
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists2 = get_int32_from_qresult();
    if(!(if_input_table_exists1&&if_input_table_exists2)){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ȷ����1�ͱ�2������ͬ������
    // ��ȡ��1������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();

    // �����ͬ�򱨴�
    if(table1_rows!=table2_rows){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-2); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ȷ����1�ͱ�2������ͬ������
    // ��ȡ��1������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();

    // �����ͬ�򱨴�
    if(table1_cols!=table2_cols){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-3); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_sub(t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        , output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_sub(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8����X2��
    ArrayType* arr_raw1 = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType* arr_raw2 = PG_GETARG_ARRAYTYPE_P(1);
    float8* arr1 = (float8 *) ARR_DATA_PTR(arr_raw1);
    float8* arr2 = (float8 *) ARR_DATA_PTR(arr_raw2);
    int size = ARRNELEMS(arr_raw1);                          // ��ARRNELEMS��Դ�����л�ȡ�����Ԫ�ظ���
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(arr1[i]-arr2[i]);
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}





/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ��������������ͣ��������==0��������ͣ��������==1�����������������С�
// ������������� ���������Ҫ��ֻ��Ϊ0��1 �������
// ���أ�0���� -1��������� -6���������Ϊ0��1
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_sumP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_sumP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_sum(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);//��ȷ������Ĳ���Ӧ������о��ǰ�װ�������
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    
    // ��������
    SPI_connect();  
    ///////////////////////////////////////////////////////////////////////// 
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    //�������Ƿ�Ϸ������Ϸ�����
    if(dim != (int32)0 && dim != (int32)1){
        // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        SPI_finish();  // ���룺�ر�����
        PG_RETURN_INT32(-6); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    char sql[MAX_SQL_LEN];
    if(dim == 0)
        sprintf(sql, "SELECT 1 as rows, t1.cols as cols, 0 as trans, inner_db4ai_sum(t1.data,t1.rows,t1.cols,%d) as data into %s from %s as t1;"
            , dim, output_table_name, input_table_name);
    else
        sprintf(sql, "SELECT t1.rows as rows, 1 as cols, 0 as trans, inner_db4ai_sum(t1.data,t1.rows,t1.cols,%d) as data into %s from %s as t1;"
            , dim, output_table_name, input_table_name);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_sum(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8����X2��
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    int32 rows = PG_GETARG_INT32(1);
    int32 cols = PG_GETARG_INT32(2);
    int32 ndim = PG_GETARG_INT32(3);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
   //�������߼�����
     if(ndim == (int32)0){//�������
         // ��palloc��̬�����ڴ�
        Datum* ans_arr_back = (Datum*) palloc(cols * sizeof(Datum)); //������
        float8* temp_acc_arr = (float8*) malloc(cols * sizeof(float8));//��ʱ���飬�����ۼ�
        for(int i=0;i<cols;i++){
            temp_acc_arr[i] = 0.0;   //Ϊ��ʱ�ۼ����鸳��ֵ
        }
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                temp_acc_arr[j]+=arr[i*cols+j]; //��ʼ�ۼ�
            }      
        }
        for(int i=0;i<cols;i++){
            ans_arr_back[i] = Float8GetDatum(temp_acc_arr[i]); //���������������
        }
        free(temp_acc_arr);
        // ���ظ�����
        ArrayType* result = construct_array(ans_arr_back, cols, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
        PG_RETURN_ARRAYTYPE_P(result);
    }
    else{//�������
           
        Datum* ans_arr_back = (Datum*) palloc(rows * sizeof(Datum)); //������
        float8* temp_acc_arr = (float8*) malloc(rows * sizeof(float8));//��ʱ���飬�����ۼ�
        for(int i=0;i<rows;i++){
            temp_acc_arr[i] = 0.0;//Ϊ��ʱ�ۼ����鸳��ֵ
            for(int j=0;j<cols;j++){//��ʼ�ۼ�
                temp_acc_arr[i] += arr[i*cols+j];
            }
            ans_arr_back[i] = Float8GetDatum(temp_acc_arr[i]);//���������������
        }
        free(temp_acc_arr);
        // ���ظ�����
        ArrayType* result = construct_array(ans_arr_back, rows, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
        PG_RETURN_ARRAYTYPE_P(result);
    }
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ���������������tensordot�����˷����㣨����2ά����
// �������������1 �������2 ά�� �������
// ���أ�0���� -1��������� -7ά�Ȳ�Ϊ1��2 -8tensordot���涨�ľ������в�ƥ��
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z21outer_db4ai_tensordotP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z21inner_db4ai_tensordotP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_tensordot(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    int32 dim = PG_GETARG_INT32(2); 
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(3));
    // ��������
    SPI_connect();  //���룺��������
    ///////////////////////////////////////////////////////////////////////// 
    // ��������ڣ�������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists1 = get_int32_from_qresult();
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists2 = get_int32_from_qresult();
    if(!(if_input_table_exists1&&if_input_table_exists2)){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    //tensordot���еļ��鲽��
        // ��ȡ��1������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();
    // ��ȡ��1������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // ��ȡ��2������
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select cols from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();
    int32 outRows = 0;
    int32 outCols = 0;
    if(dim==1){
        if(table1_cols == table2_rows || table1_cols==1 || table2_rows == 1){//����=������������ˣ�ǰһ��������Ϊ1����չ����ˣ���һ��������Ϊ1����չ����ˡ����ࣺ���Ϸ�
            outCols = table2_cols;
            outRows = table1_rows;
        }
        else{
             SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
             PG_RETURN_INT32(-8); // ���ؽ��״̬��
        }
    }
    else if(dim == 2){//��ʾ��ע�������outcols������ָ��������������������������ʾ���󳤺Ϳ���ȷ��ѭ��������
        if(table1_cols==table2_cols&&table1_rows==table2_rows){//�������С��ȫһ�£��ڻ�
            outCols = table2_cols;
            outRows = table1_rows;
        }
        else if((table1_cols==table2_cols&&(table1_rows==1||table2_rows==1))||(table1_rows==table2_rows&&(table1_cols==1||table2_cols==1))){
            outCols = table2_cols>table1_cols?table2_cols:table1_cols;//�У��У���Ϊ1�����У��У�����ȣ���չ���ڻ�
            outRows = table2_rows>table1_rows?table2_rows:table1_rows;
        }else if((table1_rows ==1 &&table1_cols==1)||(table2_rows==1&&table2_cols==1)){
            outCols = table2_cols>table1_cols?table2_cols:table1_cols;
            outRows = table2_rows>table1_rows?table2_rows:table1_rows;
        }
        else{//���಻�Ϸ�
             SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
             PG_RETURN_INT32(-8); // ���ؽ��״̬��
        }
    }
    else{
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-7); // ���ؽ��״̬��
    }

    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    if(dim == 1){
        sprintf(sql, "SELECT %d as rows, %d as cols, t1.trans as trans, inner_db4ai_tensordot(t1.cols,t2.rows,t2.cols,%d,%d,%d,t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
            ,outRows,outCols,dim, outRows,outCols, output_table_name, input_table_name1, input_table_name2);
    }else{
        sprintf(sql, "SELECT %d as rows, %d as cols, t1.trans as trans, inner_db4ai_tensordot(t1.cols,t2.rows,t2.cols,%d,%d,%d,t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
            ,1,1,dim, outRows,outCols, output_table_name, input_table_name1, input_table_name2);
    }

    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_tensordot(PG_FUNCTION_ARGS){ // ������(����) �������
    //��ȡ����������˼�壩
    int32 col1 = PG_GETARG_INT32(0);
    int32 row2 = PG_GETARG_INT32(1);
    int32 col2 = PG_GETARG_INT32(2);
    int32 dim = PG_GETARG_INT32(3);
    int32 outrows = PG_GETARG_INT32(4);
    int32 outcols = PG_GETARG_INT32(5);
    // ��ȡ������һάdouble8����X2��
    ArrayType* arr_raw1 = PG_GETARG_ARRAYTYPE_P(6);
    ArrayType* arr_raw2 = PG_GETARG_ARRAYTYPE_P(7);
    float8* arr1 = (float8 *) ARR_DATA_PTR(arr_raw1);
    float8* arr2 = (float8 *) ARR_DATA_PTR(arr_raw2);
    int size1 = ARRNELEMS(arr_raw1);     // ����1��С
    int size2 = ARRNELEMS(arr_raw2);
    int size;//��������С
    // ����һ��Datum����
    // ��Ҫ�߼�����
    if(dim == 1){
        size = outrows*outcols;
        Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
        float8* ans = (float8*)malloc(size*sizeof(float8));
        int midrow = col1>row2?col1:row2;
        for(int i=0;i<outrows;i++){
            for(int j=0;j<outcols;j++){
                float8 temp = 0;
                for(int k=0;k<midrow;k++){
                    temp += arr1[i*col1+k%col1]*arr2[(k*col2+j)%size2];
                }
                ans[i*outcols+j] = temp;
            }
        }
        for(int i=0; i<size; i++){
            ans_arr_back[i] = Float8GetDatum(ans[i]);
        }
        free(ans);
        // ���ظ�����
        ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
        PG_RETURN_ARRAYTYPE_P(result);
    }
    else{
        size = 1;
        Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
        float8 temp = 0;
        for(int i=0;i<outrows;i++){
            for(int j=0;j<outcols;j++){
                temp+=arr1[(i*col1+j%col1)%size1]*arr2[(i*col2+j%col2)%size2];//Ŀ����ȡ��Сά��������ķ�֧����ͬ
            }
        }
        for(int i=0; i<size; i++){
            ans_arr_back[i] = Float8GetDatum(temp);
        }
        // ���ظ�����
        ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
        PG_RETURN_ARRAYTYPE_P(result);
    }
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ�������ı�����Ӧ�ķ�����trace�󷵻أ�����������������ͬ��
// ����������������������
// ���أ�0����,������Ƿ����򷵻�-3��
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z17outer_db4ai_traceP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z17inner_db4ai_traceP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_trace(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // ��������
    SPI_connect();  //���룺��������

    /////////////////////////////////////////////////////////////////////////
    // ������ֲ��������������Ϊ������Ŀ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    // ��ȡ����������
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // ��ȡ����������
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����Ƿ�Ϊ��������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    if(table1_rows != table1_cols){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-11); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT 1 as rows, 1 as cols, trans, inner_db4ai_trace(%d,data) as data into %s from %s;",
        table1_rows,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_trace(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    int32 dim = PG_GETARG_INT32(0);
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(1);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = 1;
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    float8 trace = 0;
    for(int i=0; i<dim; i++){
        trace = trace + arr[i*dim+i];
    }
    ans_arr_back[0] += Float8GetDatum(trace);
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ���������������֣�����ȫ0�������
// ���������� ���� �������
// ���أ�0����
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z17outer_db4ai_zerosP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(_Z17inner_db4ai_zerosP20FunctionCallInfoData);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_zeros(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������
    
    /////////////////////////////////////////////////////////////////////////
    // ������ֲ��������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    if(rows<=0 || cols<=0){
        SPI_finish();   // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        PG_RETURN_INT32(-4); // ���ؽ��״̬��
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // ����������������ӡ������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // ����INNER����: 
    // SELECT <rows> as rows, <cols> as cols, 0 as trans, inner_db4ai_zeros(<rows>, <cols>) as data into <output_table_name>;
    char sql[MAX_SQL_LEN];
    sprintf(sql,"SELECT %d as rows, %d as cols, 0 as trans, inner_db4ai_zeros(%d) as data into %s;",
        rows, cols, rows * cols, output_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // �ر����ӷ����������������ֱ��ճ�����Լ��޸ĵĴ���Σ�
    SPI_finish();  // ���룺�ر�����
    PG_RETURN_INT32(0); // ���ؽ��״̬��
    /////////////////////////////////////////////////////////////////////////
}
// �ڲ�����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
inner_db4ai_zeros(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ������һάdouble8���飩
    int32 size = PG_GETARG_INT32(0);
    // ����һ��Datum����
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // ��palloc��̬�����ڴ�
    // ��Ҫ�߼�����
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(0.0);
    }
    // ���ظ�����
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}