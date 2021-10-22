#include "postgres.h" // ������ÿ������postgres������C�ļ���

#include "fmgr.h" // ����PG_GETARG_XXX �Լ� PG_RETURN_XXX
#include "access/hash.h"
// #include "access/htup_details.h"
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
    ESPI_init_message(); // ���룺��ʼ��MSG���
    // ���������Ƿ���ڣ��������򱨴�(ע������˫����)
    ESPI_show_message(CHECKING IF INPUT_TABLE EXISTS...);
    
    // if(!ESPI_table_exists(input_table_name)){
    //     // �ַ��������������������������ӡ�
    //     char errmsg[MAX_SQL_LEN]; // ���������飬ע�ⲻ��char*����char[]
    //     strcpy(errmsg,"TABLE NOT EXIST: "); // �ٿ����ַ���
    //     strcat(errmsg, input_table_name); // �������ַ���
    //     ESPI_show_message(errmsg); // �����ʾ�ַ���
    //     // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
    //     SPI_finish();  // ���룺�ر�����
    //     PG_RETURN_INT32(-1); // ���ؽ��״̬��
    // }

    // ��������
    // ESPI_drop_table_if_exists(output_table_name);
    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_abs(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    strcpy(sql, "SELECT rows, cols, trans, inner_db4ai_abs(data) as data into ");
    strcat(sql, output_table_name);
    strcat(sql, " from ");
    strcat(sql, input_table_name);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
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
PG_FUNCTION_INFO_V1(outer_db4ai_acc); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(inner_db4ai_acc);
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
PG_FUNCTION_INFO_V1(outer_db4ai_add); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(inner_db4ai_add);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_add(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������
    ESPI_init_message(); // ���룺��ʼ��MSG���
    // ���������Ƿ���ڣ��������򱨴�
    ESPI_show_message("CHECKING IF INPUT_TABLE EXISTS...");
    if(!ESPI_table_exists(input_table_name1) || !ESPI_table_exists(input_table_name2)){
        // �ַ��������������������������ӡ�
        char errmsg[MAX_SQL_LEN]; // ���������飬ע�ⲻ��char*����char[]
        strcpy(errmsg,"TABLE NOT EXIST! "); // �ٿ����ַ���
        ESPI_show_message(errmsg); // �����ʾ�ַ���
        // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        SPI_finish();  // ���룺�ر�����
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    // ��������
    ESPI_drop_table_if_exists(output_table_name);
    // ����INNER����: SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_add(t1.data, t2.data) as data into output_table_name from input_table_name1 as t1, input_table_name2 as t2;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_add(t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        , output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
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

////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ�������������˳����Ϣ������������С�
// ������������� ����ά�� �������
// ���أ�0���� -1��������� -6ά����Ϊ0��1
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(outer_db4ai_argsort); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(inner_db4ai_argsort);
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
// ���ܣ������������Ľ����ŵ�������С�
// �������������1 / �������2 =  �������
// ���أ�0���� -1���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(outer_db4ai_div); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(inner_db4ai_div);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_div(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������
    ESPI_init_message(); // ���룺��ʼ��MSG���
    // ���������Ƿ���ڣ��������򱨴�
    ESPI_show_message("CHECKING IF INPUT_TABLE EXISTS...");
    if(!ESPI_table_exists(input_table_name1) || !ESPI_table_exists(input_table_name2)){
        // �ַ��������������������������ӡ�
        char errmsg[MAX_SQL_LEN]; // ���������飬ע�ⲻ��char*����char[]
        strcpy(errmsg,"TABLE NOT EXIST! "); // �ٿ����ַ���
        ESPI_show_message(errmsg); // �����ʾ�ַ���
        // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        SPI_finish();  // ���룺�ر�����
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    // ��������
    ESPI_drop_table_if_exists(output_table_name);
    // ����INNER����: SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_div(t1.data, t2.data) as data into output_table_name from input_table_name1 as t1, input_table_name2 as t2;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_div(t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        , output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
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
PG_FUNCTION_INFO_V1(outer_db4ai_dot); // ע�ắ��ΪV1�汾
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
// ���ܣ��������Ԫ��ȡ��eΪ�׵�ָ���Ľ����ŵ�������С�
// ������������� �������
// ���أ�0���� -1���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(outer_db4ai_exp); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(inner_db4ai_exp);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_exp(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // ��������
    SPI_connect();  //���룺��������
    ESPI_init_message(); // ���룺��ʼ��MSG���
    // ���������Ƿ���ڣ��������򱨴�
    ESPI_show_message("CHECKING IF INPUT_TABLE EXISTS...");
    if(!ESPI_table_exists(input_table_name)){
        // �ַ��������������������������ӡ�
        char errmsg[MAX_SQL_LEN]; // ���������飬ע�ⲻ��char*����char[]
        strcpy(errmsg,"TABLE NOT EXIST: "); // �ٿ����ַ���
        strcat(errmsg, input_table_name); // �������ַ���
        ESPI_show_message(errmsg); // �����ʾ�ַ���
        // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        SPI_finish();  // ���룺�ر�����
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    // ��������
    ESPI_drop_table_if_exists(output_table_name);
    // ����INNER����: SELECT rows, cols, trans, inner_db4ai_exp(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    strcpy(sql, "SELECT rows, cols, trans, inner_db4ai_exp(data) as data into ");
    strcat(sql, output_table_name);
    strcat(sql, " from ");
    strcat(sql, input_table_name);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
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
// ���ܣ���������������˷��������������������
// �������������1 �������2 �������
// ���أ�0���� -1��������� -9�������ʱǰһ��������������ں�һ���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(outer_db4ai_matmul); // ע�ắ��ΪV1�汾
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

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// ���ܣ����������X�Ľ����ŵ�������С�
// �������������1 �������2 �������
// ���أ�0���� -1���������
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(outer_db4ai_mul); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(inner_db4ai_mul);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_mul(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������
    ESPI_init_message(); // ���룺��ʼ��MSG���
    // ���������Ƿ���ڣ��������򱨴�
    ESPI_show_message("CHECKING IF INPUT_TABLE EXISTS...");
    if(!ESPI_table_exists(input_table_name1) || !ESPI_table_exists(input_table_name2)){
        // �ַ��������������������������ӡ�
        char errmsg[MAX_SQL_LEN]; // ���������飬ע�ⲻ��char*����char[]
        strcpy(errmsg,"TABLE NOT EXIST! "); // �ٿ����ַ���
        ESPI_show_message(errmsg); // �����ʾ�ַ���
        // �ڷ�֧��λ��һ��Ҫ��ʱ�ر����ӣ�
        SPI_finish();  // ���룺�ر�����
        PG_RETURN_INT32(-1); // ���ؽ��״̬��
    }
    // ��������
    ESPI_drop_table_if_exists(output_table_name);
    // ����INNER����: SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_mul(t1.data, t2.data) as data into output_table_name from input_table_name1 as t1, input_table_name2 as t2;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_mul(t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        , output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
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
// ���أ�0����
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(_Z17outer_db4ai_zerosP20FunctionCallInfoData); // ע�ắ��ΪV1�汾
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
    ESPI_init_message(); // ���룺��ʼ��MSG���
    // ��������
    ESPI_drop_table_if_exists(output_table_name);
    // ����INNER����: 
    // SELECT <rows> as rows, <cols> as cols, 0 as trans, inner_db4ai_ones(<rows> * <cols>) as data into <output_table_name>;
    char sql[MAX_SQL_LEN];
    sprintf(sql,"SELECT %d as rows, %d as cols, 0 as trans, inner_db4ai_ones(%d) as data into %s;",
        rows, cols, rows * cols, output_table_name);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
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
// ���ܣ���������������֣�����ȫ0�������
// ���������� ���� �������
// ���أ�0����
/////////////////////////////////////////////////////////////////////////
// ����ע��
PG_FUNCTION_INFO_V1(outer_db4ai_zeros); // ע�ắ��ΪV1�汾
PG_FUNCTION_INFO_V1(inner_db4ai_zeros);
// �ⲿ����
Datum // ����Postgres�����Ĳ����ͷ������Ͷ���Datum
outer_db4ai_zeros(PG_FUNCTION_ARGS){ // ������(����) �������
    // ��ȡ����
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ��������
    SPI_connect();  //���룺��������
    ESPI_init_message(); // ���룺��ʼ��MSG���
    // ��������
    ESPI_drop_table_if_exists(output_table_name);
    // ����INNER����: 
    // SELECT <rows> as rows, <cols> as cols, 0 as trans, inner_db4ai_zeros(<rows>, <cols>) as data into <output_table_name>;
    char sql[MAX_SQL_LEN];
    sprintf(sql,"SELECT %d as rows, %d as cols, 0 as trans, inner_db4ai_zeros(%d) as data into %s;",
        rows, cols, rows * cols, output_table_name);
    SPI_exec(sql, 0);
    // �ر�����
    SPI_finish();  // ���룺�ر�����
    // ���ؽ��
    PG_RETURN_INT32(0); // ���ؽ��״̬��
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




/*
ע����һ�У���
if (SPI_processed > 0)
	{
		selected = DatumGetInt32(CStringGetDatum(SPI_getvalue(
													   SPI_tuptable->vals[0],
													   SPI_tuptable->tupdesc,
																		 1
																		)));
	}

*/