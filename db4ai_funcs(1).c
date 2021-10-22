#include "postgres.h" // 包含在每个声明postgres函数的C文件中

#include "fmgr.h" // 用于PG_GETARG_XXX 以及 PG_RETURN_XXX
#include "access/hash.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/array.h" // 用于提供ArrayType*
#include "executor/spi.h" // 用于调用SPI

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

PG_MODULE_MAGIC; // 由PostgreSQL在加载时检查兼容性

// 【在执行单行单数sql查询之后】从查询结果中获取数字，用于获取行数、列数、是否转置
// USAGE: int32 val = get_int32_from_qresult();
#define get_int32_from_qresult() atoi(SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1))

// 【在执行单行单数sql查询之后】从查询结果中获取字符串，备用
// USAGE: char* get_string_from_qresult();
#define get_string_from_qresult() SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1)

// 结果状态码：
// 0 -> 正常
// -1 -> 输入表不存在
// -2 -> 两表行数不同
// -3 -> 两表列数不同
// -4 -> 代表行数或列数的数字参数非正数
// -5 -> 新的行数和新的列数的乘积不等于原先元素个数
// -6 -> 表示操作方向的参数dim不为0或者-1

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表按元素取绝对值的结果存放到输出表中。
// 参数：输入表名 输出表名
// 返回：0正常 -1输入表不存在
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_absP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_absP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_abs(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // 启动连接
    SPI_connect();  //必须：建立连接
    
    ///////////////////////////////////////////////////////////////////////// 
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_abs(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT rows, cols, trans, inner_db4ai_abs(data) as data into %s from %s;",
        output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_abs(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = ARRNELEMS(arr_raw);                          // 用ARRNELEMS从源数据中获取数组的元素个数
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(arr[i]<0?-arr[i]:arr[i]);
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：计算两个输入表相同的元素的个数（normalize=0）或相同元素个数所占的比例（normalize = 1）
// 参数：输入表名1 输入表名2 标准化normalize 输出表名
// 返回：0正常 -1输入表不存在 -2行数不同 -3 列数不同
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_accP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_accP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_acc(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    int32 normalize = PG_GETARG_INT32(2);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(3));
    // 启动连接
    SPI_connect();  //必须：建立连接
    ///////////////////////////////////////////////////////////////////////// 
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists1 = get_int32_from_qresult();
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists2 = get_int32_from_qresult();
    if(!(if_input_table_exists1&&if_input_table_exists2)){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 确保表1和表2具有相同的行数
    // 获取表1的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // 获取表2的行数
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();

    // 如果不同则报错
    if(table1_rows!=table2_rows){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-2); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 确保表1和表2具有相同的列数
    // 获取表1的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // 获取表2的列数
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select cols from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();

    // 如果不同则报错
    if(table1_cols!=table2_cols){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-3); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT %d as rows, %d as cols, t1.trans as trans, inner_db4ai_acc(t1.data, t2.data,%d) as data into %s from %s as t1, %s as t2;"
        ,1,1,normalize, output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_acc(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组X2）
    ArrayType* arr_raw1 = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType* arr_raw2 = PG_GETARG_ARRAYTYPE_P(1);
    int32 norm = PG_GETARG_INT32(2);
    float8* arr1 = (float8 *) ARR_DATA_PTR(arr_raw1);
    float8* arr2 = (float8 *) ARR_DATA_PTR(arr_raw2);
    int size = ARRNELEMS(arr_raw1);                          // 用ARRNELEMS从源数据中获取数组的元素个数
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    float8 acc = 0;
    for(int i=0; i<size; i++){
        acc+=(arr1[i]==arr2[i]?1:0);//计算相同的个数
    }
    if(norm == 1){
        acc = acc/size;
    }
    ans_arr_back[0] = Float8GetDatum(acc);
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, 1, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表相加的结果存放到输出表中。
// 参数：输入表名1 输入表名2 输出表名
// 返回：0正常 -1输入表不存在
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_addP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_addP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_add(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接

    /////////////////////////////////////////////////////////////////////////
    // 如果表1不存在，报错并打印
    char sql_table1_exists[MAX_SQL_LEN];
    sprintf(sql_table1_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table1_exists, 0);
    int32 if_input_table1_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table1_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 如果表2不存在，报错并打印
    char sql_table2_exists[MAX_SQL_LEN];
    sprintf(sql_table2_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table2_exists, 0);
    int32 if_input_table2_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table2_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 确保表1和表2具有相同的行数
    // 获取表1的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // 获取表2的行数
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();

    // 如果不同则报错
    if(table1_rows!=table2_rows){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-2); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 确保表1和表2具有相同的列数
    // 获取表1的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // 获取表2的列数
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select cols from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();

    // 如果不同则报错
    if(table1_cols!=table2_cols){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-3); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////


    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////


    // 调用INNER函数: SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_add(t1.data, t2.data) as data into output_table_name from input_table_name1 as t1, input_table_name2 as t2;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_add(t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        , output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_add(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组X2）
    ArrayType* arr_raw1 = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType* arr_raw2 = PG_GETARG_ARRAYTYPE_P(1);
    float8* arr1 = (float8 *) ARR_DATA_PTR(arr_raw1);
    float8* arr2 = (float8 *) ARR_DATA_PTR(arr_raw2);
    int size = ARRNELEMS(arr_raw1);                          // 用ARRNELEMS从源数据中获取数组的元素个数
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(arr1[i]+arr2[i]);
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入的矩阵表的每个元素都找最大值的索引，形成输出表
// 参数：行数 列数 填充值
// 返回：0正常,如果dim不是0或1返回-6
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z18outer_db4ai_argmaxP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z18inner_db4ai_argmaxP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_argmax(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接

    /////////////////////////////////////////////////////////////////////////
    // 检测数字参数的情况——作为整体数目（用于直接粘贴和稍加修改的代码段）
    // 获取输入表的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // 获取输入表的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 检测参数dim是否正确
    if(dim != 0 && dim != 1){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-6); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
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
    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT 1 as rows,  %d as cols, trans, inner_db4ai_argmax(%d,%d,%d,data) as data into %s from %s;",
        table2_cols,table1_rows,table1_cols,dim,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_argmax(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
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
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
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
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入的矩阵表的每个元素都找最小值的索引，形成输出表
// 参数：行数 列数 填充值
// 返回：0正常,如果dim不是0或1返回-6
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z18outer_db4ai_argminP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z18inner_db4ai_argminP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_argmin(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接

    /////////////////////////////////////////////////////////////////////////
    // 检测数字参数的情况——作为整体数目（用于直接粘贴和稍加修改的代码段）
    // 获取输入表的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // 获取输入表的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 检测参数dim是否正确
    if(dim != 0 && dim != 1){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-6); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
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
    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT 1 as rows,  %d as cols, trans, inner_db4ai_argmin(%d,%d,%d,data) as data into %s from %s;",
        table2_cols,table1_rows,table1_cols,dim,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_argmin(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
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
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
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
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表排序后顺序信息保存在输出表中。
// 参数：输入表名 排序维度 输出表名
// 返回：0正常 -1输入表不存在 -6维数不为0或1
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z19outer_db4ai_argsortP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z19inner_db4ai_argsortP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_argsort(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接
    ///////////////////////////////////////////////////////////////////////// 
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    //检查参数是否合法，不合法报错
    if(dim != (int32)0 && dim != (int32)1){
        // 在分支的位置一定要及时关闭连接！
        SPI_finish();  // 必须：关闭连接
        PG_RETURN_INT32(-6); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_argsort(t1.data,t1.rows,t1.cols,%d) as data into %s from %s as t1;"
        ,dim, output_table_name, input_table_name);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_argsort(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    // 获取参数 int32类型列数，行数，维度值
    int32 rows = PG_GETARG_INT32(1);
    int32 cols = PG_GETARG_INT32(2);
    int32 dim = PG_GETARG_INT32(3);
    int size = ARRNELEMS(arr_raw);                          // 用ARRNELEMS从源数据中获取数组的元素个数
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    int32* argset = (int32*)malloc(size*sizeof(int32));
    // 主要逻辑部分
    //不让定义工具函数就没法快排，使用选择排序
    if(dim == 0){//按列排序
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

                    int32 argtmp = argset[pos*cols+k];//所在位置一样变动
                    argset[pos*cols+k] = argset[i*cols+k];
                    argset[i*cols+k] = argtmp;
                }
            }
            for(int i=0;i<rows;i++) arr[argset[i*cols+k]*cols+k] = i;//所在位置排序
        }
    }
    else{//按行排序
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
    //构建答案数组
    for(int i=0;i<size;i++) ans_arr_back[i] = Float8GetDatum(arr[i]);
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表相加的结果存放到输出表中。
// 参数：输入表名1 输入表名2 输出表名
// 返回：0正常 -1输入表不存在
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_divP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_divP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_div(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接

    /////////////////////////////////////////////////////////////////////////
    // 如果表1不存在，报错并打印
    char sql_table1_exists[MAX_SQL_LEN];
    sprintf(sql_table1_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table1_exists, 0);
    int32 if_input_table1_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table1_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 如果表2不存在，报错并打印
    char sql_table2_exists[MAX_SQL_LEN];
    sprintf(sql_table2_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table2_exists, 0);
    int32 if_input_table2_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table2_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 确保表1和表2具有相同的行数
    // 获取表1的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // 获取表2的行数
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();

    // 如果不同则报错
    if(table1_rows!=table2_rows){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-2); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 确保表1和表2具有相同的列数
    // 获取表1的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // 获取表2的列数
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select cols from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();

    // 如果不同则报错
    if(table1_cols!=table2_cols){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-3); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////


    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////


    // 调用INNER函数: SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_div(t1.data, t2.data) as data into output_table_name from input_table_name1 as t1, input_table_name2 as t2;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_div(t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        , output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_div(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组X2）
    ArrayType* arr_raw1 = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType* arr_raw2 = PG_GETARG_ARRAYTYPE_P(1);
    float8* arr1 = (float8 *) ARR_DATA_PTR(arr_raw1);
    float8* arr2 = (float8 *) ARR_DATA_PTR(arr_raw2);
    int size = ARRNELEMS(arr_raw1);                          // 用ARRNELEMS从源数据中获取数组的元素个数
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(arr1[i]/arr2[i]);
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将两个输入表矩阵做内积运算结果保存在输出表中
// 参数：输入表名1 输入表名2 输出表名
// 返回：0正常 -2两矩阵行数不同 -3 两矩阵列数不同 -10 两矩阵不为1维矩阵
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_dotP20FunctionCallInfoData); // 注册函数为V1版本
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_dot(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接
    ///////////////////////////////////////////////////////////////////////// 
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists1 = get_int32_from_qresult();
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists2 = get_int32_from_qresult();
    if(!(if_input_table_exists1&&if_input_table_exists2)){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    // 矩阵内积之前的检查步骤
    // 获取表1的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // 获取表2的行数
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();
    // 获取表1的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // 获取表2的列数
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select cols from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();
    if(table1_cols != table2_cols){//行数=列数：矩阵相乘
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-3); // 返回结果状态码
    }
    else if(table1_rows != table2_rows){
         SPI_finish();   // 在分支的位置一定要及时关闭连接！
         PG_RETURN_INT32(-2); // 返回结果状态码
    }
    if(table1_rows !=1&&table1_cols!=1){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-10); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT %d as rows, %d as cols, t1.trans as trans, inner_db4ai_tensordot(t1.cols,t2.rows,t2.cols,%d,%d,%d,t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        ,1,1,2, table1_rows,table1_cols, output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表按元素取e为底指数值的结果存放到输出表中。
// 参数：输入表名 输出表名
// 返回：0正常 -1输入表不存在
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_expP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_expP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_exp(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // 启动连接
    SPI_connect();  //必须：建立连接
    
    ///////////////////////////////////////////////////////////////////////// 
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_exp(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT rows, cols, trans, inner_db4ai_exp(data) as data into %s from %s;",
        output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_exp(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = ARRNELEMS(arr_raw);                          // 用ARRNELEMS从源数据中获取数组的元素个数
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(exp(arr[i]));
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：创建一个与输入矩阵同大小的名为output_table_name的元素均为full_value的矩阵表
// 参数：输入 填充值 输出
// 返回：0正常
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z16outer_db4ai_fullP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z16inner_db4ai_fullP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_full(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    float8 full_value = PG_GETARG_FLOAT8(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // 调用INNER函数:
    // SELECT <rows> as rows, <cols> as cols, 0 as trans, inner_db4ai_full(<rows>, <cols>) as data into <output_table_name>;
    char sql[MAX_SQL_LEN];
    sprintf(sql,"SELECT rows, cols,trans, inner_db4ai_full(%f,data) as data into %s from %s;",
        full_value, output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_full(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
    float8 full_value = PG_GETARG_FLOAT8(0);
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(1);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = ARRNELEMS(arr_raw);
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(full_value);
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入的矩阵表的每个元素都求对数，形成输出矩阵表
// 参数：输入表名 输出表名
// 返回：0正常 -1输入表不存在
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_logP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_logP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_log(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // 启动连接
    SPI_connect();  //必须：建立连接

    /////////////////////////////////////////////////////////////////////////
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_log(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT rows, cols, trans, inner_db4ai_log(data) as data into %s from %s;",
        output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_log(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = ARRNELEMS(arr_raw);                          // 用ARRNELEMS从源数据中获取数组的元素个数
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(log(arr[i]));
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入的矩阵表按选择的维度寻找最小值，形成数值表
// 参数：行数 列数 填充值
// 返回：0正常,如果dim不是0或1返回-6
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_maxP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_maxP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_max(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接

    /////////////////////////////////////////////////////////////////////////
    // 检测数字参数的情况——作为整体数目（用于直接粘贴和稍加修改的代码段）
    // 获取输入表的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // 获取输入表的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 检测参数dim是否正确
    if(dim != 0 && dim != 1){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-6); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
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
    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT 1 as rows,  %d as cols, trans, inner_db4ai_max(%d,%d,%d,data) as data into %s from %s;",
        table2_cols,table1_rows,table1_cols,dim,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_max(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
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
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
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
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入的矩阵表按选择的维度计算均值，形成数值表
// 参数：行数 列数 填充值
// 返回：0正常,如果dim不是0或1返回-6
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z16outer_db4ai_meanP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z16inner_db4ai_meanP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_mean(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接

    /////////////////////////////////////////////////////////////////////////
    // 检测数字参数的情况——作为整体数目（用于直接粘贴和稍加修改的代码段）
    // 获取输入表的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // 获取输入表的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 检测参数dim是否正确
    if(dim != 0 && dim != 1){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-6); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
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
    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT 1 as rows,  %d as cols, trans, inner_db4ai_mean(%d,%d,%d,data) as data into %s from %s;",
        table2_cols,table1_rows,table1_cols,dim,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_mean(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
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
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
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
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入的矩阵表按选择的维度寻找最小值，形成数值表
// 参数：行数 列数 填充值
// 返回：0正常,如果dim不是0或1返回-6
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_minP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_minP20FunctionCallInfoData);
// 外部函数in
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_min(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接

    /////////////////////////////////////////////////////////////////////////
    // 检测数字参数的情况——作为整体数目（用于直接粘贴和稍加修改的代码段）
    // 获取输入表的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // 获取输入表的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 检测参数dim是否正确
    if(dim != 0 && dim != 1){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-6); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
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
    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT 1 as rows,  %d as cols, trans, inner_db4ai_min(%d,%d,%d,data) as data into %s from %s;",
        table2_cols,table1_rows,table1_cols,dim,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_min(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
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
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
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
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将两个输入表矩阵乘法运算结果保存在输出表中
// 参数：输入表名1 输入表名2 输出表名
// 返回：0正常 -1输入表不存在 -9矩阵相乘时前一矩阵的列数不等于后一矩阵的行数
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z18outer_db4ai_matmulP20FunctionCallInfoData); // 注册函数为V1版本
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_matmul(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接
    ///////////////////////////////////////////////////////////////////////// 
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists1 = get_int32_from_qresult();
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists2 = get_int32_from_qresult();
    if(!(if_input_table_exists1&&if_input_table_exists2)){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    //矩阵乘法之前的检查步骤
    // 获取表1的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // 获取表2的行数
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();
    // 获取表1的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // 获取表2的列数
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select cols from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();
    int32 outRows = 0;
    int32 outCols = 0;
    if(table1_cols == table2_rows){//行数=列数：矩阵相乘
        outCols = table2_cols;
        outRows = table1_rows;
    }
    else{
         SPI_finish();   // 在分支的位置一定要及时关闭连接！
         PG_RETURN_INT32(-9); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT %d as rows, %d as cols, t1.trans as trans, inner_db4ai_tensordot(t1.cols,t2.rows,t2.cols,%d,%d,%d,t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        ,outRows,outCols,1, outRows,outCols, output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
}
// 此函数调用了其它算子的内部函数，因此无需实现自己的内部函数。

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表相乘的结果存放到输出表中。
// 参数：输入表名1 输入表名2 输出表名
// 返回：0正常 -1输入表不存在
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_mulP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_mulP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_mul(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接

    /////////////////////////////////////////////////////////////////////////
    // 如果表1不存在，报错并打印
    char sql_table1_exists[MAX_SQL_LEN];
    sprintf(sql_table1_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table1_exists, 0);
    int32 if_input_table1_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table1_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 如果表2不存在，报错并打印
    char sql_table2_exists[MAX_SQL_LEN];
    sprintf(sql_table2_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table2_exists, 0);
    int32 if_input_table2_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table2_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 确保表1和表2具有相同的行数
    // 获取表1的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // 获取表2的行数
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();

    // 如果不同则报错
    if(table1_rows!=table2_rows){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-2); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 确保表1和表2具有相同的列数
    // 获取表1的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // 获取表2的列数
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select cols from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();

    // 如果不同则报错
    if(table1_cols!=table2_cols){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-3); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////


    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////


    // 调用INNER函数: SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_mul(t1.data, t2.data) as data into output_table_name from input_table_name1 as t1, input_table_name2 as t2;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_mul(t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        , output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_mul(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组X2）
    ArrayType* arr_raw1 = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType* arr_raw2 = PG_GETARG_ARRAYTYPE_P(1);
    float8* arr1 = (float8 *) ARR_DATA_PTR(arr_raw1);
    float8* arr2 = (float8 *) ARR_DATA_PTR(arr_raw2);
    int size = ARRNELEMS(arr_raw1);                          // 用ARRNELEMS从源数据中获取数组的元素个数
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(arr1[i]*arr2[i]);
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将按照输入的数字，建立全0的输出表。
// 参数：行数 列数 输出表名
// 返回：0正常 -4数字参数不正常
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z16outer_db4ai_onesP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z16inner_db4ai_onesP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_ones(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接
    
    /////////////////////////////////////////////////////////////////////////
    // 检测数字参数的情况（用于直接粘贴和稍加修改的代码段）
    if(rows<=0 || cols<=0){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-4); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // 调用INNER函数: 
    // SELECT <rows> as rows, <cols> as cols, 0 as trans, inner_db4ai_ones(<rows>, <cols>) as data into <output_table_name>;
    char sql[MAX_SQL_LEN];
    sprintf(sql,"SELECT %d as rows, %d as cols, 0 as trans, inner_db4ai_ones(%d) as data into %s;",
        rows, cols, rows * cols, output_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_ones(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
    int32 size = PG_GETARG_INT32(0);
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(1.0);
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表的每个元素求相应的指数并将结果存放到输出表中。
// 参数：输入表名 为每个元素进行指数运算的指数值 输出表名
// 返回：0正常 -1输入表不存在
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_powP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_powP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_pow(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    float8 pow_exp = PG_GETARG_FLOAT8(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接

    /////////////////////////////////////////////////////////////////////////
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT rows, cols, trans, inner_db4ai_pow(data,%f) as data into %s from %s;",
        pow_exp,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_pow(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    float8 pow_exp = PG_GETARG_FLOAT8(1);
    int size = ARRNELEMS(arr_raw);                          // 用ARRNELEMS从源数据中获取数组的元素个数
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(pow(arr[i],pow_exp));
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}
/////////////////////////////////////////////////////////////////////////
// 功能：将输入的矩阵表按照新的行和列“重组”，形成输出矩阵表，返回状态码
// 参数：输入表名 新行数 新列数 输出表名 （要求新的行数*新的列数等于原本的元素个数）
// 返回：0正常 -4参数不正常
/////////////////////////////////////////////////////////////////////////
// 在我们的存储方式中，重组只需要改变行和列数即可，对数据没有实际改动
// 函数注册
PG_FUNCTION_INFO_V1(_Z18outer_db4ai_repeatP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z18inner_db4ai_repeatP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_repeat(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim1 = PG_GETARG_INT32(1);
    int32 dim2 = PG_GETARG_INT32(2);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(3));
    // 启动连接
    SPI_connect();  //必须：建立连接

    /////////////////////////////////////////////////////////////////////////
    // 检测数字参数的情况——作为行数列数（用于直接粘贴和稍加修改的代码段）
    if(dim1<=0 || dim2<=0){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-4); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t.rows*%d as rows, t.cols*%d as cols, trans, inner_db4ai_repeat(rows,cols,%d,%d,data) as data into %s from %s as t;",
        dim1,dim2,dim1,dim2,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
}

// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_repeat(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    int32 dim1 = PG_GETARG_INT32(2);
    int32 dim2 = PG_GETARG_INT32(3);
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(4);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = ARRNELEMS(arr_raw);                                  // 用ARRNELEMS从源数据中获取数组的元素个数
    size = size * dim2 * dim1;
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    for(int i=0; i<dim1; i++){
        for(int j=0; j<dim2; j++){
            for(int k=0; k<rows; k++){
                for(int x=0; x<cols;x++){
                    ans_arr_back[(i*dim2)*rows*cols+k*cols*dim2+j*cols+x] = Float8GetDatum(arr[k*cols+x]);
                }
            }
        }
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}
/////////////////////////////////////////////////////////////////////////
// 功能：将输入的矩阵表按照新的行和列“重组”，形成输出矩阵表，返回状态码
// 参数：输入表名 新行数 新列数 输出表名 （要求新的行数*新的列数等于原本的元素个数）
// 返回：0正常 -4参数不正常
/////////////////////////////////////////////////////////////////////////
// 在我们的存储方式中，重组只需要改变行和列数即可，对数据没有实际改动
// 函数注册
PG_FUNCTION_INFO_V1(_Z19outer_db4ai_reshapeP20FunctionCallInfoData); // 注册函数为V1版本
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_reshape(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 rows = PG_GETARG_INT32(1);
    int32 cols = PG_GETARG_INT32(2);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(3));
    // 启动连接
    SPI_connect();  //必须：建立连接
    
    /////////////////////////////////////////////////////////////////////////
    // 检测数字参数的情况——作为行数列数（用于直接粘贴和稍加修改的代码段）
    if(rows<=0 || cols<=0){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-4); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 检测数字参数的情况——作为整体数目（用于直接粘贴和稍加修改的代码段）
    // 获取输入表的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // 获取输入表的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    // 查看是否有问题
    if((table1_rows*table1_cols)!=(rows*cols)){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-5); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // 调用INNER函数: 
    // SELECT <rows> as rows, <cols> as cols, 0 as trans, inner_db4ai_reshape(<rows>, <cols>) as data into <output_table_name>;
    char sql[MAX_SQL_LEN];
    sprintf(sql,"SELECT %d as rows, %d as cols, 0 as trans, data as data into %s from %s;",
        rows, cols, output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 本算子无需内部函数

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表，按列翻转（输入参数==0）或按行翻转（输入参数==1）或全部翻转（输入参数=3）结果保存在输出表中。
// 注意：与torch.flip存在区别
// 参数：输入表名 输入参数dim，要求只能为0，1或2 输出表名
// 返回：0正常 -1输入表不存在 -6输入参数不为0，1或2
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z19outer_db4ai_reverseP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z19inner_db4ai_reverseP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_reverse(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);//不确定这里的参数应该填几，感觉是安装上面的来
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    
    // 启动连接
    SPI_connect();  
    ///////////////////////////////////////////////////////////////////////// 
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    //检查参数是否合法，不合法报错
    if(dim != (int32)0 && dim != (int32)1 && dim != (int32)2){
        // 在分支的位置一定要及时关闭连接！
        SPI_finish();  // 必须：关闭连接
        PG_RETURN_INT32(-6); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows as rows, t1.cols as cols, t1.trans as trans, inner_db4ai_reverse(t1.data,t1.rows,t1.cols,%d) as data into %s from %s as t1;"
            , dim, output_table_name, input_table_name);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_reverse(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组X2）
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    int32 rows = PG_GETARG_INT32(1);
    int32 cols = PG_GETARG_INT32(2);
    int32 ndim = PG_GETARG_INT32(3);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = ARRNELEMS(arr_raw);      
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum)); //答案数组
       //函数主逻辑部分
     if(ndim == (int32)0){//按列翻转
         // 用palloc动态分配内存
        float8* temp_acc_arr = (float8*) malloc(cols * sizeof(float8));//临时数组，用于翻转交换
        for(int i=0;i<rows/2;i++){
            for(int j=0;j<cols;j++){
                temp_acc_arr[j]=arr[i*cols+j]; //三步交换
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
    else if(ndim == 1){//按行翻转
        for(int i=0;i<rows;i++){
            float8 temp;
            for(int j=0;j<cols/2;j++){//开始累加
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
        ans_arr_back[i] = Float8GetDatum(arr[i]);//将答案输出至答案数组
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表的行数和列数存放到输出表中。
// 参数：输入表名 输出表名
// 返回：0正常 -1输入表不存在
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z17outer_db4ai_shapeP20FunctionCallInfoData); // 注册函数为V1版本
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_shape(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name =text_to_cstring(PG_GETARG_TEXT_PP(1));

    // 启动连接
    SPI_connect();  //必须：建立连接

     /////////////////////////////////////////////////////////////////////////
    // 检测数字参数的情况——作为整体数目（用于直接粘贴和稍加修改的代码段）
    // 获取输入表的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // 获取输入表的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_shape(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT 1 as rows, 2 as cols, 0 as trans, '{%d,%d}' as data into %s from %s;",
        table1_rows,table1_cols,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 本函数无需内部函数

/////////////////////////////////////////////////////////////////////////
// 功能：将输入的矩阵表切片，保存到输出表中。
// 参数：输入表名 起始行 起始列 终止行 终止列 输出表名
// 返回：0正常 -4参数不正常
/////////////////////////////////////////////////////////////////////////
// 在我们的存储方式中，重组只需要改变行和列数即可，对数据没有实际改动
// 函数注册
PG_FUNCTION_INFO_V1(_Z17outer_db4ai_sliceP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z17inner_db4ai_sliceP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_slice(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim1_start = PG_GETARG_INT32(1);
    int32 dim1_end = PG_GETARG_INT32(2);
    int32 dim2_start = PG_GETARG_INT32(3);
    int32 dim2_end = PG_GETARG_INT32(4);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(5));
    // 启动连接
    SPI_connect();  //必须：建立连接

    /////////////////////////////////////////////////////////////////////////
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 检测数字参数的情况——作为整体数目（用于直接粘贴和稍加修改的代码段）
    // 获取输入表的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // 获取输入表的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 检测参数dim是否正确
    if(!(dim1_start>=0 && dim1_start<dim1_end && dim1_end<table1_rows &&
        dim2_start>=0 && dim2_start<dim2_end && dim2_end <table1_cols)){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-12); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT %d as rows, %d as cols, trans, inner_db4ai_slice(rows,cols,%d,%d,%d,%d,data) as data into %s from %s;",
        dim1_end - dim1_start + 1,dim2_end - dim2_start + 1,dim1_start,dim1_end,dim2_start,dim2_end,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
}

// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_slice(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
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
    int size = rows2 * cols2;                                 // 用ARRNELEMS从源数据中获取数组的元素个数
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    for(int i = 0; i < rows2; i++){
        for(int j = 0; j < cols2; j++){
            ans_arr_back[i*cols2+j] = Float8GetDatum(arr[(dim1_start+i)*cols + dim2_start+j]);
        }
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表计算softmax函数结果保存在输出表中。
// 参数：输入表名 输出表名
// 返回：0正常 -1输入表不存在 -6维数不为0或1
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z19outer_db4ai_softmaxP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z19inner_db4ai_softmaxP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_softmax(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接
    ///////////////////////////////////////////////////////////////////////// 
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    //检查参数是否合法，不合法报错
    if(dim != (int32)0 && dim != (int32)1){
        // 在分支的位置一定要及时关闭连接！
        SPI_finish();  // 必须：关闭连接
        PG_RETURN_INT32(-6); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_softmax(t1.data,t1.rows,t1.cols,%d) as data into %s from %s as t1;"
        ,dim, output_table_name, input_table_name);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_softmax(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    // 获取参数 int32类型列数，行数，维度值
    int32 rows = PG_GETARG_INT32(1);
    int32 cols = PG_GETARG_INT32(2);
    int32 dim = PG_GETARG_INT32(3);
    int size = ARRNELEMS(arr_raw);                          // 用ARRNELEMS从源数据中获取数组的元素个数
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    //计算exp矩阵
    for(int i=0;i<size;i++) arr[i] = exp(arr[i]);
    if(dim == 0){//按列归一
        float8* byDiv =  (float8*)malloc(cols * sizeof(float8));//计算求和待除矩阵
        for(int i=0;i<cols;i++){
            byDiv[i] = 0;
            for(int j=0;j<rows;j++){
                byDiv[i] +=  arr[j*cols+i];
            }
        }
        for(int i=0;i<size;i++) arr[i] /= byDiv[i%cols]; 
        free(byDiv);
    }
    else{//按行归一
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
    //构建答案数组
    for(int i=0;i<size;i++) ans_arr_back[i] = Float8GetDatum(arr[i]);
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}


////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表排序结果保存在输出表中。
// 参数：输入表名 输出表名
// 返回：0正常 -1输入表不存在 -6维数不为0或1
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z16outer_db4ai_sortP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z16inner_db4ai_sortP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_sort(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接
    ///////////////////////////////////////////////////////////////////////// 
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    //检查参数是否合法，不合法报错
    if(dim != (int32)0 && dim != (int32)1){
        // 在分支的位置一定要及时关闭连接！
        SPI_finish();  // 必须：关闭连接
        PG_RETURN_INT32(-6); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_sort(t1.data,t1.rows,t1.cols,%d) as data into %s from %s as t1;"
        ,dim, output_table_name, input_table_name);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_sort(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    // 获取参数 int32类型列数，行数，维度值
    int32 rows = PG_GETARG_INT32(1);
    int32 cols = PG_GETARG_INT32(2);
    int32 dim = PG_GETARG_INT32(3);
    int size = ARRNELEMS(arr_raw);                          // 用ARRNELEMS从源数据中获取数组的元素个数
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    //不让定义工具函数就没法快排，使用选择排序
    if(dim == 0){//按列排序
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
    else{//按行排序
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
    //构建答案数组
    for(int i=0;i<size;i++) ans_arr_back[i] = Float8GetDatum(arr[i]);
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}

////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表按位取平方根结果保存在输出表中。
// 参数：输入表名 输出表名
// 返回：0正常 -1输入表不存在
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z16outer_db4ai_sqrtP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z16inner_db4ai_sqrtP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_sqrt(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // 启动连接
    SPI_connect();  //必须：建立连接
    ///////////////////////////////////////////////////////////////////////// 
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_sqrt(t1.data) as data into %s from %s as t1;"
        , output_table_name, input_table_name);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_sqrt(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = ARRNELEMS(arr_raw);                          // 用ARRNELEMS从源数据中获取数组的元素个数
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(sqrt(arr[i]));
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将两个输入表按前表（表1）减后表（表2）的顺序对对应元素做差，结果保存在输出表中。
// 参数：输入表名1 输入表名2 输出表名
// 返回：0正常 -1输入表不存在
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_subP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_subP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_sub(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接
    ///////////////////////////////////////////////////////////////////////// 
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists1 = get_int32_from_qresult();
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists2 = get_int32_from_qresult();
    if(!(if_input_table_exists1&&if_input_table_exists2)){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 确保表1和表2具有相同的行数
    // 获取表1的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // 获取表2的行数
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();

    // 如果不同则报错
    if(table1_rows!=table2_rows){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-2); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 确保表1和表2具有相同的列数
    // 获取表1的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // 获取表2的列数
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();

    // 如果不同则报错
    if(table1_cols!=table2_cols){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-3); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_sub(t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        , output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_sub(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组X2）
    ArrayType* arr_raw1 = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType* arr_raw2 = PG_GETARG_ARRAYTYPE_P(1);
    float8* arr1 = (float8 *) ARR_DATA_PTR(arr_raw1);
    float8* arr2 = (float8 *) ARR_DATA_PTR(arr_raw2);
    int size = ARRNELEMS(arr_raw1);                          // 用ARRNELEMS从源数据中获取数组的元素个数
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(arr1[i]-arr2[i]);
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}





/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表，按列求和（输入参数==0）或按行求和（输入参数==1）结果保存在输出表中。
// 参数：输入表名 输入参数，要求只能为0或1 输出表名
// 返回：0正常 -1输入表不存在 -6输入参数不为0或1
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z15outer_db4ai_sumP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z15inner_db4ai_sumP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_sum(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);//不确定这里的参数应该填几，感觉是安装上面的来
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    
    // 启动连接
    SPI_connect();  
    ///////////////////////////////////////////////////////////////////////// 
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists = get_int32_from_qresult()==0?0:1;
    if(!if_input_table_exists){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    //检查参数是否合法，不合法报错
    if(dim != (int32)0 && dim != (int32)1){
        // 在分支的位置一定要及时关闭连接！
        SPI_finish();  // 必须：关闭连接
        PG_RETURN_INT32(-6); // 返回结果状态码
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
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_sum(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组X2）
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(0);
    int32 rows = PG_GETARG_INT32(1);
    int32 cols = PG_GETARG_INT32(2);
    int32 ndim = PG_GETARG_INT32(3);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
   //函数主逻辑部分
     if(ndim == (int32)0){//按列相加
         // 用palloc动态分配内存
        Datum* ans_arr_back = (Datum*) palloc(cols * sizeof(Datum)); //答案数组
        float8* temp_acc_arr = (float8*) malloc(cols * sizeof(float8));//临时数组，用于累加
        for(int i=0;i<cols;i++){
            temp_acc_arr[i] = 0.0;   //为临时累加数组赋初值
        }
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                temp_acc_arr[j]+=arr[i*cols+j]; //开始累加
            }      
        }
        for(int i=0;i<cols;i++){
            ans_arr_back[i] = Float8GetDatum(temp_acc_arr[i]); //将答案输出至答案数组
        }
        free(temp_acc_arr);
        // 返回该数组
        ArrayType* result = construct_array(ans_arr_back, cols, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
        PG_RETURN_ARRAYTYPE_P(result);
    }
    else{//按行相加
           
        Datum* ans_arr_back = (Datum*) palloc(rows * sizeof(Datum)); //答案数组
        float8* temp_acc_arr = (float8*) malloc(rows * sizeof(float8));//临时数组，用于累加
        for(int i=0;i<rows;i++){
            temp_acc_arr[i] = 0.0;//为临时累加数组赋初值
            for(int j=0;j<cols;j++){//开始累加
                temp_acc_arr[i] += arr[i*cols+j];
            }
            ans_arr_back[i] = Float8GetDatum(temp_acc_arr[i]);//将答案输出至答案数组
        }
        free(temp_acc_arr);
        // 返回该数组
        ArrayType* result = construct_array(ans_arr_back, rows, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
        PG_RETURN_ARRAYTYPE_P(result);
    }
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将两个输入表进行tensordot张量乘法运算（限于2维），
// 参数：输入表名1 输入表名2 维度 输出表名
// 返回：0正常 -1输入表不存在 -7维度不为1或2 -8tensordot所规定的矩阵行列不匹配
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z21outer_db4ai_tensordotP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z21inner_db4ai_tensordotP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_tensordot(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    int32 dim = PG_GETARG_INT32(2); 
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(3));
    // 启动连接
    SPI_connect();  //必须：建立连接
    ///////////////////////////////////////////////////////////////////////// 
    // 如果表不存在，报错并打印（用于直接粘贴和稍加修改的代码段）
    char sql_table_exists[MAX_SQL_LEN];
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name1);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists1 = get_int32_from_qresult();
    sprintf(sql_table_exists, "select count(*) from pg_class where relname = '%s';", input_table_name2);
    SPI_exec(sql_table_exists, 0);
    int32 if_input_table_exists2 = get_int32_from_qresult();
    if(!(if_input_table_exists1&&if_input_table_exists2)){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    //tensordot独有的检验步骤
        // 获取表1的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name1);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();

    // 获取表2的行数
    char sql_table2_rows[MAX_SQL_LEN];
    sprintf(sql_table2_rows, "select rows from %s;", input_table_name2);
    SPI_exec(sql_table2_rows, 0);
    int32 table2_rows = get_int32_from_qresult();
    // 获取表1的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name1);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();

    // 获取表2的列数
    char sql_table2_cols[MAX_SQL_LEN];
    sprintf(sql_table2_cols, "select cols from %s;", input_table_name2);
    SPI_exec(sql_table2_cols, 0);
    int32 table2_cols = get_int32_from_qresult();
    int32 outRows = 0;
    int32 outCols = 0;
    if(dim==1){
        if(table1_cols == table2_rows || table1_cols==1 || table2_rows == 1){//行数=列数：矩阵相乘；前一矩阵列数为1：扩展后相乘；后一矩阵行数为1：扩展后相乘。其余：不合法
            outCols = table2_cols;
            outRows = table1_rows;
        }
        else{
             SPI_finish();   // 在分支的位置一定要及时关闭连接！
             PG_RETURN_INT32(-8); // 返回结果状态码
        }
    }
    else if(dim == 2){//提示：注意这里的outcols并不是指输出矩阵的行列数，而是用来表示矩阵长和宽以确定循环次数的
        if(table1_cols==table2_cols&&table1_rows==table2_rows){//两矩阵大小完全一致：内积
            outCols = table2_cols;
            outRows = table1_rows;
        }
        else if((table1_cols==table2_cols&&(table1_rows==1||table2_rows==1))||(table1_rows==table2_rows&&(table1_cols==1||table2_cols==1))){
            outCols = table2_cols>table1_cols?table2_cols:table1_cols;//列（行）数为1，而行（列）数相等，扩展后内积
            outRows = table2_rows>table1_rows?table2_rows:table1_rows;
        }else if((table1_rows ==1 &&table1_cols==1)||(table2_rows==1&&table2_cols==1)){
            outCols = table2_cols>table1_cols?table2_cols:table1_cols;
            outRows = table2_rows>table1_rows?table2_rows:table1_rows;
        }
        else{//其余不合法
             SPI_finish();   // 在分支的位置一定要及时关闭连接！
             PG_RETURN_INT32(-8); // 返回结果状态码
        }
    }
    else{
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-7); // 返回结果状态码
    }

    /////////////////////////////////////////////////////////////////////////
    
    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
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
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_tensordot(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    //获取参数（顾名思义）
    int32 col1 = PG_GETARG_INT32(0);
    int32 row2 = PG_GETARG_INT32(1);
    int32 col2 = PG_GETARG_INT32(2);
    int32 dim = PG_GETARG_INT32(3);
    int32 outrows = PG_GETARG_INT32(4);
    int32 outcols = PG_GETARG_INT32(5);
    // 获取参数（一维double8数组X2）
    ArrayType* arr_raw1 = PG_GETARG_ARRAYTYPE_P(6);
    ArrayType* arr_raw2 = PG_GETARG_ARRAYTYPE_P(7);
    float8* arr1 = (float8 *) ARR_DATA_PTR(arr_raw1);
    float8* arr2 = (float8 *) ARR_DATA_PTR(arr_raw2);
    int size1 = ARRNELEMS(arr_raw1);     // 矩阵1大小
    int size2 = ARRNELEMS(arr_raw2);
    int size;//输出矩阵大小
    // 构建一个Datum数组
    // 主要逻辑部分
    if(dim == 1){
        size = outrows*outcols;
        Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
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
        // 返回该数组
        ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
        PG_RETURN_ARRAYTYPE_P(result);
    }
    else{
        size = 1;
        Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
        float8 temp = 0;
        for(int i=0;i<outrows;i++){
            for(int j=0;j<outcols;j++){
                temp+=arr1[(i*col1+j%col1)%size1]*arr2[(i*col2+j%col2)%size2];//目的是取消小维矩阵带来的分支，上同
            }
        }
        for(int i=0; i<size; i++){
            ans_arr_back[i] = Float8GetDatum(temp);
        }
        // 返回该数组
        ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
        PG_RETURN_ARRAYTYPE_P(result);
    }
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入的表名对应的方阵求trace后返回，和其它张量操作不同。
// 参数：输入表名，输出表名
// 返回：0正常,如果不是方阵则返回-3。
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z17outer_db4ai_traceP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z17inner_db4ai_traceP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_trace(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // 启动连接
    SPI_connect();  //必须：建立连接

    /////////////////////////////////////////////////////////////////////////
    // 检测数字参数的情况——作为整体数目（用于直接粘贴和稍加修改的代码段）
    // 获取输入表的行数
    char sql_table1_rows[MAX_SQL_LEN];
    sprintf(sql_table1_rows, "select rows from %s;", input_table_name);
    SPI_exec(sql_table1_rows, 0);
    int32 table1_rows = get_int32_from_qresult();
    // 获取输入表的列数
    char sql_table1_cols[MAX_SQL_LEN];
    sprintf(sql_table1_cols, "select cols from %s;", input_table_name);
    SPI_exec(sql_table1_cols, 0);
    int32 table1_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 检测是否为方阵（用于直接粘贴和稍加修改的代码段）
    if(table1_rows != table1_cols){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-11); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_pow(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT 1 as rows, 1 as cols, trans, inner_db4ai_trace(%d,data) as data into %s from %s;",
        table1_rows,output_table_name, input_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_trace(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
    int32 dim = PG_GETARG_INT32(0);
    ArrayType* arr_raw = PG_GETARG_ARRAYTYPE_P(1);
    float8* arr = (float8 *) ARR_DATA_PTR(arr_raw);
    int size = 1;
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    float8 trace = 0;
    for(int i=0; i<dim; i++){
        trace = trace + arr[i*dim+i];
    }
    ans_arr_back[0] += Float8GetDatum(trace);
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将按照输入的数字，建立全0的输出表。
// 参数：行数 列数 输出表名
// 返回：0正常
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z17outer_db4ai_zerosP20FunctionCallInfoData); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(_Z17inner_db4ai_zerosP20FunctionCallInfoData);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_zeros(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接
    
    /////////////////////////////////////////////////////////////////////////
    // 检测数字参数的情况（用于直接粘贴和稍加修改的代码段）
    if(rows<=0 || cols<=0){
        SPI_finish();   // 在分支的位置一定要及时关闭连接！
        PG_RETURN_INT32(-4); // 返回结果状态码
    }
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // 清空输出表，不报错不打印（用于直接粘贴和稍加修改的代码段）
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", output_table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    /////////////////////////////////////////////////////////////////////////

    // 调用INNER函数: 
    // SELECT <rows> as rows, <cols> as cols, 0 as trans, inner_db4ai_zeros(<rows>, <cols>) as data into <output_table_name>;
    char sql[MAX_SQL_LEN];
    sprintf(sql,"SELECT %d as rows, %d as cols, 0 as trans, inner_db4ai_zeros(%d) as data into %s;",
        rows, cols, rows * cols, output_table_name);
    SPI_exec(sql, 0);

    /////////////////////////////////////////////////////////////////////////
    // 关闭连接返回正常结果（用于直接粘贴和稍加修改的代码段）
    SPI_finish();  // 必须：关闭连接
    PG_RETURN_INT32(0); // 返回结果状态码
    /////////////////////////////////////////////////////////////////////////
}
// 内部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
inner_db4ai_zeros(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数（一维double8数组）
    int32 size = PG_GETARG_INT32(0);
    // 构建一个Datum数组
    Datum* ans_arr_back = (Datum*) palloc(size * sizeof(Datum));    // 用palloc动态分配内存
    // 主要逻辑部分
    for(int i=0; i<size; i++){
        ans_arr_back[i] = Float8GetDatum(0.0);
    }
    // 返回该数组
    ArrayType* result = construct_array(ans_arr_back, size, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd');
    PG_RETURN_ARRAYTYPE_P(result);
}