#include "postgres.h" // 包含在每个声明postgres函数的C文件中

#include "fmgr.h" // 用于PG_GETARG_XXX 以及 PG_RETURN_XXX
#include "access/hash.h"
// #include "access/htup_details.h"
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
    ESPI_init_message(); // 必须：初始化MSG表格
    // 检测输入表是否存在，不存在则报错(注意无需双引号)
    ESPI_show_message(CHECKING IF INPUT_TABLE EXISTS...);
    
    // if(!ESPI_table_exists(input_table_name)){
    //     // 字符串三部曲：声明，拷贝，连接。
    //     char errmsg[MAX_SQL_LEN]; // 先声明数组，注意不是char*而是char[]
    //     strcpy(errmsg,"TABLE NOT EXIST: "); // 再拷贝字符串
    //     strcat(errmsg, input_table_name); // 再连接字符串
    //     ESPI_show_message(errmsg); // 最后显示字符串
    //     // 在分支的位置一定要及时关闭连接！
    //     SPI_finish();  // 必须：关闭连接
    //     PG_RETURN_INT32(-1); // 返回结果状态码
    // }

    // 清空输出表
    // ESPI_drop_table_if_exists(output_table_name);
    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_abs(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    strcpy(sql, "SELECT rows, cols, trans, inner_db4ai_abs(data) as data into ");
    strcat(sql, output_table_name);
    strcat(sql, " from ");
    strcat(sql, input_table_name);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
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
PG_FUNCTION_INFO_V1(outer_db4ai_acc); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(inner_db4ai_acc);
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
PG_FUNCTION_INFO_V1(outer_db4ai_add); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(inner_db4ai_add);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_add(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接
    ESPI_init_message(); // 必须：初始化MSG表格
    // 检测输入表是否存在，不存在则报错
    ESPI_show_message("CHECKING IF INPUT_TABLE EXISTS...");
    if(!ESPI_table_exists(input_table_name1) || !ESPI_table_exists(input_table_name2)){
        // 字符串三部曲：声明，拷贝，连接。
        char errmsg[MAX_SQL_LEN]; // 先声明数组，注意不是char*而是char[]
        strcpy(errmsg,"TABLE NOT EXIST! "); // 再拷贝字符串
        ESPI_show_message(errmsg); // 最后显示字符串
        // 在分支的位置一定要及时关闭连接！
        SPI_finish();  // 必须：关闭连接
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    // 清空输出表
    ESPI_drop_table_if_exists(output_table_name);
    // 调用INNER函数: SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_add(t1.data, t2.data) as data into output_table_name from input_table_name1 as t1, input_table_name2 as t2;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_add(t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        , output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
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

////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表排序后顺序信息保存在输出表中。
// 参数：输入表名 排序维度 输出表名
// 返回：0正常 -1输入表不存在 -6维数不为0或1
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(outer_db4ai_argsort); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(inner_db4ai_argsort);
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
// 功能：将输入表相除的结果存放到输出表中。
// 参数：输入表名1 / 输入表名2 =  输出表名
// 返回：0正常 -1输入表不存在
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(outer_db4ai_div); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(inner_db4ai_div);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_div(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接
    ESPI_init_message(); // 必须：初始化MSG表格
    // 检测输入表是否存在，不存在则报错
    ESPI_show_message("CHECKING IF INPUT_TABLE EXISTS...");
    if(!ESPI_table_exists(input_table_name1) || !ESPI_table_exists(input_table_name2)){
        // 字符串三部曲：声明，拷贝，连接。
        char errmsg[MAX_SQL_LEN]; // 先声明数组，注意不是char*而是char[]
        strcpy(errmsg,"TABLE NOT EXIST! "); // 再拷贝字符串
        ESPI_show_message(errmsg); // 最后显示字符串
        // 在分支的位置一定要及时关闭连接！
        SPI_finish();  // 必须：关闭连接
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    // 清空输出表
    ESPI_drop_table_if_exists(output_table_name);
    // 调用INNER函数: SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_div(t1.data, t2.data) as data into output_table_name from input_table_name1 as t1, input_table_name2 as t2;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_div(t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        , output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
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
PG_FUNCTION_INFO_V1(outer_db4ai_dot); // 注册函数为V1版本
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
// 功能：将输入表按元素取以e为底的指数的结果存放到输出表中。
// 参数：输入表名 输出表名
// 返回：0正常 -1输入表不存在
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(outer_db4ai_exp); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(inner_db4ai_exp);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_exp(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // 启动连接
    SPI_connect();  //必须：建立连接
    ESPI_init_message(); // 必须：初始化MSG表格
    // 检测输入表是否存在，不存在则报错
    ESPI_show_message("CHECKING IF INPUT_TABLE EXISTS...");
    if(!ESPI_table_exists(input_table_name)){
        // 字符串三部曲：声明，拷贝，连接。
        char errmsg[MAX_SQL_LEN]; // 先声明数组，注意不是char*而是char[]
        strcpy(errmsg,"TABLE NOT EXIST: "); // 再拷贝字符串
        strcat(errmsg, input_table_name); // 再连接字符串
        ESPI_show_message(errmsg); // 最后显示字符串
        // 在分支的位置一定要及时关闭连接！
        SPI_finish();  // 必须：关闭连接
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    // 清空输出表
    ESPI_drop_table_if_exists(output_table_name);
    // 调用INNER函数: SELECT rows, cols, trans, inner_db4ai_exp(data) as data into output_table_name from input_table_name;
    char sql[MAX_SQL_LEN];
    strcpy(sql, "SELECT rows, cols, trans, inner_db4ai_exp(data) as data into ");
    strcat(sql, output_table_name);
    strcat(sql, " from ");
    strcat(sql, input_table_name);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
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
// 功能：将两个输入表矩阵乘法运算结果保存在输出表中
// 参数：输入表名1 输入表名2 输出表名
// 返回：0正常 -1输入表不存在 -9矩阵相乘时前一矩阵的列数不等于后一矩阵的行数
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(outer_db4ai_matmul); // 注册函数为V1版本
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

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// 功能：将输入表相X的结果存放到输出表中。
// 参数：输入表名1 输入表名2 输出表名
// 返回：0正常 -1输入表不存在
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(outer_db4ai_mul); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(inner_db4ai_mul);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_mul(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接
    ESPI_init_message(); // 必须：初始化MSG表格
    // 检测输入表是否存在，不存在则报错
    ESPI_show_message("CHECKING IF INPUT_TABLE EXISTS...");
    if(!ESPI_table_exists(input_table_name1) || !ESPI_table_exists(input_table_name2)){
        // 字符串三部曲：声明，拷贝，连接。
        char errmsg[MAX_SQL_LEN]; // 先声明数组，注意不是char*而是char[]
        strcpy(errmsg,"TABLE NOT EXIST! "); // 再拷贝字符串
        ESPI_show_message(errmsg); // 最后显示字符串
        // 在分支的位置一定要及时关闭连接！
        SPI_finish();  // 必须：关闭连接
        PG_RETURN_INT32(-1); // 返回结果状态码
    }
    // 清空输出表
    ESPI_drop_table_if_exists(output_table_name);
    // 调用INNER函数: SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_mul(t1.data, t2.data) as data into output_table_name from input_table_name1 as t1, input_table_name2 as t2;
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT t1.rows, t1.cols, t1.trans, inner_db4ai_mul(t1.data, t2.data) as data into %s from %s as t1, %s as t2;"
        , output_table_name, input_table_name1, input_table_name2);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
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
// 返回：0正常
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(_Z17outer_db4ai_zerosP20FunctionCallInfoData); // 注册函数为V1版本
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
    ESPI_init_message(); // 必须：初始化MSG表格
    // 清空输出表
    ESPI_drop_table_if_exists(output_table_name);
    // 调用INNER函数: 
    // SELECT <rows> as rows, <cols> as cols, 0 as trans, inner_db4ai_ones(<rows> * <cols>) as data into <output_table_name>;
    char sql[MAX_SQL_LEN];
    sprintf(sql,"SELECT %d as rows, %d as cols, 0 as trans, inner_db4ai_ones(%d) as data into %s;",
        rows, cols, rows * cols, output_table_name);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
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
// 功能：将按照输入的数字，建立全0的输出表。
// 参数：行数 列数 输出表名
// 返回：0正常
/////////////////////////////////////////////////////////////////////////
// 函数注册
PG_FUNCTION_INFO_V1(outer_db4ai_zeros); // 注册函数为V1版本
PG_FUNCTION_INFO_V1(inner_db4ai_zeros);
// 外部函数
Datum // 所有Postgres函数的参数和返回类型都是Datum
outer_db4ai_zeros(PG_FUNCTION_ARGS){ // 函数名(参数) 必须加上
    // 获取参数
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // 启动连接
    SPI_connect();  //必须：建立连接
    ESPI_init_message(); // 必须：初始化MSG表格
    // 清空输出表
    ESPI_drop_table_if_exists(output_table_name);
    // 调用INNER函数: 
    // SELECT <rows> as rows, <cols> as cols, 0 as trans, inner_db4ai_zeros(<rows>, <cols>) as data into <output_table_name>;
    char sql[MAX_SQL_LEN];
    sprintf(sql,"SELECT %d as rows, %d as cols, 0 as trans, inner_db4ai_zeros(%d) as data into %s;",
        rows, cols, rows * cols, output_table_name);
    SPI_exec(sql, 0);
    // 关闭连接
    SPI_finish();  // 必须：关闭连接
    // 返回结果
    PG_RETURN_INT32(0); // 返回结果状态码
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




/*
注意这一行！！
if (SPI_processed > 0)
	{
		selected = DatumGetInt32(CStringGetDatum(SPI_getvalue(
													   SPI_tuptable->vals[0],
													   SPI_tuptable->tupdesc,
																		 1
																		)));
	}

*/