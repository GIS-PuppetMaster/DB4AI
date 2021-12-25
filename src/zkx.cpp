#include "postgres.h"

#include "fmgr.h"
#include "access/hash.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/array.h" // ??ArrayType*
#include "executor/spi.h" // SPI
#include "db4ai/matrix.h"
#include "/home/omm/soft/openGauss-server/src/gausskernel/storage/cmgr/cache_mgr.h"

#include <stdlib.h>
#include <cmath>
#include <string.h>
#include <stdio.h>
#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
using namespace std;

#define MAX_SQL_LEN 1024


#define DEBUG
/*
 * Taken from the intarray contrib header
 */
#define ARRPTR(x)  ( (double *) ARR_DATA_PTR(x) )
#define ARRNELEMS(x)  ArrayGetNItems( ARR_NDIM(x), ARR_DIMS(x))

extern map<string, Matrix> matrixMap;
//map<string, Matrix> matrixMap;

PG_MODULE_MAGIC;

// ?????????????sql???????????????????????????????????????????????
// USAGE: int32 val = get_int32_from_qresult();
#define get_int32_from_qresult() atoi(SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1))

// ?????????????sql?????????????????????????????
// USAGE: char* get_string_from_qresult();
#define get_string_from_qresult() SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1)

// ??????????????sql???????????????????????????????????????????
// USAGE: int32 val = get_float8_from_qresult(3,3);
// ???????????????????????
#define get_float8_from_qresult(i, j) atof(SPI_getvalue(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, (j+1)))


inline long int get_time_stamp()
{
    return std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch()).count();
}


inline void my_matrix_init(Matrix *matrix, int rows, int columns)
{
    Assert(matrix != nullptr);
    Assert(rows > 0);
    Assert(columns > 0);
    matrix->transposed = false;
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->allocated = rows * columns;
    matrix->data = new double[matrix->allocated];
}

inline void my_matrix_init_clone(Matrix *matrix, const Matrix *src)
{
    Assert(matrix != nullptr);
    Assert(src != nullptr);
    Assert(!src->transposed);
    my_matrix_init(matrix, src->rows, src->columns);
    size_t bytes = src->rows * src->columns * sizeof(gd_float);
    errno_t rc = memcpy_s(matrix->data, matrix->allocated * sizeof(gd_float), src->data, bytes);
    securec_check(rc, "", "");
}


void logger(char* message){
    char mess[MAX_SQL_LEN];
    sprintf(mess, "insert into MSG values ('%s');", message);
    SPI_connect();
    SPI_exec(mess, 0);
    SPI_finish();
}

// ????????
// 0 -> ????
// -1 -> ????????????
// -2 -> ???????????
// -3 -> ???????????
// -4 -> ?????????????????????????????
// -5 -> ???????????????????????????????????
// -6 -> ???????????????dim???0????-1
// -6 -> ?????????0??1??????reverse????0,1??2
// -7 -> tensordot???????????1??2
// -8 -> tensordot????????????????
// -9 -> ??????????????????????????????????
// -10 -> ??????????????????????????1???????????????????????1??????
// -12 -> slice???????????????
// -13 -> f1???????????????

/* Nov 23 recoding the forward ops */

inline void initTest(){
    Matrix* mtrx1 = (Matrix*)malloc(sizeof(Matrix));
    Matrix* mtrx2 = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(mtrx1,2,2);
    my_matrix_init(mtrx2,2,2);
    double* data1 = (double*)malloc(4*sizeof(double));
    double* data2 = (double*)malloc(4*sizeof(double));
    data1[0] = 1.1;
    data1[1] = 2.2;
    data1[2] = -5.5;
    data1[3] = 3.3;
    data2[0] = 6.9;
    data2[1] = 0.2;
    data2[2] = 1;
    data2[3] = -10;
    mtrx1->data = data1;
    mtrx2->data = data2;
    matrixMap["a1"] = *mtrx1;
    matrixMap["a2"] = *mtrx2;
    matrixMap.insert(pair<string,Matrix>("a1",*mtrx1));
    matrixMap.insert(pair<string,Matrix>("a2",*mtrx2));

    Matrix* emtrx = (Matrix*)malloc(sizeof(Matrix));
    double* dataematrix = (double*)malloc(6*sizeof(double));
    dataematrix[0] =1.2;
    dataematrix[1] =3.6;
    dataematrix[2] =5.5;
    dataematrix[3] =9.9;
    dataematrix[4] =0.75;
    dataematrix[5] =0.8;
    my_matrix_init(emtrx,2,3);
    emtrx->data = dataematrix;
    matrixMap["mtrx"] = *emtrx;
    matrixMap.insert(pair<string,Matrix>("mtrx",*emtrx));

    Matrix* emtrx2 = (Matrix*)malloc(sizeof(Matrix));
    double* dataematrix2 = (double*)malloc(6*sizeof(double));
    dataematrix2[0] =1.1;
    dataematrix2[1] =2.2;
    dataematrix2[2] =3.3;
    dataematrix2[3] =5.5;
    dataematrix2[4] =7.8;
    dataematrix2[5] =9.9;
    my_matrix_init(emtrx2,2,3);
    emtrx2->data = dataematrix2;
    matrixMap["mtrx2"] = *emtrx2;
    matrixMap.insert(pair<string,Matrix>("mtrx2",*emtrx2));

    Matrix* emtrx3 = (Matrix*)malloc(sizeof(Matrix));
    double* dataematrix3 =(double*)malloc(6*sizeof(double));
    dataematrix3[0] =1;
    dataematrix3[1] =2;
    dataematrix3[2] =3;
    dataematrix3[3] =4;
    dataematrix3[4] =5;
    dataematrix3[5] =6;
    my_matrix_init(emtrx3,3,2);
    emtrx3->data = dataematrix3;
    matrixMap["mtrx3"] = *emtrx3;
    matrixMap.insert(pair<string,Matrix>("mtrx3",*emtrx3));

    Matrix* ft1 = (Matrix*)malloc(sizeof(Matrix));
    double* dft1 = (double*)malloc(6*sizeof(double));
    dft1[0] =0;
    dft1[1] =1;
    dft1[2] =2;
    dft1[3] =0;
    dft1[4] =1;
    dft1[5] =2;
    my_matrix_init(ft1,1,6);
    ft1->data = dft1;
    matrixMap["ft1"] = *ft1;
    matrixMap.insert(pair<string,Matrix>("ft1",*ft1));

    Matrix* ft2 = (Matrix*)malloc(sizeof(Matrix));
    double* dft2 = (double*)malloc(6*sizeof(double));
    dft2[0] =0;
    dft2[1] =2;
    dft2[2] =1;
    dft2[3] =0;
    dft2[4] =0;
    dft2[5] =1;
    my_matrix_init(ft2,1,6);
    ft2->data = dft2;
    matrixMap["ft2"] = *ft2;
    matrixMap.insert(pair<string,Matrix>("ft2",*ft2));
}
/**
 * @brief to print mtrx into database
 *
 * @param mtrx is ptr of the matrix that you wang to see
 */
inline void printMSG(Matrix* mtrx){
    SPI_connect();
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS smy_output;");
    SPI_exec(sql_drop_table_if_exists, 0);
    const char* table_name = "smy_output";
    char sql[MAX_SQL_LEN];
    sprintf(sql, " CREATE TABLE %s(rows INT ,cols INT, mapcont INT, data float8[]);",
        table_name);
    SPI_exec(sql, 0);
    int size = mtrx->rows*mtrx->columns;
    char datainfo[MAX_SQL_LEN] = "{";
    for (int i = 0; i < size; i++)
    {
        char temp[40];
        sprintf(temp,"%f,",mtrx->data[i]);
        strcat(datainfo,temp);
    }
    int dataLen = strlen(datainfo);
    datainfo[dataLen-1] = '}';

    sprintf(sql, "  INSERT INTO %s values(%d, %d, %d, '%s');",
        table_name,mtrx->rows,mtrx->columns,matrixMap.size(),datainfo);
    SPI_exec(sql, 0);
    SPI_finish();
}

string IntToString(int i)
{
  string s;
  stringstream ss(s);
  ss<<i;
  return ss.str();
}


void printCount(){
    SPI_connect();
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    const char* table_name = "smy_count";
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;",table_name);
    SPI_exec(sql_drop_table_if_exists, 0);

    char sql[MAX_SQL_LEN];
    sprintf(sql, " CREATE TABLE %s(rows INT ,cols INT, mapcont INT, data float8[]);",
        table_name);
    SPI_exec(sql, 0);
    sprintf(sql, "  INSERT INTO %s values(%d, %d, %d, '{-1}');",
        table_name,1,1,matrixMap.size());
    SPI_exec(sql, 0);
    // sprintf(sql, "INSERT INTO MSG values('test_temp %p %d');", handle_test_temp, *handle_test_temp);
    // SPI_exec(sql, 0);
    // std::map<string, Matrix>::iterator iter;
    // for (iter=matrixMap.begin(); iter!=matrixMap.end(); iter++)
    // {
    //     char message[MAX_SQL_LEN];
    //     string strs;
    //     Matrix *t = &(iter->second);
    //     for(int i=0;i<t->columns*t->rows;i++)
    //     {
    //         int temp=t->data[i];
    //         strs+=IntToString(temp);
    //     }
    //     sprintf(message, "insert into MSG values ('matrix_name:%s, matrix_data:%s');",iter->first, strs);
    //     SPI_exec(message, 0);
    // }
    SPI_finish();
}

PG_FUNCTION_INFO_V1(_Z16outer_printCountP20FunctionCallInfoData);
Datum
outer_printCount(PG_FUNCTION_ARGS){
    SPI_connect();
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    const char* table_name = "map_count";
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;",table_name);
    SPI_exec(sql_drop_table_if_exists, 0);

    char sql[MAX_SQL_LEN];
    sprintf(sql, " CREATE TABLE %s(rows INT ,cols INT, mapcont INT, data float8[]);",
        table_name);
    SPI_exec(sql, 0);
    sprintf(sql, "  INSERT INTO %s values(%d, %d, %d, '{-1}');",
        table_name,1,1,matrixMap.size());
    SPI_exec(sql, 0);
    SPI_finish();
    PG_RETURN_INT32(matrixMap.size());
}


//??????map??size
PG_FUNCTION_INFO_V1(_Z20qp4ai_matrixMap_sizeP20FunctionCallInfoData);
Datum
qp4ai_matrixMap_size(PG_FUNCTION_ARGS){
    map<string, Matrix> ::iterator iter;
    int size = 0;
    for (iter = matrixMap.begin();iter!=matrixMap.end();iter++){
        Matrix temp= iter->second;
        size+=sizeof(iter->first) + sizeof(temp);
        size+=temp.columns*temp.rows*sizeof(double);
    }
    size += sizeof(matrixMap);
    PG_RETURN_INT32(size);
}

PG_FUNCTION_INFO_V1(_Z9qp4ai_subP20FunctionCallInfoData); // register function as V1
// ??????????????????????????1????????????2???????????????????????????????????
// ??????????????1 ????????2 ????????
// ?????0???? -1????????????
Datum
qp4ai_sub(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    long int start_time = get_time_stamp();
    string input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    if(matrixMap.count(input_table_name1)==0||matrixMap.count(input_table_name2)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name1];
    Matrix *mtrx2 = &matrixMap[input_table_name2];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    //check if lines&cols equals
    if(mtrx1->rows != mtrx2->rows)
        PG_RETURN_INT32(-2);
    if(mtrx1->columns != mtrx2->columns)
        PG_RETURN_INT32(-3);
    my_matrix_init(res, mtrx1->rows, mtrx1->columns);
   // main logic
    int size = mtrx1->rows*mtrx1->columns;
    for(int i=0;i<size;i++){
        res->data[i] = mtrx1->data[i]-mtrx2->data[i];
    }
    matrixMap[output_table_name] = *res;
    long int time_cost = get_time_stamp()-start_time;
    Matrix *tc;
    if (matrixMap.count("_time_cost_")>0)
    {
        tc = &matrixMap["_time_cost_"];
        tc->data[0] += ((double) time_cost)/1000;
    }
    else
    {
        tc = (Matrix*)malloc(sizeof(Matrix));
        my_matrix_init(tc, 1, 1);
        tc->data[0] = ((double) time_cost)/1000;
    }
    matrixMap["_time_cost_"] = *tc;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z9qp4ai_sumP20FunctionCallInfoData); // register function as V1
// ??????????????????????????????==0?????????????????==1???????????????????
// ?????????????? ?????????????????0??1 ????????
// ?????0???? -1???????????? -6???????????0??1
Datum
qp4ai_sum(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    //get para
    printCount();
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    //check
    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    if(dim != 0 && dim != 1)
        PG_RETURN_INT32(-6);
    //fetch
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    int cols = mtrx1->columns;
    int rows = mtrx1->rows;
    double* arr = mtrx1->data;
    // main logic
    if(dim == 0){//sum by cols
        my_matrix_init(res, 1, cols);
        double* temp_acc_arr = (double*) malloc(cols * sizeof(double));//???????????
        for(int i=0;i<cols;i++){
            temp_acc_arr[i] = 0.0;   //??????????????
        }
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                temp_acc_arr[j]+=arr[i*cols+j]; //??????
            }
        }
        res->data = temp_acc_arr;
    }
    else{//????????
        my_matrix_init(res, rows, 1);
        double* temp_acc_arr = (double*) malloc(rows * sizeof(double));//???????????
        for(int i=0;i<rows;i++){
            temp_acc_arr[i] = 0.0;//??????????????
            for(int j=0;j<cols;j++){//??????
                temp_acc_arr[i] += arr[i*cols+j];
            }
        }
        res->data = temp_acc_arr;
    }
    // ret
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // #endif
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(_Z10qp4ai_sqrtP20FunctionCallInfoData); // register function as V1
// ???????????????????????????????????????
// ?????????????? ????????
// ?????0???? -1????????????
Datum
qp4ai_sqrt(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    //get para
    printCount();
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    //check
    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    //fetch
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    int cols = mtrx1->columns;
    int rows = mtrx1->rows;
    int size = cols*rows;
    double* arr = (double*)malloc(size*sizeof(double));
    for(int i=0;i<size;i++)
        arr[i] = mtrx1->data[i];
    my_matrix_init(res,rows,cols);
    // main logic
    res->transposed = mtrx1->transposed;
    //matrix_square_root(res);
    for(int i=0; i<size; i++){
        arr[i] = sqrt(arr[i]);
    }
    res->data = arr;
    // ret
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // #endif
    PG_RETURN_INT32(0);
}

// ????????????????????????????????
// ?????????????? ??????? ????????
// ?????0???? -1???????????? -6??????0??1
PG_FUNCTION_INFO_V1(_Z10qp4ai_sortP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_sort(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    //check
    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    if(dim != 0 && dim != 1)
        PG_RETURN_INT32(-6);
    //fetch
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    int cols = mtrx1->columns;
    int rows = mtrx1->rows;
    int size = rows*cols;
    double* arr = (double*)malloc(size*sizeof(double));
    for(int i = 0;i<size;i++)
        arr[i] = mtrx1->data[i];
    my_matrix_init(res,rows,cols);
    //matrix_copy(res,mtrx1);
    if(dim == 0){//????????
        for(int k = 0;k<cols;k++){
            for(int i =0;i<rows-1;i++){
                int pos = i;
                for(int j=i;j<rows;j++){
                    if(arr[j*cols+k]<arr[pos*cols+k]){
                        pos = j;
                    }
                }
                if(pos != i){
                    double temp = arr[pos*cols+k];
                    arr[pos*cols+k] = arr[i*cols+k];
                    arr[i*cols+k] = temp;
                }
            }
        }
    }
    else{//????????
        for(int k = 0;k<rows;k++){
            for(int i =0;i<cols-1;i++){
                int pos = i;
                for(int j=i;j<cols;j++){
                    if(arr[k*cols+j]<arr[k*cols+pos])
                        pos = j;
                }
                if(pos != i){
                    double temp = arr[k*cols+pos];
                    arr[k*cols+pos] = arr[k*cols+i];
                    arr[k*cols+i] = temp;
                }
            }
        }
    }
    // ret
    res->data =arr;
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // #endif
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(_Z12qp4ai_argminP20FunctionCallInfoData); // register function as V1
// ????????????????????????????????????????????????
// ?????????? ???? ?????
// ?????0????,???dim????0??1????-6
Datum
qp4ai_argmin(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));

    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    //check if dim is 0/1
    if(dim != 0 && dim != 1)
        PG_RETURN_INT32(-6);
    // main logic
    int size;
    if(dim == 0){
        my_matrix_init(res, 1, mtrx1->columns);
        size = mtrx1->columns;
        for(int i=0; i<mtrx1->columns;i++){
            float8 min,argmin;
            for(int j=0; j<mtrx1->rows-1; j++){
                if(j == 0){min = mtrx1->data[j*mtrx1->columns+i]; argmin = (float8)j;}
                if(min > mtrx1->data[(j+1)*mtrx1->columns+i]){min = mtrx1->data[(j+1)*mtrx1->columns+i]; argmin = (float8) (j+1);}
            }
            res->data[i] = argmin;
        }
    }
    else if(dim == 1){
        my_matrix_init(res, 1, mtrx1->rows);
        size = mtrx1->rows;
        for(int i=0; i<mtrx1->rows;i++){
            float8 min,argmin;
            for(int j=0; j<mtrx1->columns-1; j++){
                if(j == 0){min = mtrx1->data[i*mtrx1->columns+j]; argmin = (float8)j;}
                if(min > mtrx1->data[i*mtrx1->columns+j+1]){min = mtrx1->data[i*mtrx1->columns+j+1]; argmin = (float8)(j+1);}
            }
            res->data[i] = argmin;
        }
    }
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z12qp4ai_argmaxP20FunctionCallInfoData); // register function as V1
// ???????????????????????????????????????????????
// ?????????? ???? ?????
// ?????0????,???dim????0??1????-6
Datum
qp4ai_argmax(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));

    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    //check if dim is 0/1
    if(dim != 0 && dim != 1)
        PG_RETURN_INT32(-6);
    // main logic
    int size;
    int rows = mtrx1->rows;
    int cols = mtrx1->columns;
    if(dim == 0){
        my_matrix_init(res, 1, mtrx1->columns);
        size = mtrx1->columns;
        for(int i=0; i<cols;i++){
            float8 max,argmax;
            for(int j=0; j<rows-1; j++){
                if(j == 0){max = mtrx1->data[j*cols+i]; argmax = (float8)j;}
                if(max < mtrx1->data[(j+1)*cols+i]){max = mtrx1->data[(j+1)*cols+i]; argmax = (float8) (j+1);}
            }
            res->data[i] = argmax;
        }
    }
    else if(dim == 1){
        my_matrix_init(res, 1, mtrx1->rows);
        size = mtrx1->rows;
        for(int i=0; i<rows;i++){
            float8 max,argmax;
            for(int j=0; j<cols-1; j++){
                if(j == 0){max = mtrx1->data[i*cols+j]; argmax = (float8)j;}
                if(max < mtrx1->data[i*cols+j+1]){max = mtrx1->data[i*cols+j+1]; argmax = (float8)(j+1);}
            }
            res->data[i] = argmax;
        }
    }
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z10qp4ai_fullP20FunctionCallInfoData); // register function as V1
// ???????????????????????????????output_table_name???????full_value???????
// ?????????? ????? ????
// ?????0????
Datum
qp4ai_full(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    float8 full_value = PG_GETARG_FLOAT8(1);
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));

    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, mtrx1->rows, mtrx1->columns);
    // main logic
    int size = mtrx1->rows*mtrx1->columns;
    for(int i=0;i<size;i++){
        res->data[i] = full_value;
    }
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z9qp4ai_logP20FunctionCallInfoData); // register function as V1
// ????????????????????????????????????????????
// ?????????????? ????????
// ?????0???? -1????????????
Datum
qp4ai_log(PG_FUNCTION_ARGS){
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));

    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, mtrx1->rows, mtrx1->columns);
    // main logic
    int size = mtrx1->rows*mtrx1->columns;
    for(int i=0;i<size;i++){
        res->data[i] = log(mtrx1->data[i]);
    }
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z9qp4ai_maxP20FunctionCallInfoData); // register function as V1
// ????????????????????????????????????????????
// ?????????? ???? ?????
// ?????0????,???dim????0??1????-6
Datum
qp4ai_max(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));

    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    //check if dim is 0/1
    if(dim != 0 && dim != 1)
        PG_RETURN_INT32(-6);
    // main logic
    int size;
    int rows = mtrx1->rows;
    int cols = mtrx1->columns;
    if(dim == 0){
        my_matrix_init(res, 1, mtrx1->columns);
        size = mtrx1->columns;
        for(int i=0; i<cols;i++){
            float8 max,argmax;
            for(int j=0; j<rows-1; j++){
                if(j == 0){max = mtrx1->data[j*cols+i]; argmax = (float8)j;}
                if(max < mtrx1->data[(j+1)*cols+i]){max = mtrx1->data[(j+1)*cols+i]; argmax = (float8) (j+1);}
            }
            res->data[i] = max;
        }
    }
    else if(dim == 1){
        my_matrix_init(res, 1, mtrx1->rows);
        size = mtrx1->rows;
        for(int i=0; i<rows;i++){
            float8 max,argmax;
            for(int j=0; j<cols-1; j++){
                if(j == 0){max = mtrx1->data[i*cols+j]; argmax = (float8)j;}
                if(max < mtrx1->data[i*cols+j+1]){max = mtrx1->data[i*cols+j+1]; argmax = (float8)(j+1);}
            }
            res->data[i] = max;
        }
    }
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z9qp4ai_minP20FunctionCallInfoData); // register function as V1
// ???????????????????????????????????????????
// ?????????? ???? ?????
// ?????0????,???dim????0??1????-6
Datum
qp4ai_min(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));

    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    //check if dim is 0/1
    if(dim != 0 && dim != 1)
        PG_RETURN_INT32(-6);
    // main logic
    int size;
    int rows = mtrx1->rows;
    int cols = mtrx1->columns;
    if(dim == 0){
        my_matrix_init(res, 1, mtrx1->columns);
        size = mtrx1->columns;
        for(int i=0; i<cols;i++){
            float8 min,argmin;
            for(int j=0; j<rows-1; j++){
                if(j == 0){min = mtrx1->data[j*cols+i]; argmin = (float8)j;}
                if(min > mtrx1->data[(j+1)*cols+i]){min = mtrx1->data[(j+1)*cols+i]; argmin = (float8) (j+1);}
            }
            res->data[i] = min;
        }
    }
    else if(dim == 1){
        my_matrix_init(res, 1, mtrx1->rows);
        size = mtrx1->rows;
        for(int i=0; i<rows;i++){
            float8 min,argmin;
            for(int j=0; j<cols-1; j++){
                if(j == 0){min = mtrx1->data[i*cols+j]; argmin = (float8)j;}
                if(min > mtrx1->data[i*cols+j+1]){min = mtrx1->data[i*cols+j+1]; argmin = (float8)(j+1);}
            }
            res->data[i] = min;
        }
    }
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z10qp4ai_meanP20FunctionCallInfoData); // register function as V1
// ??????????????????????????????????????????
// ?????????? ???? ?????
// ?????0????,???dim????0??1????-6
Datum
qp4ai_mean(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));

    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    //check if dim is 0/1/2
    if(dim != 0 && dim != 1 && dim != 2)
        PG_RETURN_INT32(-6);
    // main logic
    int size;
    int rows = mtrx1->rows;
    int cols = mtrx1->columns;
    if(dim == 0){
        my_matrix_init(res, 1, mtrx1->columns);
        size = mtrx1->columns;
        for(int i=0; i<cols;i++){
            double sum=0;
            double mean=0;
            for(int j=0; j<rows; j++){
                sum += mtrx1->data[j*cols+i];
            }
            mean = sum / rows;
            res->data[i] = mean;
        }
    }
    else if(dim == 1){
        my_matrix_init(res, 1, mtrx1->rows);
        size = mtrx1->rows;
        for(int i=0; i<rows;i++){
            double sum=0;
            double mean=0;
            for(int j=0; j<cols; j++){
                sum += mtrx1->data[i*cols+j];
            }
            mean = sum / cols;
            res->data[i] = mean;
        }
    }
    else if(dim == 2){
        my_matrix_init(res, 1, 1);
        size = 1;
        double sum=0;
        double mean=0;
        for(int i =0;i < rows; i++){
            for(int j = 0;j<cols;j++){
                sum += mtrx1->data[i*cols+j];
            }
        }
        mean = sum / (rows*cols);
        res->data[0] = mean;
    }
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z11qp4ai_shapeP20FunctionCallInfoData); // register function as V1
// ?????????????????????????????????????
// ?????????????? ????????
// ?????0???? -1????????????
Datum
qp4ai_shape(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    //check input_table
    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, 1, 2);
    // main logic
    int size = 2;
    res->data[0] = mtrx1->rows;
    res->data[1] = mtrx1->columns;
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z11qp4ai_sliceP20FunctionCallInfoData); // register function as V1
// ??????????????????????????????????
// ?????????????? ????? ????? ????? ????? ????????
// ?????0???? -12??????????
Datum
qp4ai_slice(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim1_start = PG_GETARG_INT32(1);
    int32 dim1_end = PG_GETARG_INT32(2);
    int32 dim2_start = PG_GETARG_INT32(3);
    int32 dim2_end = PG_GETARG_INT32(4);
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(5));
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    int rows = mtrx1->rows;
    int cols = mtrx1->columns;
    int rows2 = dim1_end - dim1_start;
    int cols2 = dim2_end - dim2_start;
    //check input_table
    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    //check dim_xxxx
    if(!(dim1_start>=0 && dim1_start<dim1_end && dim1_end<=rows &&
        dim2_start>=0 && dim2_start<dim2_end && dim2_end <=cols)){
        PG_RETURN_INT32(-12); // ???????????
    }
    my_matrix_init(res, rows2, cols2);
    // main logic
    int size = rows2 * cols2;
    for(int i = 0; i < rows2; i++){
        for(int j = 0; j < cols2; j++){
            res->data[i*cols2+j] = mtrx1->data[(dim1_start+i)*cols + dim2_start+j];
        }
    }
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z11qp4ai_traceP20FunctionCallInfoData); // register function as V1
// ????????????????????????????????????????????
// ?????????????? ????????
// ?????0???? -1????????????
Datum
qp4ai_trace(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    //check input_table
    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, mtrx1->rows, mtrx1->columns);
    //check input_table'rows&cols
    if(mtrx1->rows != mtrx1->columns){
        PG_RETURN_INT32(-11); // ???????????
    }
    my_matrix_init(res, 1, 1);
    // main logic
    int size = 1;
    float8 trace = 0;
    for(int i=0; i<mtrx1->rows; i++){
        trace = trace + mtrx1->data[i*mtrx1->rows+i];
    }
    res->data[0] += trace;
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z15qp4ai_pow_tableP20FunctionCallInfoData); // register function as V1
// ???????????????????????????????????????????????????
// ?????????????? ??????????????????????? ????????
// ?????0???? -1????????????
Datum
qp4ai_pow_table(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string pow_buttom = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // float8 pow_buttom = PG_GETARG_FLOAT8(1);
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));

    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *mtrx2 = &matrixMap[pow_buttom];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    //check if lines&cols equals
    my_matrix_init(res, mtrx1->rows, mtrx1->columns);
   // main logic
    int size = mtrx1->rows*mtrx1->columns;
    for(int i=0;i<size;i++){
        res->data[i] = pow(mtrx2->data[0],mtrx1->data[i]);
    }
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z9qp4ai_powP20FunctionCallInfoData); // register
Datum
qp4ai_pow(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    float8 pow_exp = PG_GETARG_FLOAT8(1);
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));

    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    //check if lines&cols equals
    my_matrix_init(res, mtrx1->rows, mtrx1->columns);
    // main logic
    int size = mtrx1->rows*mtrx1->columns;
    for(int i=0;i<size;i++){
        res->data[i] = pow(mtrx1->data[i],pow_exp);
    }
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z12qp4ai_repeatP20FunctionCallInfoData); // register function as V1
// ????????????????????????????????????????????????????????
// ?????????????? ?????? ?????? ???????? ????????????*????????????????????????
// ?????0???? -4??????????
Datum
qp4ai_repeat(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim1 = PG_GETARG_INT32(1);
    int32 dim2 = PG_GETARG_INT32(2);
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(3));

    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    //check dim
    if(dim1<=0 || dim2<=0)
        PG_RETURN_INT32(-4); // ???????????

    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, mtrx1->rows * dim1, mtrx1->columns * dim2);
   // main logic
    int size = mtrx1->rows*mtrx1->columns*dim1*dim2;
    int rows = mtrx1->rows;
    int cols = mtrx1->columns;
    for(int i=0; i<dim1; i++){
        for(int j=0; j<dim2; j++){
            for(int k=0; k<rows; k++){
                for(int x=0; x<cols;x++){
                    res->data[(i*dim2)*rows*cols+k*cols*dim2+j*cols+x] = mtrx1->data[k*cols+x];
                }
            }
        }
    }
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}
PG_FUNCTION_INFO_V1(_Z12qp4ai_randomP20FunctionCallInfoData); // register function as V1
// ????????????dim1??dim2??????output_table_name??????????
// ?????????? ???? ?????? ??????? ????????
// ?????0???? -4??????????s
Datum
qp4ai_random(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    int32 dim1 = PG_GETARG_INT32(0);

    int32 dim2 = PG_GETARG_INT32(1);
    int32 _distribution = PG_GETARG_INT32(2);
    float8 arg1 = PG_GETARG_FLOAT8(3);
    float8 arg2 = PG_GETARG_FLOAT8(4);
    float8 arg3 = PG_GETARG_FLOAT8(5);
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(6));
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res,  dim1, dim2);

    //check if _distribution is 0/1/2
    if(!(_distribution == 0 || _distribution == 1 || _distribution == 2)){
        PG_RETURN_INT32(-13); // ???????????
    }
    if(dim1<=0 || dim2<=0){
        PG_RETURN_INT32(-5); // ???????????
    }
    // main logic
    int size_ = dim1 * dim2;
    if(_distribution == 0){
        float8 mean = arg1;
        float8 stdc = arg2;
        float8 data = 0;
        for(int i = 0 ;i<size_; i++){
            float8 X = 0;
            float8 V1, V2, S;
            int phase = 0;
            if (phase == 0) {
                do {
                    float8 U1 = (float8) rand() / RAND_MAX;
                    float8 U2 = (float8) rand() / RAND_MAX;
                    V1 = 2 * U1 - 1;
                    V2 = 2 * U2 - 1;
                    S = V1 * V1 + V2 * V2;
                } while (S >= 1 || S == 0);

                X = V1 * sqrt(-2 * log(S) / S);
                } else
                    X = V2 * sqrt(-2 * log(S) / S);
            phase = 1 - phase;
            data = mean + X * stdc;
            res->data[i] = data;
        }
    }else if(_distribution == 1){

        srand(time(0));
        for(int i = 0;i<size_;i++){

            float8 m = (float8) rand()*1.0 / RAND_MAX;
            float8 n = arg1 + m * (arg2 - arg1);
            res->data[i] = n;
        }
        //PG_RETURN_INT32(size);
    }else if(_distribution == 2){
        int32 MAX = 1000;
        srand(time(0));
        for(int i = 0;i<size_; i++){
            int n = rand()%1000+1;
            if(n<MAX * arg3){
                res->data[i] = arg2;
            }else{
                res->data[i] = arg1;
            }
        }
    }
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // //printMSG(matrixMap["mtrx2"]);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z13qp4ai_softmaxP20FunctionCallInfoData); // register function as V1
// ????????????????softmax?????????????????????
// ?????????????? ????????
// ?????0???? -1???????????? -6??????0??1
Datum
qp4ai_softmax(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    //check
    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    if(dim != 0 && dim != 1 && dim != -1)
        PG_RETURN_INT32(-6);
    //fetch
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    int cols = mtrx1->columns;
    int rows = mtrx1->rows;
    int size = rows*cols;
    if (dim==-1){
        if (cols>1){
            dim = 1;
        }
        else{
            dim=0;
        }
    }
    my_matrix_init(res,rows,cols);
    // double* arr = (double*)malloc(size*sizeof(double));
    vector<double> arr;
    // double max_value= mtrx1->data[0];
    // for (int i=1;i<rows*cols;i++){
    //     if (mtrx1->data[i]>max_value){
    //         max_value = mtrx1->data[i];
    //     }
    // }
    // main logic
    //????exp????
    for(int i=0;i<size;i++) arr.push_back(exp(mtrx1->data[i]));
    if(dim == 0){//??????
        // double* byDiv =  (double*)malloc(cols * sizeof(double));//???????????????
        vector<double> byDiv;
        for(int i=0;i<cols;i++){
            byDiv.push_back(0.0);
            for(int j=0;j<rows;j++){
                byDiv[i] +=  arr[j*cols+i];
            }
        }
        for(int i=0;i<size;i++) arr[i] /= byDiv[i%cols];
        // free(byDiv);
    }
    else{//??????
        // double* byDiv =  (double*)malloc(rows * sizeof(double));
        vector<double> byDiv;
        for(int i=0;i<rows;i++){
            byDiv.push_back(0.0);
            for(int j=0;j<cols;j++){
                byDiv[i] +=  arr[i*cols+j];
            }
        }
        for(int i=0;i<size;i++) arr[i] /= byDiv[i/cols];
        // free(byDiv);
    }
    for (int i=0;i<rows*cols;i++){
        res->data[i] = arr[i];
    }
    // free(arr);
    // ret
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // #endif
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(_Z15qp4ai_tensordotP20FunctionCallInfoData); // register function as V1
// ????????????????????tensordot???????????????2?????
// ??????????????1 ????????2 ??? ????????
// ?????0???? -1???????????? -7?????1??2 -8tensordot?????????????????
Datum
qp4ai_tensordot(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    int32 dim = PG_GETARG_INT32(2);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(3));
    //check
    if(matrixMap.count(input_table_name1)==0||matrixMap.count(input_table_name2)==0)
        PG_RETURN_INT32(-1);
    //fetch
    Matrix *mtrx1 = &matrixMap[input_table_name1];
    Matrix *mtrx2 = &matrixMap[input_table_name2];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    int col1 = mtrx1->columns;
    int row1 = mtrx1->rows;
    int size1 = col1*row1;
    int col2 = mtrx2->columns;
    int row2 = mtrx2->rows;
    int size2 = col2*row2;
    int outcols,outrows;
    double* arr1 = mtrx1->data;
    double* arr2 = mtrx2->data;
    if(dim==1){
        //????=??????????????????????????1????????????????????????1?????????????????????
        if(col1 ==row2 || col1==1 ||row2 == 1){
            outcols = col2;
            outrows = row1;
        }
        else{
             PG_RETURN_INT32(-8); // ???????????
        }
    }
    else if(dim == 2){//?????????????outcols???????????????????????????????????????????????????????
        if(col1==col2&&row1==row2){//???????????????????
            outcols = col2;
            outrows = row1;
        }
        else if((col1==col2&&(row1==1||row2==1))||(row1==row2&&(col1==1||col2==1))){
            outcols = col2>col1?col2:col1;//?????????1?????????????????????????
            outrows = row2>row1?row2:row1;
        }else if((row1 ==1 &&col1==1)||(row2==1&&col2==1)){
            outcols = col2>col1?col2:col1;
            outrows = row2>row1?row2:row1;
        }
        else{//???????
             PG_RETURN_INT32(-8); // ???????????
        }
    }
    else{
        PG_RETURN_INT32(-7); // ???????????
    }
    my_matrix_init(res,outrows,outcols);
    // main logic
    if(dim == 1){
        int size = outrows*outcols;
        double* ans = (double*)malloc(size*sizeof(double));
        int midrow = col1>row2?col1:row2;
        for(int i=0;i<outrows;i++){
            for(int j=0;j<outcols;j++){
                double temp = 0;
                for(int k=0;k<midrow;k++){
                    temp += arr1[i*col1+k%col1]*arr2[(k*col2+j)%size2];
                }
                ans[i*outcols+j] = temp;
            }
        }
        // ?????????
        res->data = ans;
    }
    else{
        int size = 1;
        double* ans = (double*)malloc(size*sizeof(double));
        double temp = 0;
        for(int i=0;i<outrows;i++){
            for(int j=0;j<outcols;j++){
                temp+=arr1[(i*col1+j%col1)%size1]*arr2[(i*col2+j%col2)%size2];//???????????????????????????
            }
        }
        ans[0] = temp;
        // ?????????
        res->data = ans;
    }
    // ret
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // #endif
    PG_RETURN_INT32(0);
}



PG_FUNCTION_INFO_V1(_Z12qp4ai_matmulP20FunctionCallInfoData); // ??????V1??
// ??????????????????????????????????????????
// ??????????????1 ????????2 ????????
// ?????0???? -1???????????? -9?????????????????????????????????????
Datum
qp4ai_matmul(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // ???????
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ????????
    //check
    if(matrixMap.count(input_table_name1)==0||matrixMap.count(input_table_name2)==0)
        PG_RETURN_INT32(-1);
    //fetch
    Matrix *mtrx1 = &matrixMap[input_table_name1];
    Matrix *mtrx2 = &matrixMap[input_table_name2];
    // ????????????????
    int col1 = mtrx1->columns;
    int col2 = mtrx2->columns;
    int row1 = mtrx1->rows;
    int row2 = mtrx2->rows;
    int outRows = 0;
    int outCols = 0;
    if(col1 == row2){//????=??????????????
        outCols = col2;
        outRows = row1;
    }
    else{
        // #ifdef DEBUG
        // Matrix *res = &matrixMap[input_table_name2];
        // printMSG(res);
        // #endif
        PG_RETURN_INT32(-9); // ???????????
    }
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, outRows, outCols);
    for (int i=0;i<outRows*outCols;i++){
        res->data[i]=0;
    }
    for (int i=0;i<outRows;i++){
        for (int j=0;j<outCols;j++){
            for(int k=0;k<col1;k++){
                res->data[i*outCols+j] += mtrx1->data[i*col1+k]*mtrx2->data[k*col2+j];
            }
        }
    }
    matrixMap[output_table_name] = *res;
    /*
    SPI_connect();  //????????????
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT qp4ai_tensordot('%s','%s',%d,'%s');"
      , input_table_name1, input_table_name2,1, output_table_name);
    SPI_exec(sql, 0);
    // #ifdef DEBUG
    // Matrix *res = matrixMap[output_table_name];
    // printMSG(res);
    // #endif
    SPI_finish();  // ???????????
    */
    PG_RETURN_INT32(0); // ???????????
}

PG_FUNCTION_INFO_V1(_Z9qp4ai_dotP20FunctionCallInfoData); // register function as V1
// ????????????????????????????????????????????
// ??????????????1 ????????2 ????????
// ?????0???? -2????????????? -3 ????????????? -10 ???????1?????
Datum
qp4ai_dot(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    //check
    if(matrixMap.count(input_table_name1)==0||matrixMap.count(input_table_name2)==0)
        PG_RETURN_INT32(-1);
    //fetch
    Matrix *mtrx1 = &matrixMap[input_table_name1];
    Matrix *mtrx2 = &matrixMap[input_table_name2];
    int col1 = mtrx1->columns;
    int row1 = mtrx1->rows;
    int col2 = mtrx2->columns;
    int row2 = mtrx2->rows;
    if(col1 != col2){//????=??????????????
        PG_RETURN_INT32(-3); // ???????????
    }
    else if(row1 != row2){
        PG_RETURN_INT32(-2); // ???????????
    }
    if(row1 !=1&&col1!=1){
        PG_RETURN_INT32(-10); // ???????????
    }
    // main logic
    SPI_connect();
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT qp4ai_tensordot('%s','%s',%d,'%s');"
        ,  input_table_name1, input_table_name2,2,output_table_name);
    SPI_exec(sql, 0);
    SPI_finish();
    Matrix *res = &matrixMap[output_table_name];
    res->columns = 1;
    res->rows = 1;
    // ret
    // #ifdef DEBUG
    // printMSG(res);
    // #endif
    PG_RETURN_INT32(0);
}

// ??????????????????????????????????????
// ?????????????? ??????? ????????
// ?????0???? -1???????????? -6??????0??1
PG_FUNCTION_INFO_V1(_Z13qp4ai_argsortP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_argsort(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    //check
    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    if(dim != 0 && dim != 1)
        PG_RETURN_INT32(-6);
    //fetch
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    int cols = mtrx1->columns;
    int rows = mtrx1->rows;
    int size = cols*rows;
    double* arr = (double*)malloc(size*sizeof(double));
    for(int i=0;i<size;i++)
        arr[i] = mtrx1->data[i];

    // main logic
    int* argset = (int*)malloc(size*sizeof(int));
    if(dim == 0){//????????
        my_matrix_init(res,rows,1);
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
                    double temp = arr[pos*cols+k];
                    arr[pos*cols+k] = arr[i*cols+k];
                    arr[i*cols+k] = temp;

                    int argtmp = argset[pos*cols+k];//????????????
                    argset[pos*cols+k] = argset[i*cols+k];
                    argset[i*cols+k] = argtmp;
                }
            }
            for(int i=0;i<rows;i++) arr[argset[i*cols+k]*cols+k] = i;//???????????
        }
    }
    else{//????????
        my_matrix_init(res,1,cols);
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
                    double temp = arr[k*cols+pos];
                    arr[k*cols+pos] = arr[k*cols+i];
                    arr[k*cols+i] = temp;

                    int argtmp = argset[k*cols+pos];
                    argset[k*cols+pos] = argset[k*cols+i];
                    argset[k*cols+i] = argtmp;
                }
            }
            for(int i=0;i<cols;i++) arr[k*cols+argset[k*cols+i]] = i;
        }
    }
    free(argset);
    // ret
    res->data = arr;
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // #endif
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(_Z9qp4ai_accP20FunctionCallInfoData); // register function as V1
// ?????????????????????????????????normalize=0???????????????????????normalize = 1??
// ??????????????1 ????????2 ?????normalize ????????
// ?????0???? -1???????????? -2??????? -3 ???????
Datum
qp4ai_acc(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    int32 norm = PG_GETARG_INT32(2);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(3));
    //check
    if(matrixMap.count(input_table_name1)==0||matrixMap.count(input_table_name2)==0)
        PG_RETURN_INT32(-1);
    //fetch
    Matrix *mtrx1 = &matrixMap[input_table_name1];
    Matrix *mtrx2 = &matrixMap[input_table_name2];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    int col1 = mtrx1->columns;
    int row1 = mtrx1->rows;
    int col2 = mtrx2->columns;
    int row2 = mtrx2->rows;
    int size = col2*row2;
    if(col1!=col2)
        PG_RETURN_INT32(-3); // ???????????
    if(row1!=row2)
        PG_RETURN_INT32(-2); // ???????????
    my_matrix_init(res,1,1);
    double* arr1 = mtrx1->data;
    double* arr2 = mtrx2->data;
    double acc = 0;
    double* ans = (double*)malloc(sizeof(double));
    for(int i=0; i<size; i++){
        acc+=(arr1[i]==arr2[i]?1:0);//????????????
    }
    if(norm == 1){
        acc = acc/size;
    }
    // ret
    ans[0] = acc;
    res->data = ans;
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // #endif
    PG_RETURN_INT32(0);
}


/*10??2??? , 11??24?????????*/
PG_FUNCTION_INFO_V1(_Z13qp4ai_reverseP20FunctionCallInfoData); // register function as V1
// ???????????????????????????????==0??????????????????==1????????????????????=3???????????????????
// ?????torch.flip????????
// ?????????????? ???????dim?????????0??1??2 ????????
// ?????0???? -1???????????? -6???????????0??1??2
Datum
qp4ai_reverse(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 dim = PG_GETARG_INT32(1);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    //check
    if(matrixMap.count(input_table_name)==0)
        PG_RETURN_INT32(-1);
    if(dim != 0 && dim != 1 && dim!=2)
        PG_RETURN_INT32(-6);
    //fetch
    Matrix *mtrx1 = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    int cols = mtrx1->columns;
    int rows = mtrx1->rows;
    int size = cols*rows;
    double* arr = (double*)malloc(size*sizeof(double));
    for(int i=0;i<size;i++)
        arr[i] = mtrx1->data[i];
    my_matrix_init(res,rows,cols);
    matrix_copy(res,mtrx1);
    // main logic
    if(dim == 0){//??????
        double* temp_acc_arr = (double*) malloc(cols * sizeof(double));//?????????????????
        for(int i=0;i<rows/2;i++){
            for(int j=0;j<cols;j++){
                temp_acc_arr[j]=arr[i*cols+j]; //????????
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
    else if(dim == 1){//??????
        for(int i=0;i<rows;i++){
            double temp;
            for(int j=0;j<cols/2;j++){//??????
                int mid = cols-j-1;
                temp =  arr[i*cols+j];
                arr[i*cols+j] =  arr[i*cols+mid];
                arr[i*cols+mid] = temp;
            }
        }
    }
    else{
        double temp;
        for(int i=0;i<size/2;i++){
            int mid = size-i-1;
            temp = arr[i];
            arr[i] = arr[mid];
            arr[mid] = temp;
        }
    }
    // ret
    res->data = arr;
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // #endif
    PG_RETURN_INT32(0);
}

/*10??18???????? --F1????  11??24?????????*/
PG_FUNCTION_INFO_V1(_Z8qp4ai_f1P20FunctionCallInfoData); // register function as V1
// ?????????????1(?????)???2(?????)???f1??????????????????????
// ???????1
// ??????????????1(ground truth) ????????2(predict) ????????
// ?????0???? -1???????????? -2??????????? -3??????????? -13???????????1
Datum
qp4ai_f1(PG_FUNCTION_ARGS){
    // #ifdef DEBUG2
    // initTest();
    // #endif
    // get para
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    if(matrixMap.count(input_table_name1)==0||matrixMap.count(input_table_name2)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name1];
    Matrix *mtrx2 = &matrixMap[input_table_name2];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    //check if lines&cols equals
    int row1 = mtrx1->rows;
    int row2 = mtrx2->rows;
    int col1 = mtrx1->columns;
    int col2 = mtrx2->columns;
    double* arr1 = mtrx1->data;
    double* arr2 = mtrx2->data;
    if(row1 != row2)
        PG_RETURN_INT32(-2);
    if(col1!=col2)
        PG_RETURN_INT32(-3);
    if(row1!=1&&row2!=1)
        PG_RETURN_INT32(-13);
    int size = col1*row1;
    my_matrix_init(res, 1, 1);
    // main logic
    double *ans = (double*)malloc(sizeof(double));
    typedef struct node{//???????
        double data;
        int tp;
        int fp;
        int fn;
        struct node* next;
    }ND;
    ND* head = (ND*)malloc(sizeof(ND));//??????
    head -> next = NULL;
    for(int i=0;i<size;i++){
        int data1 = arr1[i];//???????
        int data2 = arr2[i];
        ND* ptr = head;
        ND* ptr1 = NULL;
        ND* ptr2 = NULL;
        while (ptr->next != NULL)//????????????????
        {
            if(ptr->next->data==data1){
                ptr1 = ptr->next;
            }
            if(ptr->next->data==data2){
                ptr2 = ptr->next;
            }
            if(ptr1 != NULL && ptr2 !=NULL){
                break;
            }
            ptr = ptr->next;
        }
        if(ptr1 == NULL){//????????????????
            ptr1 = (ND*)malloc(sizeof(ND));
            ptr1->data = data1;
            ptr1->next = NULL;
            ptr1->fn=0;
            ptr1->fp =0;
            ptr1->tp=0;
            ptr->next = ptr1;
            ptr = ptr->next;
        }
        if(data1!=data2&&ptr2 == NULL){
            ptr2 = (ND*)malloc(sizeof(ND));
            ptr2->data = data2;
            ptr2->next = NULL;
            ptr2->fn=0;
            ptr2->fp =0;
            ptr2->tp=0;
            ptr->next = ptr2;
            ptr = ptr->next;
        }
        if(data1 == data2){
            ptr1->tp = ptr1->tp+1;
        }else{
            ptr1->fn = ptr1->fn+1;
            ptr2->fp = ptr2->fp+1;
        }
    }
    int totalSum = 0;//???????????
    double totalF1 = 0;
    ND* ptr = head->next;
    ND* trash = head;
    while (ptr!=NULL)//??????????
    {
        totalSum++;
        if (ptr->tp != 0)
        {
            double precision = (double)(ptr->tp)/(ptr->tp+ptr->fp);
            double recall = (double)(ptr->tp)/(ptr->tp+ptr->fn);
            totalF1 += (2*precision*recall)/(precision+recall);
        }
        free(trash);//????????
        trash = ptr;
        ptr = ptr->next;
    }
    free(trash);
    totalF1 /= totalSum;
    ans[0] = totalF1;
    res->data = ans;
    matrixMap[output_table_name] = *res;
    // #ifdef DEBUG
    // printMSG(res);
    // #endif
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z10qp4ai_onesP20FunctionCallInfoData); // ??????V1??
// ??????
Datum // ????Postgres?????????????????????Datum
qp4ai_ones(PG_FUNCTION_ARGS){ // ??????(????) ????????
    // ???????
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));;
    my_matrix_init(matrix, rows, cols);
    matrix_ones(matrix);
    matrixMap[output_table_name] = *matrix;
    PG_RETURN_INT32(0); // ???????????
    /////////////////////////////////////////////////////////////////////////
}

//???map
PG_FUNCTION_INFO_V1(_Z15qp4ai_erase_mapP20FunctionCallInfoData); // ??????V1??
Datum
qp4ai_erase_map(PG_FUNCTION_ARGS){
    //matrixMap.clear();
    matrixMap.erase(matrixMap.begin(),matrixMap.end());
    PG_RETURN_INT32(0);
    /////////////////////////////////////////////////////////////////////////
}


PG_FUNCTION_INFO_V1(_Z19qp4ai_erase_elementP20FunctionCallInfoData); // ??????V1??
Datum
qp4ai_erase_element(PG_FUNCTION_ARGS){
    char* table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    matrixMap.erase(table_name);
    PG_RETURN_INT32(0);
}

//?????????????map
PG_FUNCTION_INFO_V1(_Z14qp4ai_init_mapP20FunctionCallInfoData); // ??????V1??
Datum
qp4ai_init_map(PG_FUNCTION_ARGS){
    initTest();
    PG_RETURN_INT32(0);
}


#define matrix_not_exists(table_name) (matrixMap.count(table_name)==0)
#define get_matrix_p_by_name(table_name) (&matrixMap[table_name])
#define malloc_new_matrix_p() (Matrix*)malloc(sizeof(Matrix))
#define store_matrix_p_as_table_name(matrix_p, name) matrixMap[name] = *matrix_p;


// ??????????????????????????????????????
PG_FUNCTION_INFO_V1(_Z12qp4ai_selectP20FunctionCallInfoData);
Datum
qp4ai_select(PG_FUNCTION_ARGS){
   char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));

    if(matrix_not_exists(input_table_name)){
        PG_RETURN_TEXT_P(cstring_to_text("Matrix not found."));
    }
    Matrix* input_matrix_p = get_matrix_p_by_name(input_table_name);
    string result="";
    int size = input_matrix_p->rows*input_matrix_p->columns;
    result = result + "{\"name\":"+input_table_name+",\"rows\":"+to_string(input_matrix_p->rows)+",\"cols\":"+to_string(input_matrix_p->columns)+",\"trans\":"+to_string((input_matrix_p->transposed?1:0))+",\"data\": [";

    //sprintf(result, "{ \"name\": %s, \"rows\": %d, \"cols\": %d, \"trans\": %d, \"data\": [", input_table_name, input_matrix_p->rows, input_matrix_p->columns, //(input_matrix_p->transposed?1:0));

    for (int i = 0; i < size; i++)
    {
         //string temp;
         result = result + to_string(input_matrix_p->data[i])+",";
         //sprintf(temp,"%f,",input_matrix_p->data[i]);
         // strcat(result,temp);
    }
    result = result + "]}";
    // strcat(result,"]}");
    // printMSG(input_matrix_p);
    // char result[128] = "result saved to table 'smy_output'";
    PG_RETURN_TEXT_P(cstring_to_text(result.c_str())); // 潩潩?
}
// ???????????????
PG_FUNCTION_INFO_V1(_Z9qp4ai_addP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_add(PG_FUNCTION_ARGS){
    // get param
    string input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    if(matrixMap.count(input_table_name1)==0||matrixMap.count(input_table_name2)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name1];
    Matrix *mtrx2 = &matrixMap[input_table_name2];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    //check if lines&cols equals
    if(mtrx1->rows != mtrx2->rows)
        PG_RETURN_INT32(-2);
    if(mtrx1->columns != mtrx2->columns)
        PG_RETURN_INT32(-3);
    my_matrix_init(res, mtrx1->rows, mtrx1->columns);
   // main logic
    int size = mtrx1->rows*mtrx1->columns;
    for(int i=0;i<size;i++){
        res->data[i] = mtrx1->data[i]+mtrx2->data[i];
    }
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0); // ?????
}

// ???????????????
PG_FUNCTION_INFO_V1(_Z9qp4ai_divP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_div(PG_FUNCTION_ARGS){
    // get para
    string input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    if(matrixMap.count(input_table_name1)==0||matrixMap.count(input_table_name2)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name1];
    Matrix *mtrx2 = &matrixMap[input_table_name2];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    //check if lines&cols equals
    if (mtrx1->rows == mtrx2->rows && mtrx1->columns == mtrx2->columns){
        my_matrix_init(res, mtrx1->rows, mtrx1->columns);
        int size = mtrx1->rows*mtrx1->columns;
        for(int i=0;i<size;i++){
            res->data[i] = mtrx1->data[i]/mtrx2->data[i];
        }
    }
    else if (mtrx1->rows*mtrx1->columns==1){
        my_matrix_init(res, mtrx2->rows, mtrx2->columns);
        int size = mtrx2->rows*mtrx2->columns;
        for(int i=0;i<size;i++){
            res->data[i] = mtrx1->data[0]/mtrx2->data[i];
        }
    }
    else if (mtrx2->rows*mtrx2->columns==1){
        my_matrix_init(res, mtrx1->rows, mtrx1->columns);
        int size = mtrx1->rows*mtrx1->columns;
        for(int i=0;i<size;i++){
            res->data[i] = mtrx1->data[i]/mtrx2->data[0];
        }
    }
    else{
        PG_RETURN_INT32(-2);
    }

   // main logic

    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0); // ?????
}
// mse
// ??????????????????????
// arg1 ?????
// arg2 ?????
// arg3 ?????
PG_FUNCTION_INFO_V1(_Z9qp4ai_mseP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_mse(PG_FUNCTION_ARGS){
    // get para
    string input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    if(matrix_not_exists(input_table_name1) || matrix_not_exists(input_table_name2)){
        PG_RETURN_INT32(-1);
    }

    Matrix *input_matrix1_p = get_matrix_p_by_name(input_table_name1);
    Matrix *input_matrix2_p = get_matrix_p_by_name(input_table_name2);
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, 1, 1);
   // main logic
    int size = input_matrix1_p->rows*input_matrix1_p->columns;
    // ???????????
    double mse_sum = 0;
    for(int i=0;i<size;i++){
        mse_sum += pow((input_matrix1_p->data[i] - input_matrix2_p->data[i]),2);
    }
    res->data[0] = mse_sum / size;
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0); // ?????
}

// auc
// ??????????????????????
// arg1 ????? ???????? 0 or 1
// arg2 ????? ????????(0-1)
// arg3 ????? 1*1
PG_FUNCTION_INFO_V1(_Z9qp4ai_aucP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_auc(PG_FUNCTION_ARGS){
    // get para
    string input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    if(matrix_not_exists(input_table_name1) || matrix_not_exists(input_table_name2)){
        PG_RETURN_INT32(-1);
    }

    Matrix *input_matrix1_p = get_matrix_p_by_name(input_table_name1);
    Matrix *input_matrix2_p = get_matrix_p_by_name(input_table_name2);
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, 1, 1);
   // main logic
    int size = input_matrix1_p->rows*input_matrix1_p->columns;
    // auc ???????
    vector<int> pos;
    vector<int> neg;
    for(int i=0; i<size; i++){
        if(abs(input_matrix1_p->data[i]-1.00)<0.1){
            pos.push_back(i);
        }else{
            neg.push_back(i);
        }
    }
    double auc = 0;
    for(int i=0; i<pos.size(); i++){
        for(int j=0; j<neg.size(); j++){
            if(input_matrix2_p->data[pos[i]] > input_matrix2_p->data[neg[j]]){
                auc += 1.0;
            }else if(input_matrix2_p->data[pos[i]] == input_matrix2_p->data[neg[j]]){
                auc += 0.5;
            }
        }
    }
    res->data[0] = auc / ( pos.size() * neg.size() );
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0); // ?????
}


// ???????????????
PG_FUNCTION_INFO_V1(_Z9qp4ai_mulP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_mul(PG_FUNCTION_ARGS){
    // get para
    string input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    if(matrixMap.count(input_table_name1)==0||matrixMap.count(input_table_name2)==0)
        PG_RETURN_INT32(-1);
    Matrix *mtrx1 = &matrixMap[input_table_name1];
    Matrix *mtrx2 = &matrixMap[input_table_name2];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    //check if lines&cols equals
    if(mtrx1->rows != mtrx2->rows)
        PG_RETURN_INT32(-2);
    if(mtrx1->columns != mtrx2->columns)
        PG_RETURN_INT32(-3);
    my_matrix_init(res, mtrx1->rows, mtrx1->columns);
   // main logic
    int size = mtrx1->rows*mtrx1->columns;
    for(int i=0;i<size;i++){
        res->data[i] = mtrx1->data[i]*mtrx2->data[i];
    }
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0); // ?????
}

// ??????????exp
PG_FUNCTION_INFO_V1(_Z9qp4ai_expP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_exp(PG_FUNCTION_ARGS){
    // get para
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    if(matrix_not_exists(input_table_name)){
        PG_RETURN_INT32(-1);
    }

    Matrix *input_matrix_p = get_matrix_p_by_name(input_table_name);
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, input_matrix_p->rows, input_matrix_p->columns);
   // main logic
    int size = input_matrix_p->rows*input_matrix_p->columns;
    for(int i=0;i<size;i++){
        res->data[i] = exp(input_matrix_p->data[i]);
    }
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0); // ?????
}

// ????????????????
PG_FUNCTION_INFO_V1(_Z9qp4ai_absP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_abs(PG_FUNCTION_ARGS){
    // get para
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    if(matrix_not_exists(input_table_name)){
        PG_RETURN_INT32(-1);
    }

    Matrix *input_matrix_p = get_matrix_p_by_name(input_table_name);
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, input_matrix_p->rows, input_matrix_p->columns);
   // main logic
    int size = input_matrix_p->rows*input_matrix_p->columns;
    for(int i=0;i<size;i++){
        res->data[i] = abs(input_matrix_p->data[i]);
    }
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0); // ?????
}

// ????????0????
PG_FUNCTION_INFO_V1(_Z11qp4ai_zerosP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_zeros(PG_FUNCTION_ARGS){
    // get para
    int32 rows = PG_GETARG_INT32(0);
    int32 cols = PG_GETARG_INT32(1);
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));

    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, rows, cols);
   // main logic
    matrix_zeroes(res);
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0); // ?????
}


// ????????
PG_FUNCTION_INFO_V1(_Z13qp4ai_reshapeP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_reshape(PG_FUNCTION_ARGS){
    // get para
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int32 rows = PG_GETARG_INT32(1);
    int32 cols = PG_GETARG_INT32(2);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(3));
    if(matrix_not_exists(input_table_name)){
        PG_RETURN_INT32(-1);
    }

    Matrix * input_matrix_p = get_matrix_p_by_name(input_table_name);
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init_clone(res, input_matrix_p);
   // main logic
    res->rows = rows;
    res->columns = cols;
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0); // ?????
}

// ????????????????????????????????
PG_FUNCTION_INFO_V1(_Z13qp4ai_shuffleP20FunctionCallInfoData);
// ??????
Datum // ????Postgres?????????????????????Datum
qp4ai_shuffle(PG_FUNCTION_ARGS){ // ??????(????) ????????
    // ???????
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    /////////////////////////////////////////////////////////////////////////
    // ???????????????????????????????????????????????
    if(matrix_not_exists(input_table_name)){
        PG_RETURN_INT32(-1); // ???????????
    }
    /////////////////////////////////////////////////////////////////////////
    Matrix* input_matrix_p = get_matrix_p_by_name(input_table_name);
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init_clone(res, input_matrix_p);
    // ???????????????????
    int32 table_rows = input_matrix_p->rows;
    int32 table_cols = input_matrix_p->columns;
    srand(time(0));
    double temp;
    for(int i=0; i<table_rows; i++){
        int swap_row = (int) rand() % (i+1) ; // ?????????????????????
        for(int j=0; j<table_cols; j++){
            temp = res->data[i * table_cols + j];
            res->data[i * table_cols + j] = res->data[swap_row * table_cols + j];
            res->data[swap_row * table_cols + j] = temp;
        }
    }
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0); // ???????????
}


// ?????????????????????????????????
PG_FUNCTION_INFO_V1(_Z10qp4ai_loadP20FunctionCallInfoData);
Datum // ????Postgres?????????????????????Datum
qp4ai_load(PG_FUNCTION_ARGS){ // ??????(????) ????????
    // ???????
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0)); // ??????????????????
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1)); // ????????????
    // ????????
    SPI_connect();  //????????????
    /////////////////////////////////////////////////////////////////////////
    // ???????????????????????????????????????????????
    // ???????????????
    char sql_table_rows[MAX_SQL_LEN];
    sprintf(sql_table_rows, "select count(*) from %s;", input_table_name);
    SPI_exec(sql_table_rows, 0);
    int32 table_rows = get_int32_from_qresult();

    if(!(table_rows)){
        SPI_finish();   // ????????????????????????
        PG_RETURN_INT32(-1); // ???????????
    }
    /////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    // ???????????
    char sql_table_cols[MAX_SQL_LEN];
    sprintf(sql_table_cols, "select count(*) from information_schema.columns where table_schema='public' and table_name = '%s';",
     input_table_name);
    SPI_exec(sql_table_cols, 0);
    int32 table_cols = get_int32_from_qresult();
    /////////////////////////////////////////////////////////////////////////
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, table_rows, table_cols);
    char sql[MAX_SQL_LEN];
    sprintf(sql, "SELECT * from %s;", input_table_name);
    SPI_execute(sql, true, 0);
    for(int i=0; i<table_rows; i++){
        for(int j=0; j<table_cols; j++){
            res->data[i*table_cols+j] = get_float8_from_qresult(i, j);
        }
    }
    matrixMap[output_table_name] = *res;
    SPI_finish();
    PG_RETURN_INT32(0); // ???????????
}

// sub ??????
PG_FUNCTION_INFO_V1(_Z14qp4ai_back_subP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_back_sub(PG_FUNCTION_ARGS){
    // get para
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, 1, 1);
   // main logic
    res->data[0] = -1.0;
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0); // ?????
}


// div ??????
PG_FUNCTION_INFO_V1(_Z14qp4ai_back_divP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_back_div(PG_FUNCTION_ARGS){
    // get para
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(2));
    char* output_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(3));

    // ??????????
    Matrix* i1 = get_matrix_p_by_name(input_table_name1);
    Matrix* i2 = get_matrix_p_by_name(input_table_name2);
    // ??????????????????????init
    Matrix* o1 = (Matrix*)malloc(sizeof(Matrix));
    Matrix* o2 = (Matrix*)malloc(sizeof(Matrix));
    // ????????????????????
    int size_i1 = i1->rows * i1->columns;
    int size_i2 = i2->rows * i2->columns;
    // ????????????
    if(size_i1 == 1){
        // ?????????????????????????
        // ??? ??????1 ?? 1*1?? ??????2????????2?????
        my_matrix_init(o1, 1, 1);
        my_matrix_init(o2, i2->rows, i2->columns);
        // o1??o2???????
        double sum = 0.0;
        for(int i=0; i<size_i2; i++){
            sum += 1 / i2->data[i];
            o2->data[i] = i1->data[0] * -1 / pow(i2->data[i], 2);
        }
        o1->data[0] = sum;
        // ??????????
        matrixMap[output_table_name1] = *o1;
        matrixMap[output_table_name2] = *o2;
    }else if(size_i2 == 1){
        // ?????????????????????????
        // ??? ??????1 ?? ??????1??????? ??????2??1*1.
        my_matrix_init(o1, i1->rows, i1->columns);
        my_matrix_init(o2, 1, 1);
        // o1??o2???????
        double sum = 0.0;
        for(int i=0; i<size_i2; i++){
            sum +=  o1->data[i] * -1 / pow(i2->data[0], 2);
            o1->data[i] = 1 / i2->data[0];
        }
        o2->data[0] = sum;
        // ??????????
        matrixMap[output_table_name1] = *o1;
        matrixMap[output_table_name2] = *o2;
    }else{
        // ??? ??????1??2??data????????????????????
        my_matrix_init(o1, i1->rows, i1->columns);
        my_matrix_init(o2, i2->rows, i2->columns);
        for(int i=0; i<size_i1; i++){
            o1->data[i] = 1 / i2->data[i];
            o2->data[i] = o1->data[i] * -1 / pow(o2->data[i], 2);
        }
        // ??????????
        matrixMap[output_table_name1] = *o1;
        matrixMap[output_table_name2] = *o2;
    }
    PG_RETURN_INT32(0); // ?????
}


// negative ??????
PG_FUNCTION_INFO_V1(_Z19qp4ai_back_negativeP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_back_negative(PG_FUNCTION_ARGS){
    // get para
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, 1, 1);
   // main logic
    res->data[0] = -1.0;
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0); // ?????
}

// log ??????
PG_FUNCTION_INFO_V1(_Z14qp4ai_back_logP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_back_log(PG_FUNCTION_ARGS){
    // get param
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));

    // ??????????
    Matrix* i1 = get_matrix_p_by_name(input_table_name);
    // ??????????????????????init
    Matrix* o1 = (Matrix*)malloc(sizeof(Matrix));
    // ????????????????
    int size_i1 = i1->rows * i1->columns;
    // ??? ?????? ???????? ?????
    my_matrix_init(o1, i1->rows, i1->columns);
    // o1??o2???????
    for(int i=0; i<size_i1; i++){
        o1->data[i] = 1.0 / i1->data[i];
    }
    // ??????????
    matrixMap[output_table_name] = *o1;
    PG_RETURN_INT32(0); // ?????
}

// mul ??????
PG_FUNCTION_INFO_V1(_Z14qp4ai_back_mulP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_back_mul(PG_FUNCTION_ARGS){
    // get param
    // self.vars[1]
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    // self.vars[2]
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(2));
    char* output_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(3));

    // ??????????
    Matrix* i1 = get_matrix_p_by_name(input_table_name1);
    Matrix* i2 = get_matrix_p_by_name(input_table_name2);
    // ??????????????????????init
    Matrix* o1 = (Matrix*)malloc(sizeof(Matrix));
    Matrix* o2 = (Matrix*)malloc(sizeof(Matrix));


    if (i1->rows==1 && i1->columns==1){
        my_matrix_init(o1, i1->rows, i1->columns);
        my_matrix_init(o2, 1, 1);
        double sum=0;
        for(int i=0; i<i2->rows*i2->columns; i++){
            sum += i2->data[i];
        }
        o1->data[0]=sum;
        o2->data[0]=i1->data[0];
    }
    else if (i2->rows==1 && i2->columns==1){
        my_matrix_init(o1, 1, 1);
        my_matrix_init(o2, i2->rows, i2->columns);
        o1->data = o2->data;
        double sum=0;
        for(int i=0; i<i1->rows*i1->columns; i++){
            sum += i1->data[i];
        }
        o2->data[0] = sum;
    }
    else if (i1->rows==i2->rows && i1->columns==i2->columns){
        my_matrix_init(o1, i1->rows, i1->columns);
        my_matrix_init(o2, i2->rows, i2->columns);
        for (int i=0;i<i1->rows*i1->columns;i++){
            o1->data[i] = i2->data[i];
        }
        for (int i=0;i<i2->rows*i2->columns;i++){
            o2->data[i] = i1->data[i];
        }
    }
    // ??????????
    matrixMap[output_table_name1] = *o1;
    matrixMap[output_table_name2] = *o2;
    PG_RETURN_INT32(0); // ?????
}

// pow ??????
PG_FUNCTION_INFO_V1(_Z14qp4ai_back_powP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_back_pow(PG_FUNCTION_ARGS){
    // self.vars[0]
    char* input_table_name0 = text_to_cstring(PG_GETARG_TEXT_PP(0));
     // self.vars[1]
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // self.vars[2]
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(2));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(3));
     // ??????????
    Matrix* i0 = get_matrix_p_by_name(input_table_name0);
    Matrix* i1 = get_matrix_p_by_name(input_table_name1);
    Matrix* i2 = get_matrix_p_by_name(input_table_name2);
    // ??????????????????????init
    Matrix* o1 = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(o1, i0->rows, i0->columns);
    int i0_size = i0->rows*i0->columns;
    int i2_size = i2->rows*i2->columns;
    if (i2_size>1){
        for(int i=0; i<i0_size; i++){
            o1->data[i]=i0->data[i]*log(i1->data[0]);
        }
    }
    else{
        for(int i=0; i<i0_size; i++){
            o1->data[i]=i0->data[i] * i2->data[0]/i1->data[i];
        }
    }
    matrixMap[output_table_name] = *o1;
    PG_RETURN_INT32(0); // ?????
}

// sqrt ??????
PG_FUNCTION_INFO_V1(_Z15qp4ai_back_sqrtP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_back_sqrt(PG_FUNCTION_ARGS){
    // self.vars[0]
    char* input_table_name0 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    // ??????????
    Matrix* i0 = get_matrix_p_by_name(input_table_name0);
    // ??????????????????????init
    Matrix* o1 = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(o1, i0->rows, i0->columns);
    for(int i=0;i<i0->rows*i0->columns;i++){
        if(i0->data[i]!=0){
            o1->data[i]=1/(2*i0->data[i]);
        }
    }
    matrixMap[output_table_name] = *o1;
    PG_RETURN_INT32(0); // ?????
}

// matmul ??????
PG_FUNCTION_INFO_V1(_Z17qp4ai_back_matmulP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_back_matmul(PG_FUNCTION_ARGS){
    // self.vars[1]
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    // self.vars[2]
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* output_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(2));
    char* output_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(3));
    char* grad_output = text_to_cstring(PG_GETARG_TEXT_PP(4));
    // ??????????
    Matrix* i1 = get_matrix_p_by_name(input_table_name1);
    Matrix* i2 = get_matrix_p_by_name(input_table_name2);
    // ??????????????????????init
    Matrix* o1 = (Matrix*)malloc(sizeof(Matrix));
    Matrix* o2 = (Matrix*)malloc(sizeof(Matrix));
    // ???py??grad_output==1??????
    if(strcmp(grad_output, "null")==0){
        vector<double> sum_data_1;
        for (int i=0;i<i2->rows;i++){
            double total = 0.0;
            for (int j=0;j<i2->columns;j++){
                total += i2->data[i*i2->columns + j];
            }
            sum_data_1.push_back(total);
        }
        vector<double> new_data_1;
        for (int i=0;i<i1->rows;i++){
            for (int j=0;j<i1->columns;j++){
                new_data_1.push_back(sum_data_1[j]);
            }
        }
        my_matrix_init(o1, i1->rows, i1->columns);
        for (int i=0;i<o1->rows*o1->columns;i++){
            o1->data[i] = new_data_1[i];
        }
        // copy(new_data_1.begin(), new_data_1.end(), o1->data);

        vector<double> sum_data_2;
        for (int i=0;i<i1->columns;i++){
            double total = 0.0;
            for (int j=0;j<i1->rows;j++){
                total += i1->data[i + j * i1->columns];
            }
            sum_data_2.push_back(total);
        }
        vector<double> new_data_2;
        for (int i=0;i<i2->rows;i++){
            for (int j=0;j<i2->columns;j++){
                new_data_2.push_back(sum_data_2[i]);
            }
        }
        my_matrix_init(o2, i2->rows, i2->columns);
        for (int i=0;i<o2->rows*o2->columns;i++){
            o2->data[i] = new_data_2[i];
        }
        // copy(new_data_2.begin(), new_data_2.end(), o2->data);
    }
    else{
        Matrix* i0 = get_matrix_p_by_name(grad_output);
        vector<double> new_data_1;
        for (int i=0;i<i0->rows;i++){
            for (int j=0;j<i2->rows;j++){
                double sum = 0.0;
                for (int k=0;k<i0->columns;k++){
                    sum += i0->data[i*i0->columns+k] * i2->data[j*i2->columns+k];
                }
                new_data_1.push_back(sum);
            }
        }
        my_matrix_init(o1, i1->rows, i1->columns);
        for (int i=0;i<o1->rows*o1->columns;i++){
            o1->data[i] = new_data_1[i];
        }
        // copy(new_data_1.begin(), new_data_1.end(), o1->data);
        vector<double> new_data_2;
        for (int i=0;i<i1->columns;i++){
            for (int j=0;j<i0->columns;j++){
                double sum = 0.0;
                for (int k=0;k<i1->rows;k++){
                    sum += i1->data[k*i1->columns+i]*i0->data[k*i2->columns+j];
                }
                new_data_2.push_back(sum);
            }
        }
        my_matrix_init(o2, i2->rows, i2->columns);
        for (int i=0;i<o2->rows*o2->columns;i++){
            o2->data[i] = new_data_2[i];
        }
        //copy(new_data_2.begin(), new_data_2.end(), o2->data);
    }
    matrixMap[output_table_name1] = *o1;
    matrixMap[output_table_name2] = *o2;
    PG_RETURN_INT32(0); // ?????
}

// mean ??????
PG_FUNCTION_INFO_V1(_Z15qp4ai_back_meanP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_back_mean(PG_FUNCTION_ARGS){
    // self.vars[1]
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    int axis = PG_GETARG_INT32(2);
    Matrix* i1 = get_matrix_p_by_name(input_table_name);
    Matrix* o1 = (Matrix*)malloc(sizeof(Matrix));
    double val;
    if (axis==0){
        val = 1.0/i1->columns;
    }
    else if(axis==1){
        val = 1.0/ i1->rows;
    }
    else{
        val = 1.0/ (i1->rows*i1->columns);
    }
    vector<double> data;
    for(int i=0;i<i1->rows;i++){
        for(int j=0;j<i1->columns;j++){
            data.push_back(val);
        }
    }
    my_matrix_init(o1, i1->rows, i1->columns);
    for (int i=0;i<o1->rows*o1->columns;i++){
        o1->data[i]=data[i];
    }
    // copy(data.begin(), data.end(), o1->data);
    matrixMap[output_table_name] = *o1;
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z18qp4ai_print_matrixP20FunctionCallInfoData);
Datum
qp4ai_print_matrix(PG_FUNCTION_ARGS){
    char* input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    Matrix* mtrx = get_matrix_p_by_name(input_table_name);
    SPI_connect();
    char sql_drop_table_if_exists[MAX_SQL_LEN];
    sprintf(sql_drop_table_if_exists, "DROP TABLE IF EXISTS %s;", table_name);
    SPI_exec(sql_drop_table_if_exists, 0);
    // const char* table_name = "output_result";
    string sql="CREATE TABLE " +(string)table_name+ " (";
    for (int i=0;i<mtrx->columns;i++){
        sql += "col_"+to_string(i)+" float8";
        if (i!=mtrx->columns-1){
            sql+=",";
        }
    }
    sql+=");";
    // sprintf(sql, " data float8[]);", table_name);
    SPI_exec(sql.c_str(), 0);
    int size = mtrx->rows*mtrx->columns;
    string table_name_ = table_name;
    for (int i = 0; i < mtrx->rows; i++)
    {
        string datainfo = "";
        for(int j=0;j<mtrx->columns;j++){
            datainfo = datainfo + to_string(mtrx->data[i*mtrx->columns+j]);
            if (j<mtrx->columns-1){
                datainfo = datainfo + ",";
            }
        }
        string sql2 = "INSERT INTO " +table_name_+ " values("+datainfo+");";
        SPI_exec(sql2.c_str(), 0);
    }
    // datainfo = datainfo + "";

    //sprintf(sql, "  INSERT INTO %s values('%s', '%s');",
    //    table_name,input_table_name, datainfo);
    SPI_finish();
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(_Z9qp4ai_valP20FunctionCallInfoData);
Datum
qp4ai_val(PG_FUNCTION_ARGS){
    float8 val = PG_GETARG_FLOAT8(0);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    Matrix* o1 = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(o1, 1, 1);
    o1->data[0]=val;
    matrixMap[output_table_name] = *o1;
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z16qp4ai_assignmentP20FunctionCallInfoData);
Datum
qp4ai_assignment(PG_FUNCTION_ARGS){
    char* mtrx_name0 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char* mtrx_name1 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    Matrix* mtrx0 = get_matrix_p_by_name(mtrx_name0);
    Matrix* o1 = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(o1, mtrx0->rows, mtrx0->columns);
    //memcpy(o1, mtrx0, o1->rows*o1->columns*sizeof(double));
    for (int i=0;i<mtrx0->rows*mtrx0->columns;i++){
        o1->data[i]=mtrx0->data[i];
    }
    matrixMap[mtrx_name1] = *o1;
    PG_RETURN_INT32(0); // ?????
}

PG_FUNCTION_INFO_V1(_Z22qp4ai_if_tensor_existsP20FunctionCallInfoData);
Datum
qp4ai_if_tensor_exists(PG_FUNCTION_ARGS){
    char* mtrx_name1 = text_to_cstring(PG_GETARG_TEXT_PP(0));
    int c = matrixMap.count(mtrx_name1);
    bool res= false;
    if(c>=1){
        res = true;
    }
    PG_RETURN_BOOL(res);
}

inline void fun_opt(int op, char* input_table_name1, char* input_table_name2, char* output_table_name){
    // ????????
    SPI_connect();
    if(op == 0){
        char sql[MAX_SQL_LEN];
        sprintf(sql, "SELECT qp4ai_add('%s','%s','%s');"
            , input_table_name1, input_table_name2, output_table_name);
        SPI_exec(sql, 0);
    }
    else if(op == 1){
        char sql[MAX_SQL_LEN];
        sprintf(sql, "SELECT qp4ai_sub('%s','%s','%s');"
            , input_table_name1, input_table_name2, output_table_name);
        SPI_exec(sql, 0);
    }
    else if(op == 2){
        char sql[MAX_SQL_LEN];
        sprintf(sql, "SELECT qp4ai_mul('%s','%s','%s');"
            , input_table_name1, input_table_name2, output_table_name);
        SPI_exec(sql, 0);
    }
    else if(op == 3){
        char sql[MAX_SQL_LEN];
        sprintf(sql, "SELECT qp4ai_div('%s','%s','%s');"
            , input_table_name1, input_table_name2, output_table_name);
        SPI_exec(sql, 0);
    }
    else if(op == 4){
        char sql[MAX_SQL_LEN];
        sprintf(sql, "SELECT qp4ai_matmul('%s','%s','%s');"
            , input_table_name1, input_table_name2, output_table_name);
        SPI_exec(sql, 0);
    }
    SPI_finish();  // ???????????
}

inline float val_opt(int op, float i, float j){
    if(op == 0){
        return i + j;
    }
    else if(op == 1){
        return i - j;
    }
    else if(op == 2){
        return i * j;
    }
    else if(op == 3){
        return i / j;
    }
}

PG_FUNCTION_INFO_V1(_Z18qp4ai_op_broadcastP20FunctionCallInfoData);
Datum
qp4ai_op_broadcast(PG_FUNCTION_ARGS){
    // ???????
    int32 op = PG_GETARG_INT32(0);
    char* input_table_name1 = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* input_table_name2 = text_to_cstring(PG_GETARG_TEXT_PP(2));
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(3));
    //check
    if(matrixMap.count(input_table_name1)==0||matrixMap.count(input_table_name2)==0)
        PG_RETURN_INT32(-1);
    // ??????????????????????????????????????????
    if(op !=0 && op !=1 && op != 2 && op != 3 && op != 4){
        PG_RETURN_INT32(-5); // ???????????
    }
    Matrix *mtrx1 = &matrixMap[input_table_name1];
    Matrix *mtrx2 = &matrixMap[input_table_name2];
    int row1 = mtrx1->rows;
    int row2 = mtrx2->rows;
    int col1 = mtrx1->columns;
    int col2 = mtrx2->columns;
    //main logic
    if(row1 == row2 && col1 == col2){
        Matrix *res = (Matrix*)malloc(sizeof(Matrix));
        my_matrix_init(res, row1, col1);
        matrixMap[output_table_name] = *res;
        fun_opt(op, input_table_name1, input_table_name2, output_table_name);
    }
    // ?????????????????????????????1????????????????
    else if(row1 == row2){
        if(col1 == 1){
            Matrix *res = (Matrix*)malloc(sizeof(Matrix));
            my_matrix_init(res, row1, col2);
            for(int i = 0;i < row1;i++){
                for(int j = 0;j < col2;j++){
                    res->data[i*col2+j] = val_opt(op, mtrx1->data[i], mtrx2->data[i*col2+j]);
                }
            }
            matrixMap[output_table_name] = *res;
        }else{
            Matrix *res = (Matrix*)malloc(sizeof(Matrix));
            my_matrix_init(res, row1, col1);
            for(int i = 0;i < row1;i++){
                for(int j = 0;j < col1;j++){
                    res->data[i*col1+j] = val_opt(op, mtrx1->data[i*col1+j], mtrx2->data[i]);
                }
            }
            matrixMap[output_table_name] = *res;
        }
    }
    // ?????????????????????????????1????????????????
    else if(col1 == col2){
        if(row1 == 1){
            Matrix *res = (Matrix*)malloc(sizeof(Matrix));
            my_matrix_init(res, row2, col1);
            for(int i = 0;i < row2;i++){
                for(int j = 0;j < col1;j++){
                    res->data[i*col1+j] = val_opt(op, mtrx1->data[j], mtrx2->data[i*col1+j]);
                }
            }
            matrixMap[output_table_name] = *res;
        }else{
            Matrix *res = (Matrix*)malloc(sizeof(Matrix));
            my_matrix_init(res, row1, col1);
            for(int i = 0;i < row1;i++){
                for(int j = 0;j < col1;j++){
                    res->data[i*col1+j] = val_opt(op, mtrx1->data[i*col1+j], mtrx2->data[j]);
                }
            }
            matrixMap[output_table_name] = *res;
        }
    }
    // ?????????????????,?????????????????1
    else if(row1 != row2 && col2 != col1){
        //
        if(row1 == 1 && col1 == 1){
            Matrix *res = (Matrix*)malloc(sizeof(Matrix));
            my_matrix_init(res, row2, col2);
            for(int i = 0;i < row2;i++){
                for(int j = 0;j < col2;j++){
                    res->data[i*col2+j] = val_opt(op, mtrx1->data[0], mtrx2->data[i*col2+j]);
                }
            }
            matrixMap[output_table_name] = *res;
        }
        else if(row2 ==1 && col2 == 1){
            Matrix *res = (Matrix*)malloc(sizeof(Matrix));
            my_matrix_init(res, row1, col1);
            for(int i = 0;i < row1;i++){
                for(int j = 0;j < col1;j++){
                    res->data[i*col1+j] = val_opt(op, mtrx1->data[i*col1+j], mtrx2->data[0]);
                }
            }
            matrixMap[output_table_name] = *res;
        }
        // 1*a??b*1
        else if(row1 == 1 && col1 != 1 && row2 !=1 && col2 ==1){
            Matrix *res = (Matrix*)malloc(sizeof(Matrix));
            my_matrix_init(res, row2, col1);
            for(int i = 0;i < row2;i++){
                for(int j = 0;j < col1;j++){
                    res->data[i*col1+j] = val_opt(op, mtrx1->data[j], mtrx2->data[i]);
                }
            }
            matrixMap[output_table_name] = *res;
        }
        else if(row1 != 1 && col1 == 1 && row2 ==1 && col2 !=1){
            Matrix *res = (Matrix*)malloc(sizeof(Matrix));
            my_matrix_init(res, row1, col2);
            for(int i = 0;i < row1;i++){
                for(int j = 0;j < col2;j++){
                    res->data[i*col2+j] = val_opt(op, mtrx1->data[i], mtrx2->data[j]);
                }
            }
            matrixMap[output_table_name] = *res;
        }
    }
    else{
        PG_RETURN_INT32(-10);//???????????
    }

    //#ifdef DEBUG
    //printMSG(res);
    //printMSG(matrixMap["mtrx2"]);
    //#endif
    PG_RETURN_INT32(0); // ?????
}




PG_FUNCTION_INFO_V1(_Z17qp4ai_update_dataP20FunctionCallInfoData);
Datum
qp4ai_update_data(PG_FUNCTION_ARGS){
    ArrayType* arr = PG_GETARG_ARRAYTYPE_P(0);
    float8* data = (float8 *) ARR_DATA_PTR(arr);
    char* output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    Matrix *mtrx = &matrixMap[output_table_name];
    for(int i=0;i<ARRNELEMS(arr);i++){
        mtrx->data[i] = data[i];
    }
    matrixMap[output_table_name] = *mtrx;
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(_Z14qp4ai_negativeP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_negative(PG_FUNCTION_ARGS){
    // get param
    // vars[0]
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    // vars[1]
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    Matrix *mtrx = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init_clone(res, mtrx);
    for (int i=0;i<res->rows*res->columns;i++){
        res->data[i] = -res->data[i];
    }
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(_Z18qp4ai_back_softmaxP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_back_softmax(PG_FUNCTION_ARGS){
    // get param
    // vars[0], softmax
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    // table_name
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    string grad_output = text_to_cstring(PG_GETARG_TEXT_PP(2));
    Matrix *mtrx = &matrixMap[input_table_name];
    Matrix *mtrx_grad = &matrixMap[grad_output];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init_clone(res, mtrx);
    if (mtrx->rows!=mtrx_grad->rows || mtrx->columns!=mtrx_grad->columns){
        PG_RETURN_INT32(-1);
    }
    // new_data == res->data
    for (int i=0;i<mtrx->rows;i++){
        for (int r=0;r<mtrx->columns;r++){
            res->data[i*mtrx->columns+r]=0;
        }
        for (int j=0;j<mtrx->columns;j++){
            for (int k=0;k<mtrx->columns;k++){
                int index = i*mtrx->columns + k;
                if(j==k){
                    res->data[index] += mtrx->data[index]*(1-mtrx->data[index])*mtrx_grad->data[i*mtrx_grad->columns+j];
                }
                else{
                    res->data[index] += -(mtrx->data[i*mtrx->columns+j])*(mtrx->data[index])*(mtrx_grad->data[i*mtrx_grad->columns+j]);
                }
            }
        }
    }
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(_Z10qp4ai_reluP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_relu(PG_FUNCTION_ARGS){
    // get param
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    Matrix *mtrx = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, mtrx->rows, mtrx->columns);
    for (int i=0;i<mtrx->rows*mtrx->columns;i++){
        res->data[i] = mtrx->data[i]>0?mtrx->data[i]:0;
    }
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(_Z15qp4ai_back_reluP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_back_relu(PG_FUNCTION_ARGS){
    // get param
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    Matrix *mtrx = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, mtrx->rows, mtrx->columns);
    for (int i=0;i<mtrx->rows*mtrx->columns;i++){
        res->data[i] = mtrx->data[i]>0?1:0;
    }
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(_Z15qp4ai_leakyreluP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_leakyrelu(PG_FUNCTION_ARGS){
    // get param
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    Matrix *mtrx = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, mtrx->rows, mtrx->columns);
    for (int i=0;i<mtrx->rows*mtrx->columns;i++){
        res->data[i] = mtrx->data[i]>0?mtrx->data[i]:0.1*mtrx->data[i];
    }
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(_Z20qp4ai_back_leakyreluP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_back_leakyrelu(PG_FUNCTION_ARGS){
    // get param
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    Matrix *mtrx = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, mtrx->rows, mtrx->columns);
    for (int i=0;i<mtrx->rows*mtrx->columns;i++){
        res->data[i] = mtrx->data[i]>0?1:0.1;
    }
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0);
}


PG_FUNCTION_INFO_V1(_Z10qp4ai_tanhP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_tanh(PG_FUNCTION_ARGS){
    // get param
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    Matrix *mtrx = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    vector<double> arr1;
    for(int i=0;i<mtrx->rows*mtrx->columns;i++) arr1.push_back(exp(mtrx->data[i]));
    vector<double> arr2;
    for(int i=0;i<mtrx->rows*mtrx->columns;i++) arr2.push_back(exp(-mtrx->data[i]));
    my_matrix_init(res, mtrx->rows, mtrx->columns);
    for (int i=0;i<mtrx->rows*mtrx->columns;i++){
        res->data[i] = (arr1[i]-arr2[i])/(arr1[i]+arr2[i]);
    }
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0);
}

PG_FUNCTION_INFO_V1(_Z15qp4ai_back_tanhP20FunctionCallInfoData); // register function as V1
Datum
qp4ai_back_tanh(PG_FUNCTION_ARGS){
    // get param
    // tanh(x)
    string input_table_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    // grad_tanh(x)
    string output_table_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
    Matrix *mtrx = &matrixMap[input_table_name];
    Matrix *res = (Matrix*)malloc(sizeof(Matrix));
    my_matrix_init(res, mtrx->rows, mtrx->columns);
    for (int i=0;i<mtrx->rows*mtrx->columns;i++){
        res->data[i] = 1 - pow(mtrx->data[i],2);
    }
    matrixMap[output_table_name] = *res;
    PG_RETURN_INT32(0);
}