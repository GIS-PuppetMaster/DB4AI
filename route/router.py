from flask import Flask, request, jsonify
from flask_cors import CORS

from SecondLevelLanguageParser import run_describe_language
from sqlProcess import ReadSqlFile
import json
import os
from FirstLevelLanguageParser import run_task_language
'''
使用request接收前端post请求
直接使用return发送后端处理好的数据给前端
'''
# flask服务启动，进行初始化
app = Flask(__name__)

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
user_kind = 'ordinary'

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,HEAD,GET,POST'
    response.headers[
        'Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response


@app.route('/api/login', methods=['GET', 'POST'])
def get_userID():
    if request.method == 'POST':
        # 获取用户身份信息
        argsJson = request.data.decode('utf-8')
        argsJson = json.loads(argsJson)
        result = process_json(argsJson)
        print(result['id'])
        # 转化为字符串格式, 如果不转化为字典
        # result = json.dumps(result, ensure_ascii=False)
        # return会直接把处理好的数据返回给前端
        return jsonify({'code': 200, 'status': 'accept'})
    else:
        print(request.data)
        return " 'it's not a POST operation! "


# 通过python装饰器的方法定义一个路由地址，如http://127.0.0.1:5000/api就是接口的url
@app.route('/api/partition', methods=['GET', 'POST'])
def get_data_partition():
    if request.method == 'POST':
        argsJson = request.data.decode('utf-8')
        print(argsJson)
        argsJson = json.loads(argsJson)
        result = process_json(argsJson)
        # 转化为字符串格式, 如果不转化为字典
        # result = json.dumps(result, ensure_ascii=False)
        # return会直接把处理好的数据返回给前端
        name = result['form']['num']
        weight = result['form']['weight']
        print(name, weight)
        """
        result 格式参考打印
        这里调用partition()函数
        """
        return result
    else:
        return " 'it's not a POST operation! "


@app.route('/api/collection', methods=['GET', 'POST'])
def get_data_collection():
    if request.method == 'POST':
        argsJson = request.data.decode('utf-8')
        argsJson = json.loads(argsJson)
        result = process_json(argsJson)
        # 转化为字符串格式, 如果不转化为字典
        # result = json.dumps(result, ensure_ascii=False)
        # return会直接把处理好的数据返回给前端
        insert_num = result['ruleForm']['insert']
        select_num = result['ruleForm']['select']
        print(insert_num, select_num)
        """
        result 格式参考打印，分别调用下列函数
        def Row(insert_num,select_num,name,num_p)
        def Colomn(insert_num,select_num,name,num_p)
        """
        return jsonify({'code': 200, 'status': 'accept'})
    else:
        return " 'it's not a POST operation! "


@app.route('/api/train', methods=['GET', 'POST'])
def get_data_train():
    if request.method == 'POST':
        argsJson = request.data.decode('utf-8')
        argsJson = json.loads(argsJson)
        # process_json()函数本来打算封装下通用操作
        result = process_json(argsJson)
        # 转化为字符串格式, 如果不转化为字典
        # result = json.dumps(result, ensure_ascii=False)
        # return会直接把处理好的数据返回给前端
        insert_num = result['ruleForm']['insert']
        select_num = result['ruleForm']['select']
        print(insert_num, select_num)
        """
        result 格式参考打印，调用函数predict()
        下面假设返回值是1
        """
        predict_result = 1
        return jsonify({
            'predict': predict_result,
            'code': 200,
            'status': 'accept'
        })
    else:
        return " 'it's not a POST operation! "


@app.route('/api/tableClass', methods=['GET', 'POST'])
def get_tableClass():
    if request.method == 'POST':
        # 获取用户身份信息
        argsJson = request.data.decode('utf-8')
        argsJson = json.loads(argsJson)
        result = process_json(argsJson)
        print(result['kind'])
        user_kind = result['kind']
        # 转化为字符串格式, 如果不转化为字典
        # result = json.dumps(result, ensure_ascii=False)
        # return会直接把处理好的数据返回给前端
        return jsonify({'code': 200, 'status': 'accept'})
    else:
        print(request.data)
        return " 'it's not a POST operation! "


@app.route('/api/sql', methods=['GET', 'POST'])
def get_user_sql():
    if request.method == 'POST':
        argsJson = request.data.decode('utf-8')
        argsJson = json.loads(argsJson)
        # process_json()函数本来打算封装下通用操作
        result = process_json(argsJson)
        sql_query = result['data']
        print(sql_query)
        if user_kind == 'ordinary':
            return jsonify({'code': 200, 'status': 'accept', 'result': run_task_language(sql_query)})
        else:
            run_describe_language(sql_query)
            return jsonify({'code': 200, 'status': 'accept', 'result': ''})
        # 转化为字符串格式, 如果不转化为字典
        # result = json.dumps(result, ensure_ascii=False)
        # return会直接把处理好的数据返回给前端
        # 在下面json返回格式中填写查询结果
    else:
        return " 'it's not a POST operation! "


@app.route('/api/files', methods=['GET', 'POST'])
def get_file():
    file = request.files['file']
    name = file.filename
    # 文件保存等操作暂时注释，按需自己写就行
    # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    """
    这里调用Create_Table函数
    """
    return "200 ok"


@app.route('/api/sqlfiles', methods=['GET', 'POST'])
def User_sql_file():
    file = request.files['file']
    name = file.filename
    # 文件保存等操作暂时注释，按需自己写就行
    file_path = os.path.abspath(os.path.join('./UserUploadSql', name))
    # print(file)
    file.save(file_path)
    return jsonify({'name': name, 'content': ReadSqlFile(file_path)})


@app.route('/api/fileList', methods=['GET', 'POST'])
def getFileList():
    if request.method == 'POST':
        current = os.path.abspath(__file__)
        route_path = os.path.abspath(
            os.path.dirname(current) + os.path.sep + ".")
        file_path = os.path.abspath(os.path.join(route_path,
                                                 './UserUploadSql'))
        # FileList = [{
        #     'filename': '测试文件1',
        #     'data': 'Alter table tablename add primary key(col)',
        # }, {
        #     'filename': '测试文件2',
        #     'data': 'select * from *',
        # }]
        FileList = []
        for file in os.listdir(file_path):
            temp_path = os.path.join(file_path, file)
            temp = {'filename': file, 'data': ReadSqlFile(temp_path)}
            FileList.append(temp)
        # return jsonify({'FileList': os.listdir(file_path)})
        print(FileList)
        return jsonify({'FileList': FileList})
    else:
        return "It's a Get request!"


# 学习索引模块的接口
@app.route('/api/learnedindex', methods=['GET', 'POST'])
def learnedInedx_sql():
    if request.method == 'POST':
        argsJson = request.data.decode('utf-8')
        argsJson = json.loads(argsJson)
        # process_json()函数本来打算封装下通用操作
        result = process_json(argsJson)
        sql_query = result['data']
        print(sql_query)
        # 转化为字符串格式, 如果不转化为字典
        # result = json.dumps(result, ensure_ascii=False)
        # return应该包括右侧展示的三个input框中内容
        # 在下面json返回格式中填写查询结果
        return jsonify({
            'querytime': '15s',
            'querymemory': '200MB',
            'querystruct': 'Random Forest'
        })
    else:
        return " 'it's not a POST operation! "


@app.route('/res/response', methods=['GET', 'POST'])
def getResponse():
    if request.method == 'POST':
        argsJson = request.data.decode('utf-8')
        argsJson = json.loads(argsJson)
        result = process_json(argsJson)
        # 转化为字符串格式, 如果不转化为字典
        # result = json.dumps(result, ensure_ascii=False)
        # return会直接把处理好的数据返回给前端
        if result['execute'] == '1':
            '''
            Col_store
            '''
            print('@')
        else:
            '''
            Row_store
            '''
            print('#')
        return jsonify({'code': 200, 'status': 'accept'})
    else:
        return " 'it's not a POST operation! "


def process_json(data):
    return data


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=10000)  # 可以设置为本机IP，或者127.0.0.1
