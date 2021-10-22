import requests
import json


class GDBC(object):
    def __init__(self):
        self.url = "http://localhost:12345/"
        self.connect_url = self.url + "connect"
        self.close_url = self.url + "close"
        self.execute_url = self.url + "execute"
        self.fetch_url = self.url + "fetch"
        self.connected = False

    """
        代理服务器建立与数据库的连接，返回ok或任何其它信息。
        return True if 该连接建立成功 else False
    """
    def connect(self):
        if self.connected:
            print("Connection already established!")
            return True
        res = requests.get(self.connect_url)
        if res.text == "ok":
            print("Connection successfully established!")
            self.connected = True
            return True
        else:
            print("Error occurred when connecting.")
            return False

    """
        代理服务器关闭与数据库的连接，返回ok或任何其它信息。
        return True if 该连接关闭成功 else False
    """
    def close(self):
        if not self.connected:
            print("No connection found!")
            return True
        res = requests.get(self.close_url)
        if res.text == "ok":
            print("Connection successfully closed!")
            return True
        else:
            print("Error occurred when disconnecting.")
            return False

    """
        代理服务器执行sql，返回ok或执行错误的信息。
        return True if sql执行无误 else False
    """
    def execute(self, sql: str):
        if not self.connected:
            print("No connection found!")
            return False
        data = {
            'sql': sql
        }
        res = requests.post(url=self.execute_url, data=data)
        if res.text == "ok":
            return True
        else:
            return False

    """
        代理服务器取数据，返回【读取到的数据的json格式字符串】。
        return None if 没数据或有问题 else 从代理服务器返回的字符串恢复的python对象。
    """
    def fetch(self):
        if not self.connected:
            print("No connection found!")
            return None
        res = requests.get(self.fetch_url)
        # print(res.text)
        data = json.loads(res.text)
        # print(type(data))
        return data


if __name__ == '__main__':
    gdbc = GDBC()
    gdbc.connect()
    vars = ["tmp", "qqq"]
    s = [[0, 0], [0, 0]]
    gdbc.execute(f"select count(*) from pg_class where relname = '_14';")
    print(gdbc.fetch())
    '''ans = gdbc.fetch()
    print(len(ans))
    if ans[0][0] == 0:
        print(True)
    else:
        gdbc.execute("create table \"A\"(val integer);")
        gdbc.execute("insert into temp values(1);")
    
    gdbc.execute("create table temp(val integer);")
    gdbc.execute("insert into temp values (1);")
    gdbc.execute("select * from temp;")
    # gdbc.execute("select count(*) from pg_class where relname = 'temp';")'''
