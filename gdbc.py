import pyodbc


class GDBC(object):
    def __init__(self):
        self.connected = False
        self.conn = None
        self.cursor = None
        self.ans = None

    """
        与数据库的连接，返回ok或任何其它信息。
    """
    def connect(self):
        if self.connected:
            print("Connection already established!")
            return True
        # 建立连接
        self.conn = pyodbc.connect('DRIVER={GaussMPP};SERVER=127.0.0.1;DATABASE=omm;UID=omm;PWD=Gauss@123')
        self.cursor = self.conn.cursor()
        if self.conn is not None:
            print("Connection successfully established!")
            self.connected = True
            return True
        else:
            print("Error occurred when connecting.")
            return False

    """
        关闭与数据库的连接。
    """
    def close(self):
        if not self.connected:
            print("No connection found!")
            return True
        #  关闭连接
        self.cursor.commit()
        self.cursor.close()
        self.conn.close()
        print("Successfully closed!")
        return True

    """
        执行sql。
    """
    def execute(self, sql: str):
        if not self.connected:
            print("No connection found!")
            return False
        #  执行sql
        self.ans = self.cursor.execute(sql)

    """
        取数据，返回。
    """
    def fetch(self):
        if not self.connected:
            print("No connection found!")
            return None
        return self.ans.fetchall()

if __name__ == '__main__':
    gdbc = GDBC()
    gdbc.connect()
    gdbc.execute("drop table if exists test1;")
    gdbc.execute("create table test1(val INT);")
    gdbc.execute("insert into test1 values (3);")
    gdbc.execute("select * from test1;")
    print(gdbc.fetch())
    gdbc.close()

