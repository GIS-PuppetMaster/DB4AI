import re


test = 'SQL(\'select * from dataset\') '
res = re.match('(SQL|sql)[(](.+)[)]', test)
if res:
    print(res.group())
else:
    print('null')