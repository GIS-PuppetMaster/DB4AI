import re
import json
# data_shape = '([(]([\d,]*(-1)*)*([\d])[)])|([(]([\d,]*(-1)*)*(-1)[)])'
data_list_reg = '[(]([\d,]?(-1,)?)*([\d])[)]|[(]([\d,]?(-1,)?)*(-1)[)]|[(]-1,[)]|[(][\d],[)]'
result=re.search('[^ ]','2')
print(result)