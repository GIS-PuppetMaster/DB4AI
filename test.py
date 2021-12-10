from gdbc import GDBC
g = GDBC()
g.connect()
g.execute(f"drop table if exists real_multi_class_x")
g.execute(f"drop table if exists real_multi_class_test_x")
g.execute(f"drop table if exists real_multi_class_y")
g.execute(f"drop table if exists real_multi_class_test_y")
g.execute(f"create table real_multi_class_x(dim1 float,dim2 float,dim3 float,dim4 float,dim5 float,dim6 float)")
g.execute(f"create table real_multi_class_test_x(dim1 float,dim2 float,dim3 float,dim4 float,dim5 float,dim6 float)")
g.execute(f"create table real_multi_class_y(flag int)")
g.execute(f"create table real_multi_class_test_y(flag int)")

with open('d:/Users/LU/Desktop/car.txt') as f:
    texts = f.readlines()
    buying = ['vhigh', 'high', 'med', 'low']
    maint = ['vhigh', 'high', 'med', 'low']
    doors = ['2', '3', '4', '5more']
    persons = ['2', '4', 'more']
    lug_boot =['small', 'med', 'big']
    safety = ['low', 'med', 'high']
    if_ac = ['acc', 'unacc']
    for i in range(len(texts)):
        text = texts[i].strip().split(',')
        if i % 10 == 0:
            s_x = "real_multi_class_test_x"
            s_y = "real_multi_class_test_y"
        else:
            s_x = "real_multi_class_x"
            s_y = "real_multi_class_y"
        g.execute(f"insert into {s_x} values ({buying.index(text[0])*0.25}, {maint.index(text[1])*0.25}, {doors.index(text[2])*0.25}"
                  f", {persons.index(text[3])*0.33}, {lug_boot.index(text[4])*0.33}, {safety.index(text[5])*0.33})")
        g.execute(f"insert into {s_y} values ({if_ac.index(text[6])})")
        print(text)
    print(texts)