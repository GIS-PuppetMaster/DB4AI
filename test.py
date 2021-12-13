# from gdbc import GDBC
# g = GDBC()
# g.connect()
# g.execute(f"drop table if exists real_multi_class_x")
# g.execute(f"drop table if exists real_multi_class_test_x")
# g.execute(f"drop table if exists real_multi_class_y")
# g.execute(f"drop table if exists real_multi_class_test_y")
# g.execute(f"create table real_multi_class_x(dim1 float,dim2 float,dim3 float,dim4 float,dim5 float,dim6 float)")
# g.execute(f"create table real_multi_class_test_x(dim1 float,dim2 float,dim3 float,dim4 float,dim5 float,dim6 float)")
# g.execute(f"create table real_multi_class_y(flag int)")
# g.execute(f"create table real_multi_class_test_y(flag int)")
#
# with open('d:/Users/LU/Desktop/car.txt') as f:
#     texts = f.readlines()
#     buying = ['vhigh', 'high', 'med', 'low']
#     maint = ['vhigh', 'high', 'med', 'low']
#     doors = ['2', '3', '4', '5more']
#     persons = ['2', '4', 'more']
#     lug_boot =['small', 'med', 'big']
#     safety = ['low', 'med', 'high']
#     if_ac = ['acc', 'unacc']
#     for i in range(len(texts)):
#         text = texts[i].strip().split(',')
#         if i % 10 == 0:
#             s_x = "real_multi_class_test_x"
#             s_y = "real_multi_class_test_y"
#         else:
#             s_x = "real_multi_class_x"
#             s_y = "real_multi_class_y"
#         g.execute(f"insert into {s_x} values ({buying.index(text[0])*0.25}, {maint.index(text[1])*0.25}, {doors.index(text[2])*0.25}"
#                   f", {persons.index(text[3])*0.33}, {lug_boot.index(text[4])*0.33}, {safety.index(text[5])*0.33})")
#         g.execute(f"insert into {s_y} values ({if_ac.index(text[6])})")
#         print(text)
#     print(texts)



with open('wdbc.txt','r') as f:
    lines = f.readlines()
    total = []
    for a in lines:
        s = []
        ta = a.split(',')
        for j in range(len(ta)):
            if j > 1:
                s.append(float(eval(ta[j])))
            else:
                s.append(ta[j])
        total.append(s)
print(len(total))
print(total[0])
for i in range(len(total[0])):
    if i > 1:
        max = 0
        min = 99999
        for j in range(len(total)):
            temp = []
            if total[j][i] > max:
                max = total[j][i]
            elif total[j][i] < min:
                min = total[j][i]
        for j in range(len(total)):
            total[j][i] = (total[j][i] - min) / (max - min)

x = open("d:/Users/LU/Desktop/x.txt", "w")
y = open("d:/Users/LU/Desktop/y.txt", "w")
test_x = open("d:/Users/LU/Desktop/test_x.txt", "w")
test_y = open("d:/Users/LU/Desktop/test_y.txt", "w")
for i in range(len(total)):
    print(i)
    if i < 500:
        for j in range(len(total[i])):
            if j > 1:
                x.write(str(total[i][j]))
                if j != len(total[i]):
                    x.write(str(','))
        if total[i][1] == 'B':
            y.write(str(1.0))
        elif total[i][1] == 'M':
            y.write(str(0.0))
        x.write('\n')
        y.write('\n')
    else:
        for j in range(len(total[i])):
            if j > 1:
                test_x.write(str(total[i][j]))
                if j != len(total[i]):
                    test_x.write(str(','))
        if total[i][1] == 'B':
            test_y.write(str(1.0))
        elif total[i][1] == 'M':
            test_y.write(str(0.0))
        test_x.write('\n')
        test_y.write('\n')
# for i in range(len(total)):
#     print(i)
#     if i < 500:
#         gdbc.execute(f"insert into real_x values({total[i][2]},{total[i][3]},{total[i][4]},{total[i][5]},{total[i][6]},{total[i][7]},{total[i][8]},{total[i][9]},{total[i][10]},{total[i][11]}"
#                      f",{total[i][12]},{total[i][13]},{total[i][14]},{total[i][15]},{total[i][16]},{total[i][17]},"
#                      f"{total[i][18]},{total[i][19]},{total[i][20]},{total[i][21]},{total[i][22]},{total[i][23]},{total[i][24]},{total[i][25]},{total[i][26]},"
#                      f"{total[i][27]},{total[i][28]},{total[i][29]},{total[i][30]},{total[i][31]})")
#         if total[i][1] == 'B':
#             gdbc.execute(f"insert into real_y values(1.0)")
#         elif total[i][1] == 'M':
#             gdbc.execute(f"insert into real_y values(0.0)")
#     else:
#         gdbc.execute(f"insert into real_test_x values({total[i][2]},{total[i][3]},{total[i][4]},{total[i][5]},{total[i][6]},{total[i][7]},{total[i][8]},{total[i][9]},{total[i][10]},{total[i][11]}"
#                      f",{total[i][12]},{total[i][13]},{total[i][14]},{total[i][15]},{total[i][16]},{total[i][17]},"
#                      f"{total[i][18]},{total[i][19]},{total[i][20]},{total[i][21]},{total[i][22]},{total[i][23]},{total[i][24]},{total[i][25]},{total[i][26]},"
#                      f"{total[i][27]},{total[i][28]},{total[i][29]},{total[i][30]},{total[i][31]})")
#         if total[i][1] == 'B':
#             gdbc.execute(f"insert into real_test_y values(1.0)")
#         elif total[i][1] == 'M':
#             gdbc.execute(f"insert into real_test_y values(0.0)")