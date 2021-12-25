from gdbc import GDBC
gdbc = GDBC()
gdbc.connect()
gdbc.execute("drop table if exists real_puf_x")
gdbc.execute("drop table if exists real_puf_y")
gdbc.execute("drop table if exists real_test_puf_x")
gdbc.execute("drop table if exists real_test_puf_y")
gdbc.execute("create table real_puf_x(dim1 float,dim2 float,dim3 float,dim4 float,dim5 float,dim6 float,dim7 float,dim8 float,"
             "dim9 float,dim10 float,dim11 float,dim12 float,dim13 float,dim14 float,dim15 float,dim16 float,dim17 float,"
             "dim18 float,dim19 float,dim20 float,dim21 float,dim22 float,dim23 float,dim24 float,dim25 float,dim26 float,"
             "dim27 float,dim28 float,dim29 float,dim30 float,dim31 float,dim32 float,dim33 float,dim34 float,dim35 float,dim36 float,dim37 float,dim38 float,"
             "dim39 float,dim40 float,dim41 float,dim42 float,dim43 float,dim44 float,dim45 float,dim46 float,dim47 float,"
             "dim48 float,dim49 float,dim50 float,dim51 float,dim52 float,dim53 float,dim54 float,dim55 float,dim56 float,"
             "dim57 float,dim58 float,dim59 float,dim60 float,dim61 float,dim62 float,dim63 float,dim64 float)")
gdbc.execute("create table real_test_puf_x(dim1 float,dim2 float,dim3 float,dim4 float,dim5 float,dim6 float,dim7 float,dim8 float,"
             "dim9 float,dim10 float,dim11 float,dim12 float,dim13 float,dim14 float,dim15 float,dim16 float,dim17 float,"
             "dim18 float,dim19 float,dim20 float,dim21 float,dim22 float,dim23 float,dim24 float,dim25 float,dim26 float,"
             "dim27 float,dim28 float,dim29 float,dim30 float,dim31 float,dim32 float,dim33 float,dim34 float,dim35 float,"
             "dim36 float,dim37 float,dim38 float,dim39 float,dim40 float,dim41 float,dim42 float,dim43 float,dim44 float,dim45 float,dim46 float,dim47 float,"
             "dim48 float,dim49 float,dim50 float,dim51 float,dim52 float,dim53 float,dim54 float,dim55 float,dim56 float,"
             "dim57 float,dim58 float,dim59 float,dim60 float,dim61 float,dim62 float,dim63 float,dim64 float)")
gdbc.execute("create table real_puf_y(dim1 float)")
gdbc.execute("create table real_test_puf_y(dim1 float)")
with open('train_data.txt','r') as f:
    lines = f.readlines()
    total = []
    for i in range(len(lines)):
        if i < 640000:
            print(i)
            s = []
            ta = lines[i].split(',')
            for j in range(len(ta)):
                if j > 1:
                    s.append(float(eval(ta[j])))
                else:
                    s.append(ta[j])
            # total.append(s)
            gdbc.execute(f"insert into real_puf_x values({s[0]},{s[1]},{s[2]},{s[3]},{s[4]},"
                         f"{s[5]},{s[6]},{s[7]},{s[8]},{s[9]},{s[10]},{s[11]}"
                         f",{s[12]},{s[13]},{s[14]},{s[15]},{s[16]},{s[17]},"
                         f"{s[18]},{s[19]},{s[20]},{s[21]},{s[22]},{s[23]},"
                         f"{s[24]},{s[25]},{s[26]},{s[27]},{s[28]},{s[29]},"
                         f"{s[30]},{s[31]},{s[32]},{s[33]},{s[34]},"
                         f"{s[35]},{s[36]},{s[37]},{s[38]},{s[39]},{s[40]},{s[41]}"
                         f",{s[42]},{s[43]},{s[44]},{s[45]},{s[46]},{s[47]},"
                         f"{s[48]},{s[49]},{s[50]},{s[51]},{s[52]},{s[53]},"
                         f"{s[54]},{s[55]},{s[56]},{s[57]},{s[58]},{s[59]},"
                         f"{s[60]},{s[61]},{s[62]},{s[63]})")
            gdbc.execute(f"insert into real_puf_y values({s[64]})")
        else:
            break
with open('test_data.txt', 'r') as f:
    lines = f.readlines()
    total = []
    for a in lines:
        if lines.index(a) < 10000:
            print(lines.index(a))
            s = []
            ta = a.split(',')
            for j in range(len(ta)):
                if j > 1:
                    s.append(float(eval(ta[j])))
                else:
                    s.append(ta[j])
            # total.append(s)
            gdbc.execute(f"insert into real_test_puf_x values({s[0]},{s[1]},{s[2]},{s[3]},{s[4]},"
                         f"{s[5]},{s[6]},{s[7]},{s[8]},{s[9]},{s[10]},{s[11]}"
                         f",{s[12]},{s[13]},{s[14]},{s[15]},{s[16]},{s[17]},"
                         f"{s[18]},{s[19]},{s[20]},{s[21]},{s[22]},{s[23]},"
                         f"{s[24]},{s[25]},{s[26]},{s[27]},{s[28]},{s[29]},"
                         f"{s[30]},{s[31]},{s[32]},{s[33]},{s[34]},"
                         f"{s[35]},{s[36]},{s[37]},{s[38]},{s[39]},{s[40]},{s[41]}"
                         f",{s[42]},{s[43]},{s[44]},{s[45]},{s[46]},{s[47]},"
                         f"{s[48]},{s[49]},{s[50]},{s[51]},{s[52]},{s[53]},"
                         f"{s[54]},{s[55]},{s[56]},{s[57]},{s[58]},{s[59]},"
                         f"{s[60]},{s[61]},{s[62]},{s[63]})")
            gdbc.execute(f"insert into real_test_puf_y values({s[64]})")
        else:
            break

