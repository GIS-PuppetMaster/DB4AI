import CaculateFeatures as cf

#函数功能： 根据result.txt 生成对应的dataset(元特征代表文件）
def GenerateDataSetaction(featurelist):
    '''

    :param featurelist:选取的元特征对应在所有可供选用的元特征中的index
    :return:
    '''
    classIndex = cf.getIndex()
    with open("Customization_result.txt", 'r') as fr:
        lines = fr.readlines()

    for line in lines:
        # print(line.split('\t')[1].strip())
        filename = line.split('\t')[0]
        clf = line.split('\t')[1]
        try:
            features = cf.calculate(featurelist,
                                    filename, classIndex)

        except Exception as exp:
            print(filename + "有异常, 跳过\n")
            print(exp)
        else:
            a = ""
            with open("dataset1.txt", 'a+') as fw:
                for feature in features:
                    a += (str(feature) + ',')
                    # fw.write(str(feature))
                    # fw.write(',')
                fw.write(a.replace('nan', '0'))
                fw.write(clf)
            print(filename + "finished\n")
            continue






