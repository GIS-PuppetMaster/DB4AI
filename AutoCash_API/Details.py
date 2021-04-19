def details():
    classifiers = []
    classifier_dict = {}
    with open("HPO.txt", 'r') as fr:
        lines = fr.readlines()
    i = 0
    for line in lines:
        if i >= 1:
            line = line.rstrip("\n")
            full_name_of_classifiers = line.split('\t')[0]
            classifiers.append(full_name_of_classifiers)

            option_temp = line.split('\t')[1]
            options = []
            for j in range(len(option_temp.split(','))):
                options.append(option_temp.split(',')[j])

            options_type_temp = line.split('\t')[2]
            options_type = []
            for j in range(len(options_type_temp.split(','))):
                options_type.append(options_type_temp.split(',')[j])

            range_temp = line.split('\t')[3]
            range_hpo = []
            for j in range(len(range_temp.split(';'))):
                range_hpo.append(range_temp.split(';')[j])

            default_temp = line.split('\t')[4]
            default_hpo = []
            for j in range(len(default_temp.split(','))):
                default_hpo.append(default_temp.split(',')[j])

            precison_temp=line.split('\t')[5]
            precision=[]
            for j in range(len(precison_temp.split(','))):
                precision.append(precison_temp.split(',')[j])

            dict_temp = []
            dict_temp.append(options)
            dict_temp.append(options_type)
            dict_temp.append(range_hpo)
            dict_temp.append(default_hpo)
            dict_temp.append(precision)

            classifier_dict[full_name_of_classifiers] = dict_temp
        i = i + 1

    classifier_dict_EA = {}
    for classifier in classifiers:
        options = classifier_dict[classifier][0]
        options_type = classifier_dict[classifier][1]
        range_hpo = classifier_dict[classifier][2]
        default_hpo = classifier_dict[classifier][3]
        precision_temp=classifier_dict[classifier][4]

        lb = []
        ub = []
        precision=[]

        for i in range(len(options_type)):
            if options_type[i] == 'int':
                lb.append(int(range_hpo[i].split(',')[0]))
                ub.append(int(range_hpo[i].split(',')[1]))
                precision.append(int(precision_temp[i]))
            elif options_type[i] == 'float':
                lb.append(float(range_hpo[i].split(',')[0]))
                ub.append(float(range_hpo[i].split(',')[1]))
                precision.append(float(precision_temp[i]))
            elif options_type[i] == 'list':
                lb.append(0)
                ub.append(len(range_hpo[i].split(',')))
                precision.append(str(precision_temp[i]))

        dict_temp_EA = []
        dict_temp_EA.append(options)
        dict_temp_EA.append(lb)
        dict_temp_EA.append(ub)
        dict_temp_EA.append(default_hpo)
        dict_temp_EA.append(precision)

        classifier_dict_EA[classifier] = dict_temp_EA

    return classifiers,classifier_dict,classifier_dict_EA
