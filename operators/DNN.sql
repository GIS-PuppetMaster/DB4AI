operator FCLayer(x,hid_units,output){
    select SHAPE(x)[1] as feature_num
    create tensor w(feature_num, hid_units) from random((feature_num, hid_units),(0,1))
    create tensor b(hid_units,) from zeros((hid_units,))
    select MATMUL(x,w)+b as output
}

operator