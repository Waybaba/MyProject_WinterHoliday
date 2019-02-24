for i in range(25):
    print("x_"+str(i)+" = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': "+str(i)+"})(x)")