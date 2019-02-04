from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import Basic_demo.ntu_date_preprocess_2 as ntu



model = models.Sequential()
# model.add(layers.Flatten(input_shape=(50,25,3)))
model.add(layers.SimpleRNN(

    units = 1,
    batch_input_shape=(None,50,2),

    # input_shape=(25,3),
    # input_length

    # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,



))
model.add(layers.Dense(1024,activation='relu' ))#这个shape好像是抛去了数据的第一维度，认为这个
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()