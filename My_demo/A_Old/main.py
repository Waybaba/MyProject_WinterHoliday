from keras.datasets import imdb


(train_date,train_lables),(test_date,test_lables)=imdb.load_data(num_words=1000)

train_date[0]