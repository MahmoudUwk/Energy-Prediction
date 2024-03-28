import scipy.io as sio
import numpy as np
from keras.layers import BatchNormalization,Dense,Dropout,BatchNormalization
from keras.models import  Sequential #,load_model
from keras.optimizers import Adam,RMSprop
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from preprocess_data import RMSE,MAE,MAPE,get_SAMFOR_data,log_results

option = 2
datatype_opt=0
seq=6
X_train,y_train,X_test,y_test,save_path = get_SAMFOR_data(option,datatype_opt,seq)

print(X_train.shape)
class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_loss', this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights= self.model.get_weights()


def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))


#%%
out_size = 1
# define the keras model
n = 2**8
# dr_rate = 0.1
model = Sequential()
model.add(Dense(n, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(n, activation='relu'))
model.add(Dense(out_size))

# compile the keras model

checkpoint = SaveBestModel()
callbacks_list = [checkpoint]


model.compile(loss='mse', optimizer='adam')


# fit the keras model on the dataset
history = model.fit(X_train, y_train, epochs=800, batch_size=2048, validation_split =0.2,callbacks=callbacks_list)
model.set_weights(checkpoint.best_weights)
best_epoch = np.argmin(history.history['val_loss'])
print("best epoch:",best_epoch)
#%%

y_test_pred = model.predict(X_test)

plt.figure(figsize=(10,5))
plt.plot(y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
plt.savefig(os.path.join(save_path,'RFR.png'))
# print('RF RMSE:',mean_squared_error(y_test,y_test_pred))
rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)
print(rmse,mae,mape)
#%%
alg_name = 'MLP'
row = [alg_name,rmse,mae,mape,seq]
log_results(row,datatype_opt,save_path)


