# Conditional-RETAIN

Conditional RETAIN Model which can be fed both static varaible and time series data.

Original article: https://mhealth.jmir.org/2021/3/e22183 

built by Tensorflow 2.x


![model](https://user-images.githubusercontent.com/45510932/113866095-dc0cd480-97e7-11eb-89fe-7d3f650fff99.PNG)



#### How to get model and fit ?
```python3 

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop ,SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from eXplainableAI.transparent_model.RNN import ConditionalRETAIN
from eXplainableAI.transparent_model.RNN.interpretation import Interpreter


#### Build model
config = {'n_features':vars_,
          'n_auxs':9,
          'steps':16,
          'hidden_units': 20
          }

conditional_retain = ConditionalRETAIN(config)
retain = conditional_retain.build_model()
retain.compile(optimizer=RMSprop(lr=0.0005), loss='mse', metrics=['mse'])
retain.summary()


#### 
history = retain.fit(x=[X_train_var, X_train_inv], 
                     y=Y_train,   
                     batch_size=16*5,
                     epochs=500,
                     validation_split=0.3)
```


---------------------------------------------------------------------
#### Interpretation

Consistent with RETAIN, interpretation of the weight prediction model involves getting the time-level attention weighting and variable-level attention weighting from each RNN gi, hi. Therefore, the equation can be rewritten, and the contributions of the predicted value of the model can be calculated as follows:

![image](https://user-images.githubusercontent.com/45510932/113866959-e67b9e00-97e8-11eb-8907-39e1428c90ea.png)

Therefore, the contribution of the Nth time-variant variable ω(ŷi, xjN)(T) and that of the mth time-fixed variable ω(ŷi, xm)(F) can also be written as

![image](https://user-images.githubusercontent.com/45510932/113867040-014e1280-97e9-11eb-9374-d5319f90ef81.png)

where each contribution coefficient of the time-variant variable is , and each contribution coefficient of the time-fixed variable is .


```
y_predict = model.predict(x=[X_test_var, X_test_inv])

# alpha (time level attention weight)
interpreter = Interpreter(model, X_test_var, X_test_inv, Y_test)
alpha = np.array(interpreter.get_model_weight('alpha'))

alpha_mean = alpha.reshape(-1, 16, 16, 1)[:, -1, :, :].mean(axis=0)
print('Alpha shape: ', alpha_mean.shape)

fig, axes = plt.subplots(1, 1, figsize=(14, 6))
sns.heatmap(alpha_mean.T, annot=True, fmt='.4f', xticklabels=list(range(1, 17)))
```

---------------------------------------------------------------
#### Contribution in timeseries part

```
beta = np.array(interpreter.get_model_weight('beta'))
beta = beta.reshape(-1, 16, 16, len(vars_))
beta_mean = beta[:, 15, :, :].mean(axis=0)

fig, axes = plt.subplots(1, 1, figsize=(13, 6))
sns.heatmap(beta_mean.T, 
            cmap='coolwarm', 
            annot=True, 
            vmin=-1, 
            vmax=1, 
            xticklabels=list(range(1, 17)),
            yticklabels=vars_
           )
plt.xticks(rotation=45)
```

![Contribution](https://user-images.githubusercontent.com/45510932/113866093-db743e00-97e7-11eb-9c67-c6a989befb36.PNG)



------------------------------------------------------------
####  Weigth in W matrix (fusion layer)

```python3

W, bias = interpreter.get_model_weight('weight')
W.reshape(-1, W.shape[0])[0][2:4] = W.reshape(-1, W.shape[0])[0][2:4] * 1000  # # calories, steps * 1K당으로 변경


fig, axes = plt.subplots(1, 1, figsize=(18, 6))
sns.heatmap(W.reshape(-1, W.shape[0]), cmap='coolwarm', fmt='.3f', annot=True, 
            xticklabels=vars_+inv_,
            vmin=-1, vmax=1)
            
```

![w](https://user-images.githubusercontent.com/45510932/113866090-da431100-97e7-11eb-8658-7bf2311df2b9.PNG)



#### Overall contribution coefficient

```
fig, axes = plt.subplots(1, 1, figsize=(14, 7))

W_time = W[:len(vars_)]
contr_coef = W_time.ravel() * (alpha_mean * beta_mean)

sns.heatmap(contr_coef.T, cmap='coolwarm', annot=True, fmt='.3f',
            yticklabels=vars_,
            xticklabels=list(range(1, 17)),
            vmin=-0.06, 
            vmax=0.06)
            
```


-------- 
### Requirement
The latest vesrion  of this repo uses TF Keras, so you only need TF 2.0+ installed 

tensorflow 2.x 


#### Install
If you want to download this code, use following commands:
```bash
$ git clone https://github.com/4pygmalion/Conditional-RETAIN.git
```
