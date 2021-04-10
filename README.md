# Conditional-RETAIN

Conditional RETAIN Model 
Original article: https://mhealth.jmir.org/2021/3/e22183 
built by Tensorflow 2.x


![model](https://user-images.githubusercontent.com/45510932/113866095-dc0cd480-97e7-11eb-89fe-7d3f650fff99.PNG)

### Requirement
The latest vesrion  of this repo uses TF Keras, so you only need TF 2.0+ installed  
tensorflow 2.x 


---------------------------------------------------------------------
Interpretation

Consistent with RETAIN, interpretation of the weight prediction model involves getting the time-level attention weighting and variable-level attention weighting from each RNN gi, hi. Therefore, the equation can be rewritten, and the contributions of the predicted value of the model can be calculated as follows:

![image](https://user-images.githubusercontent.com/45510932/113866959-e67b9e00-97e8-11eb-8907-39e1428c90ea.png)

Therefore, the contribution of the Nth time-variant variable ω(ŷi, xjN)(T) and that of the mth time-fixed variable ω(ŷi, xm)(F) can also be written as

![image](https://user-images.githubusercontent.com/45510932/113867040-014e1280-97e9-11eb-9374-d5319f90ef81.png)

where each contribution coefficient of the time-variant variable is , and each contribution coefficient of the time-fixed variable is .


---------------------------------------------------------------
Contribution in timeseries part

![Contribution](https://user-images.githubusercontent.com/45510932/113866093-db743e00-97e7-11eb-9c67-c6a989befb36.PNG)


------------------------------------------------------------
Weigth in W matrix (fusion layer)

![w](https://user-images.githubusercontent.com/45510932/113866090-da431100-97e7-11eb-8658-7bf2311df2b9.PNG)



-------- 
If you want to download this code, use following commands:
```bash
$ git clone https://github.com/4pygmalion/Conditional-RETAIN.git
```
