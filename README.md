# CON-SLU Pytorch

Pytorch implementation of contextual SLU models:

**m2mnet**

![image-20181120132345516](/Users/crluser/Library/Application Support/typora-user-images/image-20181120132345516.png)

**SDEN**

![Model](/Users/crluser/Desktop/NAACL/conslu/images/model.png "SDEN")

## 1.Data

### [kvret](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)

### [M2M_Simulated Dialogue](https://github.com/google-research-datasets/simulated-dialogue)

I have modified [Stanford Multi-turn dataset](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/) to fit this model. *So it has some noise especially slot tags.*
It consists of three domain, `Weather`, `Schedule`, `Navigate`. I did dialogue recombination for multi-domain dialogue and modified its format to BIO.

### sample

#### Single domain dialogue

```
User :  Will it be hot in Inglewood over the next few days?
BOT  :  It will be warm both Monday and Tuesday in Inglewood.
User :  Thank you very much.
BOT  :  You're welcome. Hope you have a great day.
```

#### Multi domain dialogue

```
User :  is it going to be raining this weekend
BOT  :  What city are you inquiring about?
User :  Alhambra please.
BOT  :  It will be raining on Saturday and hailing on Sunday in Alhambra.
User :  Thanks.
BOT  :  happy to help
User :  I need a gas station
BOT  :  I have one gas station listed. Want more info?
User :  What is the address?
BOT  :  76 is at 91 El Camino Real.
User :  Thank you!
BOT  :  You're welcome, stay safe.
```

## 2. requirement

### Environment

```
python 3.6
cuda 9.0
pytorch 0.4
```

### Packages

```
pip install fuzzywuzzy
pip install sklearn-crfsuite
```

## 3. Quick start

```shell
./train.sh 0 context_s2s
```

the first param 0 means choose GPU:0 for this model training process, and context_s2s is the model we want to train.

In this repository, we support s2s, sden, memnet, context_s2s four models.

## 4. Devset Result

`Intent Detection : 0.93804 (Accuracy)`

`Slot Extraction`

|                         | precision | recall | f1-score | support |
| ----------------------- | --------- | ------ | -------- | ------- |
| **B-address**           | 0         | 0      | 0        | 3       |
| **I-address**           | 0         | 0      | 0        | 6       |
| **B-agenda**            | 0         | 0      | 0        | 3       |
| **I-agenda**            | 0.8       | 0.5    | 0.615    | 8       |
| **B-date**              | 0.774     | 0.828  | 0.8      | 145     |
| **I-date**              | 0.076     | 0.854  | 0.139    | 103     |
| **B-distance**          | 0.693     | 0.859  | 0.767    | 92      |
| **I-distance**          | 0.361     | 0.393  | 0.376    | 56      |
| **B-event**             | 0.891     | 0.942  | 0.916    | 104     |
| **I-event**             | 0.932     | 0.75   | 0.831    | 92      |
| **B-location**          | 0.952     | 0.98   | 0.966    | 101     |
| **I-location**          | 0.811     | 0.977  | 0.887    | 44      |
| **B-party**             | 0.931     | 0.9    | 0.915    | 30      |
| **I-party**             | 1         | 0.571  | 0.727    | 7       |
| **B-poi**               | 0.692     | 0.562  | 0.621    | 32      |
| **I-poi**               | 0.006     | 0.111  | 0.011    | 18      |
| **B-poi_type**          | 0.831     | 0.852  | 0.841    | 81      |
| **I-poi_type**          | 0.407     | 1      | 0.578    | 50      |
| **B-room**              | 0         | 0      | 0        | 1       |
| **I-room**              | 0         | 0      | 0        | 1       |
| **B-time**              | 0.975     | 0.83   | 0.897    | 47      |
| **I-time**              | 0.933     | 0.933  | 0.933    | 45      |
| **B-traffic_info**      | 0.622     | 0.5    | 0.554    | 46      |
| **I-traffic_info**      | 0.679     | 0.581  | 0.626    | 62      |
| **B-weather_attribute** | 0.939     | 0.951  | 0.945    | 81      |
| **I-weather_attribute** | 0.818     | 1      | 0.9      | 9       |
| **avg / total**         | 0.716     | 0.803  | 0.725    | 1267    |

