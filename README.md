
# LSTMs for Human Activity Recognition

Human activity recognition using smartphones dataset and an LSTM RNN. Classifying the type of movement amongst six categories:
- WALKING,
- WALKING_UPSTAIRS,
- WALKING_DOWNSTAIRS,
- SITTING,
- STANDING,
- LAYING.

Compared to a classical approach, using a Recurrent Neural Networks (RNN) with Long Short-Term Memory cells (LSTMs) require no or almost no feature engineering. Data can be fed directly into the neural network who acts like a black box, modeling the problem correctly. Other research on the activity recognition dataset used mostly use a big amount of feature engineering, which is rather a signal processing approach combined with classical data science techniques. The approach here is rather very simple in terms of how much did the data was preprocessed. 

## Video dataset overview

Follow this link to see a video of the 6 activities recorded in the experiment with one of the participants:

<p align="center">
  <a href="http://www.youtube.com/watch?feature=player_embedded&v=XOEN9W05_4A
" target="_blank"><img src="http://img.youtube.com/vi/XOEN9W05_4A/0.jpg" 
alt="Video of the experiment" width="400" height="300" border="10" /></a>
  <a href="https://youtu.be/XOEN9W05_4A"><center>[Watch video]</center></a>
</p>

## Details about input data

I will be using an LSTM on the data to learn (as a cellphone attached on the waist) to recognise the type of activity that the user is doing. The dataset's description goes like this:

> The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. 

That said, I will use the almost raw data: only the gravity effect has been filtered out of the accelerometer  as a preprocessing step for another 3D feature as an input to help learning. 

## What is an RNN?

As explained in [this article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), an RNN takes many input vectors to process them and output other vectors. It can be roughly pictured like in the image below, imagining each rectangle has a vectorial depth and other special hidden quirks in the image below. **In our case, the "many to one" architecture is used**: we accept time series of feature vectors (one vector per time step) to convert them to a probability vector at the output for classification. Note that a "one to one" architecture would be a standard feedforward neural network. 

<img src="http://karpathy.github.io/assets/rnn/diags.jpeg" />

An LSTM is an improved RNN. It is more complex, but easier to train, avoiding what is called the vanishing gradient problem. 


## Results 

Scroll on! Nice visuals awaits. 


```python
# All Includes

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics

import os
```


```python
# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING", 
    "STANDING", 
    "LAYING"
] 

```

## Let's start by downloading the data: 


```python
# Note: Linux bash commands start with a "!" inside those "ipython notebook" cells

DATA_PATH = "data/"

!pwd && ls
os.chdir(DATA_PATH)
!pwd && ls

!python download_dataset.py

!pwd && ls
os.chdir("..")
!pwd && ls

DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
print("\n" + "Dataset is now located at: " + DATASET_PATH)

```

    /home/sowmya/LSTM-Human-Activity-Recognition-master
    1.png	 LSTM_files  README.md
    data	 LSTM.ipynb  Screenshot from 2017-11-04 20-24-29.png
    LICENSE  lstm.py     Untitled.ipynb
    /home/sowmya/LSTM-Human-Activity-Recognition-master/data
    download_dataset.py  source.txt  UCI HAR Dataset
    
    Downloading...
    --2017-11-04 20:42:40--  https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
    Resolving archive.ics.uci.edu (archive.ics.uci.edu)... 128.195.10.249
    Connecting to archive.ics.uci.edu (archive.ics.uci.edu)|128.195.10.249|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 60999314 (58M) [application/zip]
    Saving to: ‘UCI HAR Dataset.zip’
    
    UCI HAR Dataset.zip 100%[===================>]  58.17M  4.97MB/s    in 16s     
    
    2017-11-04 20:42:59 (3.60 MB/s) - ‘UCI HAR Dataset.zip’ saved [60999314/60999314]
    
    Downloading done.
    
    Extracting...
    Dataset already extracted. Did not extract twice.
    
    /home/sowmya/LSTM-Human-Activity-Recognition-master/data
    download_dataset.py  source.txt  UCI HAR Dataset  UCI HAR Dataset.zip
    /home/sowmya/LSTM-Human-Activity-Recognition-master
    1.png	 LSTM_files  README.md
    data	 LSTM.ipynb  Screenshot from 2017-11-04 20-24-29.png
    LICENSE  lstm.py     Untitled.ipynb
    
    Dataset is now located at: data/UCI HAR Dataset/


## Preparing dataset:


```python
TRAIN = "train/"
TEST = "test/"


# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []
    
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()
    
    return np.transpose(np.array(X_signals), (1, 2, 0))

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)


# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]], 
        dtype=np.int32
    )
    file.close()
    
    # Substract 1 to each output class for friendly 0-based indexing 
    return y_ - 1

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

```

## Additionnal Parameters:

Here are some core parameter definitions for the training. 

The whole neural network's structure could be summarised by enumerating those parameters and the fact an LSTM is used. 


```python
# Input Data 

training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep


# LSTM Neural Network's internal structure

n_hidden = 32 # Hidden layer num of features
n_classes = 6 # Total classes (should go up, or should go down)


# Training 

learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset
batch_size = 1500
display_iter = 30000  # To show test set accuracy during training


# Some debugging info

print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

```

    Some useful info to get an insight on dataset's shape and normalisation:
    (X shape, y shape, every X's mean, every X's standard deviation)
    (2947, 128, 9) (2947, 1) 0.0991399 0.395671
    The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.


## Utility functions for training:


```python
def LSTM_RNN(_X, _weights, _biases):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters. 
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network. 
    # Note, some code of this notebook is inspired from an slightly different 
    # RNN architecture used on another dataset, some of the credits goes to 
    # "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) 
    # new shape: (n_steps*batch_size, n_input)
    
    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier, 
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]
    
    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s


def one_hot(y_):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

```

## Let's get serious and build the neural network:


```python

# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = LSTM_RNN(x, weights, biases)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

```

## Hooray, now train the neural network:


```python
# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
while step * batch_size <= training_iters:
    batch_xs =         extract_batch_size(X_train, step, batch_size)
    batch_ys = one_hot(extract_batch_size(y_train, step, batch_size))

    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs, 
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        
        # To not spam console, show training accuracy/loss in this "if"
        print("Training iter #" + str(step*batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))
        
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy], 
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

print("Optimization Finished!")

# Accuracy for test data

one_hot_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))

```

    Training iter #1500:   Batch Loss = 2.848186, Accuracy = 0.19866666197776794
    PERFORMANCE ON TEST SET: Batch Loss = 2.5961782932281494, Accuracy = 0.20427553355693817
    Training iter #30000:   Batch Loss = 1.377615, Accuracy = 0.6853333115577698
    PERFORMANCE ON TEST SET: Batch Loss = 1.4199628829956055, Accuracy = 0.6409908533096313
    Training iter #60000:   Batch Loss = 1.117074, Accuracy = 0.7926666736602783
    PERFORMANCE ON TEST SET: Batch Loss = 1.40687894821167, Accuracy = 0.7051238417625427
    Training iter #90000:   Batch Loss = 1.002202, Accuracy = 0.8473333120346069
    PERFORMANCE ON TEST SET: Batch Loss = 1.1183960437774658, Accuracy = 0.814048171043396
    Training iter #120000:   Batch Loss = 0.957689, Accuracy = 0.8346666693687439
    PERFORMANCE ON TEST SET: Batch Loss = 1.0718357563018799, Accuracy = 0.8530709147453308
    Training iter #150000:   Batch Loss = 0.718897, Accuracy = 0.9613333344459534
    PERFORMANCE ON TEST SET: Batch Loss = 1.0000059604644775, Accuracy = 0.861214816570282
    Training iter #180000:   Batch Loss = 0.789522, Accuracy = 0.9113333225250244
    PERFORMANCE ON TEST SET: Batch Loss = 0.9518797397613525, Accuracy = 0.8778418898582458
    Training iter #210000:   Batch Loss = 0.749568, Accuracy = 0.9113333225250244
    PERFORMANCE ON TEST SET: Batch Loss = 0.9289544820785522, Accuracy = 0.882253110408783
    Training iter #240000:   Batch Loss = 0.641241, Accuracy = 0.9693333506584167
    PERFORMANCE ON TEST SET: Batch Loss = 0.9079854488372803, Accuracy = 0.88802170753479
    Training iter #270000:   Batch Loss = 0.651092, Accuracy = 0.9526666402816772
    PERFORMANCE ON TEST SET: Batch Loss = 0.9187182784080505, Accuracy = 0.8849677443504333
    Training iter #300000:   Batch Loss = 0.920353, Accuracy = 0.8806666731834412
    PERFORMANCE ON TEST SET: Batch Loss = 0.9059563875198364, Accuracy = 0.8849677443504333
    Training iter #330000:   Batch Loss = 0.740811, Accuracy = 0.9233333468437195
    PERFORMANCE ON TEST SET: Batch Loss = 1.1896973848342896, Accuracy = 0.842212438583374
    Training iter #360000:   Batch Loss = 0.640513, Accuracy = 0.9473333358764648
    PERFORMANCE ON TEST SET: Batch Loss = 0.8404675722122192, Accuracy = 0.8802171945571899
    Training iter #390000:   Batch Loss = 0.636197, Accuracy = 0.9359999895095825
    PERFORMANCE ON TEST SET: Batch Loss = 0.7898554801940918, Accuracy = 0.9043094515800476
    Training iter #420000:   Batch Loss = 0.584301, Accuracy = 0.9453333616256714
    PERFORMANCE ON TEST SET: Batch Loss = 0.7631163597106934, Accuracy = 0.9127926826477051
    Training iter #450000:   Batch Loss = 0.579528, Accuracy = 0.9480000138282776
    PERFORMANCE ON TEST SET: Batch Loss = 0.7695472240447998, Accuracy = 0.9114353656768799
    Training iter #480000:   Batch Loss = 0.561116, Accuracy = 0.9426666498184204
    PERFORMANCE ON TEST SET: Batch Loss = 0.7722400426864624, Accuracy = 0.9046487808227539
    Training iter #510000:   Batch Loss = 0.527683, Accuracy = 0.9706666469573975
    PERFORMANCE ON TEST SET: Batch Loss = 0.7183223366737366, Accuracy = 0.9087207317352295
    Training iter #540000:   Batch Loss = 0.559444, Accuracy = 0.940666675567627
    PERFORMANCE ON TEST SET: Batch Loss = 0.7261930704116821, Accuracy = 0.9077027440071106
    Training iter #570000:   Batch Loss = 0.564470, Accuracy = 0.9300000071525574
    PERFORMANCE ON TEST SET: Batch Loss = 0.7395513653755188, Accuracy = 0.9039701223373413
    Training iter #600000:   Batch Loss = 0.544714, Accuracy = 0.9359999895095825
    PERFORMANCE ON TEST SET: Batch Loss = 0.7233500480651855, Accuracy = 0.9083814024925232
    Training iter #630000:   Batch Loss = 0.466222, Accuracy = 0.9806666374206543
    PERFORMANCE ON TEST SET: Batch Loss = 0.7295655012130737, Accuracy = 0.9110960364341736
    Training iter #660000:   Batch Loss = 0.458085, Accuracy = 0.984666645526886
    PERFORMANCE ON TEST SET: Batch Loss = 0.7116966247558594, Accuracy = 0.910417377948761
    Training iter #690000:   Batch Loss = 0.432223, Accuracy = 0.981333315372467
    PERFORMANCE ON TEST SET: Batch Loss = 0.7369370460510254, Accuracy = 0.8998982310295105
    Training iter #720000:   Batch Loss = 0.555358, Accuracy = 0.9300000071525574
    PERFORMANCE ON TEST SET: Batch Loss = 0.7651142477989197, Accuracy = 0.8751272559165955
    Training iter #750000:   Batch Loss = 0.511240, Accuracy = 0.937333345413208
    PERFORMANCE ON TEST SET: Batch Loss = 0.6970201134681702, Accuracy = 0.8975229263305664
    Training iter #780000:   Batch Loss = 0.412025, Accuracy = 0.9713333249092102
    PERFORMANCE ON TEST SET: Batch Loss = 0.7389575839042664, Accuracy = 0.8924329876899719
    Training iter #810000:   Batch Loss = 0.436633, Accuracy = 0.949999988079071
    PERFORMANCE ON TEST SET: Batch Loss = 0.6819406747817993, Accuracy = 0.9093993902206421
    Training iter #840000:   Batch Loss = 0.470395, Accuracy = 0.9346666932106018
    PERFORMANCE ON TEST SET: Batch Loss = 0.6448527574539185, Accuracy = 0.9127926826477051
    Training iter #870000:   Batch Loss = 0.410217, Accuracy = 0.9706666469573975
    PERFORMANCE ON TEST SET: Batch Loss = 0.6678794026374817, Accuracy = 0.910417377948761
    Training iter #900000:   Batch Loss = 0.383603, Accuracy = 0.9773333072662354
    PERFORMANCE ON TEST SET: Batch Loss = 0.667729377746582, Accuracy = 0.9114353656768799
    Training iter #930000:   Batch Loss = 0.447928, Accuracy = 0.9240000247955322
    PERFORMANCE ON TEST SET: Batch Loss = 0.6427401900291443, Accuracy = 0.907024085521698
    Training iter #960000:   Batch Loss = 0.445182, Accuracy = 0.9346666932106018
    PERFORMANCE ON TEST SET: Batch Loss = 0.6688382625579834, Accuracy = 0.9097387194633484
    Training iter #990000:   Batch Loss = 0.373859, Accuracy = 0.9673333168029785
    PERFORMANCE ON TEST SET: Batch Loss = 0.6101267337799072, Accuracy = 0.8968442678451538
    Training iter #1020000:   Batch Loss = 0.372702, Accuracy = 0.9826666712760925
    PERFORMANCE ON TEST SET: Batch Loss = 0.6599312424659729, Accuracy = 0.8832710981369019
    Training iter #1050000:   Batch Loss = 0.357185, Accuracy = 0.9806666374206543
    PERFORMANCE ON TEST SET: Batch Loss = 0.5937492847442627, Accuracy = 0.9015948176383972
    Training iter #1080000:   Batch Loss = 0.380983, Accuracy = 0.9753333330154419
    PERFORMANCE ON TEST SET: Batch Loss = 0.613731324672699, Accuracy = 0.8998982310295105
    Training iter #1110000:   Batch Loss = 0.414877, Accuracy = 0.9380000233650208
    PERFORMANCE ON TEST SET: Batch Loss = 0.6531519889831543, Accuracy = 0.8968442678451538
    Training iter #1140000:   Batch Loss = 0.396607, Accuracy = 0.9399999976158142
    PERFORMANCE ON TEST SET: Batch Loss = 0.6406139135360718, Accuracy = 0.8958262801170349
    Training iter #1170000:   Batch Loss = 0.632654, Accuracy = 0.7973333597183228
    PERFORMANCE ON TEST SET: Batch Loss = 0.7535757422447205, Accuracy = 0.8184594511985779
    Training iter #1200000:   Batch Loss = 0.519090, Accuracy = 0.8999999761581421
    PERFORMANCE ON TEST SET: Batch Loss = 0.6180902719497681, Accuracy = 0.8744485974311829
    Training iter #1230000:   Batch Loss = 0.407589, Accuracy = 0.9279999732971191
    PERFORMANCE ON TEST SET: Batch Loss = 0.5741686820983887, Accuracy = 0.8802171945571899
    Training iter #1260000:   Batch Loss = 0.484683, Accuracy = 0.9279999732971191
    PERFORMANCE ON TEST SET: Batch Loss = 0.5544700026512146, Accuracy = 0.9110960364341736
    Training iter #1290000:   Batch Loss = 0.412714, Accuracy = 0.9293333292007446
    PERFORMANCE ON TEST SET: Batch Loss = 0.5418837070465088, Accuracy = 0.8985409140586853
    Training iter #1320000:   Batch Loss = 0.366800, Accuracy = 0.9453333616256714
    PERFORMANCE ON TEST SET: Batch Loss = 0.5303409099578857, Accuracy = 0.9185612201690674
    Training iter #1350000:   Batch Loss = 0.334662, Accuracy = 0.9673333168029785
    PERFORMANCE ON TEST SET: Batch Loss = 0.5346779823303223, Accuracy = 0.9161859750747681
    Training iter #1380000:   Batch Loss = 0.307139, Accuracy = 0.9826666712760925
    PERFORMANCE ON TEST SET: Batch Loss = 0.5430952310562134, Accuracy = 0.9073634147644043
    Training iter #1410000:   Batch Loss = 0.301987, Accuracy = 0.9793333411216736
    PERFORMANCE ON TEST SET: Batch Loss = 0.5351461172103882, Accuracy = 0.9107567071914673
    Training iter #1440000:   Batch Loss = 0.324468, Accuracy = 0.968666672706604
    PERFORMANCE ON TEST SET: Batch Loss = 0.5404599905014038, Accuracy = 0.9110960364341736
    Training iter #1470000:   Batch Loss = 0.388961, Accuracy = 0.9446666836738586
    PERFORMANCE ON TEST SET: Batch Loss = 0.5519036054611206, Accuracy = 0.9022734761238098
    Training iter #1500000:   Batch Loss = 0.337279, Accuracy = 0.9573333263397217
    PERFORMANCE ON TEST SET: Batch Loss = 0.5351852178573608, Accuracy = 0.8968442678451538
    Training iter #1530000:   Batch Loss = 0.330516, Accuracy = 0.9606666564941406
    PERFORMANCE ON TEST SET: Batch Loss = 0.6136803030967712, Accuracy = 0.8642687201499939
    Training iter #1560000:   Batch Loss = 0.317524, Accuracy = 0.9526666402816772
    PERFORMANCE ON TEST SET: Batch Loss = 0.5499243140220642, Accuracy = 0.9005768299102783
    Training iter #1590000:   Batch Loss = 0.339586, Accuracy = 0.9453333616256714
    PERFORMANCE ON TEST SET: Batch Loss = 0.5004620552062988, Accuracy = 0.9029521346092224
    Training iter #1620000:   Batch Loss = 0.297041, Accuracy = 0.9653333425521851
    PERFORMANCE ON TEST SET: Batch Loss = 0.508324921131134, Accuracy = 0.9049881100654602
    Training iter #1650000:   Batch Loss = 0.357241, Accuracy = 0.9179999828338623
    PERFORMANCE ON TEST SET: Batch Loss = 0.4903225302696228, Accuracy = 0.9053274393081665
    Training iter #1680000:   Batch Loss = 0.340250, Accuracy = 0.9286666512489319
    PERFORMANCE ON TEST SET: Batch Loss = 0.4915405809879303, Accuracy = 0.910417377948761
    Training iter #1710000:   Batch Loss = 0.352242, Accuracy = 0.9393333196640015
    PERFORMANCE ON TEST SET: Batch Loss = 0.5156731009483337, Accuracy = 0.9077027440071106
    Training iter #1740000:   Batch Loss = 0.282967, Accuracy = 0.972000002861023
    PERFORMANCE ON TEST SET: Batch Loss = 0.467829167842865, Accuracy = 0.9073634147644043
    Training iter #1770000:   Batch Loss = 0.269097, Accuracy = 0.9793333411216736
    PERFORMANCE ON TEST SET: Batch Loss = 0.468751460313797, Accuracy = 0.9080420732498169
    Training iter #1800000:   Batch Loss = 0.250395, Accuracy = 0.981333315372467
    PERFORMANCE ON TEST SET: Batch Loss = 0.5110938549041748, Accuracy = 0.9083814024925232
    Training iter #1830000:   Batch Loss = 0.323069, Accuracy = 0.9480000138282776
    PERFORMANCE ON TEST SET: Batch Loss = 0.5227196216583252, Accuracy = 0.9127926826477051
    Training iter #1860000:   Batch Loss = 0.322357, Accuracy = 0.9446666836738586
    PERFORMANCE ON TEST SET: Batch Loss = 0.49156466126441956, Accuracy = 0.9077027440071106
    Training iter #1890000:   Batch Loss = 0.338317, Accuracy = 0.9386666417121887
    PERFORMANCE ON TEST SET: Batch Loss = 0.5058750510215759, Accuracy = 0.9056667685508728
    Training iter #1920000:   Batch Loss = 0.299987, Accuracy = 0.9486666917800903
    PERFORMANCE ON TEST SET: Batch Loss = 0.6349874138832092, Accuracy = 0.8717339634895325
    Training iter #1950000:   Batch Loss = 0.281814, Accuracy = 0.949999988079071
    PERFORMANCE ON TEST SET: Batch Loss = 0.45127978920936584, Accuracy = 0.9165253043174744
    Training iter #1980000:   Batch Loss = 0.280665, Accuracy = 0.9493333101272583
    PERFORMANCE ON TEST SET: Batch Loss = 0.4655972719192505, Accuracy = 0.9155073165893555
    Training iter #2010000:   Batch Loss = 0.233872, Accuracy = 0.9706666469573975
    PERFORMANCE ON TEST SET: Batch Loss = 0.4949379563331604, Accuracy = 0.8988802433013916
    Training iter #2040000:   Batch Loss = 0.287422, Accuracy = 0.9386666417121887
    PERFORMANCE ON TEST SET: Batch Loss = 0.5221925377845764, Accuracy = 0.8985409140586853
    Training iter #2070000:   Batch Loss = 0.283807, Accuracy = 0.9426666498184204
    PERFORMANCE ON TEST SET: Batch Loss = 0.5127118229866028, Accuracy = 0.9134713411331177
    Training iter #2100000:   Batch Loss = 0.299748, Accuracy = 0.937333345413208
    PERFORMANCE ON TEST SET: Batch Loss = 0.485989511013031, Accuracy = 0.9053274393081665
    Training iter #2130000:   Batch Loss = 0.245069, Accuracy = 0.9646666646003723
    PERFORMANCE ON TEST SET: Batch Loss = 0.4559471607208252, Accuracy = 0.9182218909263611


## Training is good, but having visual insight is even better:

Okay, let's plot this simply in the notebook for now.


```python
# (Inline plots: )
%matplotlib inline

font = {
    'family' : 'Bitstream Vera Sans',
    'weight' : 'bold',
    'size'   : 18
}
matplotlib.rc('font', **font)

width = 12
height = 12
plt.figure(figsize=(width, height))

indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

indep_test_axis = np.append(
    np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
    [training_iters]
)
plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Test losses")
plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training iteration')

plt.show()
```

## And finally, the multi-class confusion matrix and metrics!


```python
# Results

predictions = one_hot_predictions.argmax(1)

print("Testing Accuracy: {}%".format(100*accuracy))

print("")
print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

print("")
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)
print("Note: training and testing data is not equally distributed amongst classes, ")
print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

# Plot Results: 
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix, 
    interpolation='nearest', 
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
```


```python
sess.close()
```

## Conclusion

Outstandingly, **the final accuracy is of 91%**! And it can peak to values such as 92.73%, at some moments of luck during the training, depending on how the neural network's weights got initialized at the start of the training, randomly. 

This means that the neural networks is almost always able to correctly identify the movement type! Remember, the phone is attached on the waist and each series to classify has just a 128 sample window of two internal sensors (a.k.a. 2.56 seconds at 50 FPS), so those predictions are extremely accurate.

I specially did not expect such good results for guessing between "SITTING" and "STANDING". Those are seemingly almost the same thing from the point of view of a device placed at waist level according to how the dataset was gathered. Thought, it is still possible to see a little cluster on the matrix between those classes, which drifts away from the identity. This is great.

It is also possible to see that there was a slight difficulty in doing the difference between "WALKING", "WALKING_UPSTAIRS" and "WALKING_DOWNSTAIRS". Obviously, those activities are quite similar in terms of movements. 

I also tried my code without the gyroscope, using only the two 3D accelerometer's features (and not changing the training hyperparameters), and got an accuracy of 87%. In general, gyroscopes consumes more power than accelerometers, so it is preferable to turn them off. 


## Improvements

In [another open-source repository of mine](https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs), the accuracy is pushed up to 94% using a special deep LSTM architecture which combines the concepts of bidirectional RNNs, residual connections and stacked cells. This architecture is also tested on another similar activity dataset. It resembles to the architecture used in "[Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf)" without an attention mechanism and with just the encoder part - still as a "many to one" architecture which is adapted to the Human Activity Recognition (HAR) problem.

If you want to learn more about deep learning, I have also built a list of the learning ressources for deep learning which have revealed to be the most useful to me [here](https://github.com/guillaume-chevalier/awesome-deep-learning-resources). You could as well learn to [learn to learn by gradient descent by gradient descent](https://arxiv.org/pdf/1606.04474.pdf) (not for the faint of heart). Ok, I pushed the joke deep enough... 


## References

The [dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) can be found on the UCI Machine Learning Repository. 

> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.

To cite my work, point to the URL of the GitHub repository: 
> Guillaume Chevalier, LSTMs for Human Activity Recognition, 2016
> https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

My code is available under the [MIT License](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/LICENSE). 

## Connect with me

- https://ca.linkedin.com/in/chevalierg 
- https://twitter.com/guillaume_che
- https://github.com/guillaume-chevalier/



```python
# Let's convert this notebook to a README for the GitHub project's title page:
!jupyter nbconvert --to markdown LSTM.ipynb
!mv LSTM.md README.md
```
