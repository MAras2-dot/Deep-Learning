import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
##from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
##from sklearn.utils import shuffle

tf.compat.v1.disable_eager_execution()


def read_dataset():
    df = pd.read_excel("BankNote.xlsx")
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.rename({'class': 'labels'}, axis=1)
    ##print(df.columns)
    ##df.info()

    X = df[df.columns[0:4]].values
    y1 = df[df.columns[4]]

    ##encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y1)
    y = encoder.transform(y1)
    Y = one_hot_encode(y)
    print('TOTAL', X.shape)
    return (X, Y, y1)


##Define teh encoder function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


##Read the dataset
X, Y, y1 = read_dataset()

##shuffle the dataset to mix the rows
##X, Y = shuffle(X, Y, random_state=1)

##convert the dataset into train and test
#train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=1)
#print(train_x.shape)
#print(train_y.shape)
#print(test_x.shape)

##DEFINE THE IMPORTANT PARAMATERS AND VARIABLE TO WORK WITH TENSORS
learning_rate = 0.3  ##steps in which weights will be updated
training_epochs = 100  ##number of iterations
cost_history = np.empty(shape=[1], dtype=float)  ##empty numpy array
n_dim = 4  ##x shape of axis 1-volume
##print('n_dim', n_dim)
n_class = 2  ##only 2 class  fake or real
model_path = "/Users/mubinaarastu/PycharmProjects/DeepLearning/BankNote"

##define the number of hidden layers and number of neurons for each layer
n_hidden_1 = 10  ##10 neurons in every hidden layer
n_hidden_2 = 10  ##10 neurons in every hidden layer
n_hidden_3 = 10  ##10 neurons in every hidden layer
n_hidden_4 = 10  ##10 neurons in every hidden layer

x = tf.compat.v1.placeholder(tf.float32, [None,
                                          n_dim])  ###shape of this placeholder will be none and n_dim and none can be any value
W = tf.Variable(tf.zeros([n_dim, n_class]))  ##initialize with zero and shape will be of n_dim and n_class
b = tf.Variable(tf.zeros([n_class]))  ###biases shape will be n_class
y_ = tf.compat.v1.placeholder(tf.float32, [None,
                                           n_class])  ###this placeholder will be used to provide us the actual output of the model


## so there will be one actual output and one model output

# define the model
def multilayer_perceptron(x, weights, biases):
    # hidden layer with RELU activations
    layer_1 = tf.add(tf.matmul(x, weights['h1']),
                     biases['b1'])  ##matrix multiplication of x and weight of h1 is hiddenlayer 1
    layer_1 = tf.nn.sigmoid(layer_1)

    # hidden layer with sigmoid activations
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # hidden layer with sigmoid activations
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # hidden layer with RELU activations
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.sigmoid(layer_4)

    ##output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


###define weights and biases for ach layer
weights = {
    'h1': tf.Variable(tf.compat.v1.random.truncated_normal([n_dim, n_hidden_1])),  ##shape of n_dim and hidden 1
    'h2': tf.Variable(tf.compat.v1.random.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.compat.v1.random.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.compat.v1.random.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.compat.v1.random.truncated_normal([n_hidden_4, n_class])),
}

biases = {
    'b1': tf.Variable(tf.compat.v1.random.truncated_normal([n_hidden_1])),  ##shapes
    'b2': tf.Variable(tf.compat.v1.random.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.compat.v1.random.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.compat.v1.random.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.compat.v1.random.truncated_normal([n_class])),
}

##initialize all the variables
init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()

## call your model defined
y = multilayer_perceptron(x, weights, biases)

###define teh cost function and optimizer
##This is the error calculated from actual output and the model output - y -actual and y_model output
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

##this is to reduce the error
training_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.compat.v1.Session()
sess.run(init)
saver.restore(sess, model_path)

###calcualte the cost and accuracy of each epoch
mse_history = []
accuracy_history = []


## Print the final mean accuracy
prediction = tf.argmax(y, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Print(accuracy_run)
print('*****************************')
print('0 stands for Fake Note & 1 stands for Real Note')
print('*****************************')
for i in range(554,790):
    prediction_run = sess.run(prediction, feed_dict={x: X[i].reshape(1, 4)})
    accuracy_run = sess.run(accuracy, feed_dict={x: X[i].reshape(1, 4), y_: Y[i].reshape(1, 2)})
    print("original class: ", y1[i], "Predicted values :", prediction_run,"Accuracy :", accuracy_run)
