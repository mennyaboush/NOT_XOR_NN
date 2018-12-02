import tensorflow as tf
import numpy
'''
        This code is written by Menny Aboush 203114798
'''


def neural_network(learning_rate, nb_hidden, short_cut):

    dim = 2
    nb_outputs = 1
    nb_epocs = 40000
    temp = 1  # Hyper Parameters

    #  values for training NOT XOR TT
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]  # n labeled XOR data
    y_train = [[1], [0], [0], [1]]

    #  values for validation
    x_validation = [[0, 0], [0, 1], [1, 0], [1, 1], [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]]
    y_validation = [[1], [0], [0], [1], [1], [0], [0], [1]]

    if short_cut:
        nb_hbridge = nb_hidden + dim                 # Bridge inputs to output (highway)
    else:
        nb_hbridge = nb_hidden

    x = tf.placeholder(tf.float32, [None, dim])      # define input placeholders and variables
    t = tf.placeholder(tf.float32, [None, 1])
    w1 = tf.Variable(tf.random_uniform([dim, nb_hidden], -1, 1), name="Weights1")  # random weights
    w2 = tf.Variable(tf.random_uniform([nb_hbridge, nb_outputs], -1, 1), name="Weights2")
    b1 = tf.Variable(tf.zeros([nb_hidden]), name="Biases1")   # biases are zeros (not random)
    b2 = tf.Variable(tf.zeros([nb_outputs]), name="Biases2")

    z1 = tf.matmul(x, w1) + b1             # Network (grah) definition
    hlayer1 = tf.sigmoid(z1/temp)
    if short_cut:
        hlayer1 = tf.concat([hlayer1, x], 1)
    z2 = tf.matmul(hlayer1, w2) + b2
    out = tf.sigmoid(z2/temp)

    loss = -tf.reduce_sum(t*tf.log(out)+(1-t)*tf.log(1-out))  # Xross Entropy Loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)  # Grad Descent Optimizer
    train = optimizer.minimize(loss)  # training is running optimizer to minimize loss



    # training and validaion
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    previous_val_loss = float('inf') #very large number
    counter_not_change = 0

    for i in range(nb_epocs):  # training iterations and evaluate test
        curr_train, curr_loss = sess.run([train, loss], {x: x_train, t: y_train})
        curr_out, curr_val_loss = sess.run([out, loss], {x: x_validation, t: y_validation})
        # check conditions
        if abs(curr_val_loss - previous_val_loss) < 0.0001 and curr_val_loss < 0.2:
            counter_not_change += 1
        else:
            counter_not_change = 0
        if counter_not_change == 10:
            return [i, curr_val_loss, curr_loss, True]
        previous_val_loss = curr_val_loss  # necessary check the delta

    return[i, curr_val_loss, curr_loss, False]
def get_data(list_of_data):
    ''' sum = 0
    std = 0
    for i in list_of_data:
        sum += i
    average = sum/10
    for i in list_of_data:
        std += (i-average)**2
    std /= 10
    std = (std**0.5)
    '''
    meanRes = numpy.mean(list_of_data)
    stdRes = numpy.std(list_of_data)
    return [meanRes, (stdRes / meanRes)*100] #[average, std]

def run_experiment(learning_rate,shortCut,nb_hidden):
    num_of_sucsses = 10
    counter_sucsses = 0
    counter_faild = 0
    list_of_epocs = []
    list_of_valLoss = []
    list_of_trainLoss = []
    result = []
    while counter_sucsses < num_of_sucsses:
        l = neural_network(learning_rate, shortCut, nb_hidden)
        if l[3] == True:
            list_of_epocs.append(l[0])
            list_of_valLoss.append(l[1])
            list_of_trainLoss.append(l[2])
            counter_sucsses += 1
        else:
            counter_faild += 1
    result.append(get_data(list_of_epocs))
    result.append(get_data(list_of_trainLoss))
    result.append(get_data(list_of_valLoss))
    result.append(counter_faild)
    return result

res = run_experiment(0.01, 4, True)
print("\nexperiment 1: hidden: 4, LR: 0.01, Bridge: True")
print("nean epocs: ", res[0][0], " std/epocs percent: ", res[0][1],"%", " failures: ", res[3],
      "\nmean train loss: ", res[1][0], "std train loss percent: ", res[1][1],
      "\nmean valid loss: ", res[2][0], "std valid loss percent: ", res[2][1])

res = run_experiment(0.01, 4, False)
print("experiment 2: hidden: 4, LR: 0.01, Bridge: False")
print("nean epocs: ", res[0][0], " std/epocs percent: ", res[0][1], "%", "failures: ", res[3],
      "\nmean train loss: ", res[1][0], "std train loss percent: ", res[1][1],
      "\nmean valid loss: ", res[2][0], "std valid loss percent: ", res[2][1])

res = run_experiment(0.1, 4, True)
print("experiment 3: hidden: 4, LR: 0.1, Bridge: True")
print("nean epocs: ", res[0][0]," std/epocs percent: ", res[0][1], "%", "failures: ", res[3],
      "\nmean train loss: ", res[1][0], "std train loss percent: ", res[1][1],
      "\nmean valid loss: ", res[2][0], "std valid loss percent: ", res[2][1])

res = run_experiment(0.1, 4, False)
print("experiment 4: hidden: 4, LR: 0.1, Bridge: False")
print("nean epocs: ", res[0][0], " std/epocs percent: ", res[0][1], "%",  "failures: ", res[3],
      "\nmean train loss: ", res[1][0], "std train loss percent: ", res[1][1],
      "\nmean valid loss: ", res[2][0], "std valid loss percent: ", res[2][1])

res = run_experiment(0.01, 2, True)
print("experiment 5: hidden: 2, LR: 0.01, Bridge: True")
print("nean epocs: ", res[0][0], " std/epocs percent: ", res[0][1], "%",  "failures: ", res[3],
      "\nmean train loss: ", res[1][0], "std train loss percent: ", res[1][1],
      "\nmean valid loss: ", res[2][0], "std valid loss percent: ", res[2][1])

res = run_experiment(0.01, 2, False)
print("experiment 6: hidden: 2, LR: 0.01, Bridge: False")
print("nean epocs: ", res[0][0], " std/epocs percent: ", res[0][1], "%",  "failures: ", res[3],
      "\nmean train loss: ", res[1][0], "std train loss percent: ", res[1][1],
      "\nmean valid loss: ", res[2][0], "std valid loss percent: ", res[2][1])

res = run_experiment(0.1, 2, True)
print("experiment 7: hidden: 2, LR: 0.1, Bridge: False")
print("nean epocs: ", res[0][0], " std/epocs percent: ", res[0][1], "%",  "failures: ", res[3],
      "\nmean train loss: ", res[1][0], "std train loss percent: ", res[1][1],
      "\nmean valid loss: ", res[2][0], "std valid loss percent: ", res[2][1])

res = run_experiment(0.1, 2, False)
print("experiment 8: hidden: 4, LR: 0.01, Bridge: False")
print("nean epocs: ", res[0][0], " std/epocs percent: ", res[0][1], "%", "failures: ", res[3],
      "\nmean train loss: ", res[1][0], "std train loss percent: ", res[1][1],
      "\nmean valid loss: ", res[2][0], "std valid loss percent: ", res[2][1])
