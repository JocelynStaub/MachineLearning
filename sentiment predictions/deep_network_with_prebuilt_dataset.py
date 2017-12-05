import tensorflow as tf

'''
input > weights > hidden layer 1 (activation function) > weights > hidden l2 (activ funct) > weights > output , this does feed forward

compare output to intended output with cost or loss fucntion (cross entopy for exemple)
optimization function/optimizer > minimize cost (AdamOptimizer...SGD, Adagrad), this does backpropagation

feed forward + backprop = epoch
'''

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) #one_hot means that only 1 output is 1 and rest is zero

'''
for exemple : 
    0 = [1,0,0,0,0,0,0,0,0,0]
    1 = [0,1,0,0,0,0,0,0,0,0]
    ...
'''

n_nodes_hl1 = 500 #number of nodes for hidden layer 1
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10  #number of classes 0 to 9 numbers (type of images)
batch_size = 100 #number of data that will be fed into the algorithm, here 100 samples/images then 100...

x = tf.placeholder('float', [None, 784]) #images input
y = tf.placeholder('float',[None, n_classes]) #label for thoses images input
'''
It is useful to set the 2nd parameter in placeholder because it forces tensorflow
to output an error if the size of the input is not correct

Here the images are 28*28 pixels which is a total of 784 pixels, it's not necessary a 28*28 matrix, it can also be a string of 784 values
The None value are here because it can be any number, because batch size is 100 it'll be 100 images of 784 so [100, 784]
'''

def neural_network(data):
    
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
   
    '''
    the weights comes from the output of hl1 to hl2 so the size are hl1, hl2
    '''
    
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    #   (input_data * weight) + bias, bias are used in case all input are zero, then somes neurons can still fire and send infos to the output
    
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1) #activation function (threshold function)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)    
    
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)    
    
    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    
    return output
    
    
def train_neural_network(x):
    prediction = neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    
    #learning_rate = 0.001 by default
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10 #how many epochs, how many feedforward + backprop
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)): #tronque la taille totale de samples par le batch_size et itère la for loop jusqu'à ce nombre
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)   #function that goes through chunk of batch_size for the total amount of samples
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y}) #c is cost
                epoch_loss+= c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1),tf.argmax(y, 1))  #seek if the prediction is equal to expected output
        
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    