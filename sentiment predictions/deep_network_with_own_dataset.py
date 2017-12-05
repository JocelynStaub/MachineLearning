import tensorflow as tf
import numpy as np

from create_sentiment_featureset import create_feature_sets_and_labels

train_x,train_y,test_x,test_y = create_feature_sets_and_labels('dataset_sentiments/pos.txt','dataset_sentiments/neg.txt')   #either load from file.pickle or use this command and redo the file
#train_x,train_y,test_x,test_y = pickle.load(open('dataset_sentiments/sentiment_set.pickle','rb')) #if you want to load data directly


n_nodes_hl1 = 500 #number of nodes for hidden layer 1
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2   #either pos or neg
batch_size = 100 #number of data that will be fed into the algorithm, here 100 samples/images then 100...

x = tf.placeholder('float',[None,len(train_x[0])]) #len of 1st train_x for exemple because they are all the same
y = tf.placeholder('float',[None,n_classes]) #label for thoses images input


def neural_network(data):
    
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
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
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            i = 0
            while i  < len(train_x):        #loop that chunk the sample into batch_size of 100
                start = i
                end = i + batch_size
                
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y}) #c is cost
                epoch_loss+= c
                i += batch_size
                
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1),tf.argmax(y, 1))  #seek if the prediction is equal to expected output
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))


train_neural_network(x)





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    