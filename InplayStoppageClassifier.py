import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from matplotlib import pyplot as plt

def DataChunking(position_file,stoppage_file,timestep = 100,fst_team = "Argentina",snd_team = "Brazil"):
    with open(position_file,'r') as w:
        original_data = json.load(w)
        len_total = len(original_data)
        # print(len(original_data))
        X = np.zeros([len_total,20,2])
        for index, i in enumerate(original_data):
            for j in range(10):
                X[index,j,0] = i[fst_team][j]["x"]
                X[index,j,1] = i[fst_team][j]["y"]
            for j in range(10,20):
                X[index,j,0] = i[snd_team][j - 10]["x"]
                X[index,j,1] = i[snd_team][j - 10]["y"]
        
        truncated_len = int(len_total - len_total%timestep)
        X = X[0:truncated_len,:,:].reshape([-1,timestep*20*2])
        print(np.shape(X))

    with open(stoppage_file,'r') as w:
        original_label = json.load(w)["Events"]

        Y = np.zeros([len_total,1])
        i = 0
        while(i < len(original_label)):
            start_frame = original_label[i]["frame"]
            end_frame = original_label[i + 1]["frame"]
            for j in range(start_frame,end_frame + 1):
                Y[j] = 1
            i += 2
    Y = Y[0:truncated_len]
    Y = np.sum(Y.reshape([timestep,-1]),axis = 0)
    Y = np.where(Y > 0.6*timestep,1,0)
    print(Y)
    # Y = np.reshape(Y,[1,-1])
    print(np.shape(Y))
    # print(Y[0:10])
    

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    return (X_train, X_test, y_train, y_test,X,Y)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X_full,y_full = DataChunking("position.json","on-off.json")
    _, _, _, _, X_full_2,y_full_2 = DataChunking("position2.json","on-off2.json",fst_team = "Argentina",snd_team = "Peru")
    print(np.shape(X_train))
    print(X_train[0][0])
    # a = [0,1,1,1]
    num_classes = 2
    # b = tf.one_hot(a,depth)
    # print(b)
    # with tf.Session() as sess:
    #     print(b.eval())   #一次能打印两个


    X = tf.placeholder(tf.float32,[None,100*20*2])
    Y = tf.placeholder(tf.float32,[None,2])

    #weights & bias for nn layers
    W1 = tf.Variable(tf.random_normal([100*20*2,256]))
    b1 = tf.Variable(tf.random_normal([256]))
    L1 = tf.nn.relu(tf.matmul(X,W1) + b1)

    W2 = tf.Variable(tf.random_normal([256,256]))
    b2 = tf.Variable(tf.random_normal([256]))
    L2 = tf.nn.relu(tf.matmul(L1,W2) + b2)

    W3 = tf.Variable(tf.random_normal([256,2]))
    b3 = tf.Variable(tf.random_normal([2]))
    hypothesis = tf.matmul(L2,W3) + b3
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


    is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    res = tf.arg_max(hypothesis, 1)

    # parameters
    training_epochs = 2000
    batch_size = 100

    sess = tf.Session()
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle

    batch_xs = X_train
    num_labels = y_train.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    batch_ys = np.zeros((num_labels, num_classes))
    batch_ys.flat[index_offset + y_train.ravel()] = 1

    num_labels = y_test.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    Y_test = np.zeros((num_labels, num_classes))
    Y_test.flat[index_offset + y_test.ravel()] = 1

    num_labels = y_full.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    Y_full = np.zeros((num_labels, num_classes))
    Y_full.flat[index_offset + y_full.ravel()] = 1

    num_labels = y_full_2.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    Y_full_2 = np.zeros((num_labels, num_classes))
    Y_full_2.flat[index_offset + y_full_2.ravel()] = 1
    for epoch in range(training_epochs):

        avg_cost = 0
        total_batch = 1

        for i in range(total_batch):
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch
        if epoch%100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
            print("Accuracy: ", accuracy.eval(session=sess,feed_dict={X: X_test, Y: Y_test}))
            if avg_cost < 0.00001:
                best_res = res.eval(session=sess,feed_dict={X: batch_xs})
                train_acc = accuracy.eval(session=sess,feed_dict={X: batch_xs, Y: batch_ys})
                best_res_0 = res.eval(session=sess,feed_dict={X: X_test})
                best_res_1 = res.eval(session=sess,feed_dict={X: X_full})
                best_res_2 = res.eval(session=sess,feed_dict={X: X_full_2})
                print("Full Accuracy: ", accuracy.eval(session=sess,feed_dict={X: X_full, Y: Y_full}))
                print("Second Full Accuracy: ", accuracy.eval(session=sess,feed_dict={X: X_full_2, Y: Y_full_2}))

        # Get one and predict
        # r = random.randint(0, mnist.test.num_examples - 1)
        # print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
        # print("Prediction:", sess.run(tf.argmax(hypothesis, 1), 
        #                     feed_dict={X: mnist.test.images[r:r + 1]}))
    print(best_res)
    print(np.sum(best_res))
    print(np.shape(best_res)[0])
    print(train_acc)
    print(best_res_0)
    print(np.sum(best_res_0))
    print(np.shape(best_res_0)[0])
    print(best_res_1)
    print(np.sum(best_res_1))
    print(np.shape(best_res_1)[0])
    print(best_res_2)
    print(np.sum(best_res_2))
    print(np.shape(best_res_2)[0])
        # plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
        # plt.show()

    cl_1_ac_1 = 0
    cl_1_ac_0 = 0
    cl_0_ac_1 = 0
    cl_0_ac_0 = 0

    for i in range(len(best_res_0)):
        if best_res_0[i] == 1 and y_test[i] == 1:
            cl_1_ac_1 += 1
        elif best_res_0[i] == 1 and y_test[i] == 0:
            cl_1_ac_0 += 1
        elif best_res_0[i] == 0 and y_test[i] == 1:
            cl_0_ac_1 += 1
        elif best_res_0[i] == 0 and y_test[i] == 0:
            cl_0_ac_0 += 1
    print(cl_1_ac_1)
    print(cl_1_ac_0)
    print(cl_0_ac_1)
    print(cl_0_ac_0)



