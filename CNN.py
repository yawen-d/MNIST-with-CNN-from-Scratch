import numpy as np
import h5py
#data file type h5py
import time
import copy
from random import randint

# cd Desktop/CS\ 398/Assignments/A2/
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))
MNIST_data.close()

####################################################################################
#Implementation of stochastic gradient descent algorithm


class CNN:
    kernals = {}
    output_layer = {}
    hppr = {}

    def __init__(self, num_iterations, l_rate, stride, padding, 
    dim_kernal, num_kernals, dim_inputs, len_outputs, input_chanl, batch_size = 1):
        # initialize the model parameters, including the first and second layer 
        # parameters and biases
        self.hppr = {
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "l_rate": l_rate,
            "stride": stride,
            "padding": padding,
            "dim_kernal": dim_kernal,
            "num_kernals": num_kernals,
            "dim_inputs" : dim_inputs,
            "len_outputs" : len_outputs,
            "input_chanl" : input_chanl
        }
        temp_dim = dim_inputs - dim_kernal + 1
        self.output_layer = {
            'para' : np.random.randn(len_outputs, num_kernals, temp_dim, temp_dim) / np.sqrt(temp_dim**2*num_kernals*len_outputs),
            'bias' : np.random.randn(len_outputs,1) / np.sqrt(len_outputs)
        }
        for i in range(num_kernals):
            self.kernals[i] = np.random.randn(input_chanl,dim_kernal,dim_kernal) / np.sqrt(dim_kernal**2)
        
    def printing(self):
        print('########## Hyperparameters ##########')
        for i,j in self.kernals.items():
            print(i,':',j.shape)
        for i,j in self.output_layer.items():
            print(i,':',j.shape)
        for i,j in self.hppr.items():
            print(i,':',j)
        print('#####################################')

    def activfunc(self,Z,type = 'ReLU',deri = False):
        # implement the activation function
        if type == 'ReLU':
            if deri == True:
                return 1*(Z>0)
            else:
                return Z*(Z>0)
        elif type == 'Sigmoid':
            if deri == True:
                return 1/(1+np.exp(-Z))*(1-1/(1+np.exp(-Z)))
            else:
                return 1/(1+np.exp(-Z))
        elif type == 'tanh':
            if deri == True:
                return 
            else:
                return 1-(np.tanh(Z))**2
        else:
            raise TypeError('Invalid type!')

    def Softmax(self,z):
        # implement the softmax function
        return 1/sum(np.exp(z)) * np.exp(z)

    def cross_entropy_error(self,v,y):
        # implement the cross entropy error
        return -np.log(v[y])

    def convolution(self,x,kernals):
        ''' input -- x: 3D-array of size (num_channels,dim_inputs,dim_inputs) e.g.(3,28,28);
                     kernals: a dictionary of kernals e.g. {0:(3,5,5) ... 4:(3,5,5)}
            output-- fm: a 3D-array of feature maps of size 
                     (num_kernals,dim_inputs-dim_kernal+1,dim_inputs-dim_kernal+1) e.g.(5,26,26)
        '''
        num_kernals = len(kernals)
        x_sp = x.shape
        k_sp = kernals[0].shape
        t_dim = x_sp[1] - k_sp[1] + 1
        result = np.zeros((num_kernals,t_dim,t_dim))
        for i in range(num_kernals):
            for j in range(t_dim):
                for k in range(t_dim):
                    result[i,j,k] = np.sum(np.multiply(kernals[i],x[:,j:j+k_sp[1],k:k+k_sp[2]]))
        return result


    def forward(self,x,y):
        ''' input -- x: training data input x, size of (784,)
                     y: training data output y, integer
            output-- a dictionary of Z, H, U, f_X, error
        '''
        dim = self.hppr['dim_inputs']
        X = x.reshape(self.hppr['input_chanl'],dim,dim)
        K = self.kernals

        temp_dim = self.hppr['dim_inputs'] - self.hppr['dim_kernal'] + 1
        Z = self.convolution(X,K)
        H = self.activfunc(Z).reshape((temp_dim**2*self.hppr['num_kernals'],1))
        U = np.matmul(self.output_layer['para'].reshape((10,temp_dim**2*self.hppr['num_kernals'])),H) + self.output_layer['bias']
        predict_list = np.squeeze(self.Softmax(U))
        # error = self.cross_entropy_error(predict_list,y)
        
        dic = {
            'Z':Z,
            'H':H,
            'U':U,
            'f_X':predict_list.reshape((1,self.hppr['len_outputs'])),
        #    'error':error
        }
        return dic

    def back_propagation(self,x,y,f_result):
        ''' input -- x: training data input x, size of (784,)
                     y: training data output y, integer
                     f_result: a dictionary of Z, H, U, f_X, error
            output--
        '''
        E = np.array([0]*self.hppr['len_outputs']).reshape((1,self.hppr['len_outputs']))
        E[0][y] = 1
        dU = (-(E - f_result['f_X'])).reshape((self.hppr['len_outputs'],1))
        db = copy.copy(dU)

        # tmp_dim = self.hppr['dim_inputs']-self.hppr['dim_kernal']+1
        delta = np.zeros((self.hppr['num_kernals'],26,26))
        for i in range(10):
            delta += self.output_layer['para'][i,:]*np.squeeze(dU)[i]
        
        dW = np.zeros((10,5,26,26))
        for i in range(10):
            dW[i]=np.squeeze(dU)[i]*f_result['H'].reshape((5,26,26))

        dK = {}
        for i in range(5):
            tmp_dic = {}
            for j in range(1):
                tmp_dic[j] = np.multiply(f_result['Z'][j],delta[j]).reshape((1,26,26))
            dK[i] = self.convolution(x.reshape((1,28,28)),tmp_dic)

        
        grad = {
            'db':db,
            'dW':dW,
            'dK':dK
        }
        return grad

    def optimize(self,b_result, learning_rate):
        # update the hyperparameters
        self.output_layer['para'] -= learning_rate*b_result['dW']
        self.output_layer['bias'] -= learning_rate*b_result['db']
        for i in range(5):
            self.kernals[i] -= learning_rate*b_result['dK'][i]

    def loss(self,X_test,Y_test):
        # implement the loss function of the training set
        loss = 0
        for n in range(len(X_test)):
            if n % 1000 == 0:
                print('computing loss',n)
            y = Y_test[n]
            x = X_test[n][:]
            loss += self.forward(x,y)['error']
        return loss

    def train(self, X_train, Y_train):
        # generate a random list of indices for the training set
        learning_rate = self.hppr['l_rate']
        num_iterations = self.hppr['num_iterations']
        rand_indices = np.random.choice(len(X_train), num_iterations, replace=True)
        
        def l_rate(base_rate, ite, num_iterations, schedule = False):
        # determine whether to use the learning schedule
            if schedule == True:
                return base_rate * 10 ** (-np.floor(ite/num_iterations*4))
            else:
                return base_rate

        count = 1
        loss_dict = {}
        test_dict = {}

        for i in rand_indices:
            f_result = self.forward(X_train[i],Y_train[i])
            b_result = self.back_propagation(X_train[i],Y_train[i],f_result)
            self.optimize(b_result,l_rate(learning_rate,i,num_iterations,False))
            
            if count % 100 == 0:
                if count % 30000 == 0:
                    loss = 'NA' # self.loss(x_test,y_test)
                    test = self.testing(x_test,y_test)
                    print('Trained for {} times,'.format(count),'loss = {}, test = {}'.format(loss,test))
                    # loss_dict[str(count)]=loss
                    test_dict[str(count)]=test
                else:
                    print('Trained for {} times,'.format(count))
            count += 1

        print('Training finished!')
        return loss_dict, test_dict

    def testing(self,X_test, Y_test):
        # test the model on the training dataset
        total_correct = 0
        for n in range(len(X_test)):
            y = Y_test[n]
            x = X_test[n][:]
            prediction = np.argmax(self.forward(x,y)['f_X'])
            if (prediction == y):
                total_correct += 1
            if n % 1000 == 0:
                print('testing data',n)
        print('Accuarcy Test: ',total_correct/len(X_test))
        return total_correct/np.float(len(X_test))


####################################################################################


# data fitting, training and accuracy evaluation
model = CNN(batch_size = 1, num_iterations=120000, l_rate=0.01, stride=1, 
padding=0, dim_kernal=3, num_kernals=5, dim_inputs=28, input_chanl=1, len_outputs=10)
model.printing()
cost_dict,tests_dict = model.train(x_train,y_train)
accu = model.testing(x_test,y_test)
model.printing()



# plotting the loss function and test accuracy corresponding to the number of iterations
import matplotlib.pyplot as plt
# plt.plot(cost_dict.keys(),cost_dict.values())
# plt.ylabel('Loss function')
# plt.xlabel('Number of iterations')
# plt.xticks(rotation=60)
# plt.title('Loss function w.r.t. number of iterations')
# plt.show()

plt.plot(tests_dict.keys(),tests_dict.values())
plt.ylabel('Test Accuracy')
plt.xlabel('Number of iterations')
plt.xticks(rotation=60)
plt.title('Test accuracy w.r.t. number of iterations')
plt.show()
