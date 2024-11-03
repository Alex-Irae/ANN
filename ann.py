import numpy as np
import pickle
import os

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    sig = sigmoid(x)
    return sig + x * sig * (1 - sig)


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)


class ArtificialNeuralNetwork:
    def __init__(self,_shape:tuple=None,init_mode:str = None,h1:int=None,fct1:str =None,h2:int=None,fct2:str =None,h3:int=None,fct3:str =None, fout:str ='softmax', loss:str='mse', name:str = 'ann_model'):
        """
        Initialise the neural Network
        
        Parameters :
        - shape, shape of the data, (features,output)
        - init_mode, method of initialisation of the weights ('xavier','he' or None)
        - h1, Number of nodes in the first layer (if not defined, no hidden layer)
        - fct1, activation function for the first layer ('sigmoid','relu', 'tanh','swish')
        - h2, Number of nodes in the second layer (if not defined, no hidden layer)
        - fct2, activation function for the second layer  ('sigmoid','relu', 'tanh','swish')
        - h3, Number of nodes in the third layer (if not defined, no hidden layer)
        - fct3, activation function for the third layer  ('sigmoid','relu', 'tanh','swish')
        - fout, output function (if None, softmax)  ('sigmoid','relu', 'tanh','swish','softmax')
        - loss, loss function ('mse','mae','cross_entropy')
        - name, name of the ANN for saving/loading
        
        Returns :
        - Artificial neural network
        """
        if h1:
            if h1 <= 0:
                h1 = None
                self.h1 = False
            else:
                self.h1 = True
        else:
            self.h1 = False
        
        if h2:
            if h2 <= 0:
                h2 = None
                self.h2 = False
            else:
                self.h2 = True
        else:
            self.h2 = False
        
        if h3:
            if h3 <= 0:
                h3 = None
                self.h3 = False
            else:
                self.h3 = True
        else:
            self.h3 = False
            
        mode_dict = ['he', 'xavier', None]
        if init_mode not in mode_dict:
             
            raise ValueError(f"Initialisation mode {init_mode} not recognised, please verify the parameters")
        
        loss_dict = ['mse', 'mae', 'cross_entropy']
        if loss not in loss_dict:
             
            raise ValueError(f"Loss function {loss} not recognised, please verify the parameters")
        
        if (not h1 and h2) or (not h2 and h3) or (not h1 and h3):
             
            raise ValueError("Cannot define a second or third layer before defining the preceding layer")
        
        if not isinstance(_shape, (list, tuple)) or len(_shape) != 2 or not all(isinstance(dim, int) for dim in _shape):
             
            raise ValueError("Incorrect data dimensions, please verify the shape parameter")

        if h1 and not fct1:
             
            raise ValueError("Activation function for first layer not declared")
        if h2 and not fct2:
             
            raise ValueError("Activation function for second layer not declared")
        if h3 and not fct3:
             
            raise ValueError("Activation function for the third layer not declared")
        if (h1 and type(h1) != int) or (h2 and type(h2) != int) or (h3 and type(h3) != int):
             
            raise ValueError("Number of nodes must be an integer")
        if (h1 and h1 <= 0) or (h2 and h2 <= 0) or (h3 and h3 <= 0):
                 
                raise ValueError("Number of nodes must be a positive integer")


        self.loss = loss
        self.init_mode = init_mode
        self.name = name
        self.shape = _shape

        if fct1 :
            self.fct1, self.d_fct1 = self.att_fcts(fct1)
            
        if fct2 :
            self.fct2, self.d_fct2 = self.att_fcts(fct2) 
        if fct3 :
            self.fct3, self.d_fct3 = self.att_fcts(fct3)
        if fout :
            self.fout,_ = self.att_fcts(fout)
        
        
        
        input_size = _shape[0]
        output_size = _shape[1]
        
        h_last_out = input_size 
        
        match init_mode :
            
            case 'he':
                if h1 :
                    self.hidden_size_1 = h1
                    self.W1 = np.random.randn(h_last_out, self.hidden_size_1) * np.sqrt(2. / h_last_out)
                    h_last_out = self.hidden_size_1
                    self.b1 = np.zeros(shape=(1, self.hidden_size_1))
                
                if h2 :
                    self.hidden_size_2 = h2
                    self.W2 = np.random.randn(h_last_out, self.hidden_size_2)* np.sqrt(2. / h_last_out)
                    h_last_out = self.hidden_size_2
                    self.b2 = np.zeros(shape=(1, self.hidden_size_2))
                    
                if h3 :
                    self.hidden_size_3 = h3
                    self.W3 = np.random.randn(h_last_out, self.hidden_size_3) * np.sqrt(2. / h_last_out)
                    h_last_out = self.hidden_size_3
                    self.b3 = np.zeros((1, self.hidden_size_3)) 
                    
                self.W_output = np.random.randn(h_last_out, output_size) * np.sqrt(2. / h_last_out)
                self.b_output = np.zeros(shape=(1, output_size))
            
            case 'xavier':
                if h1 :
                    self.hidden_size_1 = h1
                    limit = np.sqrt(6 / (input_size + self.hidden_size_1))
                    self.W1 = np.random.uniform(low=-limit, high=limit, size=(input_size, self.hidden_size_1))
                    h_last_out = self.hidden_size_1
                    self.b1 = np.zeros(shape=(1, self.hidden_size_1))
                
                if h2 : 
                    self.hidden_size_2 = h2
                    limit = np.sqrt(6 / (h_last_out + self.hidden_size_2))
                    self.W2 = np.random.uniform(low=-limit, high=limit, size=(h_last_out, self.hidden_size_2))
                    h_last_out = self.hidden_size_2
                    self.b2 = np.zeros(shape=(1, self.hidden_size_2))
                    
                if h3 : 
                    self.hidden_size_3 = h3
                    limit = np.sqrt(6 / (h_last_out + self.hidden_size_3))
                    self.W3 = np.random.uniform(low=-limit, high=limit, size=(h_last_out, self.hidden_size_3))
                    h_last_out = self.hidden_size_3
                    self.b3 = np.zeros((1, self.hidden_size_3)) 
                    
                limit = np.sqrt(6 / (h_last_out +output_size))
                self.W_output = np.random.uniform(low=-limit, high=limit, size=(h_last_out, output_size))
                self.b_output = np.zeros(shape=(1, output_size))
               
            case _:
                if h1 :
                    self.hidden_size_1 = h1
                    self.W1 = np.random.randn(input_size, self.hidden_size_1)*0.01
                    h_last_out = self.hidden_size_1
                    self.b1 = np.zeros(shape=(1, self.hidden_size_1))*0.01
                
                if h2 :
                    self.hidden_size_2 = h2
                    self.W2 = np.random.randn(h_last_out, self.hidden_size_2)*0.01
                    h_last_out = self.hidden_size_2
                    self.b2 = np.zeros(shape=(1, self.hidden_size_2))*0.01
                    
                if h3 :
                    self.hidden_size_3 = h3
                    self.W3 = np.random.randn(h_last_out, self.hidden_size_3)*0.01
                    h_last_out = self.hidden_size_3
                    self.b3 = np.zeros(shape=(1, self.hidden_size_3)) *0.01
                    
                self.W_output = np.random.randn(h_last_out, output_size) *0.01
                self.b_output = np.zeros(shape=(1, output_size))*0.01

        
    def att_fcts(self,fct):
        """
        return the functiona ssociated to the strong
        """
        valid_activations = ['relu', 'sigmoid', 'tanh', 'swish', 'linear','softmax']
        if fct not in valid_activations:
            raise ValueError(f"Incorrect activation function {fct}, please verify the parameters")
        match fct:
            case 'relu' : return relu,relu_derivative
            case 'sigmoid' : return sigmoid,sigmoid_derivative
            case 'tanh' : return tanh,tanh_derivative
            case 'swish' : return swish,swish_derivative
            case 'linear' : return linear,linear_derivative
            case 'softmax' : return softmax,None
            
    def remove_layer(self):
        """
        Remove the last hidden layer
        """
        if self.h3:
            self.W3 = None
            self.b3 = None
            self.h3 = False
        elif self.h2:
            self.W2 = None
            self.b2 = None
            self.h2 = False
        elif self.h1:
            self.W1 = None
            self.b1 = None
            self.h1 = False
        else:
            raise ValueError("Cannot remove a hidden layer if there is none")
        
    def add_layer(self, h:int, fct:str):
        """
        Add a hidden layer to the neural network
        
        Parameters:
        - h: int, number of nodes in the hidden layer
        - fct: str, activation function for the hidden layer ('relu', 'sigmoid', 'tanh', 'swish')
        """
        if not isinstance(h, int) or h <= 0:
            raise ValueError("Number of nodes must be a positive integer")
        if not fct:
            raise ValueError("Activation function for the hidden layer not declared")
        if fct not in ['relu', 'sigmoid', 'tanh', 'swish']:
            raise ValueError(f"Incorrect activation function {fct}, please verify the parameters")
        
        if not self.h1:
            self.hidden_size_1 = h
            self.W1 = np.random.randn(self.shape[0], self.hidden_size_1)*0.01
            self.b1 = np.zeros(shape=(1, self.hidden_size_1))*0.01
            self.fct1 = fct
            self.d_fct1 = relu_derivative
            self.h1 = True
        elif not self.h2 and self.h1:
            self.hidden_size_2 = h
            self.W2 = np.random.randn(self.hidden_size_1, self.hidden_size_2)*0.01
            self.b2 = np.zeros(shape=(1, self.hidden_size_2))*0.01
            self.fct2 = fct
            self.d_fct2 = relu_derivative
            self.h2 = True
        elif not self.h3 and self.h2:
            self.hidden_size_3 = h
            self.W3 = np.random.randn(self.hidden_size_2, self.hidden_size_3)*0.01
            self.b3 = np.zeros(shape=(1, self.hidden_size_3))*0.01
            self.fct3,self.d_fct3 = self.att_fcts(fct=fct)
            self.h3 = True
        else:
            raise ValueError("Cannot add more than 3 hidden layers")
    
        
    def summary(self):
        print()
        print(f"--------------Model : {self.name}--------------")
        print()
        print(f"  Loss:       {self.loss}")
        print()
        print(f"  Weight and bias initalisation mode:       {self.init_mode}")
        print()
        print(f"  Input  :       {self.shape[0]}")
        print()
        if self.h1:
            print(f"  Layer 1  :       activation function: {self.fct1}\n                   number of nodes: {self.hidden_size_1}")
            print()
        if self.h2:
            print(f"  Layer 2  :       activation function: {self.fct2}\n                   number of nodes: {self.hidden_size_2}")
            print()
        if self.h3:
            print(f"  Layer 3 :        activation function: {self.fct3}\n                   number of nodes: {self.hidden_size_3}")
            print()
        print(f"  Output :       activation function {self.fout}")
        print()
        print(f"  Output size :       {self.shape[1]}")
        print()
        print(f"--------------------------------------------")
        
        
    def forward(self, X):
        """
        Forward pass through the network
                
        Parameters:
        - X : ndarray, features 

        Returns:
        - output: probability or int corrsponding to the prediction of X by the network.
        """
        self.A_last_out = X
        if self.h1 :
            # First hidden layer 
            self.Z1 = np.dot(a=self.A_last_out, b=self.W1) + self.b1  
            self.A1 = self.fct1(x=self.Z1) 
            self.A_last_out =  self.A1
        if self.h2 :
            # Second hidden layer 
            self.Z2 = np.dot(a=self.A_last_out, b=self.W2) + self.b2  
            self.A2 = self.fct2(x=self.Z2)
            self.A_last_out = self.A2  
        
        if self.h3:
            # Third hidden layer
            self.Z3 = np.dot(a=self.A_last_out, b=self.W3) + self.b3 
            self.A3 = self.fct3(x=self.Z3) 
            self.A_last_out = self.A3
        
        # Output layer (using softmax as defaut)
        self.Z_output = np.dot(a=self.A_last_out, b=self.W_output) + self.b_output  
        self.A_output = self.fout(x=self.Z_output) 
                
        return self.A_output

 
    def compute_loss(self, y_pred, y_true):
        """
        Compute the combined loss.
        
        Parameters:
        - y_pred: ndarray, predicted probabilities (output of softmax).
        - y_true: ndarray, true one-hot encoded labels.

        Returns:
        - loss: float, computed loss value.
        """
        match self.loss :
            case 'cross_entropy':
                y_true = y_true.toarray() if hasattr(y_true, 'toarray') else y_true

                assert y_pred.shape == y_true.shape, f"Shape mismatch: {y_pred.shape} vs {y_true.shape}"

                m = y_true.shape[0]
                cross_entropy_loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m 

                return cross_entropy_loss
            case 'mse':
                mse_loss = np.mean(np.square(y_pred - y_true))
                return mse_loss
            case 'mae':
                return np.mean(np.abs(y_pred - y_true))

            
    def backward(self, X, y_true, y_pred):
        """
        Backward pass through the network
        
        Parameters :
        - X, ndarray, training features
        - y_pred: ndarray, predicted probabilities.
        - y_true: ndarray, true one-hot encoded labels.
        
        Returns:
        - Update Weights and bias
        
        """
        m = X.shape[0]  
        
        # Output layer gradients
        match self.loss :
            case 'cross_entropy': dZ_output = y_pred - y_true 
            case 'mse': dZ_output = (2 / y_true.shape[0]) * (y_pred - y_true)
            case 'mae': dZ_output = (1 / y_true.shape[0]) * np.sign(y_pred - y_true)

        
        dZ_last_out = dZ_output
        assert dZ_output.shape == (m, self.W_output.shape[1]), f"Shape mismatch: {dZ_output.shape} vs {self.W_output.shape[1]}"
        dW_output = 1 / m * np.dot(a=np.asarray(a=self.A_last_out).T, b=dZ_output)  
        db_output = 1 / m * np.sum(a=np.asarray(a=dZ_output), axis=0, keepdims=True)

        if self.h3 :
            # Third hidden layer gradients
            dA3 = np.dot(a=dZ_last_out, b=np.asarray(a=self.W_output).T)  
            assert dA3.shape == np.asarray(a=self.A3).shape, f"Shape mismatch: {dA3.shape} vs {self.A3.shape}"

            dZ3 = np.multiply(dA3, self.d_fct3(np.asarray(a=self.Z3)))  
            dW3 = 1 / m * np.dot(a=np.asarray(a=self.A2).T, b=dZ3)  
            db3 = 1 / m * np.sum(a=np.asarray(a=dZ3), axis=0, keepdims=True) 
            dZ_last_out = dZ3
            
        if self.h2:
            # Second hidden layer gradients
            if self.h3 :
                dA2 = np.dot(a=dZ_last_out, b=np.asarray(a=self.W3).T)  
                assert dA2.shape == np.asarray(a=self.A2).shape, f"Shape mismatch: {dA2.shape} vs {self.A2.shape}"
            else :
                dA2 = np.dot(a=dZ_last_out, b=np.asarray(a=self.W_output).T)  
                assert dA2.shape == np.asarray(a=self.A2).shape, f"Shape mismatch: {dA2.shape} vs {self.A2.shape}"
            dZ2 = np.multiply(dA2 ,self.d_fct2(np.asarray(a=self.Z2)))  
            dW2 = 1 / m * np.dot(a=np.asarray(a=self.A1).T, b=dZ2)  
            db2 = 1 / m * np.sum(a=np.asarray(a=dZ2), axis=0, keepdims=True) 
            dZ_last_out = dZ2
        
        if self.h1:
            # First hidden layer gradients
            if self.h2 :
                dA1 = np.dot(a=dZ_last_out, b=np.asarray(a=self.W2).T) 
                assert dA1.shape == np.asarray(a=self.A1).shape, f"Shape mismatch: {dA1.shape} vs {self.A1.shape}"
            else :
                dA1 = np.dot(a=dZ_last_out, b=np.asarray(a=self.W_output).T) 
                assert dA1.shape == np.asarray(a=self.A1).shape, f"Shape mismatch: {dA1.shape} vs {self.A1.shape}"
            dZ1 = np.multiply(dA1,self.d_fct1(np.asarray(a=self.Z1)))  
            dW1 = 1 / m * np.dot(a=np.asarray(a=X).T, b=dZ1)  
            db1 = 1 / m * np.sum(a=np.asarray(a=dZ1), axis=0, keepdims=True)  
            dZ_last_out = dZ1
  
        if self.h1:
            self.W1 -= self.learning_rate * dW1     
            self.b1 -= self.learning_rate * db1  
        if self.h2:
            self.W2 -= self.learning_rate * dW2  
            self.b2 -= self.learning_rate * db2  
        if self.h3:
            self.W3 -= self.learning_rate * dW3  
            self.b3 -= self.learning_rate * db3 

        self.W_output -= self.learning_rate * dW_output  
        self.b_output -= self.learning_rate * db_output  
        
        
        
    def train(self, X_train, y_train, epochs=1000, learning_rate=0.01, batch_size = 32):
        """
        Training method for the neural network
        
        Parameters : 
        - X_train, list of features used fo training
        - y_train, list of labels corresponding to X_train
        - epochs, number of times X_train passes through the backward propagation 
        - learning_rate, rate at which the neural network adjusts its weights
        - batch_size, size of each batch 
        """
        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError("X_train and y_train must be numpy arrays")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Number of epochs must be a positive integer")
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("Learning rate must be a positive number")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")
        if X_train.shape[1] != self.shape[0] or y_train.shape[1] != self.shape[1]:
            raise ValueError("Input and output dimensions do not match the model")
        
            
        self.learning_rate = learning_rate  
              
              
        for epoch in range(epochs):
        
            indices = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]
                
                y_pred = self.forward(X=X_batch)
                
                loss = self.compute_loss(y_pred=y_pred, y_true=y_batch)
                
                self.backward(X=X_batch, y_true=y_batch, y_pred=y_pred)

            if epoch % 50 == 0: 
                print(f'Epoch {epoch}, Loss: {loss}')
        
        

    def predict(self, X):
        """
        Predict the class probabilities for the input samples X.
        Returns:
        - probabilities: A 2D NumPy array of shape (n_samples, n_classes) containing class probabilities.
        """
        y_pred = self.forward(X=X)  
        return y_pred  


    
        

    def save_model(self, model_name):
        """
        Saves the trained model to a file using pickle.
        
        Parameters:
        - model_name: str, name to use for the saved model file.
        """
        
        model = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'W_output': self.W_output, 'b_output': self.b_output,
            'hidden_size_1': self.hidden_size_1,
            'hidden_size_2': self.hidden_size_2,
            'hidden_size_3': self.hidden_size_3,
            'fct1': self.fct1,
            'fct2': self.fct2,
            'fct3': self.fct3,
            'fout': self.fout,
            'loss': self.loss,
            'name': self.name,
            'input_size': self.W1.shape[0],
            'output_size': self.W_output.shape[1],
        }
        
        models_directory = os.path.join(os.getcwd(), 'models')

        os.makedirs(models_directory, exist_ok=True)

        model_path = os.path.join(models_directory, f'{model_name}.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        
        
def load_model(model_name):
    """
    Load a saved model from a file.
    
    Parameters:
    - model_name: str, name of the saved model file.
    
    Returns:
    - model: The loaded model.
    """
    
    models_directory = os.path.join(os.getcwd(), 'models')
    model_path = os.path.join(models_directory, f'{model_name}.pkl')
    
    with open(model_path, 'rb') as f:
        fi = pickle.load(file=f)
        model = ArtificialNeuralNetwork(inp=fi['input_size'], out=fi['output_size'],
                                        h1=fi['hidden_size_1'], h2=fi['hidden_size_2'], h3=fi['hidden_size_3'],
                                        fct1=fi['fct1'], fct2=fi['fct2'], fct3=fi['fct3'], fout=fi['fout'],
                                        loss=fi['loss'], name=fi['name'])
        model.W1 = fi['W1']
        model.W2 = fi['W2']
        model.W3 = fi['W3']
        model.W_output = fi['W_output']
        model.b1 = fi['b1']
        model.b2 = fi['b2']
        model.b3 = fi['b3']
        model.b_output = fi['b_output']
    return model  

        
        
    
X_train = np.array(object=[[2,850],[5,2000],[4,1500],[2,900],[3,1350],[4,1600],[1,700],[2,800],[3,1400]])
Y_train = np.array(object=[[220],[550],[450],[250],[370],[480],[180],[210],[390]])

#340
test = np.array(object=[3,1200])


model = ArtificialNeuralNetwork(_shape=(X_train.shape[1],Y_train.shape[1]),h1=90,fct1='relu',h2=90,fct2='relu',fout='linear',loss ='mse',name='ann_model')

model.summary()

model.add_layer(h=90,fct='sigmoid')
model.summary()

model.remove_layer()
model.summary()


# model.add_layer(h=90,fct='relu')
# model.train( X_train=X_train, y_train=Y_train, epochs=1000, learning_rate=0.0008, batch_size = 2)


# # 1
# Y= model.predict(X=test)
# print(Y)














# X_train =np.array( object=[[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])


# Y_train = np.array(object=[[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],
#                            [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
#                            [0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])

# [1,0,0]
# test = np.array(object=[[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        #  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
