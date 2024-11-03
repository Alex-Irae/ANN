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


class ArtificialNeuralNetwork:
    def __init__(self,_shape:tuple=None,init_mode:str = None,h1:int=None,fct1:str =None,h2:int=None,fct2:str =None,h3:int=None,fct3:str =None, fout:str ='softmax', pb:str = 'classification', name:str = 'ann_model'):
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
        - pb, nature of the problem, to adapt the loss function
        - name, name of the ANN for saving/loading
        
        Returns :
        - Artificial neural network
        """
        self.pb = pb
        self.init_mode = init_mode
        self.name = name
        self.shape = _shape
 
        if h1:
            if h1 <= 0 :
                h1 =None
                self.h1 =False
            else :
                self.h1 =True
        else:
            self.h1 =False        
        
            
        if h2:
            if h2 <= 0 :
                h2 = None
                self.h1 =False
            else :
                self.h2 =True
        else :
            self.h2 = False
            
        if h3:
            if h3 <= 0:
                h3 =None  
                self.h1 =False
            else :
                self.h3 =True  
        else: 
            self.h3 = False
                

            
        if (not h2 and h3) or (not h1 and h2) or (not h1 and h3):
            print("Cant define a second or third layer before defining the preceding layer")
            return
        
        if _shape[0] ==None or _shape[1] == None or type(_shape[0])!=int or type(_shape[1])!=int:
            print("Incorrect data dimensions, please verify the shape parameter")
            return
        
        if (fct1!= 'relu' and fct1 != 'sigmoid' and fct1 != 'tanh' and fct1 != 'swish') and fct1!= None:
            print(f"Incorrect activation function {fct1}, please verify the parameters")
            return
        if (fct2!= 'relu' and fct2 != 'sigmoid' and fct2 != 'tanh' and fct2 != 'swish') and fct2!= None:
            print("Incorrect activation function, please verify the parameters")
            return
        if (fct3!= 'relu' and fct3 != 'sigmoid' and fct3 != 'tanh' and fct3 != 'swish') and fct3!= None:
            print("Incorrect activation function, please verify the parameters")
            return
        if (fout!= 'relu' and fout != 'sigmoid' and fout != 'tanh' and fout != 'swish' and fout!='softmax') and fout != None:
            print("Incorrect activation function, please verify the parameters")
            return
        
       
        if h1 and not fct1:
            print("Activation function for first layer not declared")
            return
        if h2 and not fct2 :
            print("Activation function for second layer not declared")
            return
        if h3 and not fct3 :
            print("Activation function for the third layer isnt declared")
            return    
    
        
        match fct1:
            case 'relu' : 
                self.fct1 = relu
                self.d_fct1 = relu_derivative
            case 'sigmoid' : 
                self.fct1 = sigmoid
                self.d_fct1 = sigmoid_derivative
            case 'tanh' : 
                self.fct1 = tanh
                self.d_fct1 = tanh_derivative
            case 'swish' : 
                self.fct1 = swish
                self.d_fct1 = swish_derivative
    
        if fct2:
            match fct2:
                case 'relu' :
                    self.fct2 = relu
                    self.d_fct2 = relu_derivative
                case 'sigmoid' : 
                    self.fct2 = sigmoid
                    self.d_fct2 = sigmoid_derivative
                case 'tanh' : 
                    self.fct2 = tanh
                    self.d_fct2 = tanh_derivative
                case 'swish' : 
                    self.fct2 = swish
                    self.d_fct3 = swish_derivative
        if fct3 :
            match fct3:
                case 'relu' : 
                    self.fct3 = relu
                    self.d_fct3 = relu_derivative
                case 'sigmoid' : 
                    self.fct3 = sigmoid
                    self.d_fct3 = sigmoid_derivative
                case 'tanh' : 
                    self.fct3 = tanh
                    self.d_fct3 = tanh_derivative
                case 'swish' :
                    self.fct3 = swish
                    self.d_fct3 = swish_derivative
        match fout:
            case 'relu' : 
                self.fout = relu
                self.d_fout = relu_derivative
            case 'sigmoid' :
                self.fout = sigmoid
                self.d_fout = sigmoid_derivative
            case 'tanh' : 
                self.fout = tanh
                self.d_fout = tanh_derivative
            case 'swish' : 
                self.fout = swish
                self.d_fout = swish_derivative
            case 'softmax' : 
                self.fout = softmax  
        
        
        
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
                    self.W1 = np.random.randn(input_size, self.hidden_size_1)
                    h_last_out = self.hidden_size_1
                    self.b1 = np.zeros(shape=(1, self.hidden_size_1))
                
                if h2 :
                    self.hidden_size_2 = h2
                    self.W2 = np.random.randn(h_last_out, self.hidden_size_2)
                    h_last_out = self.hidden_size_2
                    self.b2 = np.zeros(shape=(1, self.hidden_size_2))
                    
                if h3 :
                    self.hidden_size_3 = h3
                    self.W3 = np.random.randn(h_last_out, self.hidden_size_3)
                    h_last_out = self.hidden_size_3
                    self.b3 = np.zeros(shape=(1, self.hidden_size_3)) 
                    
                self.W_output = np.random.randn(h_last_out, output_size) 
                self.b_output = np.zeros(shape=(1, output_size))
                
                  

    def summary(self):
        print()
        print(f"--------------Model : {self.name}--------------")
        print()
        print(f"  Use:       {self.pb}")
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
            # First hidden layer (using relu as default)
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
        match self.pb :
            case 'classification':
                y_true = y_true.toarray() if hasattr(y_true, 'toarray') else y_true

                assert y_pred.shape == y_true.shape, f"Shape mismatch: {y_pred.shape} vs {y_true.shape}"

                m = y_true.shape[0]
                cross_entropy_loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m

                diff = y_pred - y_true
                mse_loss = np.mean(np.square(diff)) 

                loss = cross_entropy_loss + 0.5 * mse_loss  

                return cross_entropy_loss
            case 'regression':
                return
                

            
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
        dZ_output = y_pred - y_true 
        assert dZ_output.shape == (m, self.W_output.shape[1]), f"Shape mismatch: {dZ_output.shape} vs {self.W_output.shape[1]}"
        dW_output = 1 / m * np.dot(a=np.asarray(a=self.A_last_out).T, b=dZ_output)  
        db_output = 1 / m * np.sum(a=np.asarray(a=dZ_output), axis=0, keepdims=True)

        if self.h3 :
            # Third hidden layer gradients
            dA3 = np.dot(a=self.dZ_last_out, b=np.asarray(a=self.W_output).T)  
            assert dA3.shape == np.asarray(a=self.A3).shape, f"Shape mismatch: {dA3.shape} vs {self.A3.shape}"

            dZ3 = np.multiply(dA3, self.d_fct3(np.asarray(a=self.Z3)))  
            dW3 = 1 / m * np.dot(a=np.asarray(a=self.A2).T, b=dZ3)  
            db3 = 1 / m * np.sum(a=np.asarray(a=dZ3), axis=0, keepdims=True) 
            
        if self.h2:
            # Second hidden layer gradients
            if self.h3 :
                dA2 = np.dot(a=dZ3, b=np.asarray(a=self.W3).T)  
                assert dA2.shape == np.asarray(a=self.A2).shape, f"Shape mismatch: {dA2.shape} vs {self.A2.shape}"
            else :
                dA2 = np.dot(a=dZ3, b=np.asarray(a=self.W_output).T)  
                assert dA2.shape == np.asarray(a=self.A2).shape, f"Shape mismatch: {dA2.shape} vs {self.A2.shape}"
            dZ2 = np.multiply(dA2 ,self.d_fct2(np.asarray(a=self.Z2)))  
            dW2 = 1 / m * np.dot(a=np.asarray(a=self.A1).T, b=dZ2)  
            db2 = 1 / m * np.sum(a=np.asarray(a=dZ2), axis=0, keepdims=True) 
        
        if self.h1:
            # First hidden layer gradients
            if self.h2 :
                dA1 = np.dot(a=dZ2, b=np.asarray(a=self.W2).T) 
                assert dA1.shape == np.asarray(a=self.A1).shape, f"Shape mismatch: {dA1.shape} vs {self.A1.shape}"
            else :
                dA1 = np.dot(a=dZ2, b=np.asarray(a=self.W_output).T) 
                assert dA1.shape == np.asarray(a=self.A1).shape, f"Shape mismatch: {dA1.shape} vs {self.A1.shape}"
            dZ1 = np.multiply(dA1,self.d_fct1(np.asarray(a=self.Z1)))  
            dW1 = 1 / m * np.dot(a=np.asarray(a=X).T, b=dZ1)  
            db1 = 1 / m * np.sum(a=np.asarray(a=dZ1), axis=0, keepdims=True)  
  
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
            'pb': self.pb,
            'name': self.name,
            'input_size': self.W1.shape[0],
            'output_size': self.W_output.shape[1],
        }
        save_model(model=model, model_name=self.name)

    def predict(self, X):
        """
        Predict the class probabilities for the input samples X.
        Returns:
        - probabilities: A 2D NumPy array of shape (n_samples, n_classes) containing class probabilities.
        """
        y_pred = self.forward(X=X)  
        return y_pred  




def save_model(model, model_name):
    """
    Saves the trained model to a file using pickle.
    
    Parameters:
    - model: Trained model to save.
    - model_name: str, name to use for the saved model file.
    """
    
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
                                        pb=fi['pb'], name=fi['name'])
        model.W1 = fi['W1']
        model.W2 = fi['W2']
        model.W3 = fi['W3']
        model.W_output = fi['W_output']
        model.b1 = fi['b1']
        model.b2 = fi['b2']
        model.b3 = fi['b3']
        model.b_output = fi['b_output']
    return model  

        
model = ArtificialNeuralNetwork(_shape=(2,6),init_mode='xavier',h1=3,fct1='relu',h2=12,fct2='sigmoid',h3=5,fct3='tanh',fout='softmax',pb='classification',name='ann_model')

model.summary()

Y= model.predict(np.array([[1,2]]))
print(Y)