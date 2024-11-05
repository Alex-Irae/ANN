# Artificial Neural Network
This project is an Artificial Neural Network (ANN) built from scratch using numpy. It allows configuring functions, layers, input/output dimensions, learning rates, and batch sizes. The ANN can train and predict, and it can be customized to any extent with a maximum of 3 layers.

## Features
Customizable layers (up to 3 hidden layers)
Various activation functions (ReLU, Sigmoid, Tanh, Swish, Softmax)
Supports different initialization methods (He, Xavier)
L2 regularization
Different loss function for different problems
Train and predict capabilities
Modifiable learning rates and batch sizes
Save and load trained models using pickle

## Installation
To use this project, you only need numpy. You can install it using pip:
```bash
pip install numpy
```

### Import
```python
import numpy as np
from ann import ArtificialNeuralNetwork
```

## How to use
### Create dataset
```python
X_train = np.random.rand(100, 10)  # 100 samples, 10 features
y_train = np.random.randint(0, 2, (100, 2))  # 100 samples, 2 classes (one-hot encoded)
```

### Initialize and train the ANN
```python
ann = ArtificialNeuralNetwork(inp=10, out=2, h1=64,h2=None,h3=None, fct1='relu',fct2=None,fct3=None fout='softmax',pb ='classification', name='My_Ann')
ann.train(X_train, y_train, epochs=500, learning_rate=0.01, batch_size=32)
```

### Make predictions
```python
predictions = ann.predict(X_train)
print(predictions)
```
### Training
To train the ANN, use the train() method:
```python
ann.train(X_train, y_train, epochs=1000, learning_rate=0.01, batch_size=32)
```
### Saving and Loading Models
Save the trained model:
```python
ann.save_model(model=ann, model_name='my_ann_model')
```
### Load a saved model:
```python
ann = load_model(model_name='my_ann_model')
```

## License
This project is licensed under the MIT License.
