class NeuralNetwork(object):
    
    def __init__(self,iters=2000,learning_rate=0.01,layer_dims=[8,7,5,1]):
        self.iters = iters
        self.learning_rate = learning_rate
        self.layer_dims = layer_dims
        self.parameters = {}
        
    def initialization(self):
        L = len(self.layer_dims)
        for l in range(1,L):
            self.parameters['W'+str(l)] = np.random.randn(self.layer_dims[l],self.layer_dims[l-1]) * 0.01
            self.parameters['b'+str(l)] = np.random.randn(self.layer_dims[l],1)
    def sigmoid(self,Z):
        A = 1 / (1 + np.exp(-Z))
        cache = Z
        return A,cache
    
    def relu(self,Z):
        A = np.maximum(0,Z)
        cache = Z
        return A,cache
    
    def relu_backward(self,dA,cache):
        Z = cache
        dZ = np.array(dA,copy=True)
        return dZ
    
    def sigmoid_backward(self,dA,cache):
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ
    
    def computeCost(self,AL,Y):
        m = Y.shape[1]
        cost = -(1.0 / m) * np.sum(np.multiply(np.log(AL), Y) + np.multiply((1 - Y), np.log(1 - AL)))
        cost = np.squeeze(cost)
        return cost
    
    def linear_forward(self,A,W,b):
        Z = np.dot(W,A) + b
        cache = (A,W,b)
        return Z,cache
    
    def forward_activation(self,A_prev, W, b, activation):
        
        Z,linear_cache = self.linear_forward(A_prev,W,b)
        if activation == 'sigmoid':
            A,activation_cache = self.sigmoid(Z)
        elif activation == 'relu':
            A,activation_cache = self.relu(Z)
        cache = (linear_cache,activation_cache)
        return A,cache
    
    def forward(self,x):
        caches = []
        A = x
        L = len(self.parameters) // 2
        for l in range(1,L):
            A_prev = A
            A,cache = self.forward_activation(A_prev,self.parameters['W'+str(l)],self.parameters['b'+str(l)],'relu')
            caches.append(cache)
        
        AL,cache = self.forward_activation(A,self.parameters['W'+str(L)],self.parameters['b'+str(L)],'sigmoid')
        caches.append(cache)
        return AL,caches
    
    def linear_backward(self,dZ,cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        
        dW = (1.0/m) * np.dot(dZ,A_prev.T)
        db = (1.0 / m) * np.sum(dZ,axis=1,keepdims=True)
        dA_prev = np.dot(W.T,dZ)
        return dA_prev,dW,db
    
    def backward_activation(self,dA,cache,activation):
        linear_cache,activation_cache = cache
        if activation == 'relu':
            dZ = self.relu_backward(dA,activation_cache)
            dA_prev,dW,db = self.linear_backward(dZ,linear_cache)
        elif activation == 'sigmoid':
            dZ = self.sigmoid_backward(dA,activation_cache)
            dA_prev,dW,db = self.linear_backward(dZ,linear_cache)
        return dA_prev,dW,db
    
    def backward(self,AL,Y,caches):
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y,AL) - np.divide(1 - Y,1 - AL))
        
        "Get last layer"
        current_cache = caches[L-1]
        grads['dA'+str(L-1)],grads['dW'+str(L)],grads['db'+str(L)] = self.backward_activation(dAL,current_cache,'sigmoid')
        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.backward_activation(grads["dA" + str(l + 1)], current_cache, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            
        return grads
    
    def updateParameters(self,grads):
        L = len(self.parameters) // 2
        for l in range(L):
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - self.learning_rate * grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - self.learning_rate * grads["db" + str(l+1)]
    def fit(self,x,y):
        
        x,y = self.reshapeInput(x,y)
        self.layer_dims.insert(0,x.shape[0])
        
        self.initialization()
        
        for i in range(self.iters):
            AL,caches = self.forward(x)
            cost = self.computeCost(AL,y)
            grads = self.backward(AL,y,caches)
            self.updateParameters(grads)
            if i % 100 == 0:
                print(f'cost at iteration {i} : {cost}')
    def reshapeInput(self,x,y):
        if y is None:
            x = x.T
            return x
        else:
            x = x.T
            y = y.reshape((1,-1))
            return x,y
    def predict(self,x):
        x = self.reshapeInput(x,None)
        m = x.shape[1]
        n = len(self.parameters) // 2
        p = np.zeros((1,m))
        probas, caches = self.forward(x)
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        return p
    def score(self,p,y):
        y = y.reshape((1,-1))
        m = y.shape[1]
        accuracy = np.sum((p == y)/m)
        return accuracy