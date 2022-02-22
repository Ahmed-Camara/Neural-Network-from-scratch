class NeuralNetwork(object):
    
    def __init__(self,iters=2000,learning_rate=0.01,layer_dims=[]):
        self.iters = iters
        self.learning_rate = learning_rate
        self.layer_dims = layer_dims
        self.parameters = {}
        
    def initialization(self):
        L = len(self.layer_dims)
        print(L)
        print('***************************************************************************************');
        for l in range(1,L):
            print(l,' => ',self.layer_dims[l])
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
    
    def forward_activation(A_prev, W, b, activation):
        
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
            A,cache = forward_activation(A_prev,self.parameters['W'+str(l)],self.parameters['b'+str(l)],'relu')
            caches.append(cache)
        AL,cache = forward_activation(A,self.parameters['W'+str(L)],self.parameters['b'+str(L)],'sigmoid')
        caches.append(cache)
        return AL,caches
    
    def updateParameters(self):
        pass
    def fit(self,x,y):
        pass
    def predict(self,x):
        pass
    def score(self,x,y):
        pass