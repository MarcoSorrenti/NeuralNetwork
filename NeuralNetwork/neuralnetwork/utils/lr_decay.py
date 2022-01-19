class Linear_decay:
    def __init__(self, lr, tau=500):
        self.eta_0 = lr
        self.tau = tau
        self.eta_t = lr * 0.01
        self.eta_s = lr

    def decay(self, iter):
        if iter < self.tau and self.eta_s > self.eta_t:
            alpha = iter/self.tau
            self.eta_s = (1-alpha)*self.eta_0 + alpha*self.eta_t
            return self.eta_s
        
        return self.eta_t
    