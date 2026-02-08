import numpy as np

class CharRNN:
    """
    Character-level RNN that learns online.
    """
    def __init__(self, vocab=None, hidden_size=128, learning_rate=1e-1):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        if vocab is None:
            self.vocab = []
        else:
            self.vocab = list(vocab)

        self.char_to_ix = {c:i for i,c in enumerate(self.vocab)}
        self.ix_to_char = {i:c for i,c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        if self.vocab_size > 0:
            self._init_weights()
        else:
            self.Wxh = None
            self.Whh = None
            self.Why = None
            self.bh = None
            self.by = None

    def _init_weights(self):
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size)*0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size)*0.01
        self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.01
        self.bh = np.zeros((self.hidden_size,1))
        self.by = np.zeros((self.vocab_size,1))

    def _update_vocab(self, text):
        for c in text:
            if c not in self.char_to_ix:
                idx = len(self.vocab)
                self.vocab.append(c)
                self.char_to_ix[c] = idx
                self.ix_to_char[idx] = c
                self.vocab_size = len(self.vocab)
        if self.Wxh is None:
            self._init_weights()

    def forward_backward(self, inputs, targets, hprev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size,1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[t],0]+1e-8)

        dWxh,dWhh,dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh,dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])

        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - hs[t]**2) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        for dparam in [dWxh,dWhh,dWhy,dbh,dby]:
            np.clip(dparam, -5,5,out=dparam)

        return loss,dWxh,dWhh,dWhy,dbh,dby,hs[len(inputs)-1]

    def sample(self, seed_idx, n, h=None):
        if h is None:
            h = np.zeros((self.hidden_size,1))
        x = np.zeros((self.vocab_size,1))
        x[seed_idx] = 1
        indices = []
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x)+np.dot(self.Whh,h)+self.bh)
            y = np.dot(self.Why,h)+self.by
            p = np.exp(y)/np.sum(np.exp(y))
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size,1))
            x[idx] = 1
            indices.append(idx)
        return indices

    def online_train(self, text, hprev=None):
        self._update_vocab(text)
        inputs = [self.char_to_ix[c] for c in text[:-1]]
        targets = [self.char_to_ix[c] for c in text[1:]]
        if hprev is None:
            hprev = np.zeros((self.hidden_size,1))
        loss,dWxh,dWhh,dWhy,dbh,dby,hprev = self.forward_backward(inputs, targets, hprev)
        for param,dparam in zip([self.Wxh,self.Whh,self.Why,self.bh,self.by],
                                [dWxh,dWhh,dWhy,dbh,dby]):
            param -= self.learning_rate*dparam
        return hprev, loss

    def save_model(self,path):
        import numpy as np
        np.save(path, {'Wxh':self.Wxh,'Whh':self.Whh,'Why':self.Why,
                       'bh':self.bh,'by':self.by,'vocab':self.vocab})

    def load_model(self,path):
        import numpy as np
        data = np.load(path, allow_pickle=True).item()
        self.Wxh,self.Whh,self.Why = data['Wxh'],data['Whh'],data['Why']
        self.bh,self.by = data['bh'],data['by']
        self.vocab = data['vocab']
        self.char_to_ix = {c:i for i,c in enumerate(self.vocab)}
        self.ix_to_char = {i:c for i,c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
