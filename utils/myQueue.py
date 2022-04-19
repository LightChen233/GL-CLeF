import torch


class Queue:
    def __init__(self, hidden_size, seq_length, maxsize=-1, batch_size=-1):
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.maxsize = maxsize * 2
        self.size = 0
        self.queue = []

    def enqueue(self, value):
        if self.maxsize > self.size:
            self.queue.insert(0, value)
            self.size = self.size + 1
        else:
            self.dequeue()
            self.size = self.size + 1
            self.queue.insert(0, value)

    def enqueue_batch_tensor(self, batch_value):
        batch_value = batch_value.reshape(-1, self.seq_length, self.hidden_size)
        N = batch_value.shape[0]
        for temp in range(N):
            self.enqueue(batch_value[temp].T)

    def negative_encode(self, batch_size):
        t = None
        for tt in self.queue:
            tt = tt.unsqueeze(2)
            if t == None:
                t = tt
            else:
                t = torch.cat((t, tt), 2)
        temp = None
        for i in range(batch_size):
            if temp == None:
                temp = t.reshape(1,self.hidden_size, self.seq_length,-1)
            else:
                temp = torch.cat((temp, t.reshape(1,self.hidden_size, self.seq_length,-1)), 0)
        return temp

    def full(self):
        return self.size == self.maxsize

    def dequeue(self):
        if self.size > 0:
            self.size = self.size - 1
            return self.queue.pop()
        else:
            return None

    def empty(self):
        return self.size == 0
