import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        self.s = torch.zeros((args.batch_size, args.state_dim)).to('cuda')
        self.a = torch.zeros((args.batch_size, 1), dtype=torch.long).to('cuda')
        self.a_logprob = torch.zeros((args.batch_size, 1)).to('cuda')
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = torch.zeros((args.batch_size, args.state_dim)).to('cuda')
        self.dw = torch.zeros((args.batch_size, 1)).to('cuda')
        self.done = torch.zeros((args.batch_size, 1)).to('cuda')
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        # s = torch.tensor(self.s, dtype=torch.float)
        # a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        # r = torch.tensor(self.r, dtype=torch.float)
        # s_ = torch.tensor(self.s_, dtype=torch.float)
        # dw = torch.tensor(self.dw, dtype=torch.float)
        # done = torch.tensor(self.done, dtype=torch.float)

        return self.s, self.a, a_logprob, self.r, self.s_, self.dw, self.done
