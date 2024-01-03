from rockpool.nn.modules import LinearTorch
from rockpool.nn.combinators import Sequential, Residual
from rockpool.parameters import Constant
from rockpool.nn.modules import LIFTorch

class GlucoseModel:
    def __init__(self, num_inputs, num_hidden, num_outputs, dt):
        self.num_in = num_inputs
        self.num_hidden = num_hidden
        self.num_out = num_outputs
        self.dt = dt
        self.tau_syn = 0.02
        self.tau_mem = 0.02

        self.net = Sequential(
            # First FC layer
            LinearTorch((self.num_in, self.num_hidden), has_bias=False),
            LIFTorch(self.num_hidden, dt=self.dt, tau_syn=self.tau_syn, tau_mem=self.tau_mem),

            # # Third FC layer
            LinearTorch((self.num_hidden, self.num_out), has_bias=False),
            LIFTorch(self.num_out, dt=self.dt, tau_syn=self.tau_syn, tau_mem=self.tau_mem),
        )
        # Scale down recurrent weights for stability
        #self.net[2][1].w_rec.data = self.net[2][1].w_rec / 10. # type: ignore
        print(self.net)

    def __call__(self, x):
        return self.net(x)
