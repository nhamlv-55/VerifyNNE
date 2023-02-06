from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy
import numpy as np
from typing import Any, Callable, List, Dict, Tuple
from maraboupy import Marabou, MarabouCore
from maraboupy.MarabouNetwork import MarabouNetwork

import tempfile

THRESHOLD = 10**-6

def get_activation(name: str, tensor_logger: Dict[str, Any], 
                    detach: bool = True, is_lastlayer:bool = False)->Callable[..., None]:
    if is_lastlayer:
        def hook(model: torch.nn.Module, input: Tensor, output: Tensor)->None:
            raw = torch.flatten(output, start_dim = 1, end_dim = -1).cpu().detach().numpy()
            #use argmax instead of broadcasting just in case comparing floating point is finicky
            
            mask = np.zeros(raw.shape, dtype = bool)
            
            mask[np.arange(raw.shape[0]), raw.argmax(axis=1)] = 1

            
            tensor_logger[name] = np.concatenate((tensor_logger[name], mask), 
                                                axis = 0) if name in tensor_logger else mask
        return hook

    if detach:
        def hook(model: torch.nn.Module, input: Tensor, output: Tensor)->None:
            raw = torch.flatten(
                output, start_dim=1, end_dim=-1).cpu().detach().numpy()
            raw = raw > 0
            logging.debug("{}, {}".format(name,raw.shape))
            tensor_logger[name] = np.concatenate((tensor_logger[name], raw), 
                                                axis = 0) if name in tensor_logger else raw
            logging.debug(tensor_logger[name].shape)
        return hook
    else:
        #keep the gradient, so cannot convert to bit here
        def hook(model: torch.nn.Module, input: Tensor, output: Tensor):
            raw = torch.sigmoid(torch.flatten(
                output, start_dim=1, end_dim=-1))
            logging.debug("{}, {}".format(name,raw.shape))
            tensor_logger[name] = torch.cat([tensor_logger[name], raw], 
                                                dim = 0) if name in tensor_logger else raw
            logging.debug(tensor_logger[name].shape)
        return hook

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.tensor_log: Dict[str, Any] = {}
        self.gradient_log = {}
        self.hooks: List[Callable[..., None]] = []
        self.bw_hooks = []
        self.marabou_net: MarabouNetwork
        self.pytorch_net: nn.Module
        self.fw_ipq: MarabouCore.InputQuery 
    def build_marabou_net(self, dummy_input: torch.Tensor)->MarabouNetwork:
        """
            convert the network to MarabouNetwork
        """
        self.eval()
        tempf = tempfile.NamedTemporaryFile()
        torch.onnx.export(self, dummy_input, tempf.name, verbose=False)
        self.marabou_net = Marabou.read_onnx_plus(tempf.name)
        return self.marabou_net

    def build_marabou_ipq(self)->MarabouCore.InputQuery:
        """
            build the Marabou Query from the internal marabou_net
            Note: addBackwardQuery assumes the correct accumulateGrad 
        """
        self.fw_ipq = MarabouCore.InputQuery(self.marabou_net.getForwardQuery())
        ipq = self.marabou_net.addBackwardQuery()
        return ipq


    def check_network_consistancy(self, verbosity:int = 0)->bool:
        """
            check if the built marabou_net is actually equivalent to the original net
            Strat: generate a random input, and run it through both network. The outputs should be similar, up 
            to a threshold
        """
        if self.marabou_forward_net is None:
            return False
        options = Marabou.createOptions(verbosity = verbosity)
        input_shape: Tuple[int] = self.marabou_forward_net.inputVars[0].shape
        dummy_inputs: List[torch.Tensor] = [torch.rand(input_shape)]
        marabou_output: List[np.ndarray] = self.marabou_forward_net.evaluateWithMarabou(inputValues=dummy_inputs, options=options)
        internal_output: torch.Tensor = self.forward(torch.stack(dummy_inputs))

        marabou_output_flat:torch.Tensor = torch.Tensor(marabou_output[0]).squeeze().flatten()
        internal_output_flat:torch.Tensor = internal_output[0].squeeze().flatten()
        for idx in range(len(marabou_output_flat)):
            if abs(marabou_output_flat[idx] - internal_output_flat[idx]) > THRESHOLD:
                print("Built marabou network is NOT consistent\n Test outputs:{} != {}".format(marabou_output_flat, internal_output_flat))
                return False
        logging.info("Built marabou network is consistent. Test outputs:{} vs {}".format(marabou_output_flat, internal_output_flat))
        return True
        

    def reset_hooks(self):
        self.tensor_log = {}
        for h in self.hooks:
            h.remove()
            
    def reset_bw_hooks(self):
        self.input_labels = None
        self.gradient_log = {}
        for h in self.bw_hooks:
            h.remove()
            
    def register_log(self, detach: bool)->None:
        raise NotImplementedError

    def register_gradient(self, detach: bool)->None:
        raise NotImplementedError
        
    def model_savename(self)->str:
        raise NotImplementedError
        
    def get_pattern(self, input: Tensor, 
                        layers: List[str], 
                        device: torch.device, 
                        detach: bool = True,
                        flatten:bool = True)->Dict[str, np.ndarray]:
        self.eval()
        self.register_log(detach)
        self.forward(input.to(device))
        tensor_log = copy.deepcopy(self.tensor_log)
        if flatten:
            return {'all': np.concatenate([tensor_log[l] for l in layers], axis=1)}
        return tensor_log
