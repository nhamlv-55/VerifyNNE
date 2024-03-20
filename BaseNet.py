from __future__ import annotations
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy
import numpy as np
from typing import Any, Callable, List, Dict, Tuple
from maraboupy import Marabou, MarabouCore  # type: ignore
from maraboupy.MarabouNetwork import MarabouNetwork
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.operators.leaf import BoundParams
from auto_LiRPA.perturbations import PerturbationLpNorm

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

class BaseNet():
    def __init__(self, pytorch_net: nn.Module):
        super(BaseNet, self).__init__()
        self.tensor_log: Dict[str, Any] = {}
        self.marabou_net: MarabouNetwork
        self.pytorch_net: nn.Module = pytorch_net
        self.forward_bounds: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.grad_bounds: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        self.fw_ipq: MarabouCore.InputQuery
        self.fused_bounds: Dict[int, Tuple[float, float]] = {}

    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.pytorch_net.forward(x)

    def build_marabou_net(self, dummy_input: torch.Tensor)->MarabouNetwork:
        """
            convert the network to MarabouNetwork
        """
        self.pytorch_net.eval()
        tempf = tempfile.NamedTemporaryFile(delete=False)
        torch.onnx.export(self.pytorch_net, dummy_input, tempf.name, verbose=False)

        self.marabou_net = Marabou.read_onnx_plus(tempf.name)
        self.fw_ipq = MarabouCore.InputQuery(self.marabou_net.getForwardQuery())
        return self.marabou_net

    def fusion(self, layer_map: Dict[str, List[int]], label:int):
        """
        match the bounds computer by auto_lirpa with the corresponding 
        node in the marabou network
        Currently required a manually prepare layer_map, since pytorch does not
        preserve node name when exporting the model to ONNX, so matching node between
        marabou (ONNX) and auto-lirpa(Pytorch) hard.
        """
        auto_lirpa_node: str
        for auto_lirpa_node in layer_map:
            if auto_lirpa_node in self.forward_bounds:
                print(f"fusing bounds for forward node {auto_lirpa_node}")
                print(self.forward_bounds[auto_lirpa_node])
                _shape1 = len(layer_map[auto_lirpa_node])
                _shape2 = self.forward_bounds[auto_lirpa_node][0].shape
                assert _shape1 == _shape2[-1], print(_shape1, "!~", _shape2)
                for idx, var in enumerate(layer_map[auto_lirpa_node]):
                    self.fused_bounds[var] = (self.forward_bounds[auto_lirpa_node][0][idx].item(),
                                              self.forward_bounds[auto_lirpa_node][1][idx].item())
            elif auto_lirpa_node in self.grad_bounds:
                print(f"fusing bounds for grad node {auto_lirpa_node}")
                print(self.grad_bounds[auto_lirpa_node])
                assert len(layer_map[auto_lirpa_node]) == self.grad_bounds[auto_lirpa_node][0].shape[-1]
                for idx, var in enumerate(layer_map[auto_lirpa_node]):
                    self.fused_bounds[var] = (self.grad_bounds[auto_lirpa_node][0][idx].item(),
                                              self.grad_bounds[auto_lirpa_node][1][idx].item())
        return self.fused_bounds

    def load_jacobian_bounds(self, pre_computed_bounds: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]):
        print("loading pre computed bounds...")
        self.forward_bounds = pre_computed_bounds["forward_bounds"]
        self.grad_bounds = pre_computed_bounds["grad_bounds"]

    def compute_jacobian_bounds(self, input: torch.autograd.Variable, 
                                eps: float, 
                                label: int, 
                                option: Dict[str, Any]):
        print(f"COMPUTING JACOBIAN WITH EPS={eps}")

        self.lirpa_model = BoundedModule(self.pytorch_net, input)
        self.lirpa_model.set_bound_opts(option) 
        
        self.lirpa_forward_model = BoundedModule(self.pytorch_net, input)
        self.lirpa_forward_model.set_bound_opts(option)
        
        print(self.lirpa_model.node_name_map)
        print(self.marabou_net.shapeMap)
        self.lirpa_model.augment_gradient_graph(input)

        x = BoundedTensor(input, PerturbationLpNorm(norm=np.inf, eps = eps))
        lirpa_layers = list(self.lirpa_model.modules())
        for i in range(1, len(lirpa_layers)):
            if isinstance(lirpa_layers[i], BoundParams): continue
            if "grad" in lirpa_layers[i].name and "tmp" not in lirpa_layers[i].name:
                print(f"Bounding the backward node {lirpa_layers[i]}")
                lb, ub = self.lirpa_model.compute_jacobian_bounds(x, optimize=False, 
                                                                  final_node_name=lirpa_layers[i].name)
                self.grad_bounds[lirpa_layers[i].name] = (lb[label], ub[label])
            elif "grad" not in lirpa_layers[i].name:
                print(f"Bounding the forward node {lirpa_layers[i]}")
                lb, ub = self.lirpa_forward_model.compute_bounds(x=(x,), final_node_name=lirpa_layers[i].name)
                self.forward_bounds[lirpa_layers[i].name] = (lb.squeeze(), ub.squeeze())


    def add_backward_query(self, to_be_abstracted: List[int], abs_domain: str, top_k:int)->None:
        self.marabou_net.addBackwardQuery(to_be_abstracted, self.fused_bounds, 
                                        #   abs_domain=abs_domain, top_k = top_k
                                          )

    def check_network_consistancy(self, verbosity:int = 0)->bool:
        """
            check if the built marabou_net is actually equivalent to the original net
            Strat: generate a random input, and run it through both network. The outputs should be similar, up 
            to a threshold
        """
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
