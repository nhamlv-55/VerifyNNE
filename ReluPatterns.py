import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
from BaseNet import BaseNet
import numpy as np
import logging
class Patterns:
    def __init__(self, model: BaseNet, 
                dataloader: DataLoader[Tuple[Tensor, Tensor]], 
                labels: List[int], 
                layers: List[str],
                device: torch.device = torch.device('cpu')):
        self._model = model
        self.label2patterns: Dict[str, np.ndarray] = {}
        self._labels = labels
        self._layers = layers
        self._device = device
        self._dataloader = dataloader
        self._populate()    
        
    def _populate(self):
        _label2patterns = {}
        for label in self._labels:
            patterns: List[np.ndarray] = []
            for data, target in self._dataloader:
                flter = target == label
                data = data[flter]
                logging.debug(data.shape[0])
                pattern = self._model.get_pattern(data, 
                                                self._layers, 
                                                self._device,
                                                flatten = True)['all']
                patterns.append(pattern)

            _label2patterns[label] = np.squeeze(np.concatenate(patterns, axis = 0))

            
            logging.info(_label2patterns[label].shape)
        
        self.label2patterns = _label2patterns 