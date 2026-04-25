import torch

class MiniMindKVCache:
    def __init__(self, layers=None):
        self.layers = list(layers) if layers is not None else []
    
    @classmethod
    def from_past_key_values(cls, past_key_values):
        return cls(list(past_key_values) if past_key_values is not None else [])
    
    def to_past_key_values(self):
        return tuple(self.layers)
    
    def update_from_past_key_values_(self, past_key_values):
        self.layers[:] = list(past_key_values)
        return self
    
    def get_layer(self, layer_idx):
        if layer_idx < len(self.layers):
            return self.layers[layer_idx]
        return None
    
    def update_layer(self, layer_idx, key, value):
        while len(self.layers) <= layer_idx:
            self.layers.append(None)
        self.layers[layer_idx] = (key, value)
    
    def num_layers(self):
        return len(self.layers)
    
    def clear(self):
        self.layers = []
    
    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, idx):
        return self.layers[idx]
    
    def __setitem__(self, idx, value):
        while len(self.layers) <= idx:
            self.layers.append(None)
        self.layers[idx] = value