import torch.nn as nn

def change_output_size(model, output_size):
	output_module, output_layer = list(model.named_modules())[-1]
	assert type(output_layer) == nn.Linear, 'Output layer is not a linear layer'
	in_features = output_layer.in_features
	bias = output_layer.bias is not None
	setattr(model, output_module, nn.Linear(in_features, output_size, bias))
	return model
