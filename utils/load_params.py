import torch

def load_params(params, dst_object, object_type, modules='all'):

	# if the path to params is submitted (rather than params themselves), load params
	if type(params) == str:
		params = torch.load(params, weights_only=False,
							map_location=torch.device('cpu'))

	key_config = {
		'model': ['model', 'state_dict'],
		'swa_model': ['swa_model'],
		'optimizer': ['optimizer'],
		'scheduler': ['scheduler'],
		'swa_optimizer': ['swa_optimizer'],
		'swa_scheduler': ['swa_scheduler']}

	# if first level of params is model weights
	if sum(['weight' in key for key in params.keys()]):
		src_params = params
	# if model weights are nested in a key
	else:
		state_dict_found = False
		for key in key_config[object_type]:
			if key in params:
				src_params = params[key]
				state_dict_found = True
				break
		assert state_dict_found, (f'No state dict found in params file, '
								  f'searched for {key_config[object_type]}')

	# resolve key errors arising when 'module.' is prefixed to either the source or dest keys
	dst_keys = dst_object.state_dict().keys()
	dst_wrapped = list(dst_keys)[0].startswith('module')
	src_wrapped = list(src_params.keys())[0].startswith('module')
	if dst_wrapped and not src_wrapped:
		src_params = {f'module.{key}': values for key, values in src_params.items()}
	elif src_wrapped and not dst_wrapped:
		while list(src_params.keys())[0].startswith('module'):
			src_params = {key[7:]: values for key, values in src_params.items()}
	src_keys = src_params.keys()

	# load entire parameter set
	if modules == 'all':
		try:
			dst_object.load_state_dict(src_params)
		except:
		#if set(dst_keys) != set(src_keys):
			for dst_key in dst_keys:
				if dst_key not in src_keys:
					UserWarning(f'no weights for layer {dst_key} found in '
								f'params file, original weights will be kept')
					src_params[dst_key] = dst_object.state_dict()[dst_key]
			dst_object.load_state_dict(src_params)

	# load subset of params e.g. for transfer learning or partially 
	# matching architectures
	else:
		for module in modules:

			# get destination module and state dict
			dst_module = getattr(dst_object, module)
			dst_params_mod = dst_module.state_dict()

			# create source state dict for this module
			src_params_mod = {key[len(module) + 1:]: src_params[key] for key in src_params
							  if key.startswith(module)}
			src_keys_mod = src_params_mod.keys()

			# if state dicts have matching keys, load params
			if dst_params_mod.keys() == set(src_keys_mod):
				dst_module.load_state_dict(src_params_mod)
			else:
				# make new src dict with appropriate keys
				new_src_params_mod = {}
				for dst_key, dst_tensor in dst_params_mod.items():
					# copy params over if key matches
					if dst_key in src_keys_mod:
						new_src_params_mod[dst_key] = src_params_mod[dst_key]
					else:
						# try finding params by matching ending of key and shape of tensor
						for src_key, src_tensor in src_params_mod.items():
							if src_key.endswith(key) and src_tensor.shape == dst_tensor.shape:
								new_src_params_mod[dst_key] = src_params_mod[src_key]
								break
						# if this fails, copy the weights
						UserWarning(f'no weights for layer {dst_key} found in params file, '
									f'original weights will be kept')
						new_src_params_mod[dst_key] = dst_params_mod[dst_key]
				dst_module.load_state_dict(new_src_params_mod)
	
	return dst_object
