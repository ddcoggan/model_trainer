import os
import os.path as op
import sys
from types import SimpleNamespace

import torch.nn as nn
import math
from pathlib import Path
sys.path.append(op.expanduser('~/david/master_scripts/DNN'))
import zoo
from torchvision import models

def get_model(architecture, kwargs):

	if architecture in models.list_models():
		try:
			model = getattr(models, architecture)(**kwargs)
		except:
			ValueError('kwargs not accepted for this model')
	else:
		try:
			model = getattr(zoo, architecture)(**kwargs)
		except:
			try:
				model = getattr(zoo, architecture)(kwargs)
			except:
				ValueError('kwargs not accepted for this model')

	return model
