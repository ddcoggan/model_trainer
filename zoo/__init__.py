import os.path as op
import sys
from types import SimpleNamespace

from .cornet_r import CORnet_R as cornet_r
from .cornet_rt import CORnet_RT as cornet_rt
from .cornet_rt_hw3 import CORnet_RT as cornet_rt_hw3
from .cornet_rt_output_avpool import CORnet_RT as cornet_rt_output_avpool
from .cornet_s import CORnet_S as cornet_s
from .cornet_s_V1 import CORnet_S as cornet_s_V1
from .cornet_s_V1_v2 import CORnet_S as cornet_s_V1_v2
from .cornet_s_V1_v3 import CORnet_S as cornet_s_V1_v3
from .cornet_s_V1_v4 import CORnet_S as cornet_s_V1_v4
from .cornet_s_V1_v5 import CORnet_S as cornet_s_V1_v5
from .cornet_s_V1_v6 import CORnet_S as cornet_s_V1_v6
from .cornet_s_FovealBlock import CORnet_S as cornet_s_FovealBlock
from .cornet_s_unshared import CORnet_S as cornet_s_unshared
from .cornet_s_output_avpool import CORnet_S as cornet_s_output_avpool
from .cornet_s_cont import CORnet_S_cont as cornet_s_cont
from .cornet_s_custom import CORnet_S_custom as cornet_s_custom
from .cornet_s_plus import CORnet_S_custom as cornet_s_plus
from .cornet_st import CORnet_ST as cornet_st
from .cornet_flab import CORnet_FLaB as cornet_flab
from .cornet_z import CORnet_Z as cornet_z
from .cornet_s_hw7 import CORnet_S as cornet_s_hw7
from .cornet_s_hw3 import CORnet_S as cornet_s_hw3
from .cornet_s_hd2_hw3 import CORnet_S as cornet_s_hd2_hw3
#from .cornet_s_custom_predify import *
from .locCon1HL import *
from .cognet.cognet import CogNet as cognet
from .cognet.cognet_v2 import CogNet as cognet_v2
from .cognet.cognet_v3 import CogNet as cognet_v3
from .cognet.cognet_v4 import CogNet as cognet_v4
from .cognet.cognet_v5 import CogNet as cognet_v5
from .cognet.cognet_v6 import CogNet as cognet_v6
from .cognet.cognet_v7 import CogNet as cognet_v7
from .cognet.cognet_v8 import CogNet as cognet_v8
from .cognet.cognet_v9 import CogNet as cognet_v9
from .cognet.cognet_v10 import CogNet as cognet_v10
from .cognet.cognet_v11 import CogNet as cognet_v11
from .cognet.cognet_v12 import CogNet as cognet_v12
from .cognet.cognet_v13 import CogNet as cognet_v13
from .cognet.cognet_v14 import CogNet as cognet_v14
from .cognet.cognet_v15 import CogNet as cognet_v15
from .cognet.cognet_v16 import CogNet as cognet_v16
from .cognet.cognet_v17 import CogNet as cognet_v17
from .cognet.cognet_v18 import CogNet as cognet_v18
from .cognet.cognet_v19 import CogNet as cognet_v19
from .cognet.cognet_v20 import FLaB as cognet_v20
from .cognet.cognet_v21 import FLaB as cognet_v21
from .cognet.cognet_v22 import FLaB as cognet_v22
from .cognet.cognet_v23 import FLaB as cognet_v23
from .cognet.cognet_v24 import FLaB as cognet_v24
from .cognet.cognet_v25 import FLaB as cognet_v25
from .cognet.cognet_v26 import FLaB as cognet_v26
from .cognet.cognet_v27 import FLaB as cognet_v27
from .GaborFilterBank import GaborFilterBank
from .alexnet_V1 import AlexNet as alexnet_V1

#from . import segmentation
#from . import detection
#from . import video
#from . import quantization

if op.isdir(op.expanduser('~/data/repos')):
    sys.path.append(op.expanduser('~/data/repos/PredNet_pytorch'))
    from prednet import PredNet as prednet

    sys.path.append(op.expanduser('~/data/repos/pix2pix/gan'))
    from generator import UnetGenerator as pix2pix

    sys.path.append(op.expanduser('~/data/repos/BLT_recurrent_models'))
    from models.build_model import build_model as blt

    sys.path.append(op.expanduser('~/data/repos/pytorch_hmax'))
    from hmax import HMAX
    hmax = lambda: HMAX.HMAX(sys.path.append(op.expanduser('~/data/repos/pytorch_hmax'
                          '/universal_patch_set.mat')))


