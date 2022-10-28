import os

from copy import deepcopy
import math
from collections import OrderedDict

from mindspore.common.tensor import Tensor
from mindspore.train.callback import Callback
from mindspore.common.parameter import Parameter
from mindspore._checkparam import Validator
from mindspore.train.serialization import _get_merged_param_data, _exec_save

from threading import Thread, Lock
import time

_ckpt_mutex = Lock()

class ModelEMA(Callback):
    """ 
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    """

    def __init__(self, model, save_checkpoint_steps, save_path, decay, updates = 0):
        # Create EMA
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        self.updates = updates  # number of EMA updates

        self.model = model
        self.save_checkpoint_path =  save_path
        self.save_checkpoint_steps = save_checkpoint_steps

    def begin(self, run_context):
        """
        Called once before the network executing.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.ema = self.get_weight(self.model)

    def step_end(self, run_context):
        """
        Called after each epoch finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        cb_params = run_context.original_args()
  
        network = cb_params.train_network

        epoch_num = cb_params.epoch_num
        batch_num = cb_params.batch_num

        self.update(network)
        if cb_params.cur_step_num % self.save_checkpoint_steps == 0  or cb_params.cur_step_num == epoch_num * batch_num: 
            self.save_ckpt(network)
        

    def get_weight(self, save_obj):
        param_dict = OrderedDict()
        for _, param in save_obj.parameters_and_names():
            param_dict[param.name] = param
        return param_dict
        

    def update(self, current_obj):
        self.updates += 1
        d = self.decay(self.updates)
        current_weight = self.get_weight(current_obj)
        
        for k, v in self.ema.items():
            v *= d
            v += (1. - d) * current_weight[k]
            self.ema[k] = v

    def tolist(self):
        param_list = []
        for (key, value) in self.ema.items():
            each_param = {"name": key}
            param_data = Tensor(value.data.asnumpy())
            each_param["data"] = param_data
            param_list.append(each_param)
        
        return param_list

    def save_ckpt(self, enc_key = None, enc_mode="AES-GCM"):
        
        # Update EMA parameters

        save_obj = self.tolist()

        data_list = OrderedDict()
        with _ckpt_mutex:
            for param in save_obj:
                key = param["name"]
                data_list[key] = []
                if isinstance(param["data"], Parameter):
                    param["data"].init_data()
                dims = []
                if param['data'].shape == ():
                    dims.append(0)
                else:
                    for dim in param['data'].shape:
                        dims.append(dim)
                data_list[key].append(dims)
                tensor_type = str(param["data"].dtype)
                data_list[key].append(tensor_type)
                data = param["data"].asnumpy().reshape(-1)
                data_list[key].append(data)
            
        enc_key = Validator.check_isinstance('enc_key', enc_key, (type(None), bytes))
        enc_mode = Validator.check_isinstance('enc_mode', enc_mode, str)

        ckpt_file_name = os.path.join(self.save_checkpoint_path, "ema.ckpt" ) 
        _exec_save(ckpt_file_name, data_list, enc_key, enc_mode)
