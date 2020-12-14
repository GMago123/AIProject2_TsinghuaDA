import os

class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"


class Config():
    def __init__(self, args):
        self.global_running = vars(args)
        # hyper parameters
        self.HYPER_MODEL_MAP = {
            'unet': self.__unet
        }
    
    def __unet(self):
        params = {
            'ModelParam': {
                
                'n_channels': 3,
                'n_classes': 1,
                'feature_maps': 32,
                'bilinear': False,
            },

            'DataParam' : {
                'crop_size': 384,
                'crop' : True,
                'batch_size': 5,
            },

            'TrainParam': {
                'early_stop': 6,
                'scheduler': True,
                'scheduler_criterion': 'f1',

                'valid' : 0.15,
                
                'optim': 'rmsprop',

                'lr': 0.0001,
                'momentum': 0.9,
                'weight_decay': 1e-8,
                'epochs': 40,
                'save_figure_path': './figure',         # 训练指标图像保存目录

                'center_enforcement': False,

            },

            'ProcessingParam': {
                'display': False,
                'colored': False,
                'MORPH_KERNEL' : 3,         # CLOSE操作膨胀核大小
                'MORPH_ITERATIONS' : 3,     # CLOSE操作迭代次数
                'DILATE_ITERATIONS': 3,     # DILATE膨胀迭代次数(获取背景)
                'ERODE_ITERATIONS': 4,      # ERODE腐蚀迭代次数
                'DIST_THRESHOLD' : 0.4,     # 寻找中心区域时，距离阈值
                'REAL_THRESHOLD' : 0.5,     # 细胞区域概率阈值
            }
        }

        return params


    def get_config(self):
        model_name = str.lower(self.global_running['modelName'])

        # integrate all parameters.
        res = Storage(dict(self.global_running,
                    **self.HYPER_MODEL_MAP[model_name]()['ModelParam'],
                    **self.HYPER_MODEL_MAP[model_name]()['TrainParam'],
                    **self.HYPER_MODEL_MAP[model_name]()['ProcessingParam'],
                    **self.HYPER_MODEL_MAP[model_name]()['DataParam']))
        return res