import torch.nn as nn

# 该函数用于判断传入的网络 net 是否已经被包装为多 GPU 版本
def is_multi_gpu(net):
    return isinstance(net, (MultiGPU, nn.DataParallel)) # 如果 net 是 MultiGPU 或 nn.DataParallel 的实例，则返回 True，否则返回 False


class MultiGPU(nn.DataParallel):
    """Wraps a network to allow simple multi-GPU training."""
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)