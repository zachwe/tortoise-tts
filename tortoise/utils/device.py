import torch

def has_dml():
    import importlib
    loader = importlib.find_loader('torch_directml')
    return loader is not None

def get_device_name():
    name = 'cpu'

    if has_dml():
        name = 'dml'
    elif torch.cuda.is_available():
        name = 'cuda'

    return name

def get_device(verbose=False):
    name = get_device_name()

    if verbose:
        if name == 'cpu':
            print("No hardware acceleration is available, falling back to CPU...")    
        else:
            print(f"Hardware acceleration found: {name}")

    if name == "dml":
        import torch_directml
        return torch_directml.device()

    return torch.device(name)

def get_device_batch_size():
    if torch.cuda.is_available():
        _, available = torch.cuda.mem_get_info()
        availableGb = available / (1024 ** 3)
        if availableGb > 14:
            return 16
        elif availableGb > 10:
            return 8
        elif availableGb > 7:
            return 4
    return 1

if has_dml():
    _cumsum = torch.cumsum
    _repeat_interleave = torch.repeat_interleave
    _multinomial = torch.multinomial
    
    _Tensor_new = torch.Tensor.new
    _Tensor_cumsum = torch.Tensor.cumsum
    _Tensor_repeat_interleave = torch.Tensor.repeat_interleave
    _Tensor_multinomial = torch.Tensor.multinomial

    torch.cumsum = lambda input, *args, **kwargs: ( _cumsum(input.to("cpu"), *args, **kwargs).to(input.device) )
    torch.repeat_interleave = lambda input, *args, **kwargs: ( _repeat_interleave(input.to("cpu"), *args, **kwargs).to(input.device) )
    torch.multinomial = lambda input, *args, **kwargs: ( _multinomial(input.to("cpu"), *args, **kwargs).to(input.device) )
    
    torch.Tensor.new = lambda self, *args, **kwargs: ( _Tensor_new(self.to("cpu"), *args, **kwargs).to(self.device) )
    torch.Tensor.cumsum = lambda self, *args, **kwargs: ( _Tensor_cumsum(self.to("cpu"), *args, **kwargs).to(self.device) )
    torch.Tensor.repeat_interleave = lambda self, *args, **kwargs: ( _Tensor_repeat_interleave(self.to("cpu"), *args, **kwargs).to(self.device) )
    torch.Tensor.multinomial = lambda self, *args, **kwargs: ( _Tensor_multinomial(self.to("cpu"), *args, **kwargs).to(self.device) )