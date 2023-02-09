import torch

def has_dml():
    """
    # huggingface's transformer/GPT2 model will just lead to a long track of problems
    # I will suck off a wizard if he gets this remedied somehow
    """
    """
    # Note 1:
    # self.inference_model.generate will lead to this error in torch.LongTensor.new:
    #   RuntimeError: new(): expected key in DispatchKeySet(CPU, CUDA, HIP, XLA, MPS, IPU, XPU, HPU, Lazy, Meta) but got: PrivateUse1
    # Patching "./venv/lib/site-packages/transformers/generation_utils.py:1906" with:
    #   unfinished_sequences = input_ids.new_tensor(input_ids.shape[0], device=input_ids.device).fill_(1)
    # "fixes" it, but meets another error/crash about an unimplemented functions.........
    """
    """
    # Note 2:
    # torch.load() will gripe about something CUDA not existing
    # remedy this with passing map_location="cpu"
    """
    """
    # Note 3:
    # stft requires device='cpu' or it'll crash about some error about an unimplemented function I do not remember
    """
    """
    # Note 4:
    # 'Tensor.multinominal' and 'Tensor.repeat_interleave' throws errors about being unimplemented and falls back to CPU and crashes
    """
    return False
    """
    import importlib
    loader = importlib.find_loader('torch_directml')
    return loader is not None
    """

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