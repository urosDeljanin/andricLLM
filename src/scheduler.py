import math
from torch.optim.lr_scheduler import LambdaLR


def create_cosine_with_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_fraction: float = 0.1):
	
	def lr_lambda(current_step: int):
		if current_step < warmup_steps:
			return float(current_step) / float(max(1, warmup_steps))
		
		progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
		progress = min(1.0, progress)
		cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
		return min_lr_fraction + (1.0 - min_lr_fraction) * cosine_decay
	
	return LambdaLR(optimizer, lr_lambda)
