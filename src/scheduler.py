import math
from torch.optim.lr_scheduler import LambdaLR


def create_cosine_with_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_fraction: float = 0.1):
	"""
	Crea LR scheduler sa linear warmup-om i cosine decay-om.
	
	Args:
		optimizer: Optimizer instance
		warmup_steps: Number of steps for linear warmup phase
		total_steps: Total number of training steps
		min_lr_fraction: Minimum LR as fraction of initial LR (default: 0.1)
	
	Returns:
		LambdaLR scheduler that adjusts learning rate
	"""
	
	def lr_lambda(current_step: int):
		if current_step < warmup_steps:
			# Linear warmup phase
			return float(current_step) / float(max(1, warmup_steps))
		
		# Cosine decay phase
		progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
		# Clamp progress na [0, 1] da izbegnes oscilacije
		progress = min(1.0, progress)
		# Cosine annealing: from 1 to min_lr_fraction
		cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
		return min_lr_fraction + (1.0 - min_lr_fraction) * cosine_decay
	
	return LambdaLR(optimizer, lr_lambda)
