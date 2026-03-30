import torch
from torch import nn


class TransformerLM(nn.Module):
	def __init__(self, vocab_size: int, block_size: int, embed_dim: int, num_layers: int, num_heads: int, dropout: float):
		super().__init__()
		self.token_emb = nn.Embedding(vocab_size, embed_dim)
		self.pos_emb = nn.Embedding(block_size, embed_dim)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=embed_dim,
			nhead=num_heads,
			dim_feedforward=4 * embed_dim,
			dropout=dropout,
			activation="gelu",
			batch_first=True,
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.ln_f = nn.LayerNorm(embed_dim)
		self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
		self.block_size = block_size

	def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
		bsz, seq_len = idx.size()
		pos = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
		x = self.token_emb(idx) + self.pos_emb(pos)
		mask = torch.triu(torch.ones(seq_len, seq_len, device=idx.device), diagonal=1).bool()
		x = self.encoder(x, mask)
		x = self.ln_f(x)
		logits = self.lm_head(x)
		loss = None
		if targets is not None:
			loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
		return logits, loss

	@torch.no_grad()
	def generate(
		self,
		idx: torch.Tensor,
		max_new_tokens: int,
		temperature: float = 0.7,
		top_k: int | None = None,
		eos_id: int | None = None,
	):
		self.eval()
		for _ in range(max_new_tokens):
			idx_cond = idx[:, -self.block_size :]
			logits, _ = self(idx_cond)
			logits = logits[:, -1, :] / max(temperature, 1e-8)
			if top_k is not None:
				v, _ = torch.topk(logits, top_k)
				logits[logits < v[:, [-1]]] = -float("inf")
			probs = nn.functional.softmax(logits, dim=-1)
			next_id = torch.multinomial(probs, num_samples=1)
			idx = torch.cat([idx, next_id], dim=1)
			if eos_id is not None and next_id.item() == eos_id:
				break
		return idx
