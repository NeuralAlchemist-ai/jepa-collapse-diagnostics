import torch
from torch import nn


class AutoEncoder(nn.Module):
	def __init__(self, input_dim: int = 392, latent_dim: int = 64, hidden_dim: int = 107) -> None:
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim)
		)
		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim)
		)

	def encode(self, x: torch.Tensor) -> torch.Tensor:
		return self.encoder(x)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.decoder(self.encode(x))


class JEPA(nn.Module):
	def __init__(self, input_dim: int = 392, latent_dim: int = 64) -> None:
		super().__init__()
		hidden_dim = input_dim // 2
		self.context_encoder = nn.Sequential(
			nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim)
		)
		self.target_encoder = nn.Sequential(
			nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim)
		)
		self.predictor = nn.Sequential(
			nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
		)
		self.target_encoder.load_state_dict(self.context_encoder.state_dict())
		for parameter in self.target_encoder.parameters():
			parameter.requires_grad = False

	def encode(self, context_data: torch.Tensor) -> torch.Tensor:
		return self.context_encoder(context_data)

	def forward(self, context_data: torch.Tensor, target_data: torch.Tensor):
		context_latent = self.context_encoder(context_data)
		with torch.no_grad():
			target_latent = self.target_encoder(target_data)
		return self.predictor(context_latent), target_latent

	@torch.no_grad()
	def update_target_encoder(self, beta: float = 0.99) -> None:
		for context_parameter, target_parameter in zip(
			self.context_encoder.parameters(), self.target_encoder.parameters()
		):
			target_parameter.mul_(beta).add_(context_parameter, alpha=1.0 - beta)


class CollapsedControl(nn.Module):
	"""Negative control that emits an identical representation for every input."""

	def __init__(self, latent_dim: int = 64) -> None:
		super().__init__()
		self.latent_dim = latent_dim

	def encode(self, x: torch.Tensor) -> torch.Tensor:
		return torch.ones(x.size(0), self.latent_dim, device=x.device)
