import torch
from torch import nn

from .data import flatten_images, split_context_target
from .diagnostics import compute_diagnostics


def evaluate_linear_probe(
	encoder: nn.Module,
	train_loader,
	test_loader,
	latent_dim: int = 64,
	is_jepa: bool = False,
	use_context: bool = False,
	device: torch.device | str = "cpu",
	epochs: int = 2,
	learning_rate: float = 1e-3,
) -> tuple[float, dict[str, float]]:
	"""Train a classifier on frozen representations and report test metrics."""
	device = torch.device(device)
	probe = nn.Linear(latent_dim, 10).to(device)
	optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss()
	encoder.eval()
	for _ in range(epochs):
		probe.train()
		for images, labels in train_loader:
			images, labels = images.to(device), labels.to(device)
			inputs = split_context_target(images)[0] if is_jepa or use_context else flatten_images(images)
			with torch.no_grad():
				latents = encoder.encode(inputs)
			optimizer.zero_grad()
			criterion(probe(latents), labels).backward()
			optimizer.step()

	correct = total = 0
	all_latents = []
	probe.eval()
	with torch.no_grad():
		for images, labels in test_loader:
			images, labels = images.to(device), labels.to(device)
			inputs = split_context_target(images)[0] if is_jepa or use_context else flatten_images(images)
			latents = encoder.encode(inputs)
			correct += (probe(latents).argmax(dim=1) == labels).sum().item()
			total += labels.size(0)
			all_latents.append(latents.cpu())
	diagnostics = compute_diagnostics(torch.cat(all_latents, dim=0))
	return round(correct / max(total, 1), 4), diagnostics
