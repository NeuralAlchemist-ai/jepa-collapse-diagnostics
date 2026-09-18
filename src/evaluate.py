import torch
from torch import nn

from .data import flatten_images, split_context_target
from .diagnostics import compute_diagnostics


def _prepare_inputs(images: torch.Tensor, is_jepa: bool = False, use_context: bool = False) -> torch.Tensor:
	if is_jepa or use_context:
		return split_context_target(images)[0]
	return flatten_images(images)


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
	weight_decay: float = 0.0,
) -> tuple[float, dict[str, float]]:
	"""Train a classifier on frozen representations and report test metrics."""
	result = evaluate_linear_probe_on_loaders(
		encoder=encoder,
		train_loader=train_loader,
		evaluation_loaders={"standard": test_loader},
		latent_dim=latent_dim,
		is_jepa=is_jepa,
		use_context=use_context,
		device=device,
		epochs=epochs,
		learning_rate=learning_rate,
		weight_decay=weight_decay,
	)
	return result["standard"]["accuracy"], result["standard"]["diagnostics"]


def evaluate_linear_probe_on_loaders(
	encoder: nn.Module,
	train_loader,
	evaluation_loaders: dict[str, object],
	latent_dim: int = 64,
	is_jepa: bool = False,
	use_context: bool = False,
	device: torch.device | str = "cpu",
	epochs: int = 2,
	learning_rate: float = 1e-3,
	weight_decay: float = 0.0,
) -> dict[str, dict[str, float | dict[str, float]]]:
	"""Train one probe and evaluate it on multiple loaders, including shifted data."""
	device = torch.device(device)
	probe = nn.Linear(latent_dim, 10).to(device)
	optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
	criterion = nn.CrossEntropyLoss()
	encoder.eval()
	for _ in range(epochs):
		probe.train()
		for images, labels in train_loader:
			images, labels = images.to(device), labels.to(device)
			inputs = _prepare_inputs(images, is_jepa=is_jepa, use_context=use_context)
			with torch.no_grad():
				latents = encoder.encode(inputs)
			optimizer.zero_grad()
			criterion(probe(latents), labels).backward()
			optimizer.step()

	results: dict[str, dict[str, float | dict[str, float]]] = {}
	probe.eval()
	with torch.no_grad():
		for name, loader in evaluation_loaders.items():
			correct = total = 0
			all_latents = []
			for images, labels in loader:
				images, labels = images.to(device), labels.to(device)
				inputs = _prepare_inputs(images, is_jepa=is_jepa, use_context=use_context)
				latents = encoder.encode(inputs)
				correct += (probe(latents).argmax(dim=1) == labels).sum().item()
				total += labels.size(0)
				all_latents.append(latents.cpu())
			accuracy = round(correct / max(total, 1), 4)
			diagnostics = compute_diagnostics(torch.cat(all_latents, dim=0))
			results[name] = {"accuracy": accuracy, "diagnostics": diagnostics}
	return results
