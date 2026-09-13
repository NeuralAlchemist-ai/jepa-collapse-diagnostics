import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from src.data import get_mnist_loaders, split_context_target
from src.evaluate import evaluate_linear_probe
from src.models import AutoEncoder, CollapsedControl, JEPA


def train_autoencoder(model, loader, device, epochs, learning_rate):
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	model.train()
	for epoch in range(epochs):
		total_loss = 0.0
		for images, _ in loader:
			images = split_context_target(images.to(device))[0]
			optimizer.zero_grad()
			loss = nn.functional.mse_loss(model(images), images)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		print(f"AutoEncoder epoch {epoch + 1}/{epochs}: {total_loss / len(loader):.4f}")


def train_jepa(model, loader, device, epochs, learning_rate):
	optimizer = torch.optim.Adam(
		list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
		lr=learning_rate,
	)
	model.train()
	for epoch in range(epochs):
		total_loss = 0.0
		for images, _ in loader:
			context, target = split_context_target(images.to(device))
			optimizer.zero_grad()
			prediction, target_latent = model(context, target)
			loss = nn.functional.mse_loss(prediction, target_latent)
			loss.backward()
			optimizer.step()
			model.update_target_encoder()
			total_loss += loss.item()
		print(f"JEPA epoch {epoch + 1}/{epochs}: {total_loss / len(loader):.4f}")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--probe-epochs", type=int, default=2)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--max-train-samples", type=int)
	parser.add_argument("--max-test-samples", type=int)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.use_deterministic_algorithms(True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	train_loader, test_loader = get_mnist_loaders(
		batch_size=args.batch_size,
		max_train_samples=args.max_train_samples,
		max_test_samples=args.max_test_samples,
		seed=args.seed,
	)

	autoencoder = AutoEncoder().to(device)
	train_autoencoder(autoencoder, train_loader, device, args.epochs, 1e-3)
	ae_accuracy, ae_diagnostics = evaluate_linear_probe(
		autoencoder, train_loader, test_loader, use_context=True, device=device, epochs=args.probe_epochs
	)

	jepa = JEPA().to(device)
	train_jepa(jepa, train_loader, device, args.epochs, 1e-3)
	jepa_accuracy, jepa_diagnostics = evaluate_linear_probe(
		jepa, train_loader, test_loader, is_jepa=True, device=device, epochs=args.probe_epochs
	)

	collapsed = CollapsedControl().to(device)
	collapsed_accuracy, collapsed_diagnostics = evaluate_linear_probe(
		collapsed, train_loader, test_loader, device=device, epochs=args.probe_epochs
	)
	def parameter_count(model, trainable_only=False):
		return sum(
			parameter.numel()
			for parameter in model.parameters()
			if not trainable_only or parameter.requires_grad
		)

	report = {
		"seed": args.seed,
		"device": str(device),
		"primary_metric": "effective_rank",
		"config": {
			"seed": args.seed,
			"epochs": args.epochs,
			"probe_epochs": args.probe_epochs,
			"batch_size": args.batch_size,
			"max_train_samples": args.max_train_samples,
			"max_test_samples": args.max_test_samples,
			"deterministic_algorithms": True,
		},
		"parameter_counts": {
			"autoencoder_trainable": parameter_count(autoencoder, trainable_only=True),
			"jepa_trainable": parameter_count(jepa, trainable_only=True),
			"jepa_total": parameter_count(jepa),
		},
		"autoencoder": {"linear_probe_accuracy": ae_accuracy, **ae_diagnostics},
		"jepa": {"linear_probe_accuracy": jepa_accuracy, **jepa_diagnostics},
		"collapsed_control": {"linear_probe_accuracy": collapsed_accuracy, **collapsed_diagnostics},
	}
	base_name = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%fZ")
	run_dir = Path("results") / base_name
	suffix = 1
	while run_dir.exists():
		run_dir = Path("results") / f"{base_name}_{suffix}"
		suffix += 1
	run_dir.mkdir(parents=True)
	(run_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
	print(json.dumps(report, indent=2))
	print(f"Saved: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
	main()
