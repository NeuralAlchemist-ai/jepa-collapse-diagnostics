import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from src.data import PROTOCOL_SEEDS, PROTOCOL_SHIFT_BOUNDS, PROTOCOL_SHIFT_SEED, PROTOCOL_TEST_SAMPLES, PROTOCOL_TRAIN_SAMPLES, get_mnist_loaders, split_context_target
from src.evaluate import evaluate_linear_probe
from src.models import AutoEncoder, CollapsedControl, JEPA


SEEDS = (42, 100, 2026, 3141, 404)
EPOCHS = 3
PROBE_EPOCHS = 2
BATCH_SIZE = 128
MAX_TRAIN_SAMPLES = 10_000
MAX_TEST_SAMPLES = 2_000
METRICS = (
	"linear_probe_accuracy",
	"feature_variance",
	"effective_rank",
	"avg_pairwise_cosine_sim",
)
DISTRIBUTION_SHIFT_PROTOCOL = {
	"protocol_version": "distribution_shift_v1",
	"experiment_name": "distribution_shift_v1",
	"dataset": "MNIST",
	"train_samples": PROTOCOL_TRAIN_SAMPLES,
	"test_samples": PROTOCOL_TEST_SAMPLES,
	"seeds": list(PROTOCOL_SEEDS),
	"batch_size": BATCH_SIZE,
	"optimizer": "Adam",
	"learning_rate": 1e-3,
	"weight_decay": 0.0,
	"epochs": EPOCHS,
	"scheduler": "none",
	"probe_config": {
		"input_dimension": 64,
		"output_dimension": 10,
		"optimizer": "Adam",
		"learning_rate": 1e-3,
		"epochs": PROBE_EPOCHS,
		"weight_decay": 0.0,
		"scheduler": "none",
	},
	"preprocessing": {
		"to_tensor": True,
		"normalize_mean": 0.5,
		"normalize_std": 0.5,
	},
	"shift_type": "translation",
	"shift_parameters": {
		"x_displacement": {"min": -2, "max": 2},
		"y_displacement": {"min": -2, "max": 2},
		"interpolation": "bilinear",
		"fill_value": 0,
		"bounds": list(PROTOCOL_SHIFT_BOUNDS),
		"generation_seed": PROTOCOL_SHIFT_SEED,
	},
	"primary_metric": "accuracy_drop = accuracy_standard - accuracy_shifted",
	"secondary_metrics": [
		"centered_effective_rank",
		"feature_variance",
		"avg_pairwise_cosine_similarity",
	],
	"decision_rule": {
		"threshold": 0.05,
		"criterion": "mean(D) > 0.05 and 95% paired confidence interval excludes 0",
	},
	"training_condition": "standard_train_only",
	"matched_augmentation_condition": {
		"available": True,
		"default": False,
		"requires_explicit_selection": True,
	},
	"output_artifacts": [
		"aggregate.json",
		"paired_summary.json",
		"seed_<seed>/metrics.json",
	],
	"status": "pre_outcome_protocol_not_executed",
}


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


def parameter_count(model, trainable_only=False):
	return sum(
		parameter.numel()
		for parameter in model.parameters()
		if not trainable_only or parameter.requires_grad
	)


def run_seed(seed: int, output_dir: Path) -> dict:
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.use_deterministic_algorithms(True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device} for seed {seed}")
	train_loader, test_loader = get_mnist_loaders(
		batch_size=BATCH_SIZE,
		max_train_samples=MAX_TRAIN_SAMPLES,
		max_test_samples=MAX_TEST_SAMPLES,
		seed=seed,
	)

	autoencoder = AutoEncoder().to(device)
	train_autoencoder(autoencoder, train_loader, device, EPOCHS, 1e-3)
	ae_accuracy, ae_diagnostics = evaluate_linear_probe(
		autoencoder, train_loader, test_loader, use_context=True, device=device, epochs=PROBE_EPOCHS
	)

	jepa = JEPA().to(device)
	train_jepa(jepa, train_loader, device, EPOCHS, 1e-3)
	jepa_accuracy, jepa_diagnostics = evaluate_linear_probe(
		jepa, train_loader, test_loader, is_jepa=True, device=device, epochs=PROBE_EPOCHS
	)

	collapsed = CollapsedControl().to(device)
	collapsed_accuracy, collapsed_diagnostics = evaluate_linear_probe(
		collapsed, train_loader, test_loader, device=device, epochs=PROBE_EPOCHS
	)
	report = {
		"seed": seed,
		"device": str(device),
		"primary_metric": "effective_rank",
		"config": {
			"seed": seed,
			"epochs": EPOCHS,
			"probe_epochs": PROBE_EPOCHS,
			"batch_size": BATCH_SIZE,
			"max_train_samples": MAX_TRAIN_SAMPLES,
			"max_test_samples": MAX_TEST_SAMPLES,
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
	output_dir.mkdir(parents=True)
	(output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
	return report


def summarize(reports: list[dict]) -> dict:
	model_metrics = {}
	for model_name in ("autoencoder", "jepa", "collapsed_control"):
		model_summary = {}
		for metric in METRICS:
			values = [report[model_name][metric] for report in reports]
			average = sum(values) / len(values)
			deviation = (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5
			model_summary[metric] = {"mean": round(average, 6), "std": round(deviation, 6)}
		model_metrics[model_name] = model_summary

	paired = {}
	for metric in METRICS:
		deltas = [report["jepa"][metric] - report["autoencoder"][metric] for report in reports]
		average = sum(deltas) / len(deltas)
		deviation = (sum((delta - average) ** 2 for delta in deltas) / len(deltas)) ** 0.5
		paired[metric] = {
			"deltas": [round(delta, 6) for delta in deltas],
			"mean": round(average, 6),
			"std": round(deviation, 6),
		}
	return {"model_metrics": model_metrics, "paired_jepa_minus_autoencoder": paired}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--protocol",
		choices=("closure", "distribution_shift_v1"),
		default="closure",
		help="Which experiment protocol to configure. The distribution-shift protocol is frozen but intentionally not executed here.",
	)
	args = parser.parse_args()
	if args.protocol == "distribution_shift_v1":
		print(json.dumps(DISTRIBUTION_SHIFT_PROTOCOL, indent=2))
		print("Distribution-shift protocol is frozen and intentionally not executed in this repository state.")
		return
	base_name = datetime.now(timezone.utc).strftime("closure_%Y%m%d_%H%M%S_%fZ")
	closure_dir = Path("results") / base_name
	closure_dir.mkdir(parents=True)
	reports = []
	for seed in SEEDS:
		seed_dir = closure_dir / f"seed_{seed}"
		reports.append(run_seed(seed, seed_dir))
	aggregate = {
		"seeds": list(SEEDS),
		"config": {
			"epochs": EPOCHS,
			"probe_epochs": PROBE_EPOCHS,
			"batch_size": BATCH_SIZE,
			"max_train_samples": MAX_TRAIN_SAMPLES,
			"max_test_samples": MAX_TEST_SAMPLES,
			"primary_metric": "effective_rank",
		},
		**summarize(reports),
	}
	(closure_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
	print(json.dumps(aggregate, indent=2))
	print(f"Saved closure results: {closure_dir}")


if __name__ == "__main__":
	main()
