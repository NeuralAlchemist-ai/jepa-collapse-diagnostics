import argparse
import hashlib
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from src.data import PROTOCOL_SEEDS, PROTOCOL_SHIFT_BOUNDS, PROTOCOL_SHIFT_SEED, PROTOCOL_TEST_SAMPLES, PROTOCOL_TRAIN_SAMPLES, generate_translation_offsets, get_mnist_loaders, get_shifted_test_loader, get_standard_test_loader, split_context_target
from src.evaluate import evaluate_linear_probe, evaluate_linear_probe_on_loaders
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


def load_distribution_shift_protocol() -> dict:
	protocol_path = Path(__file__).resolve().parent / "protocols" / "distribution_shift_v1.json"
	with protocol_path.open("r", encoding="utf-8") as file:
		return json.load(file)


def set_deterministic_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.use_deterministic_algorithms(True)


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
	set_deterministic_seed(seed)
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


def run_distribution_shift_seed(
	seed: int,
	output_dir: Path,
	matched_training_augmentation: bool = False,
	protocol: dict | None = None,
) -> dict:
	protocol = protocol or load_distribution_shift_protocol()
	set_deterministic_seed(seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	training = protocol["training"]
	probe_config = protocol["linear_probe"]
	train_kwargs = {
		"batch_size": training["batch_size"],
		"max_train_samples": protocol["dataset"]["train_samples"],
		"max_test_samples": protocol["dataset"]["test_samples"],
		"seed": seed,
		"matched_training_augmentation": matched_training_augmentation,
		"translation_seed": protocol["training_augmentation"]["translation_seed"],
	}
	offsets = generate_translation_offsets(
		protocol["dataset"]["test_samples"],
		seed=protocol["shift"]["translation_seed"],
		bounds=tuple(protocol["shift"]["dx_range"]),
	)
	standard_test_loader = get_standard_test_loader(
		batch_size=training["batch_size"],
		max_test_samples=protocol["dataset"]["test_samples"],
	)
	shifted_test_loader = get_shifted_test_loader(
		batch_size=training["batch_size"],
		max_test_samples=protocol["dataset"]["test_samples"],
		shift_seed=protocol["shift"]["translation_seed"],
		offsets=offsets,
	)

	def evaluate_model(model: nn.Module, model_name: str, use_context: bool = False, is_jepa: bool = False) -> dict:
		set_deterministic_seed(seed)
		model = model.to(device)
		model_train_loader, _ = get_mnist_loaders(**train_kwargs)
		if model_name == "autoencoder":
			train_autoencoder(model, model_train_loader, device, training["epochs"], training["learning_rate"])
		else:
			train_jepa(model, model_train_loader, device, training["epochs"], training["learning_rate"])
		probe_train_loader, _ = get_mnist_loaders(**train_kwargs)
		probe_results = evaluate_linear_probe_on_loaders(
			encoder=model,
			train_loader=probe_train_loader,
			evaluation_loaders={"standard": standard_test_loader, "shifted": shifted_test_loader},
			latent_dim=protocol["linear_probe"]["latent_dim"],
			is_jepa=is_jepa,
			use_context=use_context,
			device=device,
			epochs=probe_config["epochs"],
			learning_rate=probe_config["learning_rate"],
			weight_decay=probe_config["weight_decay"],
		)
		standard = probe_results["standard"]
		shifted = probe_results["shifted"]
		return {
			"model": model_name,
			"parameter_count": parameter_count(model, trainable_only=True),
			"standard_accuracy": standard["accuracy"],
			"shifted_accuracy": shifted["accuracy"],
			"accuracy_drop": round(standard["accuracy"] - shifted["accuracy"], 6),
			"standard_diagnostics": standard["diagnostics"],
			"shifted_diagnostics": shifted["diagnostics"],
		}

	autoencoder_report = evaluate_model(AutoEncoder(), "autoencoder", use_context=True)
	jepa_report = evaluate_model(JEPA(), "jepa", is_jepa=True)
	report = {
		"seed": seed,
		"protocol": protocol["protocol_id"],
		"protocol_sha256": protocol_sha256(),
		"device": str(device),
		"training_condition": "matched_translation_augmentation" if matched_training_augmentation else "standard_unshifted",
		"training_config": training,
		"probe_config": probe_config,
		"shift_config": protocol["shift"],
		"autoencoder": autoencoder_report,
		"jepa": jepa_report,
		"D_seed": round(autoencoder_report["accuracy_drop"] - jepa_report["accuracy_drop"], 6),
	}
	output_dir.mkdir(parents=True, exist_ok=True)
	(output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
	return report


def protocol_sha256() -> str:
	protocol_path = Path(__file__).resolve().parent / "protocols" / "distribution_shift_v1.json"
	return hashlib.sha256(protocol_path.read_bytes()).hexdigest()


def run_distribution_shift(output_dir: Path, matched_training_augmentation: bool = False) -> dict:
	protocol = load_distribution_shift_protocol()
	output_dir.mkdir(parents=True, exist_ok=False)
	offsets = generate_translation_offsets(
		protocol["dataset"]["test_samples"],
		seed=protocol["shift"]["translation_seed"],
		bounds=tuple(protocol["shift"]["dx_range"]),
	)
	(output_dir / "translation_offsets.json").write_text(json.dumps(offsets, indent=2) + "\n")
	reports = [
		run_distribution_shift_seed(
			seed,
			output_dir / f"seed_{seed}",
			matched_training_augmentation=matched_training_augmentation,
			protocol=protocol,
		)
		for seed in protocol["seeds"]
	]
	differences = [report["D_seed"] for report in reports]
	mean_difference = statistics.mean(differences)
	std_difference = statistics.stdev(differences)
	margin = protocol["analysis"]["t_critical_df_4"] * std_difference / math.sqrt(len(differences))
	paired_summary = {
		"D_seed": [{"seed": report["seed"], "value": report["D_seed"]} for report in reports],
		"mean": round(mean_difference, 6),
		"std": round(std_difference, 6),
		"confidence_level": protocol["analysis"]["confidence_level"],
		"ci_method": protocol["analysis"]["ci_method"],
		"ci": [round(mean_difference - margin, 6), round(mean_difference + margin, 6)],
		"decision_rule": protocol["decision_rule"],
	}
	aggregate = {
		"protocol": protocol["protocol_id"],
		"protocol_sha256": protocol_sha256(),
		"reproduction_command": protocol["reproduction_command"],
		"training_condition": "matched_translation_augmentation" if matched_training_augmentation else "standard_unshifted",
		"seeds": protocol["seeds"],
		"models": {
			model_name: {
				"standard_accuracy_mean": round(statistics.mean(report[model_name]["standard_accuracy"] for report in reports), 6),
				"standard_accuracy_std": round(statistics.stdev(report[model_name]["standard_accuracy"] for report in reports), 6),
				"shifted_accuracy_mean": round(statistics.mean(report[model_name]["shifted_accuracy"] for report in reports), 6),
				"shifted_accuracy_std": round(statistics.stdev(report[model_name]["shifted_accuracy"] for report in reports), 6),
				"accuracy_drop_mean": round(statistics.mean(report[model_name]["accuracy_drop"] for report in reports), 6),
				"accuracy_drop_std": round(statistics.stdev(report[model_name]["accuracy_drop"] for report in reports), 6),
			}
			for model_name in ("autoencoder", "jepa")
		},
		"D_seed_mean": round(mean_difference, 6),
		"D_seed_std": round(std_difference, 6),
	}
	(output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
	(output_dir / "paired_summary.json").write_text(json.dumps(paired_summary, indent=2) + "\n")
	return aggregate


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
		help="Which experiment protocol to execute.",
	)
	parser.add_argument("--output-dir", type=Path)
	parser.add_argument("--matched-training-augmentation", action="store_true")
	args = parser.parse_args()
	if args.protocol == "distribution_shift_v1":
		output_dir = args.output_dir or Path("results") / f"distribution_shift_v1_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')}"
		aggregate = run_distribution_shift(output_dir, matched_training_augmentation=args.matched_training_augmentation)
		print(json.dumps(aggregate, indent=2))
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
