import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


PROTOCOL_SHIFT_BOUNDS = (-2, 2)
PROTOCOL_SHIFT_SEED = 12345
PROTOCOL_TRAIN_SAMPLES = 10_000
PROTOCOL_TEST_SAMPLES = 2_000
PROTOCOL_SEEDS = (42, 100, 2026, 3141, 404)


def _repo_root() -> Path:
	return Path(__file__).resolve().parent.parent


def _resolve_data_root(root: str | Path) -> Path:
	path = Path(root)
	if not path.is_absolute():
		path = _repo_root() / path
	return path


def _load_split_indices(split_name: str, root: str | Path = "data") -> list[int]:
	with (_resolve_data_root(root) / "splits" / f"{split_name}_indices.json").open(
		"r", encoding="utf-8"
	) as file:
		payload = json.load(file)
	if isinstance(payload, dict) and "indices" in payload:
		payload = payload["indices"]
	return [int(index) for index in payload]


def generate_translation_offsets(
	count: int,
	seed: int = PROTOCOL_SHIFT_SEED,
	bounds: tuple[int, int] = PROTOCOL_SHIFT_BOUNDS,
) -> list[tuple[int, int]]:
	"""Generate a deterministic set of bounded image translations.

	The exact translation bounds and generation seed are frozen for the
	distribution-shift protocol and are intentionally not randomized at runtime.
	"""
	lower, upper = bounds
	generator = random.Random(seed)
	return [(generator.randint(lower, upper), generator.randint(lower, upper)) for _ in range(count)]


def apply_translation(
	images: torch.Tensor,
	dx: int,
	dy: int,
	fill_value: float = 0.0,
) -> torch.Tensor:
	"""Apply a bounded bilinear translation to a batch of MNIST tensors."""
	transformed = []
	for image in images:
		transformed.append(
			TF.affine(
				image,
				angle=0.0,
				translate=(dx, dy),
				scale=1.0,
				shear=0.0,
				interpolation=InterpolationMode.BILINEAR,
				fill=fill_value,
			)
		)
	return torch.stack(transformed, dim=0)


class DeterministicTranslationDataset(torch.utils.data.Dataset):
	"""Dataset wrapper that applies a deterministic bounded translation to each example."""

	def __init__(
		self,
		base_dataset: torch.utils.data.Dataset,
		offsets: list[tuple[int, int]] | None = None,
		seed: int = PROTOCOL_SHIFT_SEED,
		bounds: tuple[int, int] = PROTOCOL_SHIFT_BOUNDS,
		fill_value: float = 0.0,
	) -> None:
		super().__init__()
		self.base_dataset = base_dataset
		if offsets is None:
			offsets = generate_translation_offsets(len(base_dataset), seed=seed, bounds=bounds)
		self.offsets = offsets
		self.fill_value = fill_value

	def __len__(self) -> int:
		return len(self.base_dataset)

	def __getitem__(self, index: int):
		image, label = self.base_dataset[index]
		dx, dy = self.offsets[index]
		shifted = apply_translation(image.unsqueeze(0), dx, dy, fill_value=self.fill_value).squeeze(0)
		return shifted, label


def get_mnist_loaders(
	root: str = "data",
	batch_size: int = 128,
	max_train_samples: int | None = None,
	max_test_samples: int | None = None,
	num_workers: int = 0,
	seed: int = 42,
	use_fixed_indices: bool = True,
	matched_training_augmentation: bool = False,
	translation_seed: int = PROTOCOL_SHIFT_SEED,
) -> tuple[DataLoader, DataLoader]:
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,)),
	])
	train_dataset = datasets.MNIST(_resolve_data_root(root), train=True, download=True, transform=transform)
	test_dataset = datasets.MNIST(_resolve_data_root(root), train=False, download=True, transform=transform)
	if use_fixed_indices:
		train_indices = _load_split_indices("train", root)[:max_train_samples or len(_load_split_indices("train", root))]
		test_indices = _load_split_indices("test", root)[:max_test_samples or len(_load_split_indices("test", root))]
		train_dataset = Subset(train_dataset, train_indices)
		test_dataset = Subset(test_dataset, test_indices)
	elif max_train_samples is not None:
		train_dataset = Subset(train_dataset, range(min(max_train_samples, len(train_dataset))))
	elif max_test_samples is not None:
		test_dataset = Subset(test_dataset, range(min(max_test_samples, len(test_dataset))))
	if matched_training_augmentation:
		train_offsets = generate_translation_offsets(len(train_dataset), seed=translation_seed + seed, bounds=PROTOCOL_SHIFT_BOUNDS)
		train_dataset = DeterministicTranslationDataset(train_dataset, offsets=train_offsets, seed=translation_seed + seed)
	generator = torch.Generator()
	generator.manual_seed(seed)
	return (
		DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=True,
			num_workers=num_workers,
			generator=generator,
		),
		DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
	)


def get_standard_test_loader(
	root: str = "data",
	batch_size: int = 128,
	max_test_samples: int | None = PROTOCOL_TEST_SAMPLES,
	num_workers: int = 0,
	use_fixed_indices: bool = True,
) -> DataLoader:
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,)),
	])
	dataset = datasets.MNIST(_resolve_data_root(root), train=False, download=True, transform=transform)
	indices = _load_split_indices("test", root)
	if max_test_samples is not None:
		indices = indices[:max_test_samples]
	return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_shifted_test_loader(
	root: str = "data",
	batch_size: int = 128,
	max_test_samples: int | None = PROTOCOL_TEST_SAMPLES,
	num_workers: int = 0,
	shift_seed: int = PROTOCOL_SHIFT_SEED,
	use_fixed_indices: bool = True,
) -> DataLoader:
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,)),
	])
	dataset = datasets.MNIST(_resolve_data_root(root), train=False, download=True, transform=transform)
	indices = _load_split_indices("test", root)
	if max_test_samples is not None:
		indices = indices[:max_test_samples]
	base_subset = Subset(dataset, indices)
	offsets = generate_translation_offsets(len(base_subset), seed=shift_seed, bounds=PROTOCOL_SHIFT_BOUNDS)
	shifted_dataset = DeterministicTranslationDataset(base_subset, offsets=offsets, seed=shift_seed)
	return DataLoader(shifted_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def flatten_images(images: torch.Tensor) -> torch.Tensor:
	return images.view(images.size(0), -1)


def split_context_target(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	context = images[:, :, :, :14].contiguous().view(images.size(0), -1)
	target = images[:, :, :, 14:].contiguous().view(images.size(0), -1)
	return context, target
