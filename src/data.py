from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist_loaders(
	root: str = "data",
	batch_size: int = 128,
	max_train_samples: int | None = None,
	max_test_samples: int | None = None,
	num_workers: int = 0,
	seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,)),
	])
	train_dataset = datasets.MNIST(Path(root), train=True, download=True, transform=transform)
	test_dataset = datasets.MNIST(Path(root), train=False, download=True, transform=transform)
	if max_train_samples is not None:
		train_dataset = Subset(train_dataset, range(min(max_train_samples, len(train_dataset))))
	if max_test_samples is not None:
		test_dataset = Subset(test_dataset, range(min(max_test_samples, len(test_dataset))))
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


def flatten_images(images: torch.Tensor) -> torch.Tensor:
	return images.view(images.size(0), -1)


def split_context_target(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	context = images[:, :, :, :14].contiguous().view(images.size(0), -1)
	target = images[:, :, :, 14:].contiguous().view(images.size(0), -1)
	return context, target
