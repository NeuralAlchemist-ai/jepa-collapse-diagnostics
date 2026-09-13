import torch
import torch.nn.functional as F


def compute_diagnostics(z: torch.Tensor) -> dict[str, float]:
	"""Return variance, effective rank, and mean off-diagonal cosine similarity."""
	z = z.detach().float()
	centered = z - z.mean(dim=0, keepdim=True)
	feature_variance = centered.var(dim=0, unbiased=False).mean().item()
	singular_values = torch.linalg.svdvals(centered)
	probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
	entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
	effective_rank = torch.exp(entropy).item() if singular_values.sum() > 0 else 0.0
	normalized = F.normalize(z, p=2, dim=1)
	similarities = normalized @ normalized.T
	mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
	average_cosine = similarities[mask].mean().item() if mask.any() else 1.0
	return {
		"feature_variance": round(feature_variance, 6),
		"effective_rank": round(effective_rank, 6),
		"avg_pairwise_cosine_sim": round(average_cosine, 6),
	}
