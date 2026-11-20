"""N1 observable difference product and fixed marginal-preserving common noise."""
import numpy as np

PATTERNS = ((20, 20, 20), (20, 80, 80), (80, 20, 20))


def triple_inputs(gt255, photons, common_fraction, rng):
    low = (np.clip(gt255, 0, 255) / 255) ** 2.2 * .35
    common_rate = common_fraction * min(photons) * low
    common_count = rng.poisson(common_rate)
    common_read = rng.normal(0, .008, low.shape)
    frames = []
    for p in photons:
        independent_count = rng.poisson(np.maximum(p * low - common_rate, 0))
        read = np.sqrt(common_fraction) * common_read + np.sqrt(1 - common_fraction) * rng.normal(0, .008, low.shape)
        image = (common_count + independent_count) / p + read
        frames.append((np.clip(image, 0, 1) * 255).astype(np.uint8).astype(np.float32))
    return np.stack(frames)


def tiles(value):
    """RGB mean AFTER forming per-coordinate products, center96, fixed16 pooling."""
    assert value.shape[-3:] == (3, 128, 128)
    central = value[..., 16:-16, 16:-16].mean(axis=-3)
    return central.reshape(*central.shape[:-2], 6, 16, 6, 16).mean(axis=(-3, -1))


def describe(predictions, references, photons):
    assert predictions.shape == (64, 3, 3, 128, 128)
    output = predictions.astype(np.float64)
    current, first, second = output[:, 0], output[:, 1], output[:, 2]
    q = (current - first) * (current - second)
    signed_tiles = tiles(q)
    positive_tiles = np.maximum(signed_tiles, 0)
    target = tiles(references[photons[0]]['variance'])
    denominator = max(float(target.sum()), 1e-8)
    relative_L1 = np.abs(positive_tiles - target).sum(axis=(1, 2)) / denominator
    global_reference = float(target.mean())
    global_signed_ratio = float(signed_tiles.mean() / max(global_reference, 1e-8))
    negative_fraction = float((signed_tiles < 0).mean())
    local_median = float(np.median(relative_L1))
    means = output.mean(axis=0)
    centered = output - means
    covariances = np.stack([(centered[:, a] * centered[:, b]).sum(axis=0) / 63 for a, b in ((0, 1), (0, 2), (1, 2))])
    reference_means = [references[p]['mean'] for p in photons]
    mean_bias = (reference_means[0] - reference_means[1]) * (reference_means[0] - reference_means[2])
    covariance_bias = -covariances[0] - covariances[1] + covariances[2]
    expected = references[photons[0]]['variance'] + covariance_bias + mean_bias
    gates = dict(reference_nonzero=global_reference > 1e-8,
                 global_calibration=.85 <= global_signed_ratio <= 1.15,
                 local_estimation=local_median <= .35,
                 negative_tiles=negative_fraction <= .20)
    metrics = dict(reference_global_variance=global_reference,
        signed_global_estimate=float(signed_tiles.mean()), signed_global_reference_ratio=global_signed_ratio,
        positive_map_mean_relative_L1=float(np.abs(positive_tiles.mean(0) - target).sum() / denominator),
        per_triplet_local_relative_L1_median=local_median,
        per_triplet_local_relative_L1_quantiles=np.quantile(relative_L1, [0, .1, .5, .9, 1]).tolist(),
        mean_negative_tile_fraction=negative_fraction,
        reference_mean_bias_over_variance=float(tiles(mean_bias).mean() / max(global_reference, 1e-8)),
        probe_covariance_bias_over_variance=float(tiles(covariance_bias).mean() / max(global_reference, 1e-8)),
        finite_sample_formula_relative_gap=float((q.mean(0) - expected)[..., 16:-16, 16:-16].mean() / max(global_reference, 1e-8)),
        gates=gates, all_gates_passed=all(gates.values()))
    maps = dict(reference_variance=references[photons[0]]['variance'], reference_tiles=target,
        signed_tiles=signed_tiles, positive_tiles=positive_tiles, per_triplet_relative_L1=relative_L1,
        observable_product_mean=q.mean(0), probe_means=means, probe_covariances=covariances,
        independent_reference_mean_bias=mean_bias, finite_sample_expected_product=expected)
    assert all(np.isfinite(value).all() for value in maps.values())
    return metrics, maps
