"""Fixed colour-keyed pre-square feature bank, a bilateral-propagation control."""
import itertools
import torch
import torch.nn.functional as F

SIDE, CHANNELS, EPS, BLOCK = 17, 12, 1e-8, 4096
CORNERS = tuple(itertools.product((0, 1), repeat=3))
STATE_BYTES_FP32 = SIDE**3 * (CHANNELS+1) * 4


def guide(low):
    assert low.ndim == 4 and low.shape[:2] == (1, 3)
    return F.avg_pool2d(F.pad(low, (2, 2, 2, 2), mode='replicate'), 5, 1).clamp(0, 1).sqrt()[0].flatten(1).T


def weights(keys):
    scaled = keys * (SIDE-1)
    lower = scaled.floor().long().clamp(0, SIDE-2)
    fraction = scaled - lower
    for corner in CORNERS:
        offset = torch.tensor(corner, dtype=torch.long, device=keys.device)
        coords = lower + offset
        index = (coords[:, 0]*SIDE + coords[:, 1])*SIDE + coords[:, 2]
        weight = torch.where(offset.bool(), fraction, 1-fraction).prod(-1)
        yield index, weight


def splat(keys, z):
    assert z.shape[:2] == (1, CHANNELS) and keys.shape == (z.shape[-2]*z.shape[-1], 3)
    values = z[0].flatten(1)
    table = z.new_zeros(CHANNELS+1, SIDE**3)
    for start in range(0, len(keys), BLOCK):
        value = values[:, start:start+BLOCK]
        for index, weight in weights(keys[start:start+BLOCK]):
            table[:CHANNELS].index_add_(1, index, value * (weight / len(keys))[None])
            table[CHANNELS].index_add_(0, index, weight / len(keys))
    return table


def lookup(table, keys, z):
    estimate = torch.empty_like(z[0].flatten(1))
    valid_count, mass_min, mass_max = 0, float('inf'), 0.
    original = z[0].flatten(1)
    for start in range(0, len(keys), BLOCK):
        key = keys[start:start+BLOCK]
        numerator, mass = z.new_zeros(CHANNELS, len(key)), z.new_zeros(len(key))
        for index, weight in weights(key):
            numerator += table[:CHANNELS, index] * weight[None]
            mass += table[CHANNELS, index] * weight
        valid = mass > EPS
        value = numerator / mass.clamp_min(EPS)[None]
        estimate[:, start:start+BLOCK] = torch.where(valid[None], value, original[:, start:start+BLOCK])
        valid_count += int(valid.sum()); mass_min = min(mass_min, float(mass.min())); mass_max = max(mass_max, float(mass.max()))
    return estimate.reshape_as(z), dict(accepted_fraction=valid_count/len(keys), minimum_lookup_mass=mass_min, maximum_lookup_mass=mass_max)


class FeatureBank:
    def __init__(self, current_only=False):
        self.current_only, self.table, self.frames = current_only, None, 0

    def step(self, low, z):
        assert z.dtype == low.dtype == torch.float32 and z.device.type == low.device.type == 'cpu'
        keys = guide(low)
        if self.frames == 0:
            output, details = z, dict(accepted_fraction=0., bootstrap=True)
            current = splat(keys, z)
        elif self.current_only:
            current = splat(keys, z)
            past, details = lookup(current, keys, z)
            output = .5*z + .5*past
        else:
            past, details = lookup(self.table, keys, z)
            output = .5*z + .5*past
            current = splat(keys, z)  # Read original history before building/updating current statistics.
        if not self.current_only:
            self.table = (current if self.table is None else .5*self.table + .5*current).detach().clone()
            assert self.table.is_contiguous() and self.table.untyped_storage().nbytes() == STATE_BYTES_FP32
        self.frames += 1
        return output, dict(**details, frames_seen=self.frames, stored_bytes=0 if self.table is None else self.table.untyped_storage().nbytes(),
                            current_only=self.current_only)


def continue_from_head(net, z):
    head = (net.head.weight1*z)*(net.head.weight2*z)+net.head.bias
    body = net.body(head)
    channel = net.att(body)
    maximum = (channel*body).max(1, keepdim=True).values
    spatial = net.att1(maximum)
    return net.tail((spatial*channel)*body)


def qualification():
    low = torch.full((1, 3, 8, 8), .16)
    z = torch.full((1, 12, 8, 8), 3.)
    keys = guide(low)
    total = sum(weight for _, weight in weights(keys))
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6, rtol=0)
    bank = FeatureBank(); first, _ = bank.step(low, z)
    assert torch.equal(first, z)
    old = bank.table.clone()
    second, details = bank.step(low, torch.full_like(z, 2.))
    assert torch.allclose(second, torch.full_like(z, 2.5), atol=1e-6, rtol=0)
    assert not torch.equal(old, bank.table)
    empty, d = lookup(torch.zeros_like(old), keys, z)
    assert torch.equal(empty, z) and d['accepted_fraction'] == 0
    static = FeatureBank(current_only=True)
    static.step(low, z); constant, _ = static.step(low, z)
    assert static.table is None and torch.allclose(constant, z, atol=1e-6, rtol=0)
    return dict(passed=True, fixed_state_bytes=STATE_BYTES_FP32, weight_partition_max_error=float((total-1).abs().max()),
                first_frame_exact=True, constant_restored=True, old_table_read_before_update=True, empty_fallback_exact=True,
                constant_bank_steps=4, direct_empty_lookup_calls=1, model_calls=0, optimizer_updates=0)
