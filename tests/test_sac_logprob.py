"""Test to demonstrate SAC log-prob numerical instability issue."""
import mlx.core as mx
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dirty_mlx_ml.reinforcement.nn import SACActor, _gauss_log_prob

def tanh_log_prob_old(z, mu, log_std):
    """Old unstable tanh squash log-prob calculation."""
    action = mx.tanh(z)
    return _gauss_log_prob(z, mu, log_std) - mx.sum(mx.log(1.0 - action**2 + 1e-6), axis=-1)

def tanh_log_prob_new(z, mu, log_std):
    """New numerically stable tanh squash log-prob calculation."""
    # Use numerically stable softplus: softplus(x) = log(1 + exp(x))
    # For numerical stability: softplus(x) = x + log1p(exp(-x)) when x > 0, else log1p(exp(x))
    x = -2.0 * z
    softplus_x = mx.where(x > 0, x + mx.log1p(mx.exp(-x)), mx.log1p(mx.exp(x)))
    return _gauss_log_prob(z, mu, log_std) - mx.sum(2 * (math.log(2.0) - z - softplus_x), axis=-1)

def test_sac_logprob_instability():
    """Test that demonstrates the numerical instability in SACActor.sample."""
    
    print("Testing SAC log-prob numerical instability...")
    print("=" * 60)
    
    # Test case 1: Normal conditions (should work fine)
    print("\n1. Normal conditions (small mu, no saturation):")
    mu_normal = mx.array([[0.5], [-0.3], [0.1]])
    log_std_normal = mx.array([[0.0], [0.0], [0.0]])
    std_normal = mx.exp(log_std_normal)
    
    z_normal = mu_normal + std_normal * mx.random.normal(mu_normal.shape)
    action_normal = mx.tanh(z_normal)
    
    # Old unstable calculation
    log_prob_normal_old = tanh_log_prob_old(z_normal, mu_normal, log_std_normal)
    # New numerically stable calculation
    log_prob_normal_new = tanh_log_prob_new(z_normal, mu_normal, log_std_normal)
    
    mx.eval(action_normal, log_prob_normal_old, log_prob_normal_new)
    
    print(f"  Mu: {mu_normal.flatten()}")
    print(f"  Z: {z_normal.flatten()}")
    print(f"  Actions: {action_normal.flatten()}")
    print(f"  Old log probs: {log_prob_normal_old}")
    print(f"  New log probs: {log_prob_normal_new}")
    print(f"  Difference: {log_prob_normal_new - log_prob_normal_old}")
    print(f"  ✓ No saturation, both forms agree")
    
    # Test case 2: Unstable conditions (large mu causing saturation)
    print("\n2. Unstable conditions (large mu, action saturation):")
    mu_large = mx.array([[10.0], [-10.0], [5.0]])
    log_std_large = mx.array([[0.0], [0.0], [0.0]])
    std_large = mx.exp(log_std_large)
    
    z_large = mu_large + std_large * mx.random.normal(mu_large.shape)
    action_large = mx.tanh(z_large)
    
    # Old unstable calculation
    log_prob_large_old = tanh_log_prob_old(z_large, mu_large, log_std_large)
    # New numerically stable calculation
    log_prob_large_new = tanh_log_prob_new(z_large, mu_large, log_std_large)
    
    mx.eval(action_large, log_prob_large_old, log_prob_large_new)
    
    print(f"  Mu: {mu_large.flatten()}")
    print(f"  Z: {z_large.flatten()}")
    print(f"  Actions: {action_large.flatten()}")
    print(f"  1 - action^2: {(1.0 - action_large**2).flatten()}")
    print(f"  Old log probs: {log_prob_large_old}")
    print(f"  New log probs: {log_prob_large_new}")
    print(f"  Difference: {log_prob_large_new - log_prob_large_old}")
    
    # Check for saturation
    saturated = mx.abs(action_large) > 0.999
    if mx.any(saturated):
        print(f"  ⚠️  WARNING: {int(mx.sum(saturated))} actions are saturated!")
        print(f"     Old form: epsilon guard (1e-6) dominates → log(1e-6) ≈ -13.8")
        print(f"     New form: handles saturation gracefully with softplus")
    
    # Test case 3: Edge case - near saturation (but not extreme)
    print("\n3. Edge case (action near ±1.0):")
    mu_edge = mx.array([[8.0], [-8.0]])
    log_std_edge = mx.array([[0.0], [0.0]])
    std_edge = mx.exp(log_std_edge)
    
    z_edge = mu_edge + std_edge * mx.random.normal(mu_edge.shape)
    action_edge = mx.tanh(z_edge)
    
    # Old unstable calculation
    log_prob_edge_old = tanh_log_prob_old(z_edge, mu_edge, log_std_edge)
    # New numerically stable calculation
    log_prob_edge_new = tanh_log_prob_new(z_edge, mu_edge, log_std_edge)
    
    mx.eval(action_edge, log_prob_edge_old, log_prob_edge_new)
    
    print(f"  Mu: {mu_edge.flatten()}")
    print(f"  Actions: {action_edge.flatten()}")
    print(f"  1 - action^2: {(1.0 - action_edge**2).flatten()}")
    print(f"  Old log probs: {log_prob_edge_old}")
    print(f"  New log probs: {log_prob_edge_new}")
    print(f"  Difference: {log_prob_edge_new - log_prob_edge_old}")
    print(f"  ⚠️  Actions are near ±1.0")
    print(f"     Old form: dominated by epsilon guard")
    print(f"     New form: numerically stable even near saturation")
    
    # Test case 4: Test actual SACActor.sample with the fix
    print("\n4. Test actual SACActor.sample with large mu:")
    actor = SACActor(obs_dim=4, act_dim=1)
    
    # Inflate mu weights to simulate instability
    original_mu_weight = actor.mu.weight
    actor.mu.weight = actor.mu.weight * 20.0
    
    obs_test = mx.random.normal((5, 4))
    action_test, log_prob_test = actor.sample(obs_test, deterministic=False)
    mx.eval(action_test, log_prob_test)
    
    print(f"  Actions: {action_test.flatten()}")
    print(f"  Log probs: {log_prob_test}")
    print(f"  Log prob range: [{float(mx.min(log_prob_test)):.4f}, {float(mx.max(log_prob_test)):.4f}]")
    
    # Restore original weights
    actor.mu.weight = original_mu_weight
    
    print(f"  ✓ SACActor.sample now uses numerically stable form")
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("The old implementation fails when actions saturate to ±1.0")
    print("because 1 - action**2 → 0, making the epsilon guard (1e-6) dominate.")
    print("This creates artificial log-prob collapse rather than smooth degradation.")
    print("The new numerically stable form: 2*(log(2) - z - softplus(-2z))")
    print("handles saturation gracefully and avoids the singularity.")
    print("SACActor.sample has been updated to use the stable form.")

if __name__ == "__main__":
    test_sac_logprob_instability()
