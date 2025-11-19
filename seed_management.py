"""
Seed Management for Train/Validation/Test Split
训练集/验证集/测试集的种子管理

This module ensures proper separation of train/val/test datasets
to avoid data leakage.
"""

from PRNG import PRNG
from typing import List


class SeedManager:
    """
    Manages seeds for train/validation/test splits

    Ensures:
    - Training uses random seeds (different each time)
    - Validation uses fixed seeds (for monitoring during training)
    - Test uses different fixed seeds (never seen during training)
    """

    def __init__(
        self,
        val_seed: str = "VALIDATION_2024",
        test_seed: str = "TEST_2024",
        num_val: int = 50,
        num_test: int = 200
    ):
        """
        Initialize seed manager

        Args:
            val_seed: Master seed for validation set
            test_seed: Master seed for test set (must be different!)
            num_val: Number of validation episodes
            num_test: Number of test episodes
        """
        assert val_seed != test_seed, "Validation and test seeds must be different!"

        self.val_seed_name = val_seed
        self.test_seed_name = test_seed
        self.num_val = num_val
        self.num_test = num_test

        # Generate validation seeds
        val_prng = PRNG()
        val_prng.seed(val_seed)
        self.val_seeds = [val_prng.random() for _ in range(num_val)]

        # Generate test seeds
        test_prng = PRNG()
        test_prng.seed(test_seed)
        self.test_seeds = [test_prng.random() for _ in range(num_test)]

        print(f"✅ Seed Manager initialized:")
        print(f"   Validation: {num_val} episodes (seed: '{val_seed}')")
        print(f"   Test: {num_test} episodes (seed: '{test_seed}')")

    def get_train_seed(self) -> None:
        """
        Get seed for training

        Returns None to use random seed (different each episode)
        """
        return None

    def get_val_seeds(self) -> List[int]:
        """
        Get all validation seeds

        Returns:
            List of validation seeds (fixed)
        """
        return self.val_seeds.copy()

    def get_test_seeds(self) -> List[int]:
        """
        Get all test seeds

        Returns:
            List of test seeds (fixed, different from validation)
        """
        return self.test_seeds.copy()

    def verify_no_overlap(self) -> bool:
        """
        Verify that validation and test sets don't overlap

        Returns:
            True if no overlap, False otherwise
        """
        val_set = set(self.val_seeds)
        test_set = set(self.test_seeds)
        overlap = val_set & test_set

        if len(overlap) > 0:
            print(f"⚠️  WARNING: {len(overlap)} seeds overlap between val and test!")
            return False
        else:
            print(f"✅ Verified: No overlap between validation and test sets")
            return True

    def save_seeds(self, filepath: str = "seeds.txt"):
        """
        Save seed configuration to file for reproducibility

        Args:
            filepath: Path to save seed information
        """
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("Seed Configuration for Train/Val/Test Split\n")
            f.write("="*70 + "\n\n")

            f.write(f"Validation Master Seed: {self.val_seed_name}\n")
            f.write(f"Test Master Seed: {self.test_seed_name}\n\n")

            f.write(f"Validation Seeds ({len(self.val_seeds)} episodes):\n")
            f.write(f"{self.val_seeds[:10]} ... (showing first 10)\n\n")

            f.write(f"Test Seeds ({len(self.test_seeds)} episodes):\n")
            f.write(f"{self.test_seeds[:10]} ... (showing first 10)\n\n")

            f.write(f"Full validation seeds:\n{self.val_seeds}\n\n")
            f.write(f"Full test seeds:\n{self.test_seeds}\n")

        print(f"💾 Seed configuration saved to: {filepath}")


# ===== Convenience Functions =====

def get_default_seed_manager() -> SeedManager:
    """
    Get default seed manager with standard configuration

    Returns:
        Initialized SeedManager
    """
    return SeedManager(
        val_seed="VALIDATION_2024",
        test_seed="TEST_2024",
        num_val=50,
        num_test=200
    )


def demo_usage():
    """Demonstrate proper usage"""
    print("="*70)
    print("Seed Management Demo")
    print("="*70 + "\n")

    # Initialize
    seed_mgr = get_default_seed_manager()
    seed_mgr.verify_no_overlap()

    print("\n" + "="*70)
    print("Usage Examples")
    print("="*70 + "\n")

    # Training
    print("1. During Training:")
    print(f"   train_seed = seed_mgr.get_train_seed()  # {seed_mgr.get_train_seed()}")
    print("   env.reset(seed=train_seed)  # None = random seed")

    # Validation
    print("\n2. During Validation (monitoring training progress):")
    val_seeds = seed_mgr.get_val_seeds()
    print(f"   val_seeds = seed_mgr.get_val_seeds()  # {len(val_seeds)} seeds")
    print(f"   First 5: {val_seeds[:5]}")
    print("   for seed in val_seeds:")
    print("       env.reset(seed=seed)")
    print("       # evaluate...")

    # Test
    print("\n3. Final Testing (after training complete):")
    test_seeds = seed_mgr.get_test_seeds()
    print(f"   test_seeds = seed_mgr.get_test_seeds()  # {len(test_seeds)} seeds")
    print(f"   First 5: {test_seeds[:5]}")
    print("   for seed in test_seeds:")
    print("       env.reset(seed=seed)")
    print("       # evaluate...")

    # Save
    print("\n4. Save for Reproducibility:")
    seed_mgr.save_seeds("demo_seeds.txt")

    print("\n" + "="*70)
    print("✅ Demo Complete")
    print("="*70)


if __name__ == "__main__":
    demo_usage()
