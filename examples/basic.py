"""Basic usage example for forgetnet."""
from src.core import Forgetnet

def main():
    instance = Forgetnet(config={"verbose": True})

    print("=== forgetnet Example ===\n")

    # Run primary operation
    result = instance.search(input="example data", mode="demo")
    print(f"Result: {result}")

    # Run multiple operations
    ops = ["search", "index", "rank]
    for op in ops:
        r = getattr(instance, op)(source="example")
        print(f"  {op}: {"✓" if r.get("ok") else "✗"}")

    # Check stats
    print(f"\nStats: {instance.get_stats()}")

if __name__ == "__main__":
    main()
