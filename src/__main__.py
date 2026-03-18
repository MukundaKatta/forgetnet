"""CLI for forgetnet."""
import sys, json, argparse
from .core import Forgetnet

def main():
    parser = argparse.ArgumentParser(description="ForgetNet — AI Memory and Forgetting. Research on selective forgetting and memory management in neural networks.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Forgetnet()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.search(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"forgetnet v0.1.0 — ForgetNet — AI Memory and Forgetting. Research on selective forgetting and memory management in neural networks.")

if __name__ == "__main__":
    main()
