"""Integration tests for Forgetnet."""
from src.core import Forgetnet

class TestForgetnet:
    def setup_method(self):
        self.c = Forgetnet()
    def test_10_ops(self):
        for i in range(10): self.c.search(i=i)
        assert self.c.get_stats()["ops"] == 10
    def test_service_name(self):
        assert self.c.search()["service"] == "forgetnet"
    def test_different_inputs(self):
        self.c.search(type="a"); self.c.search(type="b")
        assert self.c.get_stats()["ops"] == 2
    def test_config(self):
        c = Forgetnet(config={"debug": True})
        assert c.config["debug"] is True
    def test_empty_call(self):
        assert self.c.search()["ok"] is True
    def test_large_batch(self):
        for _ in range(100): self.c.search()
        assert self.c.get_stats()["ops"] == 100
