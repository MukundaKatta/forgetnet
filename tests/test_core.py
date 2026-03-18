"""Tests for Forgetnet."""
from src.core import Forgetnet
def test_init(): assert Forgetnet().get_stats()["ops"] == 0
def test_op(): c = Forgetnet(); c.search(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Forgetnet(); [c.search() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Forgetnet(); c.search(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Forgetnet(); r = c.search(); assert r["service"] == "forgetnet"
