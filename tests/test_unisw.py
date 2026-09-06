import importlib.util
import os
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "unisw"
LOADER = SourceFileLoader("unisw", str(SCRIPT))
SPEC = importlib.util.spec_from_loader("unisw", LOADER)
unisw = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(unisw)


class UniswTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.config_home = self.root / "config"
        self.data_home = self.root / "data"
        self.state = self.root / "state"
        self.paths = [self.state / "first", self.state / "second"]
        self.config = {"chatgpt": {"paths": [str(path) for path in self.paths]}}
        self.environment = patch.dict(os.environ, {
            "XDG_CONFIG_HOME": str(self.config_home),
            "XDG_DATA_HOME": str(self.data_home),
        })
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.tempdir.cleanup()

    def make_state(self, first="one", second="two"):
        for path, value in zip(self.paths, (first, second)):
            path.mkdir(parents=True)
            (path / "account.txt").write_text(value)

    def test_snatch_new_and_switch_round_trip(self):
        self.make_state()
        unisw.execute_snatch("chatgpt", "work", self.config)

        for path in self.paths:
            self.assertTrue(path.is_symlink())
        self.assertEqual(
            (unisw.profile_dir("chatgpt", "work") / "path_0" / "account.txt").read_text(),
            "one",
        )

        unisw.execute_new("chatgpt", self.config)
        self.make_state("personal-one", "personal-two")
        unisw.execute_snatch("chatgpt", "personal", self.config)
        unisw.execute_switch("chatgpt", "work", self.config)
        self.assertEqual((self.paths[0] / "account.txt").read_text(), "one")
        self.assertEqual((self.paths[1] / "account.txt").read_text(), "two")
        unisw.execute_switch("chatgpt", "personal", self.config)
        self.assertEqual((self.paths[0] / "account.txt").read_text(), "personal-one")

    def test_snatch_refuses_existing_profile_and_path_escape(self):
        self.make_state()
        unisw.execute_snatch("chatgpt", "work", self.config)
        with self.assertRaises(unisw.UniswError):
            unisw.execute_snatch("chatgpt", "work", self.config)
        with self.assertRaises(unisw.UniswError):
            unisw.execute_switch("chatgpt", "../outside", self.config)

    def test_unexpected_target_symlink_is_never_unlinked(self):
        external = self.root / "external"
        external.mkdir()
        self.paths[0].parent.mkdir(parents=True)
        self.paths[0].symlink_to(external, target_is_directory=True)
        with self.assertRaises(unisw.UniswError):
            unisw.execute_snatch("chatgpt", "work", self.config)
        self.assertTrue(self.paths[0].is_symlink())
        self.assertTrue(external.is_dir())

    def test_switch_preflight_does_not_partially_replace_targets(self):
        self.make_state()
        unisw.execute_snatch("chatgpt", "work", self.config)
        unisw.execute_new("chatgpt", self.config)
        self.paths[1].mkdir(parents=True)
        with self.assertRaises(unisw.UniswError):
            unisw.execute_switch("chatgpt", "work", self.config)
        self.assertFalse(self.paths[0].exists())
        self.assertTrue(self.paths[1].is_dir())

    def test_active_profile_cannot_be_renamed(self):
        self.make_state()
        unisw.execute_snatch("chatgpt", "work", self.config)
        with self.assertRaises(unisw.UniswError):
            unisw.execute_mv("chatgpt", "work", "renamed", self.config)
        self.assertTrue(unisw.profile_dir("chatgpt", "work").is_dir())
        self.assertFalse(unisw.profile_dir("chatgpt", "renamed").exists())

    def test_missing_config_is_not_created(self):
        with self.assertRaises(unisw.UniswError):
            unisw.load_config()
        self.assertFalse((self.config_home / "unisw" / "config.toml").exists())


if __name__ == "__main__":
    unittest.main()
