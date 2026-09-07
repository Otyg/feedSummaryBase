import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from feedsummary_core.persistence import create_store


class ReadOnlyStoreFactoryTests(unittest.TestCase):
    def test_read_only_sqlite_does_not_create_database_or_parent(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing" / "store.sqlite"
            with self.assertRaises(FileNotFoundError):
                create_store(
                    {
                        "provider": "sqlite",
                        "path": str(path),
                        "initialize_schema": False,
                    }
                )
            self.assertFalse(path.parent.exists())

    def test_read_only_tinydb_does_not_create_database_or_parent(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing" / "store.json"
            with self.assertRaises(FileNotFoundError):
                create_store(
                    {
                        "provider": "tinydb",
                        "path": str(path),
                        "initialize_schema": False,
                    }
                )
            self.assertFalse(path.parent.exists())


if __name__ == "__main__":
    unittest.main()
