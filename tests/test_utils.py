import unittest
from unittest.mock import patch

from app.utils import open_file_locally


class TestUtils(unittest.TestCase):
    @patch("app.utils.platform.system", return_value="Darwin")
    @patch("app.utils.subprocess.run")
    def test_open_file_locally_mac(self, mock_run, mock_system):
        open_file_locally("/tmp/test.png", check=True)
        mock_run.assert_called_once_with(["open", "/tmp/test.png"], check=True)

    @patch("app.utils.platform.system", return_value="Windows")
    @patch("os.startfile", create=True)
    def test_open_file_locally_windows(self, mock_startfile, mock_system):
        open_file_locally("/tmp/test.png", check=False)
        mock_startfile.assert_called_once_with("/tmp/test.png")

    @patch("app.utils.platform.system", return_value="Linux")
    @patch("app.utils.subprocess.run")
    def test_open_file_locally_linux(self, mock_run, mock_system):
        open_file_locally("/tmp/test.png", check=False)
        mock_run.assert_called_once_with(["xdg-open", "/tmp/test.png"], check=False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
