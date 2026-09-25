import unittest
import unittest.mock
from fastapi.testclient import TestClient
from app.api import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_search_limit_bound(self):
        # Even if limit is large, it shouldn't crash, and max results wouldn't exceed limit logic bounds
        # We can just test it doesn't 500 when limit is massive
        response = self.client.get("/api/search?q=test&limit=9999")
        self.assertEqual(response.status_code, 200)

        response = self.client.get("/api/search?q=test&limit=-5")
        self.assertEqual(response.status_code, 200)

    def test_categories(self):
        response = self.client.get("/api/categories")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("categories", data)
        self.assertIn("counts", data)
        self.assertIn("total", data)

    def test_open_file_not_found(self):
        response = self.client.get("/api/open?path=/does/not/exist/file.txt")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "File not found"})

    def test_thumbnail_not_found(self):
        response = self.client.get("/api/thumbnail/does_not_exist_id")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "No thumbnail"})

    @unittest.mock.patch("app.api.get_store")
    def test_correct_not_found(self, mock_get_store):
        mock_store = unittest.mock.MagicMock()
        mock_store.update_category.return_value = False
        mock_get_store.return_value = mock_store

        from app.config import CFG
        valid_cat = CFG.category_names[0]
        response = self.client.post("/api/correct", json={"id": "missing_id", "category": valid_cat})
        self.assertEqual(response.status_code, 404)

    def test_correct_unknown_category(self):
        response = self.client.post("/api/correct", json={"id": "some_id", "category": "NonExistentCategory!"})
        self.assertEqual(response.status_code, 400)

    @unittest.mock.patch("app.api.get_classifier")
    def test_reload_classifier(self, mock_get_classifier):
        mock_classifier = unittest.mock.MagicMock()
        mock_classifier.is_trained = True
        mock_get_classifier.return_value = mock_classifier

        response = self.client.post("/api/reload-classifier")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "trained": True})
        mock_classifier.reload.assert_called_once()


if __name__ == "__main__":
    unittest.main()
