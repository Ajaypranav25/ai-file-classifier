import unittest
from starlette.testclient import TestClient
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


if __name__ == "__main__":
    unittest.main()
