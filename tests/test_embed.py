import unittest
import numpy as np

from app.embed import get_embedder


class TestEmbedder(unittest.TestCase):
    def test_embed_text(self):
        embedder = get_embedder()
        vec = embedder.embed_text("test string")
        self.assertEqual(vec.shape, (512,))
        self.assertEqual(vec.dtype, np.float32)

    def test_embed_texts(self):
        embedder = get_embedder()
        vecs = embedder.embed_texts(["test string 1", "test string 2"])
        self.assertEqual(vecs.shape, (2, 512))
        self.assertEqual(vecs.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
