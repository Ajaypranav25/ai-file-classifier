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

    def test_embed_image(self):
        embedder = get_embedder()
        from PIL import Image
        import io
        img = Image.new('RGB', (10, 10), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        vec = embedder.embed_image(img_bytes)
        self.assertEqual(vec.shape, (512,))
        self.assertEqual(vec.dtype, np.float32)

    def test_pick_device(self):
        from app.embed import _pick_device
        from app.config import CFG
        import unittest.mock

        with unittest.mock.patch.object(CFG, 'embedding_device', 'cpu'):
            self.assertEqual(_pick_device(), 'cpu')

        with unittest.mock.patch.object(CFG, 'embedding_device', 'auto'):
            with unittest.mock.patch('torch.cuda.is_available', return_value=True):
                self.assertEqual(_pick_device(), 'cuda')

            with unittest.mock.patch('torch.cuda.is_available', return_value=False):
                with unittest.mock.patch('torch.backends.mps.is_available', return_value=True):
                    self.assertEqual(_pick_device(), 'mps')

            with unittest.mock.patch('torch.cuda.is_available', return_value=False):
                with unittest.mock.patch('torch.backends.mps.is_available', return_value=False):
                    self.assertEqual(_pick_device(), 'cpu')

    def test_embed_empty_text(self):
        embedder = get_embedder()
        vec = embedder.embed_text("")
        self.assertEqual(vec.shape, (512,))

    def test_embed_none_text(self):
        embedder = get_embedder()
        vec = embedder.embed_text(None)
        self.assertEqual(vec.shape, (512,))

    def test_embed_empty_texts(self):
        embedder = get_embedder()
        vecs = embedder.embed_texts(["", " ", None])
        self.assertEqual(vecs.shape, (3, 512))


if __name__ == "__main__":
    unittest.main()
