import unittest
import unittest.mock
import numpy as np

from app.classify import Classifier, _zero_shot_classify
from app.config import CFG, Category


class TestClassify(unittest.TestCase):
    @unittest.mock.patch("app.classify.get_embedder")
    def test_category_centroids(self, mock_get_embedder):
        original_categories = CFG.categories
        CFG.categories = [
            Category(name="Cat1", prompts=["prompt 1"]),
            Category(name="Cat2", prompts=["prompt 2"])
        ]

        # Since it uses @functools.lru_cache we need to clear it so it picks up mocked values
        from app.classify import _category_centroids
        _category_centroids.cache_clear()

        mock_embedder = unittest.mock.MagicMock()
        mock_get_embedder.return_value = mock_embedder

        # Mock embed_texts to return vectors where we know the average and normalized output
        # For Cat1
        vec1 = np.array([[2.0, 0.0]])
        # For Cat2
        vec2 = np.array([[0.0, 3.0]])
        mock_embedder.embed_texts.side_effect = [vec1, vec2]

        try:
            centroids = _category_centroids()
            self.assertEqual(centroids.shape, (2, 2))
            # Should be normalized
            self.assertTrue(np.allclose(centroids[0], [1.0, 0.0]))
            self.assertTrue(np.allclose(centroids[1], [0.0, 1.0]))
        finally:
            CFG.categories = original_categories
            _category_centroids.cache_clear()

    @unittest.mock.patch("app.classify._category_centroids")
    def test_zero_shot_classify(self, mock_centroids):
        # We need to mock CFG.categories to match our mocked centroids
        original_categories = CFG.categories
        CFG.categories = [
            Category(name="Cat1", prompts=["prompt 1"]),
            Category(name="Cat2", prompts=["prompt 2"])
        ]

        mock_centroids.return_value = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        try:
            # emb closer to Cat1
            emb1 = np.array([0.9, 0.1])
            emb1 = emb1 / np.linalg.norm(emb1)

            label, conf = _zero_shot_classify(emb1)
            self.assertEqual(label, "Cat1")
            self.assertTrue(0.0 <= conf <= 1.0)

            # emb closer to Cat2
            emb2 = np.array([0.1, 0.9])
            emb2 = emb2 / np.linalg.norm(emb2)

            label2, conf2 = _zero_shot_classify(emb2)
            self.assertEqual(label2, "Cat2")
            self.assertTrue(0.0 <= conf2 <= 1.0)
        finally:
            CFG.categories = original_categories

    @unittest.mock.patch("app.classify.CFG")
    @unittest.mock.patch("app.classify.joblib.load")
    def test_classifier_with_model(self, mock_load, mock_cfg):
        mock_cfg.classifier_path.exists.return_value = True
        mock_cfg.label_encoder_path.exists.return_value = True

        mock_model = unittest.mock.MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])

        mock_encoder = unittest.mock.MagicMock()
        mock_encoder.inverse_transform.return_value = ["Cat2"]

        # joblib.load is called twice, return model then encoder
        mock_load.side_effect = [mock_model, mock_encoder]

        classifier = Classifier()

        self.assertTrue(classifier.is_trained)

        emb = np.array([0.5, 0.5])
        label, conf, source = classifier.classify(emb)

        self.assertEqual(label, "Cat2")
        self.assertEqual(conf, 0.9)
        self.assertEqual(source, "trained")
        mock_model.predict_proba.assert_called_once()
        mock_encoder.inverse_transform.assert_called_once_with([1])

    @unittest.mock.patch("app.classify._zero_shot_classify")
    @unittest.mock.patch("app.classify.CFG")
    def test_classifier_fallback(self, mock_cfg, mock_zsc):
        mock_cfg.classifier_path.exists.return_value = False
        mock_cfg.label_encoder_path.exists.return_value = False

        mock_zsc.return_value = ("FallbackCat", 0.75)

        classifier = Classifier()
        self.assertFalse(classifier.is_trained)

        emb = np.array([0.5, 0.5])
        label, conf, source = classifier.classify(emb)

        self.assertEqual(label, "FallbackCat")
        self.assertEqual(conf, 0.75)
        self.assertEqual(source, "zero-shot")
        mock_zsc.assert_called_once_with(emb)

    def test_classifier_reload(self):
        c = Classifier()
        c.reload()

    def test_get_classifier_singleton(self):
        from app.classify import get_classifier
        c1 = get_classifier()
        c2 = get_classifier()
        self.assertIs(c1, c2)


if __name__ == "__main__":
    unittest.main()
