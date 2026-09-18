import numpy as np
from app.classify import Classifier


def test_classifier_zero_shot(monkeypatch):
    monkeypatch.setattr(Classifier, "_try_load_trained", lambda self: None)
    c = Classifier()
    assert c.is_trained is False

    # Mock _zero_shot_classify
    monkeypatch.setattr("app.classify._zero_shot_classify", lambda x: ("MockCat", 0.99))
    label, conf, src = c.classify(np.zeros(512))
    assert label == "MockCat"
    assert src == "zero-shot"
