from app.embed import ClipEmbedder


def test_clip_embedder(monkeypatch):
    class MockModel:
        def to(self, *args, **kwargs):
            return self

        def eval(self):
            pass

    monkeypatch.setattr("open_clip.create_model_and_transforms", lambda *a, **k: (MockModel(), None, None))
    monkeypatch.setattr("open_clip.get_tokenizer", lambda *a, **k: None)

    e = ClipEmbedder()
    assert e.model is not None
