"""call_groq must skip a decommissioned model immediately instead of retrying it."""
import requests

from backend.app import groq_client


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Client Error")


DECOMMISSIONED = {
    "error": {
        "message": "The model `bad-model` has been decommissioned and is no longer supported.",
        "code": "model_decommissioned",
    }
}
OK = {"choices": [{"message": {"content": "[]"}}]}


def test_decommissioned_model_is_tried_once_then_falls_back(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json["model"])
        if json["model"] == "bad-model":
            return _Response(400, DECOMMISSIONED)
        return _Response(200, OK)

    monkeypatch.setattr(groq_client.requests, "post", fake_post)
    monkeypatch.setattr(groq_client, "get_live_groq_models", lambda api_key=None: ["good-model"])
    monkeypatch.setattr(groq_client, "_WARNED_UNAVAILABLE_MODELS", set())

    text, model = groq_client.call_groq(
        "key", {"messages": [{"role": "user", "content": "hi"}]}, preferred_model="bad-model"
    )

    assert model == "good-model"
    assert calls.count("bad-model") == 1
