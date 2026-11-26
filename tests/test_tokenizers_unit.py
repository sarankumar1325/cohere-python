import asyncio
import threading
import typing
import unittest
from unittest import mock

import requests

from cohere.manually_maintained import tokenizers

_TOKENIZER_JSON = (
    '{"version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null, '
    '"pre_tokenizer": null, "post_processor": null, "decoder": null, '
    '"model": {"type": "WordLevel", "vocab": {"hello": 0, "[UNK]": 1}, "unk_token": "[UNK]"}}'
)


class _FakeModels:
    def __init__(self, url: typing.Optional[str]) -> None:
        self._url = url

    def get(self, model: str) -> typing.Any:
        return mock.Mock(tokenizer_url=self._url)


class _FakeAsyncModels(_FakeModels):
    async def get(self, model: str) -> typing.Any:  # type: ignore[override]
        return super().get(model)


class _FakeClient:
    def __init__(self, models: typing.Any) -> None:
        self.models = models
        self._cache: typing.Dict[str, typing.Any] = {}

    def _cache_get(self, key: str) -> typing.Any:
        return self._cache.get(key)

    def _cache_set(self, key: str, value: typing.Any) -> None:
        self._cache[key] = value


def _response(
    status_code: int, text: str = "", headers: typing.Optional[typing.Dict[str, str]] = None
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = text.encode()
    response.headers.update(headers or {})
    return response


class TestTokenizerDownload(unittest.TestCase):
    def test_requests_use_a_timeout(self) -> None:
        client = _FakeClient(_FakeModels("https://example.invalid/tokenizer.json"))
        with (
            mock.patch.object(
                tokenizers.requests, "head", return_value=_response(200, headers={"Content-Length": "1048576"})
            ) as head,
            mock.patch.object(tokenizers.requests, "get", return_value=_response(200, _TOKENIZER_JSON)) as get,
        ):
            tokenizers.get_hf_tokenizer(client, "m")  # type: ignore[arg-type]

        self.assertIsNotNone(head.call_args.kwargs.get("timeout"))
        self.assertIsNotNone(get.call_args.kwargs.get("timeout"))

    def test_http_error_status_is_reported(self) -> None:
        client = _FakeClient(_FakeModels("https://example.invalid/tokenizer.json"))
        with (
            mock.patch.object(
                tokenizers.requests, "head", return_value=_response(200, headers={"Content-Length": "9"})
            ),
            mock.patch.object(tokenizers.requests, "get", return_value=_response(404, "NoSuchKey")),
        ):
            with self.assertRaises(requests.HTTPError) as ctx:
                tokenizers.get_hf_tokenizer(client, "m")  # type: ignore[arg-type]

        self.assertIn("404", str(ctx.exception))
        self.assertEqual(client._cache, {})

    def test_async_requests_do_not_run_on_the_event_loop_thread(self) -> None:
        client = _FakeClient(_FakeAsyncModels("https://example.invalid/tokenizer.json"))
        threads: typing.List[int] = []

        def fake_head(*args: typing.Any, **kwargs: typing.Any) -> requests.Response:
            threads.append(threading.get_ident())
            return _response(200, headers={"Content-Length": "1048576"})

        def fake_get(*args: typing.Any, **kwargs: typing.Any) -> requests.Response:
            threads.append(threading.get_ident())
            return _response(200, _TOKENIZER_JSON)

        async def run() -> int:
            with (
                mock.patch.object(tokenizers.requests, "head", fake_head),
                mock.patch.object(tokenizers.requests, "get", fake_get),
            ):
                await tokenizers.async_get_hf_tokenizer(client, "m")  # type: ignore[arg-type]
            return threading.get_ident()

        loop_thread = asyncio.run(run())

        self.assertEqual(len(threads), 2)
        self.assertNotIn(loop_thread, threads)

    def test_async_http_error_status_is_reported(self) -> None:
        client = _FakeClient(_FakeAsyncModels("https://example.invalid/tokenizer.json"))

        async def run() -> None:
            with (
                mock.patch.object(
                    tokenizers.requests, "head", return_value=_response(200, headers={"Content-Length": "9"})
                ),
                mock.patch.object(tokenizers.requests, "get", return_value=_response(403, "Forbidden")),
            ):
                await tokenizers.async_get_hf_tokenizer(client, "m")  # type: ignore[arg-type]

        with self.assertRaises(requests.HTTPError) as ctx:
            asyncio.run(run())

        self.assertIn("403", str(ctx.exception))
