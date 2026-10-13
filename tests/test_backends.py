import base64
import json
from io import BytesIO

import httpx
import pytest
from openai import OpenAI
from PIL import Image

from challenger import Challenger
from main_agent import MainAgent
from model_client import ModelError, OpenAIClient
from models import Settings
from strong_solver import SOLVER, StrongSolver
from verifier import Verifier
from weak_solver import OPENROUTER_BASE_URL, WeakSolver


def response_body(text, status="completed"):
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "model": "test-model",
        "status": status,
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 10,
            "total_tokens": 20,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def chat_body(text, finish="stop"):
    return {
        "id": "chat_test",
        "object": "chat.completion",
        "created": 0,
        "model": Settings().weak_model,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": text},
            }
        ],
    }


def catalog_body(model_id=None, modalities=None, parameters=None):
    return {
        "data": [
            {
                "id": model_id or Settings().weak_model,
                "architecture": {"input_modalities": modalities or ["text", "image"]},
                "supported_parameters": parameters or ["response_format", "structured_outputs"],
            }
        ]
    }


def mock_client(handler, base_url=None):
    return OpenAI(
        api_key="test-only",
        base_url=base_url,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_real_sdk_requests_keep_labels_and_marked_images_away_from_solvers(image):
    requests = {}

    def respond(request):
        requests[request.url.host] = json.loads(request.content)
        text = '{"bbox_2d":[100,200,500,800]}'
        body = chat_body(text) if request.url.host == "openrouter.ai" else response_body(text)
        return httpx.Response(200, json=body)

    settings = Settings()
    client = OpenAIClient(settings, client=mock_client(respond))
    weak_solver = WeakSolver(settings, client=mock_client(respond, OPENROUTER_BASE_URL))
    strong_solver = StrongSolver(client, settings.strong_model)
    weak = weak_solver.solve(image, "cup beside the plate")
    strong = strong_solver.solve(image, "cup beside the plate")
    assert weak == strong
    weak_request = requests["openrouter.ai"]
    strong_request = requests["api.openai.com"]
    assert weak_request["messages"][0]["content"] == strong_request["instructions"] == SOLVER
    assert strong_request["store"] is False
    assert weak_request["model"] == settings.weak_model
    assert strong_request["model"] == settings.strong_model
    for content in [weak_request["messages"][1]["content"], strong_request["input"][0]["content"]]:
        assert len(content) == 2
        assert json.loads(content[0]["text"]) == {"referring_expression": "cup beside the plate"}
        image_url = content[1]["image_url"]
        if isinstance(image_url, dict):
            image_url = image_url["url"]
        raw = base64.b64decode(image_url.split(",")[1])
        decoded = Image.open(BytesIO(raw))
        assert decoded.tobytes() == image.tobytes()


@pytest.mark.parametrize("text", ['{"bbox_2d":[900,200,100,800]}', '{"bbox_2d":[0,0,2000,900]}'])
def test_openai_rejects_invalid_model_boxes(image, text):
    client = mock_client(lambda _: httpx.Response(200, json=response_body(text)))
    transport = OpenAIClient(Settings(), client=client)
    solver = StrongSolver(transport, Settings().strong_model)
    with pytest.raises(ModelError, match="invalid structured"):
        solver.solve(image, "cup")


def test_openai_refusal_is_not_scored(image):
    body = response_body("")
    body["output"][0]["content"] = [{"type": "refusal", "refusal": "Refused"}]
    client = OpenAIClient(Settings(), client=mock_client(lambda _: httpx.Response(200, json=body)))
    with pytest.raises(ModelError, match="refused"):
        StrongSolver(client, Settings().strong_model).solve(image, "cup")


@pytest.mark.parametrize("code", ["rate_limit_exceeded", "insufficient_quota", "slow_down"])
def test_openai_429_reports_model_code_and_safe_rate_headers(image, code):
    def respond(_):
        return httpx.Response(
            429,
            json={
                "error": {
                    "message": "Server body with sk-private-credential must not be saved.",
                    "type": "rate_limit_error",
                    "code": code,
                }
            },
            headers={
                "x-ratelimit-limit-requests": "10",
                "x-ratelimit-remaining-requests": "8",
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-tokens": "6m0s",
                "retry-after": "25",
                "x-ratelimit-reset-requests": "sk-private-credential",
            },
        )

    transport = OpenAIClient(Settings(), client=mock_client(respond))
    with pytest.raises(ModelError) as caught:
        StrongSolver(transport, "gpt-5.6-luna").solve(image, "cup")
    message = str(caught.value)
    assert "strong_solver" in message and "model=gpt-5.6-luna" in message
    assert f"code={code}" in message
    assert "x-ratelimit-remaining-tokens=0" in message
    assert "x-ratelimit-limit-requests=10" in message
    assert "retry-after=25" in message
    assert "sk-private-credential" not in message
    assert "Server body" not in message
    if code == "insufficient_quota":
        assert "waiting does not fix" in message


def test_openrouter_uses_vision_chat_api_and_json_schema(image, sample):
    requests = []

    def respond(request):
        if request.method == "GET":
            assert str(request.url) == OPENROUTER_BASE_URL + "/models"
            return httpx.Response(200, json=catalog_body())
        assert str(request.url) == OPENROUTER_BASE_URL + "/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-only"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=chat_body('{"bbox_2d":[100,200,500,800]}'))

    solver = WeakSolver(Settings(), client=mock_client(respond, OPENROUTER_BASE_URL))
    solver.check_connection()
    prediction = solver.solve(image, "cup on the table")
    assert prediction.bbox == sample.target
    request = requests[0]
    assert request["messages"][0]["content"] == SOLVER
    assert json.loads(request["messages"][1]["content"][0]["text"]) == {
        "referring_expression": "cup on the table"
    }
    assert request["model"] == "qwen/qwen3-vl-8b-instruct"
    assert request["response_format"]["type"] == "json_schema"
    schema = request["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["properties"]["bbox_2d"]
    assert request["provider"] == {"require_parameters": True}


def test_qwen_credentials_are_separate(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test-secret")
    solver = WeakSolver(Settings())
    assert solver.client.api_key == "router-test-secret"
    assert str(solver.client.base_url) == OPENROUTER_BASE_URL + "/"
    solver.client.close()


@pytest.mark.parametrize("key", ["", "EMPTY", "your-openrouter-key"])
def test_openrouter_key_required_even_with_openai_key(monkeypatch, key):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        WeakSolver(Settings())


@pytest.mark.parametrize(
    "catalog, message",
    [
        ({"data": []}, "OpenRouter model ID"),
        (catalog_body(modalities=["text"]), "image inputs"),
        (catalog_body(parameters=["temperature"]), "JSON-schema"),
    ],
)
def test_openrouter_rejects_incompatible_models(catalog, message):
    client = mock_client(lambda _: httpx.Response(200, json=catalog), OPENROUTER_BASE_URL)
    with pytest.raises(ModelError, match=message):
        WeakSolver(Settings(), client=client).check_connection()


def test_openrouter_api_failure_is_not_a_weak_solver_failure(image):
    client = mock_client(
        lambda _: httpx.Response(402, json={"error": {"message": "Insufficient credits"}}),
        OPENROUTER_BASE_URL,
    )
    with pytest.raises(ModelError, match="credits"):
        WeakSolver(Settings(), client=client).solve(image, "cup")


def test_complete_openai_qwen_loop_through_sdk_transports(tmp_path, image, sample):
    openai_requests = []

    def openai_response(request):
        body = json.loads(request.content)
        openai_requests.append(body)
        schema = body["text"]["format"]["name"]
        outputs = {
            "ChallengerDirection": {
                "assessment": "Use a visible relation to distinguish this cup.",
                "challenger_prompt": "Describe the cup by its position relative to the plate.",
            },
            "Candidate": {
                "referring_expression": "cup beside the white plate",
                "strategy": "relation",
            },
            "GroundingAnswer": {"bbox_2d": [100, 200, 500, 800]},
        }
        return httpx.Response(200, json=response_body(json.dumps(outputs[schema])))

    def qwen_response(request):
        return httpx.Response(200, json=chat_body('{"bbox_2d":[700,100,900,300]}'))

    settings = Settings()
    weak = WeakSolver(settings, client=mock_client(qwen_response, OPENROUTER_BASE_URL))
    client = OpenAIClient(settings, client=mock_client(openai_response))
    main_agent = MainAgent(
        client,
        settings,
        Challenger(client, settings.challenger_model),
        weak,
        StrongSolver(client, settings.strong_model),
        Verifier(settings),
    )
    result = main_agent.run(sample, image, lambda _: None)
    assert result.status == "accepted"
    assert len(result.rounds) == 1
    assert [len(r["input"][0]["content"]) for r in openai_requests] == [3, 3, 2]
    challenger_input = json.loads(openai_requests[1]["input"][0]["content"][0]["text"])
    assert challenger_input["challenger_prompt"] == result.rounds[0].direction.challenger_prompt
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "finish,content",
    [("length", '{"bbox_2d":[100,200,500,800]}'), ("stop", '{"bbox_2d":[500,200,100,800]}')],
)
def test_qwen_invalid_or_truncated_answer_is_an_error(image, finish, content):
    body = chat_body(content, finish)
    solver = WeakSolver(
        Settings(),
        client=mock_client(lambda _: httpx.Response(200, json=body), OPENROUTER_BASE_URL),
    )
    with pytest.raises(ModelError):
        solver.solve(image, "cup")
