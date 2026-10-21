import base64
import json
from io import BytesIO

import httpx
import pytest
from conftest import make_settings
from openai import OpenAI
from PIL import Image
from pydantic import ValidationError

from challenger import Challenger
from judge import INSTRUCTIONS as JUDGE_INSTRUCTIONS
from judge import Judge
from main_agent import MainAgent
from model_client import OPENROUTER_BASE_URL, ModelError, OpenAIClient, OpenRouterClient
from models import Box, Candidate, ChallengerDirection, GroundingAnswer, Settings, Verdict
from solver import Solver, grounding_prompt
from verifier import Verifier


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
        "model": make_settings().weak_model,
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
                "id": model_id or make_settings().weak_model,
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

    settings = make_settings()
    client = OpenAIClient(settings, client=mock_client(respond))
    weak_solver = Solver(
        OpenRouterClient(settings, client=mock_client(respond, OPENROUTER_BASE_URL)),
        settings.weak_model,
        "weak_solver",
    )
    strong_solver = Solver(client, settings.strong_model, "strong_solver")
    weak = weak_solver.solve(image, "cup beside the plate")
    strong = strong_solver.solve(image, "cup beside the plate")
    assert weak == strong
    weak_request = requests["openrouter.ai"]
    strong_request = requests["api.openai.com"]
    assert "instructions" not in strong_request
    assert [m["role"] for m in weak_request["messages"]] == ["user"]
    assert weak_request["temperature"] == 0
    assert strong_request["store"] is False
    assert weak_request["model"] == settings.weak_model
    assert strong_request["model"] == settings.strong_model
    for content in [weak_request["messages"][0]["content"], strong_request["input"][0]["content"]]:
        assert len(content) == 2
        assert content[-1]["text"] == grounding_prompt("cup beside the plate")
        image_url = content[0]["image_url"]
        if isinstance(image_url, dict):
            image_url = image_url["url"]
        raw = base64.b64decode(image_url.split(",")[1])
        decoded = Image.open(BytesIO(raw))
        assert decoded.tobytes() == image.tobytes()


@pytest.mark.parametrize("text", ['{"bbox_2d":[900,200,100,800]}', '{"bbox_2d":[0,0,2000,900]}'])
def test_openai_rejects_invalid_model_boxes(image, text):
    client = mock_client(lambda _: httpx.Response(200, json=response_body(text)))
    transport = OpenAIClient(make_settings(), client=client)
    solver = Solver(transport, make_settings().strong_model, "strong_solver")
    with pytest.raises(ModelError, match="invalid structured"):
        solver.solve(image, "cup")


def test_openai_refusal_is_not_scored(image):
    body = response_body("")
    body["output"][0]["content"] = [{"type": "refusal", "refusal": "Refused"}]
    client = OpenAIClient(
        make_settings(), client=mock_client(lambda _: httpx.Response(200, json=body))
    )
    with pytest.raises(ModelError, match="refused"):
        Solver(client, make_settings().strong_model, "strong_solver").solve(image, "cup")


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

    transport = OpenAIClient(make_settings(), client=mock_client(respond))
    with pytest.raises(ModelError) as caught:
        Solver(transport, "gpt-5.6-luna", "strong_solver").solve(image, "cup")
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

    settings = make_settings()
    router = OpenRouterClient(settings, client=mock_client(respond, OPENROUTER_BASE_URL))
    router.check_model(settings.weak_model, "weak_solver")
    solver = Solver(router, settings.weak_model, "weak_solver")
    prediction = solver.solve(image, "cup on the table")
    assert prediction.bbox == sample.target
    request = requests[0]
    assert [m["role"] for m in request["messages"]] == ["user"]
    content = request["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[-1]["text"] == grounding_prompt("cup on the table")
    assert request["temperature"] == 0
    assert request["model"] == "qwen/qwen3-vl-8b-instruct"
    assert request["response_format"]["type"] == "json_schema"
    schema = request["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["properties"]["bbox_2d"]
    assert request["provider"] == {"require_parameters": True}


def test_judge_sends_both_images_and_parses_a_verdict(image, sample):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        reply = {
            "claims": [{"claim": "red cup", "evidence": "seen", "target": "true"}],
            "also_fits": ["the red cup on the shelf"],
            "reason": "A second red cup.",
        }
        return httpx.Response(200, json=chat_body(json.dumps(reply)))

    settings = make_settings()
    router = OpenRouterClient(settings, client=mock_client(respond, OPENROUTER_BASE_URL))
    chosen = Box(x_min=600, y_min=100, x_max=900, y_max=400)
    verdict = Judge(router, settings.judge_model).review(image, sample, "the red cup", chosen)
    assert verdict == Verdict(
        claims=[{"claim": "red cup", "evidence": "seen", "target": "true"}],
        also_fits=["the red cup on the shelf"],
        reason="A second red cup.",
    )
    request = requests[0]
    assert request["model"] == settings.judge_model
    assert request["messages"][0] == {"role": "system", "content": JUDGE_INSTRUCTIONS}
    content = request["messages"][1]["content"]
    assert [part["type"] for part in content] == ["image_url", "image_url", "text"]
    assert "the red cup" in content[-1]["text"] and "target_bbox_2d" in content[-1]["text"]
    assert "weak_solver_bbox_2d" in content[-1]["text"]
    assert request["response_format"]["json_schema"]["name"] == "Verdict"


def test_qwen_credentials_are_separate(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test-secret")
    router = OpenRouterClient(make_settings())
    assert router.client.api_key == "router-test-secret"
    assert str(router.client.base_url) == OPENROUTER_BASE_URL + "/"
    router.client.close()


@pytest.mark.parametrize("key", ["", "EMPTY", "your-openrouter-key"])
def test_openrouter_key_required_even_with_openai_key(monkeypatch, key):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterClient(make_settings())


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
        OpenRouterClient(make_settings(), client=client).check_model(
            make_settings().weak_model, "weak_solver"
        )


def test_openrouter_api_failure_is_not_a_weak_solver_failure(image):
    client = mock_client(
        lambda _: httpx.Response(402, json={"error": {"message": "Insufficient credits"}}),
        OPENROUTER_BASE_URL,
    )
    with pytest.raises(ModelError, match="credits"):
        Solver(
            OpenRouterClient(make_settings(), client=client),
            make_settings().weak_model,
            "weak_solver",
        ).solve(image, "cup")


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
            },
            "GroundingAnswer": {"bbox_2d": [100, 200, 500, 800]},
            "Verdict": {
                "claims": [{"claim": "cup beside the plate", "evidence": "seen", "target": "true"}],
                "also_fits": [],
                "reason": "Only one cup.",
            },
        }
        return httpx.Response(200, json=response_body(json.dumps(outputs[schema])))

    def qwen_response(request):
        return httpx.Response(200, json=chat_body('{"bbox_2d":[700,100,900,300]}'))

    settings = make_settings()
    weak = Solver(
        OpenRouterClient(settings, client=mock_client(qwen_response, OPENROUTER_BASE_URL)),
        settings.weak_model,
        "weak_solver",
    )
    client = OpenAIClient(settings, client=mock_client(openai_response))
    main_agent = MainAgent(
        client,
        settings,
        Challenger(client, settings.challenger_model),
        weak,
        Solver(client, settings.strong_model, "strong_solver"),
        Judge(client, settings.judge_model),
        Verifier(settings),
    )
    result = main_agent.run(sample, image, lambda _: None)
    assert result.status == "accepted"
    assert len(result.rounds) == 1
    assert [len(r["input"][0]["content"]) for r in openai_requests] == [2, 2, 2, 3]
    challenger_input = json.loads(openai_requests[1]["input"][0]["content"][-1]["text"])
    for request in openai_requests[:2]:
        assert request["instructions"]
        assert "temperature" not in request
    assert challenger_input["challenger_prompt"] == result.rounds[0].direction.challenger_prompt
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "finish,content",
    [("length", '{"bbox_2d":[100,200,500,800]}'), ("stop", '{"bbox_2d":[500,200,100,800]}')],
)
def test_qwen_invalid_or_truncated_answer_is_an_error(image, finish, content):
    body = chat_body(content, finish)
    settings = make_settings()
    solver = Solver(
        OpenRouterClient(
            settings,
            client=mock_client(lambda _: httpx.Response(200, json=body), OPENROUTER_BASE_URL),
        ),
        settings.weak_model,
        "weak_solver",
    )
    with pytest.raises(ModelError):
        solver.solve(image, "cup")


def test_strong_solver_on_openrouter_uses_chat_api_with_strong_model(image, sample):
    requests = []

    def respond(request):
        assert str(request.url) == OPENROUTER_BASE_URL + "/chat/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=chat_body('{"bbox_2d":[100,200,500,800]}'))

    settings = make_settings(strong_provider="openrouter")
    router = OpenRouterClient(settings, client=mock_client(respond, OPENROUTER_BASE_URL))
    prediction = Solver(router, settings.strong_model, "strong_solver").solve(image, "cup")
    assert prediction.bbox == sample.target
    request = requests[0]
    assert request["model"] == "qwen/qwen3-vl-235b-a22b-instruct"
    assert [m["role"] for m in request["messages"]] == ["user"]
    assert request["messages"][0]["content"][-1]["text"] == grounding_prompt("cup")
    assert request["temperature"] == 0


def test_openai_preflight_rejects_openrouter_model_id_before_generation():
    model = "qwen/qwen3-vl-235b-a22b-instruct"

    def respond(request):
        if request.method == "POST":
            pytest.fail("Pre-flight must not make a generation call")
        assert request.url.path == f"/v1/models/{model}"
        error = {"message": "not found", "type": "invalid_request_error", "code": "model_not_found"}
        return httpx.Response(404, json={"error": error})

    client = OpenAIClient(make_settings(), client=mock_client(respond))
    with pytest.raises(ModelError) as raised:
        client.check_model(model, "strong_solver")
    assert model in str(raised.value)
    assert "AUTODATA_STRONG_PROVIDER" in str(raised.value)


def test_openrouter_catalogue_checked_once_for_both_solvers():
    settings = make_settings()
    methods = []
    catalogue = catalog_body(settings.weak_model)
    catalogue["data"] += catalog_body(settings.strong_model)["data"]

    def respond(request):
        methods.append(request.method)
        return httpx.Response(200, json=catalogue)

    router = OpenRouterClient(settings, client=mock_client(respond, OPENROUTER_BASE_URL))
    router.check_model(settings.weak_model, "weak_solver")
    router.check_model(settings.strong_model, "strong_solver")
    assert methods == ["GET"]


def test_openai_preflight_reports_auth_failure_without_blaming_the_model_id():
    def respond(request):
        assert request.method == "GET"
        error = {"message": "bad key", "type": "invalid_request_error", "code": "invalid_api_key"}
        return httpx.Response(401, json={"error": error})

    client = OpenAIClient(make_settings(), client=mock_client(respond))
    with pytest.raises(ModelError) as raised:
        client.check_model("gpt-5.4-nano", "main_agent")
    message = str(raised.value)
    assert "HTTP 401" in message and "main_agent" in message
    assert "does not offer" not in message


# Keywords OpenAI accepts in a strict response schema; anything else is rejected with HTTP 400.
STRICT_SCHEMA_KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "anyOf",
    "$ref",
    "$defs",
    "title",
    "description",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "pattern",
    "format",
}


def schema_keywords(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"properties", "$defs"}:
                for child in value.values():
                    yield from schema_keywords(child)
            else:
                yield key
                yield from schema_keywords(value)
    elif isinstance(node, list):
        for item in node:
            yield from schema_keywords(item)


@pytest.mark.parametrize("schema", [GroundingAnswer, Candidate, ChallengerDirection, Verdict])
def test_response_schemas_use_only_openai_strict_keywords(schema):
    unsupported = set(schema_keywords(schema.model_json_schema())) - STRICT_SCHEMA_KEYWORDS
    assert unsupported == set()


@pytest.mark.parametrize("value", ["", "   ", "x" * 401])
def test_candidate_text_limits_still_apply_after_parsing(value):
    with pytest.raises(ValidationError):
        Candidate(referring_expression=value)
    assert len(Candidate(referring_expression="x" * 400).referring_expression) == 400


def test_direction_text_limits_still_apply_after_parsing():
    with pytest.raises(ValidationError):
        ChallengerDirection(assessment="x" * 1001, challenger_prompt="ok")
    with pytest.raises(ValidationError):
        ChallengerDirection(assessment="ok", challenger_prompt="x" * 1601)


def test_model_ids_come_from_env_with_no_defaults(monkeypatch):
    names = [
        "AUTODATA_MAIN_MODEL",
        "AUTODATA_CHALLENGER_MODEL",
        "AUTODATA_STRONG_MODEL",
        "AUTODATA_WEAK_MODEL",
        "AUTODATA_JUDGE_MODEL",
    ]
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AUTODATA_MAIN_MODEL", "gpt-5.6-luna")
    with pytest.raises(ValueError) as raised:
        Settings.from_env()
    assert "AUTODATA_CHALLENGER_MODEL" in str(raised.value)
    assert "AUTODATA_MAIN_MODEL" not in str(raised.value)
    for name in names:
        monkeypatch.setenv(name, "model-" + name.lower())
    settings = Settings.from_env(max_rounds=1)
    assert settings.weak_model == "model-autodata_weak_model" and settings.max_rounds == 1


def test_rate_limit_is_waited_out_and_the_call_then_succeeds(image, rate_limit_waits):
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "rate-limited upstream"}})
        return httpx.Response(200, json=chat_body('{"bbox_2d":[100,200,500,800]}'))

    router = OpenRouterClient(make_settings(), client=mock_client(respond, OPENROUTER_BASE_URL))
    prediction = Solver(router, make_settings().weak_model, "weak_solver").solve(image, "cup")
    assert prediction.bbox.as_xyxy() == [100, 200, 500, 800]
    assert rate_limit_waits == [30]
    assert len(calls) == 2


def test_persistent_rate_limit_fails_after_bounded_waits(image, rate_limit_waits):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(429, json={"error": {"message": "rate-limited upstream"}})

    router = OpenRouterClient(make_settings(), client=mock_client(respond, OPENROUTER_BASE_URL))
    with pytest.raises(ModelError, match="RateLimitError"):
        Solver(router, make_settings().weak_model, "weak_solver").solve(image, "cup")
    assert rate_limit_waits == [30, 60, 120]
    assert len(calls) == 4
