from langchain_openai import ChatOpenAI

from agent import llms
from agent.config import get_config


def test_all_roles_point_at_mlflow_gateway():
    config = get_config()
    getters = {
        "small": llms.get_small,
        "heavy": llms.get_heavy,
        "critic": llms.get_critic,
        "testgen": llms.get_testgen,
        "evaluator": llms.get_evaluator,
    }
    for role, getter in getters.items():
        model = getter()
        expected = config.models[role]
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == expected.name
        assert model.openai_api_base == config.gateway.base_url
        assert model.temperature == expected.temperature
        assert model.max_tokens == expected.max_tokens
        assert model.request_timeout == expected.timeout_s


def test_empty_api_key_gets_placeholder():
    model = llms.get_heavy()
    assert model.openai_api_key.get_secret_value()


def test_role_constants_match_config():
    config = get_config()
    assert llms.HEAVY_MODEL == config.models["heavy"].name
    assert llms.SMALL_MODEL == config.models["small"].name
    assert llms.CRITIC_MODEL == config.models["critic"].name
    assert llms.TESTGEN_MODEL == config.models["testgen"].name
    assert llms.EVALUATOR_MODEL == config.models["evaluator"].name
