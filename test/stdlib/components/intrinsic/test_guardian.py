"""Tests of the code in ``mellea.stdlib.components.intrinsic.guardian``"""

import gc
import json
import os
import pathlib

import pytest
import torch

from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_4_MICRO_3B
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.context import ChatContext

# Skip entire module in CI since all tests are qualitative
pytestmark = [
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping Guardian tests in CI - all qualitative tests",
    ),
    pytest.mark.huggingface,
    pytest.mark.requires_gpu,
    pytest.mark.requires_heavy_ram,
    pytest.mark.llm,
]

DATA_ROOT = pathlib.Path(os.path.dirname(__file__)) / "testdata"
"""Location of data files for the tests in this file."""

# Local path to the guardian-core LoRA adapter.  When the adapter is published
# on Hugging Face Hub this fixture can be removed and the ``lora_path`` kwarg
# dropped from the test calls.
_LORA_PATH = (
    pathlib.Path(__file__).resolve().parents[5]
    / "granitelib-guardian-r1.0"
    / "guardian-core"
    / "granite-4.0-micro"
    / "lora"
)


@pytest.fixture(name="backend", scope="module")
def _backend():
    """Backend used by the tests in this file. Module-scoped to avoid reloading the model for each test."""
    torch.set_num_threads(4)

    backend_ = LocalHFBackend(model_id=IBM_GRANITE_4_MICRO_3B.hf_model_name)  # type: ignore
    yield backend_

    del backend_
    gc.collect()
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()


def _read_guardian_input(file_name: str) -> ChatContext:
    """Read test input and convert to a ChatContext."""
    with open(DATA_ROOT / "input_json" / file_name, encoding="utf-8") as f:
        json_data = json.load(f)

    context = ChatContext()
    for m in json_data["messages"]:
        context = context.add(Message(m["role"], m["content"]))
    return context


@pytest.mark.qualitative
def test_guardian_check_harm(backend):
    """Verify that guardian_check detects harmful prompts."""
    context = _read_guardian_input("guardian_core.json")

    # First call triggers adapter loading
    result = guardian.guardian_check(
        context, backend, criteria="harm", target_role="user",
        lora_path=_LORA_PATH,
    )
    assert isinstance(result, float)
    assert 0.7 <= result <= 1.0, f"Expected high risk score, got {result}"

    # Second call hits a different code path from the first one
    result = guardian.guardian_check(
        context, backend, criteria="harm", target_role="user",
        lora_path=_LORA_PATH,
    )
    assert isinstance(result, float)
    assert 0.7 <= result <= 1.0, f"Expected high risk score, got {result}"


@pytest.mark.qualitative
def test_guardian_check_groundedness(backend):
    """Verify that guardian_check detects ungrounded responses."""
    context = (
        ChatContext()
        .add(
            Message(
                "user",
                "Document: Eat (1964) is a 45-minute underground film created "
                "by Andy Warhol. The film was first shown by Jonas Mekas on "
                "July 16, 1964, at the Washington Square Gallery.",
            )
        )
        .add(
            Message(
                "assistant",
                "The film Eat was first shown by Jonas Mekas on December 24, "
                "1922 at the Washington Square Gallery.",
            )
        )
    )

    result = guardian.guardian_check(
        context, backend, criteria="groundedness", lora_path=_LORA_PATH,
    )
    assert isinstance(result, float)
    assert 0.7 <= result <= 1.0, f"Expected high risk score, got {result}"


@pytest.mark.qualitative
def test_guardian_check_function_call(backend):
    """Verify that guardian_check detects function call hallucinations."""
    tools = [
        {
            "name": "comment_list",
            "description": "Fetches a list of comments for a specified IBM video.",
            "parameters": {
                "aweme_id": {
                    "description": "The ID of the IBM video.",
                    "type": "int",
                    "default": "7178094165614464282",
                },
                "cursor": {
                    "description": "The cursor for pagination. Defaults to 0.",
                    "type": "int, optional",
                    "default": "0",
                },
                "count": {
                    "description": "The number of comments to fetch. Maximum is 30. Defaults to 20.",
                    "type": "int, optional",
                    "default": "20",
                },
            },
        }
    ]
    tools_text = "Available tools:\n" + json.dumps(tools, indent=2)
    user_text = "Fetch the first 15 comments for the IBM video with ID 456789123."
    # Deliberately wrong: uses "video_id" instead of "aweme_id"
    response_text = str(
        [{"name": "comment_list", "arguments": {"video_id": 456789123, "count": 15}}]
    )

    context = (
        ChatContext()
        .add(Message("user", f"{tools_text}\n\n{user_text}"))
        .add(Message("assistant", response_text))
    )

    result = guardian.guardian_check(
        context, backend, criteria="function_call", lora_path=_LORA_PATH,
    )
    assert isinstance(result, float)
    assert 0.7 <= result <= 1.0, f"Expected high risk score, got {result}"


if __name__ == "__main__":
    pytest.main([__file__])
