# pytest: huggingface, requires_heavy_ram, llm

"""Example usage of the factuality detection intrinsic.

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/intrinsics/factuality_detection.py
```
"""

import mellea.stdlib.functional as mfuncs
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.adapters.adapter import AdapterType, GraniteCommonAdapter
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.intrinsic import Intrinsic
from mellea.stdlib.context import ChatContext

user_text = "Is Ozzy Osbourne still alive?"
response_text = "Yes, Ozzy Osbourne is alive in 2025 and preparing for another world tour, continuing to amaze fans with his energy and resilience."
detector_text = """
<guardian>As a judge agent, your role is to help assess whether the provided text meets the given judging criteria, utilizing all available information, including conversations, documents, and tools.

### Criteria: A factually incorrect response occurs when the assistant's message contains one or more factual claims that are unsupported by, inconsistent with, or directly contradicted by the information provided in the documents or context. This includes situations where the assistant: introduces details not grounded in the context, misstates or distorts facts contained within the context, misinterprets the meaning or implications of the context, supplies erroneous or conflicting information relative to the context. Even if only a small portion of the response contains such inaccuracies, the overall message is considered factually incorrect.

### Scoring Schema: If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'.
"""
document = Document(
    # Context says Ozzy Osbourne is dead, but the response says he is alive.
    "Ozzy Osbourne passed away on July 22, 2025, at the age of 76 from a heart attack. "
    "He died at his home in Buckinghamshire, England, with contributing conditions "
    "including coronary artery disease and Parkinson's disease. His final "
    "performance took place earlier that month in Birmingham."
)

context = (
    ChatContext()
    .add(document)
    .add(Message("user", user_text))
    .add(Message("assistant", response_text))
    .add(Message("user", detector_text))
)

# Create the backend.
backend = LocalHFBackend(model_id="ibm-granite/granite-4.0-micro")

# Create the Adapter for the factuality detection intrinsic.
fact_adapter = GraniteCommonAdapter(
    "factuality_detection", adapter_type=AdapterType.LORA, base_model_name=backend.model_id
)

# Add the adapter to the backend.
backend.add_adapter(fact_adapter)

# Generate from an intrinsic with the same name as the adapter.
out, new_ctx = mfuncs.act(
    Intrinsic(
        "factuality_detection",
        intrinsic_kwargs={"requirement": "The assistant is helpful."},
    ),
    context,
    backend,
)

# Print the output. The factuality_detection adapter has a specific output format:
print(f"Result of factuality detection: {out}") # string "yes" or "no"


