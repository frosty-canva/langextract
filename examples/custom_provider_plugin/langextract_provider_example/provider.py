# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal example of a custom provider plugin for LangExtract."""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Iterator, Sequence

import langextract as lx


@lx.providers.registry.register(
    r"^gemini",  # Matches Gemini model IDs (same as default provider)
)
@dataclasses.dataclass(init=False)
class CustomGeminiProvider(lx.inference.BaseLanguageModel):
    """Example custom LangExtract provider implementation.

    This demonstrates how to create a custom provider for LangExtract
    that can intercept and handle model requests. This example uses
    Gemini as the backend, but you would replace this with your own
    API or model implementation.

    Note: Since this registers the same pattern as the default Gemini provider,
    you must explicitly specify this provider when creating a model:

    config = lx.factory.ModelConfig(
        model_id="gemini-2.5-flash",
        provider="CustomGeminiProvider"
    )
    model = lx.factory.create_model(config)
    """

    model_id: str
    api_key: str | None
    project: str | None
    location: str | None
    temperature: float
    _client: Any = dataclasses.field(repr=False, compare=False)

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the custom provider.

        Args:
          model_id: The model ID.
          api_key: API key for the service.
          temperature: Sampling temperature.
          **kwargs: Additional parameters.
        """
        # TODO: Replace with your own client initialization
        try:
            from google import genai  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise lx.exceptions.InferenceConfigError(
                "This example requires google-genai package. "
                "Install with: pip install google-genai"
            ) from e

        self.model_id = model_id
        self.api_key = api_key
        self.project = project
        self.location = location
        self.temperature = temperature

        # Store any additional kwargs for potential use
        self._extra_kwargs = kwargs

        # Support both Google AI API (api_key) and Vertex AI via ADC (no api_key)
        if self.api_key:
            self._client = genai.Client(api_key=self.api_key)
        else:
            project_id = (
                self.project
                or os.getenv("GOOGLE_CLOUD_PROJECT")
                or os.getenv("GCLOUD_PROJECT")
            )
            resolved_location = (
                self.location
                or os.getenv("GOOGLE_CLOUD_LOCATION")
                or os.getenv("VERTEX_AI_LOCATION")
                or "us-central1"
            )

            if not project_id:
                try:
                    import google.auth  # pylint: disable=import-outside-toplevel

                    _, detected_project = google.auth.default()
                    project_id = detected_project
                except Exception:
                    project_id = None

            if not project_id:
                raise lx.exceptions.InferenceConfigError(
                    "No API key provided and no Google Cloud project detected. "
                    'To use Vertex AI, run "gcloud auth application-default login" and '
                    "set GOOGLE_CLOUD_PROJECT or pass project/location to the provider."
                )

            try:
                self._client = genai.Client(
                    vertexai=True, project=project_id, location=resolved_location
                )
            except Exception as e:
                raise lx.exceptions.InferenceConfigError(
                    f"Failed to initialize Vertex AI client (project={project_id}, location={resolved_location})."
                ) from e

        super().__init__()

    def infer(
        self, batch_prompts: Sequence[str], **kwargs: Any
    ) -> Iterator[Sequence[lx.inference.ScoredOutput]]:
        """Run inference on a batch of prompts.

        Args:
          batch_prompts: Input prompts to process.
          **kwargs: Additional generation parameters.

        Yields:
          Lists of ScoredOutputs, one per prompt.
        """
        config = {
            "temperature": kwargs.get("temperature", self.temperature),
        }

        # Add other parameters if provided
        for key in ["max_output_tokens", "top_p", "top_k"]:
            if key in kwargs:
                config[key] = kwargs[key]

        for prompt in batch_prompts:
            try:
                # TODO: Replace this with your own API/model calls
                response = self._client.models.generate_content(
                    model=self.model_id, contents=prompt, config=config
                )
                output = response.text.strip()
                yield [lx.inference.ScoredOutput(score=1.0, output=output)]

            except Exception as e:
                raise lx.exceptions.InferenceRuntimeError(
                    f"API error: {str(e)}", original=e
                ) from e
