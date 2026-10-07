import unittest

from pydantic import ValidationError

from inference_control.configuration import load_endpoints
from inference_control.contracts import EndpointSnapshot


class ReasoningConfigTests(unittest.TestCase):
    def test_thinking_setting_is_frozen_and_changes_evidence_identity(self):
        original = load_endpoints("examples/nvidia-live/endpoints.json")[0]
        data = original.model_dump()
        data["inference_config"] = {"chat_template_kwargs": {"enable_thinking": False}}
        endpoint = EndpointSnapshot.model_validate(data)
        self.assertEqual(endpoint.inference_config["chat_template_kwargs"]["enable_thinking"], False)
        self.assertNotEqual(original.capability_revision, endpoint.capability_revision)
        with self.assertRaises(TypeError):
            endpoint.inference_config["chat_template_kwargs"]["enable_thinking"] = True

    def test_template_settings_cannot_smuggle_unapproved_parameters(self):
        original = load_endpoints("examples/nvidia-live/endpoints.json")[0].model_dump()
        for template in (None, {"enable_thinking": "false"}, {"enable_thinking": False, "api_key": "dummy"}, {}):
            with self.subTest(template=template), self.assertRaises(ValidationError):
                EndpointSnapshot.model_validate({**original, "inference_config": {"chat_template_kwargs": template}})
