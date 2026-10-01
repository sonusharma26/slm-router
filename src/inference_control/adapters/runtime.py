"""Explicit bridge from provider messages to bounded execution results."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from inference_control.execution.runtime import RuntimeResult
from inference_control.util import digest


class ProviderAccountingError(RuntimeError):
    pass


@dataclass
class RuntimeProviderAdapter:
    endpoints: dict
    providers: dict
    messages: list | None = None
    request: object = None
    response_config: dict | None = None

    def for_request(self, request, messages, response_config=None):
        if messages is None:
            raise ValueError("execution requires messages; raw prompts are not retained in the ledger")
        if not isinstance(messages,list) or not messages:
            raise ValueError("non-empty message list required")
        response_config = response_config or {}
        if set(response_config) - {"response_format", "tools", "tool_choice", "parallel_tool_calls"}:
            raise ValueError("unapproved per-request generation override")
        for key, expected, capability in (
            ("tools", request.tool_schema_hash, "tools"),
            ("response_format", request.structured_output_schema_hash, "structured_output"),
        ):
            value = response_config.get(key)
            actual = digest(value) if value is not None else None
            if actual != expected:
                raise ValueError(f"{key} differs from the request used for routing")
            if value is not None and capability not in request.required_capabilities:
                raise ValueError(f"{capability} capability was not requested before routing")
        if (response_config.get("tool_choice") is not None or response_config.get("parallel_tool_calls") is not None) and not response_config.get("tools"):
            raise ValueError("tool controls require declared tools")
        response_format=response_config.get("response_format")
        if response_format and response_format.get("type")=="json_schema":
            from jsonschema import Draft202012Validator
            schema=response_format.get("json_schema",{}).get("schema")
            if not isinstance(schema,dict):raise ValueError("missing JSON schema")
            if len(json.dumps(schema).encode())>65536:raise ValueError("JSON schema exceeds 64 KiB admission limit")
            pending=[schema]
            while pending:
                item=pending.pop()
                if isinstance(item,dict):
                    for key,value in item.items():
                        if key in {"$ref","$dynamicRef"} and (not isinstance(value,str) or not value.startswith("#")):
                            raise ValueError("external JSON schema references are forbidden")
                        pending.append(value)
                elif isinstance(item,(tuple,list)):pending.extend(item)
            Draft202012Validator.check_schema(schema)
        # Common conservative admission bound, not a substitute for an operator-verified tokenizer.
        if len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > request.input_tokens:
            raise ValueError("messages exceed the declared byte-based input admission bound")
        return RuntimeProviderAdapter(self.endpoints, self.providers, messages, request, response_config)

    def call(self, endpoint_id: str, generation_config: dict) -> RuntimeResult:
        ep = self.endpoints[endpoint_id]
        config = dict(generation_config)
        allowed_response = {"response_format", "tools", "tool_choice", "parallel_tool_calls"}
        if set(self.response_config or {}) - allowed_response:
            raise ValueError("unapproved per-request generation override")
        config.update(self.response_config or {})
        result = self.providers[endpoint_id].call(ep.upstream_model, self.messages, config)
        if not result.usage_known:
            raise ProviderAccountingError("provider returned no token usage; spend is unknown, reservation remains charged")
        if result.input_tokens < 0 or result.output_tokens < 0:
            raise ProviderAccountingError("invalid negative usage")
        snapshot_cost = (result.input_tokens*ep.input_price_per_million + result.output_tokens*ep.output_price_per_million)/1e6
        spend = result.spend if result.spend is not None else snapshot_cost
        if not math.isfinite(spend) or spend < 0:
            raise ProviderAccountingError("invalid provider cost")
        violations = []
        if result.input_tokens > self.request.input_tokens + ep.input_token_overhead:
            violations.append("INPUT_TOKEN_BOUND_EXCEEDED")
        if result.output_tokens > self.request.max_output_tokens:
            violations.append("OUTPUT_TOKEN_BOUND_EXCEEDED")
        if result.observed_model is not None and result.observed_model != ep.upstream_model:
            violations.append("ENDPOINT_MODEL_MISMATCH")
        if result.observed_revision is not None and result.observed_revision != ep.revision:
            violations.append("ENDPOINT_REVISION_MISMATCH")
        message = {"role":"assistant", "content":result.text}
        try:
            raw_message = result.raw.get("choices", [{}])[0].get("message", {})
            if isinstance(raw_message, dict):
                message.update({k:v for k,v in raw_message.items() if k in {"role","content","tool_calls","refusal"}})
        except (AttributeError, IndexError, TypeError):
            pass
        response_format = (self.response_config or {}).get("response_format")
        if response_format and response_format.get("type") in {"json_object", "json_schema"}:
            try:
                parsed = json.loads(message.get("content") or "")
                if response_format.get("type") == "json_schema":
                    from jsonschema import Draft202012Validator
                    schema = response_format.get("json_schema", {}).get("schema")
                    if not isinstance(schema, dict): raise ValueError("missing JSON schema")
                    Draft202012Validator.check_schema(schema)
                    Draft202012Validator(schema).validate(parsed)
            except Exception:
                # Preserve cost/usage even for invalid outputs; the runtime rejects the result.
                violations.append("MALFORMED_STRUCTURED_OUTPUT")
        tool_calls = message.get("tool_calls")
        if tool_calls:
            allowed = {t.get("function", {}).get("name") for t in (self.response_config or {}).get("tools", [])}
            try:
                for call in tool_calls:
                    function = call["function"]
                    if function["name"] not in allowed: raise ValueError("undeclared tool")
                    if not isinstance(json.loads(function["arguments"]), dict): raise ValueError("tool arguments must be a JSON object")
            except (ValueError, KeyError, TypeError):
                violations.append("MALFORMED_TOOL_CALL")
        return RuntimeResult("sha256:"+digest(message), result.provider_request_id,
            result.observed_revision or ep.revision, result.input_tokens, result.output_tokens,
            spend, result.time_to_first_token_ms, endpoint_id=endpoint_id,
            message=message, provider_latency_ms=result.latency_ms,
            constraint_violations=tuple(violations), revision_observed=result.observed_revision is not None)
