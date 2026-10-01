from inference_control.contracts import EndpointSnapshot, PolicySpec, RequestContext


def validate_policy(policy: PolicySpec) -> None:
    # Construction performs structural validation; this explicit seam supports future DSL parsing.
    PolicySpec.model_validate(policy.model_dump())


def eligibility_reasons(
    endpoint: EndpointSnapshot, request: RequestContext, policy: PolicySpec
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not endpoint.available:
        reasons.append("ENDPOINT_UNAVAILABLE")
    if not endpoint.healthy:
        reasons.append("ENDPOINT_UNHEALTHY")
    if policy.allowed_providers and endpoint.provider not in policy.allowed_providers:
        reasons.append("PROVIDER_NOT_ALLOWED")
    if policy.allowed_endpoints and endpoint.endpoint_id not in policy.allowed_endpoints:
        reasons.append("ENDPOINT_NOT_ALLOWED")
    if request.modality not in endpoint.modalities:
        reasons.append("MODALITY_UNSUPPORTED")
    if not (policy.required_capabilities | request.required_capabilities) <= endpoint.capabilities:
        reasons.append("CAPABILITY_MISSING")
    if policy.data_boundary != "any" and policy.data_boundary not in endpoint.governance:
        reasons.append("DATA_BOUNDARY_VIOLATION")
    if request.privacy_classification != "public" and request.privacy_classification not in endpoint.governance:
        reasons.append("REQUEST_PRIVACY_VIOLATION")
    if policy.required_traffic_slices and not request.traffic_slices <= policy.required_traffic_slices:
        reasons.append("TRAFFIC_SLICE_NOT_ALLOWED")
    if request.input_tokens + endpoint.input_token_overhead + request.max_output_tokens > endpoint.context_window:
        reasons.append("CONTEXT_LIMIT")
    return tuple(reasons)
