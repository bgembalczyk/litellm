"""
Enforce TPM/RPM or separate ITPM/OTPM rate limits set on model deployments.

When enabled via router_settings.optional_pre_call_checks: ["enforce_model_rate_limits"]

- litellm_params.rate_limits: rps, rpm, rph, rpd, rpm_month, tpm, tph, tpd, tpm_month, plus limits shared by a group
- tpm/rpm/tpd/rpd as top-level litellm_params, model_info or deployment keys: the same limits in the legacy spelling
- itpm/otpm: separate input/output tokens per minute

When a deployment sets both itpm/otpm and tpm/rpm, both are enforced. A warning
is logged the first time such a deployment is seen.
"""

import contextlib
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import (
    async_io_token_pre_call_check,
    async_io_token_reconcile_success,
    async_io_token_refund_failure,
    deployment_has_io_token_limits,
    get_io_token_rate_limit_request_kwargs,
    io_token_pre_call_check,
    io_token_reconcile_success,
    io_token_refund_failure,
)
from litellm.router_utils.pre_call_checks.window_rate_limits import (
    WindowDeployment,
    WindowRateLimits,
    window_deployment_from_usage,
)
from litellm.router_utils.routing_read_batch import ROUTER_USAGE_TARGET
from litellm.types.utils import StandardLoggingPayload
from litellm.utils import get_utc_datetime

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


def _litellm_params_of(kwargs: Mapping[str, object]) -> object:
    return kwargs.get("litellm_params")


class ModelRateLimitingCheck(CustomLogger):
    """
    Pre-call check that enforces TPM/RPM or ITPM/OTPM limits on model deployments.

    This check runs before each request and raises a RateLimitError
    if the deployment has exceeded its configured TPM or RPM limits.

    Unlike the usage-based-routing strategy which uses limits for routing decisions,
    this check actively enforces those limits across ALL routing strategies.
    """

    def __init__(self, dual_cache: DualCache, clock: Callable[[], datetime] = get_utc_datetime):
        self.dual_cache = dual_cache
        self.window_limits = WindowRateLimits(dual_cache)
        self.clock = clock
        # model_ids already warned about itpm/otpm + tpm/rpm on the same deployment,
        # so the warning is logged once per deployment rather than per request.
        self._io_token_conflict_warned_ids: set[str] = set()

    def _warn_io_token_and_tpm_rpm_coexist_once(self, deployment: dict) -> None:
        limits: Final = WindowDeployment.model_validate(deployment).limits()
        if "tpm" not in limits and "rpm" not in limits:
            return
        model_id: Final = deployment.get("model_info", {}).get("id")
        # Dedup per deployment id; if there is no id (degenerate config) don't
        # collapse every such deployment onto one key - warn each time instead.
        if model_id is not None:
            if model_id in self._io_token_conflict_warned_ids:
                return
            self._io_token_conflict_warned_ids.add(str(model_id))
        verbose_router_logger.warning(
            "Deployment '%s' configures itpm/otpm alongside tpm/rpm; both limit types are enforced on this deployment",
            model_id,
        )

    def _refund_io_token_reservation_if_any(self) -> None:
        request_kwargs: Final = get_io_token_rate_limit_request_kwargs()
        if request_kwargs is not None:
            io_token_refund_failure(self.dual_cache, request_kwargs)

    async def _async_refund_io_token_reservation_if_any(
        self,
        parent_otel_span: Span | None = None,
    ) -> None:
        request_kwargs: Final = get_io_token_rate_limit_request_kwargs()
        if request_kwargs is not None:
            await async_io_token_refund_failure(
                self.dual_cache,
                request_kwargs,
                parent_otel_span=parent_otel_span,
            )

    @with_service_target(ROUTER_USAGE_TARGET)
    def pre_call_check(self, deployment: dict) -> dict | None:
        """
        Synchronous pre-call check for model rate limits.

        Raises RateLimitError if deployment exceeds its rate limits.
        """
        io_reservation_made = False
        try:
            if deployment_has_io_token_limits(deployment):
                self._warn_io_token_and_tpm_rpm_coexist_once(deployment)
                io_token_pre_call_check(
                    self.dual_cache,
                    deployment,
                )
                io_reservation_made = True

            self.window_limits.charge(WindowDeployment.model_validate(deployment), self.clock())
            return deployment

        except litellm.RateLimitError:
            if io_reservation_made:
                self._refund_io_token_reservation_if_any()
            raise
        except Exception as e:
            verbose_router_logger.debug("Error in ModelRateLimitingCheck.pre_call_check: %s", e)
            # Don't fail the request if rate limit check fails
            return deployment

    @with_service_target(ROUTER_USAGE_TARGET)
    async def async_pre_call_check(self, deployment: dict, parent_otel_span: Span | None = None) -> dict | None:
        """
        Async pre-call check for model rate limits.

        Raises RateLimitError if deployment exceeds its rate limits or its ITPM/OTPM limits.
        """
        io_reservation_made = False
        try:
            if deployment_has_io_token_limits(deployment):
                self._warn_io_token_and_tpm_rpm_coexist_once(deployment)
                await async_io_token_pre_call_check(
                    self.dual_cache,
                    deployment,
                    parent_otel_span=parent_otel_span,
                )
                io_reservation_made = True

            await self.window_limits.async_charge(
                WindowDeployment.model_validate(deployment), self.clock(), parent_otel_span
            )
            return deployment

        except litellm.RateLimitError:
            if io_reservation_made:
                await self._async_refund_io_token_reservation_if_any(parent_otel_span=parent_otel_span)
            raise
        except Exception as e:
            verbose_router_logger.debug("Error in ModelRateLimitingCheck.async_pre_call_check: %s", e)
            # Don't fail the request if rate limit check fails
            return deployment

    @with_service_target(ROUTER_USAGE_TARGET)
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.litellm_core_utils.core_helpers import (
            get_parent_otel_span_from_kwargs,
        )

        try:
            standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object")
            await async_io_token_reconcile_success(
                self.dual_cache,
                kwargs,
                response_obj,
                parent_otel_span=get_parent_otel_span_from_kwargs(kwargs),
            )

            if standard_logging_object is None:
                return

            model_id: Final = standard_logging_object.get("model_id")
            if model_id is None:
                return

            total_tokens: Final = standard_logging_object.get("total_tokens", 0)
            model: Final = standard_logging_object.get("hidden_params", {}).get("litellm_model_name")

            verbose_router_logger.debug(
                "[TPM TRACKING] model_id=%s, total_tokens=%s, model=%s", model_id, total_tokens, model
            )

            if not model or not total_tokens:
                return

            await self.window_limits.async_record_tokens(
                window_deployment_from_usage(
                    f"{model_id}",
                    f"{model}",
                    _litellm_params_of(kwargs),  # pyright: ignore[reportUnknownArgumentType]  # CustomLogger passes an untyped kwargs dict
                ),
                int(total_tokens),
                self.clock(),
            )

        except Exception as e:
            verbose_router_logger.debug("Error in ModelRateLimitingCheck.async_log_success_event: %s", e)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.litellm_core_utils.core_helpers import (
            get_parent_otel_span_from_kwargs,
        )

        # Never fail the primary logging pipeline over an io-token refund error.
        with contextlib.suppress(Exception):
            await async_io_token_refund_failure(
                self.dual_cache,
                kwargs,
                parent_otel_span=get_parent_otel_span_from_kwargs(kwargs),
            )

    @with_service_target(ROUTER_USAGE_TARGET)
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Sync version of tracking TPM usage after successful request.
        Always tracks tokens - the pre-call check handles enforcement.
        """
        try:
            standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object")
            io_token_reconcile_success(self.dual_cache, kwargs, response_obj)

            if standard_logging_object is None:
                return

            model_id: Final = standard_logging_object.get("model_id")
            if model_id is None:
                return

            total_tokens: Final = standard_logging_object.get("total_tokens", 0)
            model: Final = standard_logging_object.get("hidden_params", {}).get("litellm_model_name")

            if not model or not total_tokens:
                return

            self.window_limits.record_tokens(
                window_deployment_from_usage(
                    f"{model_id}",
                    f"{model}",
                    _litellm_params_of(kwargs),  # pyright: ignore[reportUnknownArgumentType]  # CustomLogger passes an untyped kwargs dict
                ),
                int(total_tokens),
                self.clock(),
            )

        except Exception as e:
            verbose_router_logger.debug("Error in ModelRateLimitingCheck.log_success_event: %s", e)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        with contextlib.suppress(Exception):
            io_token_refund_failure(
                self.dual_cache,
                kwargs,
            )
