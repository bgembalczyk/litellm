from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias, assert_never

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.router import DeploymentRateLimits, RouterErrors, parse_daily_reset_time

if TYPE_CHECKING:
    from opentelemetry.trace import Span

Window: TypeAlias = Literal["second", "minute", "hour", "day", "month"]
Metric: TypeAlias = Literal["requests", "tokens"]

DAY_FORMAT: Final = "%Y-%m-%d"


@dataclass(frozen=True, slots=True)
class LimitKind:
    metric: Metric
    window: Window


_COUNT: Final = TypeAdapter[float | None](float | None)
_PARAMS: Final = TypeAdapter[Mapping[str, object] | None](Mapping[str, object] | None)


class RedisReader(Protocol):
    def get_cache(self, key: str) -> object: ...

    async def async_get_cache(self, key: str, parent_otel_span: "Span | None" = None) -> object: ...


class CounterStore(Protocol):
    @property
    def redis_cache(self) -> RedisReader | None: ...

    def get_cache(self, key: str, *, local_only: bool = False) -> object: ...

    async def async_get_cache(self, key: str, *, local_only: bool = False) -> object: ...

    def increment_cache(self, key: str, value: int, *, ttl: int) -> int: ...

    async def async_increment_cache_pipeline(
        self,
        increment_list: list[RedisPipelineIncrementOperation],  # mutable-ok: DualCache takes a list
        *,
        parent_otel_span: "Span | None" = None,
    ) -> Sequence[float] | None: ...


LIMIT_KINDS: Final = MappingProxyType(
    {
        "rps": LimitKind("requests", "second"),
        "rpm": LimitKind("requests", "minute"),
        "rph": LimitKind("requests", "hour"),
        "rpd": LimitKind("requests", "day"),
        "rpm_month": LimitKind("requests", "month"),
        "tpm": LimitKind("tokens", "minute"),
        "tph": LimitKind("tokens", "hour"),
        "tpd": LimitKind("tokens", "day"),
        "tpm_month": LimitKind("tokens", "month"),
    }
)


def daily_window_day(now: datetime, reset_offset: timedelta) -> str:
    return (now - reset_offset).strftime(DAY_FORMAT)


def seconds_until_daily_reset(now: datetime, reset_offset: timedelta = timedelta(0)) -> int:
    window_start: Final = (now - reset_offset).replace(hour=0, minute=0, second=0, microsecond=0)
    next_reset: Final = window_start + timedelta(days=1) + reset_offset
    return max(1, int((next_reset - now).total_seconds()))


def _seconds_until_next_month(now: datetime) -> int:
    next_month: Final = (now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) + timedelta(days=32)).replace(
        day=1
    )
    return max(1, int((next_month - now).total_seconds()))


def seconds_until_window_end(window: Window, now: datetime, reset_offset: timedelta) -> int:
    match window:
        case "second":
            return 1
        case "minute":
            return 60
        case "hour":
            return 3600 - now.minute * 60 - now.second
        case "day":
            return seconds_until_daily_reset(now, reset_offset)
        case "month":
            return _seconds_until_next_month(now)
        case _:
            assert_never(window)


def window_ttl(window: Window, now: datetime, reset_offset: timedelta) -> int:
    match window:
        case "minute":
            return 60
        case "day":
            return 86400
        case "second" | "hour" | "month":
            return seconds_until_window_end(window, now, reset_offset) + 1
        case _:
            assert_never(window)


def window_bucket(window: Window, now: datetime, reset_offset: timedelta) -> str:
    match window:
        case "second":
            return now.strftime("%Y-%m-%dT%H:%M:%S")
        case "minute":
            return now.strftime("%H-%M")
        case "hour":
            return now.strftime("%Y-%m-%dT%H")
        case "day":
            return daily_window_day(now, reset_offset)
        case "month":
            return now.strftime("%Y-%m")
        case _:
            assert_never(window)


@dataclass(frozen=True, slots=True)
class WindowCounter:
    key: str
    kind: str
    limit: int
    metric: Metric
    window: Window
    ttl: int
    retry_after: int
    shared_group: str | None = None

    def increment(self, value: int) -> RedisPipelineIncrementOperation:
        return RedisPipelineIncrementOperation(key=self.key, increment_value=value, ttl=self.ttl)


class _WindowParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)
    model: str | None = None
    daily_reset_time: str | None = None
    rate_limits: DeploymentRateLimits | None = None
    rpm: int | None = None
    tpm: int | None = None
    rpd: int | None = None
    tpd: int | None = None


class _WindowModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)
    id: str | None = None
    rpm: int | None = None
    tpm: int | None = None


def _first(*limits: int | None) -> int | None:
    return next((limit for limit in limits if limit is not None), None)


class WindowDeployment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)
    model_name: str = ""
    rpm: int | None = None
    tpm: int | None = None
    litellm_params: _WindowParams = _WindowParams()
    model_info: _WindowModelInfo = _WindowModelInfo()

    def limits(self) -> Mapping[str, int]:
        params: Final = self.litellm_params
        legacy: Final = {
            "rpm": _first(self.rpm, params.rpm, self.model_info.rpm),
            "tpm": _first(self.tpm, params.tpm, self.model_info.tpm),
            "rpd": params.rpd,
            "tpd": params.tpd,
        }
        configured: Final = {} if params.rate_limits is None else params.rate_limits.windows()
        merged: Final = {**{kind: limit for kind, limit in legacy.items() if limit is not None}, **configured}
        return {kind: merged[kind] for kind in LIMIT_KINDS if kind in merged}

    def counters(self, now: datetime) -> tuple[WindowCounter, ...]:
        return self._counters(now, self.limits())

    def recorded_counters(self, now: datetime) -> tuple[WindowCounter, ...]:
        return _tokens(self._counters(now, {"tpm": 0, "tpd": 0, **self.limits()}))

    def _counters(self, now: datetime, limits: Mapping[str, int]) -> tuple[WindowCounter, ...]:
        reset_offset: Final = parse_daily_reset_time(self.litellm_params.daily_reset_time)
        shared: Final = None if self.litellm_params.rate_limits is None else self.litellm_params.rate_limits.shared
        return _counters(f"{self.model_info.id}:{self.litellm_params.model}", limits, now, reset_offset) + (
            ()
            if shared is None
            else _counters(f"shared:{shared.group}", shared.windows(), now, reset_offset, shared_group=shared.group)
        )

    def limit_error(self, counter: WindowCounter, usage: int, fail_fast: bool = False) -> litellm.RateLimitError:
        scope: Final = "" if counter.shared_group is None else f" shared_group={counter.shared_group}"
        return litellm.RateLimitError(
            message=(
                f"Model rate limit exceeded. {counter.kind.upper()} limit={counter.limit}, current usage={usage}{scope}"
            ),
            llm_provider="",
            model=self.litellm_params.model or "",
            response=httpx.Response(
                status_code=429,
                content=(
                    f"{RouterErrors.user_defined_ratelimit_error.value} {counter.kind} limit={counter.limit}. "
                    f"current usage={usage}.{scope} id={self.model_info.id}, model_group={self.model_name}"
                ),
                headers={"retry-after": str(counter.retry_after)},
                request=httpx.Request(method="model_rate_limit_check", url="https://github.com/BerriAI/litellm"),
            ),
            num_retries=0 if fail_fast else None,
        )


def rpm_tpm_limits(deployment: object) -> tuple[int | None, int | None]:
    try:
        limits: Final = WindowDeployment.model_validate(deployment).limits()
    except ValidationError:
        return None, None
    return limits.get("rpm"), limits.get("tpm")


def _counters(
    prefix: str,
    limits: Mapping[str, int],
    now: datetime,
    reset_offset: timedelta,
    shared_group: str | None = None,
) -> tuple[WindowCounter, ...]:
    return tuple(
        WindowCounter(
            key=f"{prefix}:{kind}:{window_bucket(LIMIT_KINDS[kind].window, now, reset_offset)}",
            kind=kind,
            limit=limit,
            metric=LIMIT_KINDS[kind].metric,
            window=LIMIT_KINDS[kind].window,
            ttl=window_ttl(LIMIT_KINDS[kind].window, now, reset_offset),
            retry_after=seconds_until_window_end(LIMIT_KINDS[kind].window, now, reset_offset),
            shared_group=shared_group,
        )
        for kind, limit in limits.items()
    )


def window_deployment_from_usage(model_id: str, model: str, litellm_params: object) -> WindowDeployment:
    return WindowDeployment.model_validate(
        {
            "litellm_params": {**(_PARAMS.validate_python(litellm_params) or {}), "model": model},
            "model_info": {"id": model_id},
        }
    )


def _fails_fast(counter: WindowCounter) -> bool:
    return counter.kind in ("rpm", "tpm")


def _requests(counters: tuple[WindowCounter, ...]) -> tuple[WindowCounter, ...]:
    return tuple(counter for counter in counters if counter.metric == "requests")


def _tokens(counters: tuple[WindowCounter, ...]) -> tuple[WindowCounter, ...]:
    return tuple(counter for counter in counters if counter.metric == "tokens")


def _first_exceeded(counters: tuple[WindowCounter, ...], counts: Sequence[float]) -> tuple[WindowCounter, int] | None:
    return next(
        ((counter, int(count)) for counter, count in zip(counters, counts) if count > counter.limit),
        None,
    )


class WindowRateLimits:
    def __init__(self, dual_cache: CounterStore) -> None:
        self.dual_cache = dual_cache

    def _usage(self, counter: WindowCounter) -> int:
        local: Final = _COUNT.validate_python(self.dual_cache.get_cache(key=counter.key, local_only=True))
        redis_cache: Final = self.dual_cache.redis_cache
        if redis_cache is None or (local is not None and local >= counter.limit):
            return int(local or 0)
        try:
            return int(_COUNT.validate_python(redis_cache.get_cache(key=counter.key)) or 0)
        except RedisCircuitBreakerOpenError:
            return int(local or 0)

    async def _async_usage(self, counter: WindowCounter, parent_otel_span: "Span | None") -> int:
        local: Final = _COUNT.validate_python(await self.dual_cache.async_get_cache(key=counter.key, local_only=True))
        redis_cache: Final = self.dual_cache.redis_cache
        if redis_cache is None or (local is not None and local >= counter.limit):
            return int(local or 0)
        try:
            fetched: Final = await redis_cache.async_get_cache(key=counter.key, parent_otel_span=parent_otel_span)
            return int(_COUNT.validate_python(fetched) or 0)
        except RedisCircuitBreakerOpenError:
            return int(local or 0)

    def charge(self, deployment: WindowDeployment, now: datetime) -> None:
        counters: Final = deployment.counters(now)
        spent: Final = next(
            ((counter, usage) for counter in _tokens(counters) if (usage := self._usage(counter)) >= counter.limit),
            None,
        )
        if spent is not None:
            raise deployment.limit_error(*spent)
        requests: Final = _requests(counters)
        counts: Final = [
            float(self.dual_cache.increment_cache(key=counter.key, value=1, ttl=counter.ttl)) for counter in requests
        ]
        exceeded: Final = _first_exceeded(requests, counts)
        if exceeded is None:
            return
        for counter in requests:
            self.dual_cache.increment_cache(key=counter.key, value=-1, ttl=counter.ttl)
        raise deployment.limit_error(*exceeded)

    async def async_charge(
        self, deployment: WindowDeployment, now: datetime, parent_otel_span: "Span | None" = None
    ) -> None:
        counters: Final = deployment.counters(now)
        token_counters: Final = _tokens(counters)
        usages: Final = [await self._async_usage(counter, parent_otel_span) for counter in token_counters]
        spent: Final = next(
            ((counter, usage) for counter, usage in zip(token_counters, usages) if usage >= counter.limit), None
        )
        if spent is not None:
            raise deployment.limit_error(*spent, fail_fast=_fails_fast(spent[0]))
        requests: Final = _requests(counters)
        if not requests:
            return
        counts: Final = await self.dual_cache.async_increment_cache_pipeline(
            increment_list=[counter.increment(1) for counter in requests], parent_otel_span=parent_otel_span
        )
        if counts is None or len(counts) != len(requests):
            return
        exceeded: Final = _first_exceeded(requests, counts)
        if exceeded is None:
            return
        await self.dual_cache.async_increment_cache_pipeline(
            increment_list=[counter.increment(-1) for counter in requests], parent_otel_span=parent_otel_span
        )
        raise deployment.limit_error(*exceeded, fail_fast=_fails_fast(exceeded[0]))

    def record_tokens(self, deployment: WindowDeployment, tokens: int, now: datetime) -> None:
        for counter in deployment.recorded_counters(now):
            self.dual_cache.increment_cache(key=counter.key, value=tokens, ttl=counter.ttl)

    async def async_record_tokens(self, deployment: WindowDeployment, tokens: int, now: datetime) -> None:
        operations: Final = [counter.increment(tokens) for counter in deployment.recorded_counters(now)]
        if operations:
            await self.dual_cache.async_increment_cache_pipeline(increment_list=operations)
