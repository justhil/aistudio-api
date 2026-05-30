"""Domain errors for AI Studio interactions."""

from enum import IntEnum


class ErrorCode(IntEnum):
    USAGE_LIMIT_EXCEEDED = 429
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    RATE_LIMITED = 429
    INTERNAL_ERROR = 500
    BAD_REQUEST = 400


class AistudioError(Exception):
    pass


class AuthError(AistudioError):
    pass


class AccountAuthExpired(AuthError):
    """账号 Cookie 失效，浏览器被重定向到 Google 登录页。

    与一般 AuthError 区分：这类错误重试同一账号无意义，应直接轮换账号。
    """
    pass


class UsageLimitExceeded(AistudioError):
    pass


class SnapshotExpired(AistudioError):
    pass


class ModelNotFoundError(AistudioError):
    pass


class RequestError(AistudioError):
    def __init__(self, status: int, message: str = ""):
        self.status = status
        super().__init__(f"HTTP {status}: {message}")


def classify_error(status: int, body: str) -> AistudioError:
    if status == 429:
        return UsageLimitExceeded(f"配额用完: {body[:200]}")
    if status == 401:
        return AuthError(f"认证失败: {body[:200]}")
    if status == 403:
        return AuthError(f"禁止访问: {body[:200]}")
    return RequestError(status, body[:200])

