"""本地回环 Gateway 与 Telegram 渠道的最小适配层。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from dataclasses import dataclass
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request

from dotenv import load_dotenv

from hermes_lite.chat_runtime import ChatRuntime, ChatRuntimeError
from hermes_lite.config import DOTENV_PATH
from hermes_lite.domain import TaskStatus


class GatewayError(ValueError):
    """Gateway 或 Telegram 配置、请求或投递失败。"""


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    """只允许本机回环监听的 Gateway 配置。"""

    token: str
    host: str = "127.0.0.1"
    port: int = 18791


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    """Telegram 私聊白名单与长轮询配置。"""

    bot_token: str
    allowed_user_id: int
    poll_timeout_seconds: int = 20


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value.strip():
        raise GatewayError(f"缺少必备配置：{name}")
    return value.strip()


def load_gateway_config(environment: Mapping[str, str] | None = None) -> GatewayConfig:
    if environment is None:
        load_dotenv(DOTENV_PATH, override=False)
        environment = os.environ
    token = _required(environment, "HERMES_GATEWAY_TOKEN")
    if len(token) < 16:
        raise GatewayError("HERMES_GATEWAY_TOKEN 至少需要 16 个字符")
    host = environment.get("HERMES_GATEWAY_HOST", "127.0.0.1")
    if host != "127.0.0.1":
        raise GatewayError("HERMES_GATEWAY_HOST 只能是 127.0.0.1")
    try:
        port = int(environment.get("HERMES_GATEWAY_PORT", "18791"))
    except ValueError as error:
        raise GatewayError("HERMES_GATEWAY_PORT 必须是端口号") from error
    if not 1024 <= port <= 65535:
        raise GatewayError("HERMES_GATEWAY_PORT 必须在 1024 到 65535 之间")
    return GatewayConfig(token=token, host=host, port=port)


def load_telegram_config(environment: Mapping[str, str] | None = None) -> TelegramConfig:
    if environment is None:
        load_dotenv(DOTENV_PATH, override=False)
        environment = os.environ
    try:
        user_id = int(_required(environment, "TELEGRAM_ALLOWED_USER_ID"))
    except ValueError as error:
        raise GatewayError("TELEGRAM_ALLOWED_USER_ID 必须是正整数") from error
    if user_id <= 0:
        raise GatewayError("TELEGRAM_ALLOWED_USER_ID 必须是正整数")
    raw_timeout = environment.get("TELEGRAM_POLL_TIMEOUT_SECONDS", "20")
    try:
        timeout = int(raw_timeout)
    except ValueError as error:
        raise GatewayError("TELEGRAM_POLL_TIMEOUT_SECONDS 必须是整数") from error
    if not 1 <= timeout <= 50:
        raise GatewayError("TELEGRAM_POLL_TIMEOUT_SECONDS 必须在 1 到 50 之间")
    return TelegramConfig(_required(environment, "TELEGRAM_BOT_TOKEN"), user_id, timeout)


class MessageRuntime(Protocol):
    def run_turn(self, session_id: str, user_request: str, skill_name: object | None = None): ...


class GatewayService:
    """将所有渠道消息收敛为已有 ChatRuntime 的单次调用。"""

    def __init__(self, runtime: MessageRuntime) -> None:
        self._runtime = runtime

    def handle_message(self, session_id: object, text: object) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise GatewayError("session_id 必须是非空文本")
        if not isinstance(text, str) or not text.strip():
            raise GatewayError("text 必须是非空文本")
        if text.strip().startswith("/confirm"):
            return "外部渠道不接受确认令牌；请在本地交互聊天中重新发起并确认操作。"
        try:
            result = self._runtime.run_turn(session_id.strip(), text.strip())
        except ChatRuntimeError as error:
            raise GatewayError("Agent 状态保存失败") from error
        if result.turn.task.status is TaskStatus.BLOCKED:
            return "该操作需要本地交互确认，未在外部渠道执行。"
        if result.turn.task.status is TaskStatus.FAILED:
            return "Agent 当前无法完成该请求。"
        assert result.turn.answer is not None
        return result.turn.answer


class TelegramApi(Protocol):
    def get_updates(self, offset: int | None, timeout: int) -> list[dict[str, object]]: ...
    def send_message(self, chat_id: int, text: str) -> None: ...


class TelegramHttpApi:
    """仅封装 Telegram Bot API 所需的两个 JSON 请求。"""

    def __init__(self, config: TelegramConfig) -> None:
        self._base_url = f"https://api.telegram.org/bot{config.bot_token}"

    def _post(self, method: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(f"{self._base_url}/{method}", data=body, method="POST", headers={"Content-Type": "application/json"})
        try:
            with request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (error.URLError, json.JSONDecodeError) as exc:
            raise GatewayError("Telegram 服务请求失败") from exc
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise GatewayError("Telegram 服务返回失败")
        return result

    def get_updates(self, offset: int | None, timeout: int) -> list[dict[str, object]]:
        payload: dict[str, object] = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        result = self._post("getUpdates", payload).get("result")
        return result if isinstance(result, list) else []

    def send_message(self, chat_id: int, text: str) -> None:
        self._post("sendMessage", {"chat_id": chat_id, "text": text[:4000]})


class TelegramChannel:
    """仅接受指定用户的私聊文本，并映射到稳定的渠道会话。"""

    def __init__(self, config: TelegramConfig, api: TelegramApi, service: GatewayService) -> None:
        self._config, self._api, self._service, self._offset = config, api, service, None

    def poll_once(self) -> int:
        handled = 0
        for update in self._api.get_updates(self._offset, self._config.poll_timeout_seconds):
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self._offset = update_id + 1
            message = update.get("message")
            if not isinstance(message, dict):
                continue
            sender = message.get("from")
            chat = message.get("chat")
            text = message.get("text")
            if not isinstance(sender, dict) or not isinstance(chat, dict) or not isinstance(text, str):
                continue
            user_id, chat_id = sender.get("id"), chat.get("id")
            if user_id != self._config.allowed_user_id or chat.get("type") != "private" or not isinstance(chat_id, int):
                continue
            reply = self._service.handle_message(f"telegram:{user_id}", text)
            self._api.send_message(chat_id, reply)
            handled += 1
        return handled


def run_gateway_http_server(config: GatewayConfig, service: GatewayService) -> None:
    """在本机回环地址提供最小健康检查和消息转发 HTTP 接口。"""
    if not isinstance(config, GatewayConfig) or not isinstance(service, GatewayService):
        raise GatewayError("Gateway 服务配置无效")

    class Handler(BaseHTTPRequestHandler):
        server_version = "HermesLiteGateway"

        def _send_json(self, status: int, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _authorized(self) -> bool:
            return self.headers.get("Authorization") == f"Bearer {config.token}"

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send_json(404, {"error": "not_found"})
                return
            if not self._authorized():
                self._send_json(401, {"error": "unauthorized"})
                return
            self._send_json(200, {"status": "ok"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/messages":
                self._send_json(404, {"error": "not_found"})
                return
            if not self._authorized():
                self._send_json(401, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16_000:
                    raise ValueError
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError
                answer = service.handle_message(payload.get("session_id"), payload.get("text"))
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError, GatewayError):
                self._send_json(400, {"error": "invalid_request"})
                return
            self._send_json(200, {"answer": answer})

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    try:
        with ThreadingHTTPServer((config.host, config.port), Handler) as server:
            server.serve_forever()
    except OSError as error:
        raise GatewayError("无法启动本机 Gateway") from error
