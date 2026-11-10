"""claude_cli：让小智设备通过本机 Claude Code（claude-agent-sdk）对话和干活。

语音层（填充语、进度提示、卡住判断、打断、转后台、待汇报、接续、热加载退役……）都在
core/agent/voice_provider.py，这里只有 Claude Code 自己的部分：找 claude 命令、自检、会话配置和语音规则。
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from config.config_loader import get_project_dir
from config.logger import setup_logging
from core.agent.agent_session import AgentSession
from core.agent.claude_session import ClaudeSession, ClaudeSessionConfig
from core.agent.session_store import SessionStore
from core.agent.voice_provider import (  # noqa: F401 —— 其余名字供管理后台、grok_bot 旧代码和测试从这里导入
    BACKGROUND_NOTICE,
    BUSY_NOTICE_TEMPLATE,
    DEFAULT_PROGRESS_REPLY,
    PROGRESS_MAX_FACTOR,
    SENTENCE_ENDS,
    STOPPED_NOTICE_TEMPLATE,
    AgentVoiceProvider,
    ConnState,
    Turn,
    _num,
    elapsed_text,
    single_sentence,
)
from core.utils import textUtils  # noqa: F401 —— 测试通过 provider_mod.textUtils 替换 get_emotion
from core.utils.prompt_manager import EMOJI_List

TAG = __name__
logger = setup_logging()

VOICE_RULES = f"""
# 语音助手规则（最高优先级）
你现在通过一个语音设备和用户对话，你说的每个字都会被朗读出来。
1. 每次回复的第一个字符必须是一个表示心情的 emoji，并且只能从这个列表中选：{" ".join(EMOJI_List)}。列表以外的 emoji 一律不要用，回复中间和结尾也不要再出现 emoji。
2. 回答要简短口语化，一般不超过 3 句话。
3. 不要使用 Markdown、列表、编号、代码块、链接、表格等任何格式符号。
4. 需要调用工具时，调用前最多说一句过渡语；完成后用一两句话汇报结果，不要念出代码、命令、路径细节和文件内容。
5. 用户没问时，不要主动解释自己有哪些工具、权限或限制。
6. 不要用后台方式运行命令或任务（例如 run_in_background）：这一轮说完后你没法再主动开口，后台任务的结果用户永远听不到。需要等的命令就在前台运行，把超时设够；设备会陪着等，期间会自动提示"还在处理"。
"""

SPOKEN_NAME = "Claude"
BUSY_NOTICE = BUSY_NOTICE_TEMPLATE.format(name=SPOKEN_NAME)
STOPPED_NOTICE = STOPPED_NOTICE_TEMPLATE.format(name=SPOKEN_NAME)

COMMON_CLI_PATHS = [
    "~/.claude/local/claude",
    "~/.local/bin/claude",
    "/opt/homebrew/bin/claude",
    "/usr/local/bin/claude",
]


def find_claude_cli(configured: Optional[str] = None) -> Optional[str]:
    """优先用配置的路径，其次 PATH，再其次常见安装位置（从图形界面启动时 PATH 可能不完整）。"""
    if configured:
        p = Path(os.path.expanduser(configured))
        return str(p) if p.exists() else None
    names = ["claude.exe", "claude"] if sys.platform == "win32" else ["claude"]
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    for candidate in COMMON_CLI_PATHS:
        p = Path(os.path.expanduser(candidate))
        if p.exists():
            return str(p)
    return None


def claude_cli_version(cli_path: str, timeout: float = 15) -> str:
    """运行 `claude --version`，返回第一行非空输出；失败时抛出异常。"""
    out = subprocess.run([cli_path, "--version"], capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    lines = [l.strip() for l in (out.stdout or out.stderr or "").splitlines() if l.strip()]
    return lines[0] if lines else ""


class LLMProvider(AgentVoiceProvider):
    AGENT_LABEL = "Claude Code"
    SPOKEN_NAME = SPOKEN_NAME
    BACKGROUND_TURNS = True
    REPORT_ID = "claude_cli"
    PROVIDER_TYPE = "claude_cli"
    LOOP_THREAD_NAME = "claude-cli-loop"
    LOG_TAG = TAG

    def __init__(self, config: dict):
        self.model_name = f"claude_cli:{config.get('model') or 'default'}"
        self.cli_path = find_claude_cli(config.get("cli_path"))
        workdir = Path(os.path.expanduser(config.get("workdir") or "~/xiaozhi-workspace"))
        workdir.mkdir(parents=True, exist_ok=True)
        self.session_config = ClaudeSessionConfig(
            cli_path=self.cli_path,
            model=config.get("model") or None,
            workdir=str(workdir),
            allowed_tools=list(config.get("allowed_tools") or ["Read", "Grep", "Glob"]),
            permission_mode=config.get("permission_mode") or "default",
            # 默认 low：语音场景以快速应答为主，实测工具类问题从约 6 秒降到约 4 秒
            effort=str(config.get("effort", "low") or "").strip() or None,
        )
        self._init_common(config)
        self.store = SessionStore(Path(get_project_dir()) / "data" / "agent_sessions.json", "claude_cli")
        self._start_loop()
        self._self_check()

    def _self_check(self) -> None:
        if not self.cli_path:
            logger.bind(tag=TAG).error("未找到 claude 命令，请先安装 Claude Code 并登录，或在配置中指定 cli_path")
            return
        try:
            logger.bind(tag=TAG).info(
                f"Claude Code: {claude_cli_version(self.cli_path)} ({self.cli_path})，"
                f"工作目录 {self.session_config.workdir}，允许工具 {self.session_config.allowed_tools}，"
                f"思考力度 {self.session_config.effort or '默认'}"
            )
        except Exception as e:
            logger.bind(tag=TAG).error(f"无法运行 {self.cli_path} --version: {e}")

    def _create_session(self, st: ConnState, system_prompt: str, resume_id: Optional[str] = None) -> AgentSession:
        cfg = ClaudeSessionConfig(**{**self.session_config.__dict__,
                                     "system_prompt_append": (system_prompt or "") + "\n" + VOICE_RULES})
        return ClaudeSession(cfg, resume_id=resume_id, logger=logger.bind(tag=TAG))
