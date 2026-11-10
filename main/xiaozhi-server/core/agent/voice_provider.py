"""Agent 的语音层：让一个 AgentSession（Claude Code、Grok Bot……）在小智设备上"陪着干活"。

和具体 Agent 无关的部分都在这里，各 Agent 的 provider 继承 AgentVoiceProvider，只需要提供怎么创建会话
（_create_session），以及各自的自检、错误播报等。

线程模型：
- 服务端在每个连接的线程池里同步迭代 response() 生成器；
- provider 自带一个后台线程运行 asyncio 事件循环，所有会话调用都在这个循环里执行；
- 两边通过 queue.Queue 传递事件。

每个设备连接最多对应一个常驻的会话：首句话到来时启动（顺便决定新开还是接续），
空闲超时、连接关闭或服务退出时结束。

设备这边结束一轮（唤醒词打断、点结束、退出词、连接超时断开、单轮超时）时，Agent 如果在干正经活
（用过工具，或已经做了 background_after_seconds 以上），不中断它：这一轮转到后台接着做，做完把回复放进
待汇报箱（core/agent/reports.py），设备空闲时念；后台最多做 background_max_seconds。只是在聊天、刚开口
就被打断的，照旧马上中断。后台在做时同一台设备又问它：说"还在做上一件事"，不发出去；说"停下"就停掉它。

管理后台热加载大模型时，旧实例会被调用 shutdown()：已连接的设备继续用旧实例对话，
等最后一个连接关闭、手上的任务都结束后，再停止事件循环，后台线程随之退出。
"""

import asyncio
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from config.logger import setup_logging
from core.agent import reports
from core.agent.agent_session import (
    STARTING_LABELS,
    Activity,
    AgentSession,
    AgentSessionError,
    Done,
    Error,
    ErrorKind,
    TextDelta,
)
from core.agent.session_store import choose_resume
from core.agent.voice_commands import DEFAULT_RESUME_KEYWORDS, match_resume, match_stop
from core.providers.llm.base import LLMProviderBase
from core.utils import textUtils, turn_gen

TAG = __name__
logger = setup_logging()

SENTENCE_ENDS = "。！？!?.;；~～…"
DEFAULT_PROGRESS_REPLY = "还在处理，请稍等"
# 后台做事相关的播报；{name} 是 Agent 的播报名（SPOKEN_NAME）
BACKGROUND_NOTICE = "这件事要做一阵子，我先不等了，做完了我再告诉你。"
BUSY_NOTICE_TEMPLATE = "😶{name} 还在做上一件事，做完了我告诉你。想让它停下，就说停下。"
STOPPED_NOTICE_TEMPLATE = "😶好，已经让 {name} 停下了。"
PROGRESS_MAX_FACTOR = 4  # 进度提示的间隔逐次翻倍，最多拉长到 4 倍（默认 30 → 60 → 120 秒）


def elapsed_text(seconds: float) -> str:
    """把已用时间说成口语：95 秒 → "1 分半"，185 秒 → "3 分钟"。"""
    minutes, rest = divmod(int(seconds), 60)
    if minutes == 0:
        return f"{rest} 秒"
    return f"{minutes} 分半" if rest >= 30 else f"{minutes} 分钟"


def single_sentence(text: str) -> str:
    """把一句话整理成 TTS 只会切出一段的形式。

    原项目为了降低首字延迟，第一句遇到逗号、顿号等就会切开，填充语会因此变成两段（两个气泡）。
    这里把句中的停顿标点换成空格，结尾统一用句号。
    """
    body = re.sub(r"[，,、~～；;：:]+", " ", text or "").strip()
    body = body.rstrip("。.！!？? ")
    return f"{body}。" if body else ""


def _num(config: dict, key: str, default: float) -> float:
    """读取数字配置。只有没填（None 或空字符串）才用默认值，填 0 就是 0。"""
    value = config.get(key)
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@dataclass
class ConnState:
    key: str  # 设备 ID；没有时退回连接的 session_id
    conn: object = None
    agent: Optional[AgentSession] = None
    lock: Optional[asyncio.Lock] = None
    idle_handle: Optional[asyncio.TimerHandle] = None
    intent_warned: bool = False
    closed: bool = False  # 连接已关闭：不再启动或保留会话
    turn_gen: int = 0  # 每开始一轮加 1，用于让过期的空闲定时器失效
    turn_task: Optional[asyncio.Task] = None  # 正在执行的一轮，卡死时要取消它，否则它一直占着锁
    turn: Optional["Turn"] = None  # 正在执行的一轮（含转到后台的）


class Turn:
    """一轮对话的事件去向：设备在听时交给 response()（经 out 队列），转到后台后自己收着，做完交给待汇报箱。

    emit 在事件循环线程里调用，detach 在连接的线程里调用，用锁保证一条事件要么交给了设备、要么留在这里。"""

    def __init__(self, st: ConnState, out: queue.Queue):
        self.st = st
        self.out = out
        self.gen = st.turn_gen
        self.agent = None
        self.started = time.monotonic()
        self.last_event = self.started
        self.tools_running = 0
        self.used_tool = False
        self.detached = False
        self.finished = False
        self.cancelled = False  # 用户说了"停下"：不再汇报
        self.stop_kind: Optional[ErrorKind] = None  # 后台判定卡死或做太久，被我们停下
        self.texts = []  # 转到后台后（以及转的时候还没念）的文字
        self.error: Optional[Error] = None
        self.ended = asyncio.Event()  # 在事件循环线程里创建
        self._lock = threading.Lock()

    def emit(self, ev) -> None:
        with self._lock:
            if isinstance(ev, Activity) and ev.tools_running >= 0:
                self.tools_running = ev.tools_running
                self.used_tool = self.used_tool or ev.tools_running > 0
            self.last_event = time.monotonic()
            if not self.detached:
                self.out.put(ev)
            elif isinstance(ev, TextDelta):
                self.texts.append(ev.text)
            elif isinstance(ev, Error):
                self.error = ev

    def is_task(self, after: float) -> bool:
        """在干正经活：用过工具，或者已经做了 after 秒以上（想得久）。只是聊天、刚开口的不算。"""
        return self.used_tool or time.monotonic() - self.started >= after

    def detach(self) -> bool:
        """设备不听了，转到后台。队列里还没念的文字也收回来。已经结束或已经转过返回 False。"""
        with self._lock:
            if self.finished or self.detached:
                return False
            self.detached = True
            while True:
                try:
                    ev = self.out.get_nowait()
                except queue.Empty:
                    break
                if isinstance(ev, TextDelta):
                    self.texts.append(ev.text)
                elif isinstance(ev, Error):
                    self.error = ev
            return True

    def finish(self) -> bool:
        """这一轮结束了；返回它是不是在后台结束的。"""
        with self._lock:
            self.finished = True
            self.ended.set()
            return self.detached


class AgentVoiceProvider(LLMProviderBase):
    """各 Agent provider 的基类。子类在 __init__ 里设好 self.store（SessionStore），调用 _init_common、
    _start_loop，并实现 _create_session；不支持接续或启动流程不同的（如 grok_bot）可以整个重写 _ensure_agent。"""

    # 日志和提示里对这个 Agent 的称呼
    AGENT_LABEL = "Agent"
    # 播报给用户时的称呼（"Claude 做完了"）
    SPOKEN_NAME = "Agent"
    # 设备这边结束一轮时，在干正经活的转到后台接着做（见模块说明）。会话自己有后台盯着机制的 Agent 关掉
    BACKGROUND_TURNS = True
    REPORT_ID = "agent"  # 待汇报箱里标记是谁的话
    PROVIDER_TYPE = "agent"  # 日志里的 provider 类型名
    LOOP_THREAD_NAME = "agent-loop"
    LOG_TAG = TAG

    @property
    def _log(self):
        return logger.bind(tag=self.LOG_TAG)

    def _init_common(self, config: dict, filler_default: str = "好的，我看一下", filler_delay_default: float = 2) -> None:
        """和具体 Agent 无关的部分：超时、填充语、进度提示、连接状态表。"""
        # 超时类配置填 0 表示不限制。单轮默认 10 分钟：Agent 干长任务时设备陪着等，期间连接不会因"没人说话"断开
        self.turn_timeout = _num(config, "turn_timeout_seconds", 600) or float("inf")
        # 无响应超时：Agent 连续这么久没有任何事件（文字、思考、工具调用……）就判定卡住。
        # 工具执行期间（例如跑一条很慢的命令）不算，只受单轮超时限制
        self.stall_timeout = _num(config, "stall_timeout_seconds", 30) or float("inf")
        # 干活期间这么久没开口，就说一句进度，让人知道还在做；填 0 关闭。之后间隔逐渐拉长（见 _progress_gap），并报已用时间
        self.progress_interval = _num(config, "progress_interval_seconds", 30)
        self.progress_reply = single_sentence(config.get("progress_reply") or DEFAULT_PROGRESS_REPLY)
        # 第二次起的提示："还在处理，已经 3 分钟了"。自定义了提示语时用它开头
        self.progress_prefix = "还在处理" if not config.get("progress_reply") else self.progress_reply.rstrip("。")
        self.idle_timeout = _num(config, "idle_timeout_seconds", 300)  # 0：每轮结束就关闭会话
        self.resume_window = _num(config, "resume_window_minutes", 120)  # 0：不自动接续
        self.resume_keywords = list(config.get("resume_keywords") or DEFAULT_RESUME_KEYWORDS)
        self.filler = single_sentence(config.get("first_reply_filler", filler_default))
        self.filler_delay = _num(config, "filler_delay_seconds", filler_delay_default)
        self.abandon_grace = 5.0  # 发出中断后等这么久，这一轮还没停下就强制关闭会话
        # 设备不听了以后，Agent 在后台最多再做多久（从这一轮开始算），到时间就停下并汇报；填 0 不转后台（照旧中断）
        self.background_max = _num(config, "background_max_seconds", 7200)
        # 做了这么久（或用过工具）才算"正经活"、值得转后台；更短的聊天被打断就直接中断
        self.background_after = _num(config, "background_after_seconds", 30)
        self._background: Dict[str, Turn] = {}  # 设备 -> 在后台做的一轮

        self._states: Dict[str, ConnState] = {}
        self._states_lock = threading.Lock()
        self._retiring = False  # 已被新实例替换：没有连接后就停止事件循环
        self._stopping = False  # 已安排停止事件循环

    # ---------- 子类实现 ----------

    def _create_session(self, st: ConnState, system_prompt: str, resume_id: Optional[str] = None) -> AgentSession:
        """创建（还没启动的）会话。resume_id 不为空时接续那个会话。"""
        raise NotImplementedError

    def _self_check(self) -> None:
        """启动自检：只写日志。"""

    def _voice_error(self, kind: ErrorKind) -> str:
        """超时、卡住等由 provider 自己判定的错误，播报给用户的话。"""
        return Error(kind).voice_message

    def _stall_note(self, st: ConnState) -> str:
        """看门狗判卡住时附在日志后面的说明。grok_bot 用它写清卡在发送前的哪一步。"""
        return ""

    # ---------- 事件循环 ----------

    def _start_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, args=(self._loop,), name=self.LOOP_THREAD_NAME, daemon=True)
        self._thread.start()

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            loop.close()

    # ---------- 退役（管理后台热加载后调用） ----------

    def shutdown(self) -> None:
        """实例被替换后调用：不再接新连接，现有连接照常对话；没有连接时停止事件循环，后台线程退出。"""
        with self._states_lock:
            self._retiring = True
        self._log.info(f"{self.PROVIDER_TYPE} 旧实例退役，剩余连接 {len(self._states)} 个，全部关闭后结束后台线程")
        self._maybe_stop()

    def _maybe_stop(self) -> None:
        with self._states_lock:
            if not self._retiring or self._stopping or self._states or self._background:
                return
            self._stopping = True
            loop = self._loop
        asyncio.run_coroutine_threadsafe(self._stop_loop(), loop)

    async def _stop_loop(self, timeout: float = 30) -> None:
        """等循环里剩下的任务（关闭会话、收尾中的一轮对话）结束，再停止循环。"""
        loop = asyncio.get_running_loop()
        pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        if pending:
            _, still = await asyncio.wait(pending, timeout=timeout)
            for t in still:
                t.cancel()
        loop.stop()
        self._log.info(f"{self.PROVIDER_TYPE} 旧实例的后台线程已结束")

    # ---------- 连接钩子（由原项目代码通过 getattr 调用） ----------

    def _state(self, session_id: str) -> ConnState:
        with self._states_lock:
            if self._stopping:
                # 极少见：替换实例的瞬间刚好有设备连上旧实例，而循环已经在停止。重新开一个循环给它用，
                # 它关闭后会再次停止。
                self._stopping = False
                self._start_loop()
            st = self._states.get(session_id)
            if st is None:
                st = ConnState(key=session_id)
                self._states[session_id] = st
            return st

    def on_conn_open(self, conn) -> None:
        st = self._state(conn.session_id)
        st.conn = conn
        if getattr(conn, "device_id", None):
            st.key = conn.device_id

    def abort(self, conn) -> None:
        st = self._states.get(conn.session_id)
        if st and self._let_go(st, "设备打断"):
            return
        if st and st.agent:
            asyncio.run_coroutine_threadsafe(st.agent.interrupt(), self._loop)
            # 卡在工具调用里时 interrupt 可能无效，兜底防止下一句话一直排队等这一轮
            asyncio.run_coroutine_threadsafe(self._abandon_if_stuck(st, st.turn_gen), self._loop)

    def on_conn_close(self, conn) -> None:
        with self._states_lock:
            st = self._states.pop(conn.session_id, None)
        if st:
            st.closed = True
            if self._let_go(st, "连接关闭") or self._in_background(st):
                self._log.info(f"连接关闭，{self.AGENT_LABEL} 还在后台做，做完再关进程，设备 {st.key}")
            else:
                asyncio.run_coroutine_threadsafe(self._close_agent(st, "连接关闭"), self._loop)
        self._maybe_stop()

    # ---------- 设备不听了：转后台接着做 ----------

    @staticmethod
    def _in_background(st: ConnState) -> bool:
        turn = st.turn
        return turn is not None and turn.detached and not turn.finished

    def _let_go(self, st: ConnState, reason: str) -> bool:
        """设备这边结束这一轮。Agent 在干正经活就不中断，转后台接着做、做完进待汇报箱，返回 True；
        没有进行中的一轮、只是聊天、或不支持后台的 Agent 返回 False（调用方照旧中断或关闭）。可以从任何线程调用。"""
        turn = st.turn
        if (not self.BACKGROUND_TURNS or not self.background_max or turn is None
                or not turn.is_task(self.background_after) or not turn.detach()):
            return False
        with self._states_lock:
            self._background[st.key] = turn
        self._log.info(
            f"{reason}：{self.AGENT_LABEL} 还在干活（已 {time.monotonic() - turn.started:.0f} 秒），不中断，转后台，"
            f"做完进待汇报，设备 {st.key}")
        asyncio.run_coroutine_threadsafe(self._watch_background(turn), self._loop)
        return True

    async def _watch_background(self, turn: Turn) -> None:
        """后台这一轮的看门狗：真卡死（连续 stall_timeout 没有任何事件、也没有工具在跑）或做满 background_max 就停下它。"""
        while not turn.finished:
            try:
                await asyncio.wait_for(turn.ended.wait(), timeout=1)
            except asyncio.TimeoutError:
                pass
            if turn.finished:
                return
            now = time.monotonic()
            stalled = not turn.tools_running and now - turn.last_event >= self.stall_timeout
            if stalled or now - turn.started >= self.background_max:
                turn.stop_kind = ErrorKind.STALLED if stalled else ErrorKind.TIMEOUT
                self._log.warning(
                    f"后台的 {self.AGENT_LABEL} {'连续 %d 秒没有任何响应' % int(self.stall_timeout) if stalled else '做满 %d 秒' % int(self.background_max)}，停下它")
                if turn.agent is not None:
                    await turn.agent.interrupt()
                asyncio.ensure_future(self._abandon_if_stuck(turn.st, turn.gen))
                return

    def _background_report(self, turn: Turn) -> str:
        """后台这一轮结束后要念给用户的话。"""
        name = self.SPOKEN_NAME
        if turn.cancelled:
            return ""
        if turn.stop_kind == ErrorKind.STALLED:
            return f"{name} 刚才那件事一直没有回应，我让它停下了。想接着做，跟我说继续就行。"
        if turn.stop_kind == ErrorKind.TIMEOUT:
            return f"{name} 刚才那件事做了太久，我让它停下了。想接着做，跟我说继续就行。"
        body = textUtils.check_emoji("".join(turn.texts)).strip().lstrip("。")  # 汇报时不需要换表情
        if turn.error is not None and not body:
            return f"{name} 刚才那件事没做完：{turn.error.voice_message}"
        return f"{name} 做完了：{body}" if body else f"{name} 做完了。"

    async def _finish_background(self, st: ConnState, turn: Turn) -> None:
        with self._states_lock:
            if self._background.get(st.key) is turn:
                self._background.pop(st.key, None)
        text = self._background_report(turn)
        self._log.info(f"后台的 {self.AGENT_LABEL} 这一轮结束（{time.monotonic() - turn.started:.0f} 秒）"
                       f"{'，待汇报：' + text[:80] if text else '，不汇报'}")
        if text:
            reports.add(st.key, self.REPORT_ID, text)
        if st.closed:
            await self._close_agent(st, "后台这一轮做完，连接早已关闭")
        self._maybe_stop()

    async def _busy_in_background(self, st: ConnState, text: str, out: queue.Queue) -> bool:
        """这台设备有一轮还在后台做（可能是上一个连接留下的）：这句话不发给 Agent。说"停下"就停掉它。"""
        with self._states_lock:
            bg = self._background.get(st.key)
        if bg is None or bg.finished:
            return False
        if match_stop(text):
            bg.cancelled = True
            self._log.info(f"用户让后台的 {self.AGENT_LABEL} 停下，设备 {st.key}")
            if bg.agent is not None:
                await bg.agent.interrupt()
            asyncio.ensure_future(self._abandon_if_stuck(bg.st, bg.gen))
            out.put(TextDelta(STOPPED_NOTICE_TEMPLATE.format(name=self.SPOKEN_NAME)))
        else:
            self._log.info(f"{self.AGENT_LABEL} 还在后台做上一件事，这句话不发：{text[:40]}")
            out.put(TextDelta(BUSY_NOTICE_TEMPLATE.format(name=self.SPOKEN_NAME)))
        out.put(Done())
        return True

    # ---------- 后台循环中的逻辑 ----------

    async def _close_agent(self, st: ConnState, reason: str) -> None:
        if st.idle_handle:
            st.idle_handle.cancel()
            st.idle_handle = None
        agent, st.agent = st.agent, None
        if agent:
            await agent.close()
            self._log.info(f"已关闭 {self.AGENT_LABEL} 会话（{reason}），设备 {st.key}")

    def _schedule_idle_close(self, st: ConnState) -> None:
        if st.idle_handle:
            st.idle_handle.cancel()
        gen = st.turn_gen
        st.idle_handle = self._loop.call_later(
            self.idle_timeout, lambda: asyncio.ensure_future(self._idle_close(st, gen))
        )

    async def _idle_close(self, st: ConnState, gen: int) -> None:
        async with st.lock:
            if st.turn_gen != gen:
                return  # 定时器触发前已经开始了新的一轮
            st.idle_handle = None
            await self._close_agent(st, f"空闲 {int(self.idle_timeout)} 秒")

    async def _abandon_if_stuck(self, st: ConnState, gen: int) -> None:
        """超时或设备打断后已发送中断；如果这一轮过了 abandon_grace 秒还没结束，说明会话卡死了，直接关闭。
        下一轮会自动重新启动（在接续窗口内会接续原会话）。

        关闭会话不一定能让卡住的那一轮退出（它可能还在等一个永远不来的消息），所以还要取消它，
        否则它一直占着锁，下一句话会排队等到超时。
        """
        await asyncio.sleep(self.abandon_grace)
        if st.turn_gen == gen and st.lock is not None and st.lock.locked():
            self._log.warning(f"中断后 {self.AGENT_LABEL} 仍未结束本轮，强制关闭")
            task = st.turn_task
            await self._close_agent(st, "中断无效")
            if task is not None and not task.done() and st.turn_gen == gen:
                task.cancel()

    async def _ensure_agent(self, st: ConnState, text: str, out: queue.Queue, system_prompt: str) -> bool:
        """确保有可用的会话。返回 True 表示这句话已经作为"接续指令"处理完，不必再发给 Agent。

        接续指令后面还带着别的内容时（如"接着上次聊那个 bug"），接续后把整句话照常发给 Agent，
        不能只回一句"好的"就把后半句吞掉。
        """
        if st.agent and st.agent.alive:
            return False
        remainder = match_resume(text, self.resume_keywords)
        keyword = remainder is not None
        resume_id, reason = choose_resume(self.store.get(st.key), keyword, self.resume_window)
        agent = self._create_session(st, system_prompt, resume_id)
        t0 = time.time()
        try:
            await agent.start()
        except AgentSessionError as e:
            if e.kind != ErrorKind.SESSION_MISSING:
                raise
            self._log.warning(f"接续会话 {resume_id} 失败，改为新开会话")
            agent = self._create_session(st, system_prompt)
            await agent.start()  # 新会话也启动失败时（如未登录、网络问题）保留旧记录，下次还能接续
            self.store.forget(st.key)
            if keyword:
                out.put(TextDelta("😶没有找到之前的对话，我们重新开始吧。"))
            resume_id, reason = None, "resume_failed"
        if st.closed:
            await agent.close()  # 启动期间连接已经关闭
            raise AgentSessionError(ErrorKind.PROCESS_EXITED, "连接已关闭")
        st.agent = agent
        self._log.info(
            f"{self.AGENT_LABEL} 进程已启动（{time.time() - t0:.1f}s），设备 {st.key}，"
            f"{'接续会话 ' + resume_id if resume_id else '新会话 ' + agent.session_id}（{reason}）"
        )
        if keyword and not remainder:
            if resume_id:
                out.put(TextDelta("😉好的，我们接着上次聊。"))
            elif reason == "no_record":
                out.put(TextDelta("😶没有找到之前的对话，我们重新开始吧。"))
            out.put(Done(session_id=agent.session_id))
            return True
        if keyword and reason == "no_record":
            out.put(TextDelta("😶没有找到之前的对话，我们重新开始吧。"))
        return False

    async def _run_turn(self, st: ConnState, text: str, system_prompt: str, out: queue.Queue) -> None:
        if self.BACKGROUND_TURNS and await self._busy_in_background(st, text, out):
            return
        if st.lock is None:
            st.lock = asyncio.Lock()
        async with st.lock:
            st.turn_gen += 1
            st.turn_task = asyncio.current_task()
            turn = st.turn = Turn(st, out)
            if st.idle_handle:
                st.idle_handle.cancel()
                st.idle_handle = None
            try:
                if self.BACKGROUND_TURNS:
                    pending = reports.take_text(st.key, self.REPORT_ID)  # 上一件事在后台做完、还没念的，先念
                    if pending:
                        out.put(TextDelta(pending))
                if await self._ensure_agent(st, text, out, system_prompt):
                    if st.agent:  # grok_bot 在连接已关闭时把话交出去后也返回 True，那时没有会话
                        self.store.touch(st.key, st.agent.session_id)
                    return
                agent, recorded, working, failed, previous = st.agent, False, False, False, None
                turn.agent = agent
                async for ev in agent.send(text):
                    if isinstance(ev, Error) and not working and ev.kind != ErrorKind.PROCESS_EXITED:
                        failed = True  # 还没开始干活就出错：没登录、额度用完、限流或过载重试用完等
                    elif not failed and isinstance(ev, (TextDelta, Activity)):
                        if not recorded:
                            # 会话已经建好（例如 Claude Code 在 init 时就已把用户的话存进本机会话）。马上记下来：
                            # 这一轮如果卡死（例如网络卡住，只收到 init）、会话被强制关闭，下一句话重新启动时也能接回这个会话
                            previous = self.store.get(st.key)
                            self.store.touch(st.key, agent.session_id)
                            recorded = True
                        working = working or not (isinstance(ev, Activity) and ev.label in STARTING_LABELS)
                    turn.emit(ev)
                    if isinstance(ev, Error) and ev.kind == ErrorKind.PROCESS_EXITED:
                        st.agent = None
                if failed:
                    # 还没开始干活就出错：恢复原来的记录，免得新会话覆盖之前能接续的会话（包括最后活跃时间）
                    if recorded:
                        self._restore_record(st.key, previous)
                elif recorded and st.agent:
                    self.store.touch(st.key, st.agent.session_id)  # 更新最后活跃时间
            except AgentSessionError as e:
                turn.emit(Error(e.kind, e.detail, voice=self._voice_error(e.kind)))
            except asyncio.CancelledError:
                # 会话卡死被强制关闭（_abandon_if_stuck 取消了这一轮）：告诉还在等的一方这一轮结束了
                turn.emit(Done(interrupted=True))
                raise
            except Exception as e:
                self._log.exception(f"{self.AGENT_LABEL} 处理出错: {e}")
                turn.emit(Error(ErrorKind.UNKNOWN, str(e), voice=self._voice_error(ErrorKind.UNKNOWN)))
            finally:
                if turn.finish():
                    await self._finish_background(st, turn)
                if st.agent and not st.closed:
                    self._schedule_idle_close(st)

    # ---------- 服务端调用的同步接口 ----------

    def response(self, session_id, dialogue, **kwargs):
        text = next((m.get("content") for m in reversed(dialogue) if m.get("role") == "user"), "") or ""
        system_prompt = dialogue[0].get("content", "") if dialogue and dialogue[0].get("role") == "system" else ""
        st = self._state(session_id)
        self._warn_intent(st)
        # [xiaozhi-hub] 这一轮的代次（turn_gen）：打断后新的一轮一开始就清掉 client_abort，只看它会漏
        gen = turn_gen.current(st.conn) if st.conn is not None else None

        out: queue.Queue = queue.Queue()
        asyncio.run_coroutine_threadsafe(self._run_turn(st, text, system_prompt, out), self._loop)

        t0 = time.time()
        last_event_at = t0
        last_spoken_at = t0       # 上一次说话（填充语、正式回复、进度提示）的时间
        tail = ""                 # 已说出的最后一个字，判断能不能在这里插一句进度
        tools_running = 0
        progress_count = 0        # 这一轮已经说了几次进度
        first_at = None
        filler_sent = False
        emotion_buf = ""          # 填充语之后，正式回复开头的几个字，用来找表情
        emotion_resent = False
        finished = False
        try:
            while True:
                if self._aborted(st, gen):
                    # 设备打断了（abort() 已经让 Agent 中断）：这一轮到此为止，不再保活、不再说进度。
                    # 不能等 Agent 自己结束：它卡住不理中断时，这里会一直等到单轮超时（最长 10 分钟），
                    # 期间连接一直不会因为没人说话断开；用户开口说下一句后，打断标志被清掉，
                    # 这里还会接着吐出"还在处理""做得太久"
                    finished = True
                    return
                try:
                    ev = out.get(timeout=0.2)
                except queue.Empty:
                    self._keep_alive(st)
                    elapsed = time.time() - t0
                    if first_at is None and self.filler and not filler_sent and elapsed >= self.filler_delay:
                        filler_sent = True
                        last_spoken_at, tail = time.time(), self.filler[-1]
                        yield f"😶{self.filler}"
                    elif self._progress_due(last_spoken_at, tail, first_at is not None or filler_sent, progress_count):
                        text = self.progress_reply if progress_count == 0 else \
                            f"{self.progress_prefix}，已经 {elapsed_text(time.time() - t0)}了。"
                        notice = self._before_notice(tail, first_at is not None or filler_sent) + text
                        progress_count += 1
                        last_spoken_at, tail = time.time(), text[-1]
                        yield notice
                    stalled = not tools_running and time.time() - last_event_at >= self.stall_timeout
                    if elapsed >= self.turn_timeout and not stalled and self._let_go(st, "单轮超时"):
                        # Agent 还在正常干活：不中断，转后台，做完进待汇报
                        yield self._before_notice(tail, first_at is not None or filler_sent) + BACKGROUND_NOTICE
                        finished = True
                        return
                    if elapsed >= self.turn_timeout or stalled:
                        kind = ErrorKind.STALLED if stalled else ErrorKind.TIMEOUT
                        self._log.warning(
                            f"已中断本轮：{'连续 %d 秒没有任何响应' % int(self.stall_timeout) + self._stall_note(st) if stalled else '超过 %d 秒' % int(self.turn_timeout)}"
                        )
                        if st.agent:
                            asyncio.run_coroutine_threadsafe(st.agent.interrupt(), self._loop)
                            asyncio.run_coroutine_threadsafe(self._abandon_if_stuck(st, st.turn_gen), self._loop)
                        yield self._before_notice(tail, first_at is not None or filler_sent) + self._voice_error(kind)
                        finished = True
                        return
                    continue
                last_event_at = time.time()
                self._keep_alive(st)
                if isinstance(ev, Activity):
                    if ev.tools_running >= 0:
                        tools_running = ev.tools_running
                    continue
                if isinstance(ev, TextDelta):
                    if first_at is None:
                        first_at = time.time() - t0
                    if filler_sent and not emotion_resent:
                        # 原项目每轮只按第一段文字设一次表情，填充语会把表情锁成 😶。
                        # 正式回复开头带了表情时，再补发一次情绪消息，让设备换成真正的表情。
                        emotion_buf += ev.text
                        emoji = next((ch for ch in emotion_buf[:8] if ch in textUtils.EMOJI_MAP), None)
                        if emoji or len(emotion_buf) >= 8:
                            emotion_resent = True
                            if emoji:
                                self._send_emotion(st, emoji)
                    if ev.text.strip():
                        last_spoken_at, tail = time.time(), ev.text.strip()[-1]
                    yield ev.text
                elif isinstance(ev, Error):
                    self._log.error(f"{self.AGENT_LABEL} 错误 [{ev.kind.value}]: {ev.detail[:500]}")
                    yield self._before_notice(tail, first_at is not None or filler_sent) + ev.voice_message
                    finished = True
                    return
                elif isinstance(ev, Done):
                    finished = True
                    self._log.info(
                        f"{self.AGENT_LABEL} 本轮完成：首字 {first_at if first_at is None else round(first_at, 2)}s，"
                        f"总耗时 {time.time() - t0:.1f}s{'，已被打断' if ev.interrupted else ''}"
                    )
                    return
        finally:
            if not finished and st.agent and not self._let_go(st, "停止播报"):
                # 生成器被提前关闭（例如设备打断后服务端停止迭代），让 Agent 也停下来（在干正经活的除外，转后台）
                asyncio.run_coroutine_threadsafe(st.agent.interrupt(), self._loop)

    @staticmethod
    def _aborted(st: ConnState, gen=None) -> bool:
        """设备是否打断了这一轮。打断时服务端会把连接的 client_abort 置为 True，新的一句话开始时清掉；
        清掉之前这里没轮询到也不要紧，打断和新的一轮都会让代次变（gen 是这一轮开始时的代次）。"""
        if bool(getattr(st.conn, "client_abort", False)):
            return True
        return gen is not None and st.conn is not None and turn_gen.stale(st.conn, gen)

    def _restore_record(self, key: str, previous: Optional[dict]) -> None:
        """把这台设备的会话记录恢复成这一轮开始前的样子；原来没有记录就删掉。"""
        if previous and previous.get("session_id"):
            self.store.touch(key, previous["session_id"], now=previous.get("last_active"))
        else:
            self.store.forget(key)

    @staticmethod
    def _keep_alive(st: ConnState) -> None:
        """Agent 干活期间刷新连接的活动时间。

        服务端在"没有任何动静"一段时间后会兜底断开连接（core/connection.py 的 _check_timeout，
        默认 180 秒）。干活时设备在等、服务端也可能很久不说话，不刷新的话长任务做到一半连接就断了，
        会话也会随之关闭。
        """
        if st.conn is not None:
            st.conn.last_activity_time = time.time() * 1000

    @staticmethod
    def _before_notice(tail: str, spoke: bool) -> str:
        """服务端自己插一句话（超时、出错、进度提示）前要加的东西。

        还没说过话：加一个表情，设备才会换脸；已经说了半句（结尾不是句末标点）：先补个句号把那半句收住，
        否则 TTS 会把两句粘成一句念，比如"我先看一下文件这件事做得太久"。
        """
        if not spoke:
            return "😶"
        return "" if not tail or tail in SENTENCE_ENDS or tail in textUtils.EMOJI_MAP else "。"

    def _progress_gap(self, count: int) -> float:
        """第 count+1 次进度提示前要等多久：每次翻倍，最多拉长到 PROGRESS_MAX_FACTOR 倍。
        默认依次在第 30、90、210、330… 秒提示，10 分钟的任务说 6 次，而不是每 30 秒一次说 20 次。"""
        return self.progress_interval * min(2 ** count, PROGRESS_MAX_FACTOR)

    def _progress_due(self, last_spoken_at: float, tail: str, spoke: bool, count: int = 0) -> bool:
        """该不该说一句进度：超过间隔没开口，而且上一句已经说完（不在半句话中间插话）。"""
        if not self.progress_reply or not self.progress_interval:
            return False
        if time.time() - last_spoken_at < self._progress_gap(count):
            return False
        return not spoke or not tail or tail in SENTENCE_ENDS or tail in textUtils.EMOJI_MAP

    def _send_emotion(self, st: ConnState, emoji: str) -> None:
        conn = st.conn
        loop = getattr(conn, "loop", None)
        if conn is None or loop is None or not (getattr(conn, "features", None) or {}).get("emoji", True):
            return
        asyncio.run_coroutine_threadsafe(textUtils.get_emotion(conn, emoji), loop)
        self._log.debug(f"填充语之后补发表情 {emoji}")

    def response_no_stream(self, system_prompt, user_prompt, **kwargs):
        self._log.warning(f"{self.AGENT_LABEL} 不支持非流式调用（意图识别、记忆总结等），请为这些模块单独配置 API 模型")
        return ""

    def _warn_intent(self, st: ConnState) -> None:
        if st.intent_warned or st.conn is None:
            return
        intent_type = getattr(st.conn, "intent_type", None)
        if intent_type and intent_type != "nointent":
            self._log.warning(f"使用 {self.AGENT_LABEL} 时 Intent 应设为 nointent，当前为 {intent_type}")
        st.intent_warned = True
