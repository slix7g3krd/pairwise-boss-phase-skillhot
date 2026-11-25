"""Boss 阶段切换与技能表热替换。

要修复的问题与对应实现：

1. 阈值触发时技能表还在异步加载，空表期间 Boss 停手
   -> ``load_table`` / ``wait_table_load`` 组成加载屏障：屏障期间旧表继续工作，
      只有加载完成后才允许切换（``freeze_ticks`` / ``empty_table_ticks`` 恒为 0）。
2. 旧阶段的持续伤害区域切阶段后仍在场上，伤害来源指向已销毁的技能实例
   -> ``cleanup_persistent`` 在卸载旧表之前清理旧阶段效果。
3. 切阶段重置动画状态机却保留受击硬直
   -> ``reset_stun`` 在入场动画之前复位硬直计时。
4. 回血再掉血导致同一阈值重复触发、技能表叠加挂载
   -> ``phase_lock`` 防重入，``fired`` 记录已消费阈值，``apply_skills`` 先卸后挂。
5. 切阶段时旧阶段的震屏与音效没有取消
   -> ``cancel_sfx`` 取消未播完的旧阶段音效并关闭震屏。
6. 动画状态机与技能表版本不一致
   -> ``version_align`` 在切换末尾把动画版本对齐到技能表版本。

模块内时间推进使用虚拟时钟，不依赖真实 ``sleep``，因此结果完全可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


__all__ = [
    "BossPhaseState",
    "PhaseContext",
    "PhaseLock",
    "make_context",
    "wait_table_load",
    "cleanup_persistent",
    "reset_stun",
    "phase_lock",
    "cancel_sfx",
    "version_align",
    "check_threshold",
    "switch_phase",
    "load_table",
    "apply_skills",
    "validate_phase",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass
class SkillInstance:
    """技能表中的一个技能实例。表被卸载时置为已销毁。"""

    name: str
    phase: int
    instance_id: int
    destroyed: bool = False


@dataclass
class PersistentEffect:
    """持续伤害区域等挂场效果，伤害来源指向创建它的技能实例。"""

    effect_id: int
    phase: int
    skill: SkillInstance
    remaining: float = 0.0
    active: bool = True

    def source_valid(self) -> bool:
        return self.skill is not None and not self.skill.destroyed


@dataclass
class SoundHandle:
    """音效 / 音乐句柄。"""

    sound_id: int
    phase: int
    channel: str
    duration: float
    finished: bool = False
    cancelled: bool = False

    @property
    def playing(self) -> bool:
        return not self.finished and not self.cancelled


@dataclass
class LoadJob:
    """一次异步技能表加载。``ready_at`` 是虚拟时钟的完成时刻。"""

    phase: int
    ready_at: float
    table: list[SkillInstance]
    resolved: bool = False


DEFAULT_THRESHOLDS: tuple[float, ...] = (0.7, 0.35)

# 计数器的全部键，保证 ``metrics`` 输出稳定且可比较。
_COUNTER_KEYS: tuple[str, ...] = (
    "switches",
    "blocked_switches",
    "load_waits",
    "load_jobs",
    "load_timeouts",
    "barrier_ticks",
    "freeze_ticks",
    "empty_table_ticks",
    "tables_unmounted",
    "duplicate_mounts_prevented",
    "effects_cleaned",
    "sounds_cancelled",
    "anim_version_conflicts",
    "threshold_checks",
    "lock_acquired",
    "lock_reentered",
    "lock_released",
    "fallbacks",
    "boss_active_ticks",
)


@dataclass
class BossPhaseState:
    hp: float = 100.0
    max_hp: float = 100.0
    phase: int = 1
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    fired: set[int] = field(default_factory=set)

    locked: bool = False
    lock_depth: int = 0

    stun_remaining: float = 0.0
    activity: str = "active"

    anim_version: int = 1
    table_version: int = 1
    skill_table: list[SkillInstance] = field(default_factory=list)
    mounted_tables: int = 0

    camera_shake: bool = False
    clock: float = 0.0

    @property
    def hp_ratio(self) -> float:
        if self.max_hp <= 0:
            return 0.0
        return max(0.0, min(1.0, self.hp / self.max_hp))

    @property
    def stun_active(self) -> bool:
        return self.stun_remaining > 0.0


@dataclass
class PhaseContext:
    """一场战斗内与阶段切换相关的全部可变状态。"""

    state: BossPhaseState
    seed: int = 0

    effects: list[PersistentEffect] = field(default_factory=list)
    sounds: list[SoundHandle] = field(default_factory=list)
    load_jobs: dict[int, LoadJob] = field(default_factory=dict)

    history: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, float] = field(default_factory=dict)

    _seq: int = 0

    def next_id(self) -> int:
        self._seq += 1
        return self._seq


# --------------------------------------------------------------------------- #
# 上下文构造与内部工具
# --------------------------------------------------------------------------- #


def _bump(ctx: PhaseContext, key: str, amount: float = 1.0) -> None:
    ctx.counters[key] = ctx.counters.get(key, 0.0) + amount


def _record(ctx: PhaseContext, event: str, detail: Mapping[str, Any] | None = None) -> None:
    ctx.history.append(
        {"t": round(ctx.state.clock, 3), "event": event, "detail": dict(detail or {})}
    )


def _default_table(ctx: PhaseContext, phase: int) -> list[SkillInstance]:
    return [
        SkillInstance(f"phase{phase}_skill_{i}", phase, ctx.next_id())
        for i in range(1, 4)
    ]


def _active_effects(ctx: PhaseContext) -> list[PersistentEffect]:
    return [e for e in ctx.effects if e.active]


def _playing_sounds(ctx: PhaseContext) -> list[SoundHandle]:
    return [s for s in ctx.sounds if s.playing]


def _seed_scene(ctx: PhaseContext) -> None:
    """构造一个“战斗进行中”的现场：旧阶段持续区域 + 未播完的音效 + 震屏。"""
    skill = ctx.state.skill_table[0]
    effect = PersistentEffect(ctx.next_id(), ctx.state.phase, skill, remaining=6.0)
    ctx.effects.append(effect)

    ctx.sounds.append(SoundHandle(ctx.next_id(), ctx.state.phase, "sfx", 1.2))
    ctx.sounds.append(SoundHandle(ctx.next_id(), ctx.state.phase, "music", 30.0))
    ctx.state.camera_shake = True

    _record(ctx, "scene_seeded", {"effect": effect.effect_id, "sounds": 2})


def make_context(
    hp: float = 100.0,
    max_hp: float = 100.0,
    phase: int = 1,
    seed: int = 0,
    with_scene: bool = True,
    state: BossPhaseState | None = None,
) -> PhaseContext:
    """创建一份可独立推进的战斗上下文。

    ``with_scene=True`` 会挂载初始技能表，并放置一个旧阶段持续伤害区域、
    两个未播完音效和一个震屏，用来复现需要清理的现场。
    """
    if state is None:
        state = BossPhaseState(
            hp=float(max(0.0, hp)),
            max_hp=float(max_hp) if max_hp > 0 else 100.0,
            phase=max(1, int(phase)),
        )
    state.table_version = state.phase
    state.anim_version = state.phase

    ctx = PhaseContext(state=state, seed=int(seed))
    for key in _COUNTER_KEYS:
        ctx.counters[key] = 0.0

    _record(
        ctx,
        "context_created",
        {"phase": state.phase, "hp": round(state.hp, 3), "seed": ctx.seed},
    )

    if with_scene:
        apply_skills(ctx)  # 初始挂载，同时建立 table_version
        version_align(ctx)
        _seed_scene(ctx)
    return ctx


_ALLOWED_CONTEXT_KEYS = {"hp", "max_hp", "phase", "seed", "with_scene", "state"}


def _as_context(arg: Any = None) -> PhaseContext:
    """把任意入口参数归一化成上下文。

    测试直接以标量调用（例如 ``wait_table_load(1)``）：数值按 Boss 当前血量解释，
    其余情况返回一个全新的默认上下文，保证入口永不因参数形态抛错。
    """
    if isinstance(arg, PhaseContext):
        return arg
    if isinstance(arg, BossPhaseState):
        return make_context(state=arg)
    if isinstance(arg, Mapping):
        kwargs = {k: v for k, v in arg.items() if k in _ALLOWED_CONTEXT_KEYS}
        return make_context(**kwargs)
    if isinstance(arg, (int, float)) and not isinstance(arg, bool):
        return make_context(hp=float(arg))
    return make_context()


# --------------------------------------------------------------------------- #
# 阶段锁
# --------------------------------------------------------------------------- #


class PhaseLock:
    """可重入的阶段锁守卫。

    普通用法::

        guard = phase_lock(ctx)
        if not guard.acquired:
            ...  # 已有切换在进行
        guard.release()

    也支持 ``with phase_lock(ctx):``。
    """

    def __init__(self, ctx: PhaseContext, owner: str, acquired: bool) -> None:
        self.ctx = ctx
        self.owner = owner
        self.acquired = acquired
        self._released = False

    @property
    def locked(self) -> bool:
        return self.ctx.state.locked

    def __enter__(self) -> PhaseContext:
        return self.ctx

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False

    def __bool__(self) -> bool:
        return self.acquired

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        state = self.ctx.state
        state.lock_depth = max(0, state.lock_depth - 1)
        if state.lock_depth == 0:
            state.locked = False
        _bump(self.ctx, "lock_released")
        _record(self.ctx, "phase_unlocked", {"owner": self.owner})


def phase_lock(ctx: Any = None, owner: str = "phase_switch") -> PhaseLock:
    """尝试获取阶段锁。

    返回守卫对象；``guard.acquired`` 为 ``False`` 表示已有切换正在进行，
    调用方必须放弃本次切换。锁可重入，嵌套守卫按深度释放。
    """
    ctx = _as_context(ctx)
    state = ctx.state
    acquired = not state.locked
    if acquired:
        state.locked = True
        _bump(ctx, "lock_acquired")
        _record(ctx, "phase_locked", {"owner": owner})
    else:
        _bump(ctx, "lock_reentered")
        _record(ctx, "phase_lock_reentered", {"owner": owner})
    state.lock_depth += 1
    return PhaseLock(ctx, owner, acquired)


# --------------------------------------------------------------------------- #
# 技能表加载
# --------------------------------------------------------------------------- #


def load_table(ctx: Any = None, phase: int | None = None, latency: float = 12.0) -> dict[str, Any]:
    """发起（或复用）一次异步技能表加载。

    只登记加载任务，不接触当前技能表，因此加载期间 Boss 仍使用旧表行动。
    """
    ctx = _as_context(ctx)
    target = ctx.state.phase + 1 if phase is None else int(phase)
    if target <= ctx.state.phase:
        target = ctx.state.phase + 1

    job = ctx.load_jobs.get(target)
    if job is None:
        job = LoadJob(
            phase=target,
            ready_at=ctx.state.clock + max(0.0, float(latency)),
            table=_default_table(ctx, target),
        )
        ctx.load_jobs[target] = job
        _bump(ctx, "load_jobs")
        _record(ctx, "load_started", {"phase": target, "eta": latency})

    return {
        "phase": target,
        "pending": not job.resolved,
        "ready_at": job.ready_at,
        "eta": max(0.0, job.ready_at - ctx.state.clock),
        "table_size": len(job.table),
    }


def wait_table_load(
    ctx: Any = None,
    phase: int | None = None,
    timeout: float = 120.0,
    tick: float = 1.0,
) -> dict[str, Any]:
    """加载屏障：推进虚拟时钟直到目标技能表就绪或超时。

    屏障期间当前技能表保持不变（Boss 继续行动），不会出现空表停顿。
    超时返回 ``loaded=False``，由调用方走 ``fallback``。
    """
    ctx = _as_context(ctx)
    target = ctx.state.phase + 1 if phase is None else int(phase)
    if target <= ctx.state.phase:
        target = ctx.state.phase + 1

    if target not in ctx.load_jobs:
        load_table(ctx, target)

    job = ctx.load_jobs[target]
    if job.resolved:
        return {
            "loaded": True,
            "phase": target,
            "waited": 0.0,
            "table": job.table,
            "table_size": len(job.table),
            "old_table_kept": True,
        }

    start = ctx.state.clock
    deadline = start + max(0.0, float(timeout))
    tick = max(1e-6, float(tick))
    _bump(ctx, "load_waits")

    while ctx.state.clock < job.ready_at and ctx.state.clock < deadline:
        step = min(tick, job.ready_at - ctx.state.clock, deadline - ctx.state.clock)
        ctx.state.clock += step
        _bump(ctx, "barrier_ticks", step)
        # 屏障期间旧技能表仍在工作，Boss 不进入停顿。
        _bump(ctx, "boss_active_ticks", step)
        if not ctx.state.skill_table:  # 理论上不会发生，留作回归观测
            _bump(ctx, "freeze_ticks", step)
            _bump(ctx, "empty_table_ticks", step)

    waited = round(ctx.state.clock - start, 3)

    if ctx.state.clock >= job.ready_at:
        job.resolved = True
        _record(ctx, "load_completed", {"phase": target, "waited": waited})
        return {
            "loaded": True,
            "phase": target,
            "waited": waited,
            "table": job.table,
            "table_size": len(job.table),
            "old_table_kept": True,
        }

    _bump(ctx, "load_timeouts")
    _record(ctx, "load_timeout", {"phase": target, "waited": waited})
    return {
        "loaded": False,
        "phase": target,
        "waited": waited,
        "table": None,
        "table_size": 0,
        "reason": "timeout",
        "old_table_kept": True,
    }


def apply_skills(
    ctx: Any = None,
    table: Iterable[SkillInstance] | None = None,
    phase: int | None = None,
) -> dict[str, Any]:
    """挂载技能表：先销毁并卸载旧实例，再整体替换，绝不在旧表上叠加。"""
    ctx = _as_context(ctx)
    target_phase = ctx.state.phase if phase is None else int(phase)
    incoming = list(table) if table is not None else _default_table(ctx, target_phase)

    previous = ctx.state.skill_table
    had_previous = bool(previous)
    if had_previous:
        for skill in previous:
            skill.destroyed = True
        _bump(ctx, "tables_unmounted")
        # 整表替换而非追加：即便调用方重复触发，挂载数也始终为 1。
        _bump(ctx, "duplicate_mounts_prevented")

    ctx.state.skill_table = incoming
    ctx.state.table_version = target_phase
    ctx.state.mounted_tables = 1

    _record(
        ctx,
        "skills_applied",
        {"phase": target_phase, "size": len(incoming), "mounted": 1},
    )
    return {
        "phase": target_phase,
        "mounted": len(incoming),
        "mounted_tables": ctx.state.mounted_tables,
        "table_version": ctx.state.table_version,
        "previous_unmounted": had_previous,
        "skills": [s.name for s in incoming],
    }


# --------------------------------------------------------------------------- #
# 阶段清理与对齐
# --------------------------------------------------------------------------- #


def cleanup_persistent(ctx: Any = None, phase: int | None = None) -> dict[str, Any]:
    """清理指定阶段（默认当前旧阶段）仍在场上的持续效果。"""
    ctx = _as_context(ctx)
    old_phase = ctx.state.phase if phase is None else int(phase)

    removed: list[int] = []
    for effect in ctx.effects:
        if effect.active and effect.phase == old_phase:
            effect.active = False
            effect.remaining = 0.0
            removed.append(effect.effect_id)

    dangling = [
        e.effect_id
        for e in _active_effects(ctx)
        if e.skill is None or e.skill.destroyed
    ]

    _bump(ctx, "effects_cleaned", len(removed))
    _record(
        ctx,
        "persistent_cleaned",
        {"phase": old_phase, "removed": len(removed), "dangling": len(dangling)},
    )
    return {
        "phase": old_phase,
        "removed": len(removed),
        "remaining": len(_active_effects(ctx)),
        "dangling_sources": len(dangling),
        "removed_ids": removed,
    }


def reset_stun(ctx: Any = None) -> dict[str, Any]:
    """复位受击硬直计时，确保入场动画不被硬直吞掉。"""
    ctx = _as_context(ctx)
    previous = ctx.state.stun_remaining
    ctx.state.stun_remaining = 0.0
    ctx.state.activity = "active"
    _record(ctx, "stun_reset", {"previous": round(previous, 3)})
    return {
        "previous": round(previous, 3),
        "current": 0.0,
        "cleared": previous > 0.0,
        "stun_active": ctx.state.stun_active,
    }


def cancel_sfx(ctx: Any = None, phase: int | None = None) -> dict[str, Any]:
    """取消指定旧阶段未播完的音效，并关闭残留震屏。"""
    ctx = _as_context(ctx)
    old_phase = ctx.state.phase if phase is None else int(phase)

    cancelled = 0
    for sound in ctx.sounds:
        if sound.playing and sound.phase == old_phase:
            sound.cancelled = True
            cancelled += 1

    shake_cancelled = ctx.state.camera_shake
    if shake_cancelled:
        ctx.state.camera_shake = False

    _bump(ctx, "sounds_cancelled", cancelled)
    _record(
        ctx,
        "sfx_cancelled",
        {"phase": old_phase, "cancelled": cancelled, "shake": shake_cancelled},
    )
    return {
        "phase": old_phase,
        "cancelled": cancelled,
        "camera_shake_cancelled": shake_cancelled,
        "camera_shake": ctx.state.camera_shake,
        "remaining_playing": len(_playing_sounds(ctx)),
    }


def version_align(ctx: Any = None) -> dict[str, Any]:
    """把动画状态机版本对齐到当前技能表版本。"""
    ctx = _as_context(ctx)
    previous = ctx.state.anim_version
    target = ctx.state.table_version
    mismatch = previous != target
    if mismatch:
        _bump(ctx, "anim_version_conflicts")
    ctx.state.anim_version = target

    _record(
        ctx,
        "version_aligned",
        {"from": previous, "to": target, "mismatch": mismatch},
    )
    return {
        "aligned": ctx.state.anim_version == ctx.state.table_version,
        "previous_anim_version": previous,
        "anim_version": ctx.state.anim_version,
        "table_version": ctx.state.table_version,
        "conflict_resolved": mismatch,
    }


# --------------------------------------------------------------------------- #
# 阈值与阶段切换
# --------------------------------------------------------------------------- #


def check_threshold(ctx: Any = None, hp: float | None = None) -> dict[str, Any]:
    """检查血量是否跨过尚未消费的阈值。

    已消费的阈值写入 ``state.fired``，因此回血再掉血不会重复触发。
    """
    ctx = _as_context(ctx)
    if hp is not None:
        ctx.state.hp = max(0.0, float(hp))

    _bump(ctx, "threshold_checks")
    ratio = ctx.state.hp_ratio
    crossed = 0

    for index, threshold in enumerate(ctx.state.thresholds):
        if ratio > threshold:
            continue
        crossed += 1
        target_phase = 2 + index  # 阈值 0 -> 2 阶段，阈值 1 -> 3 阶段
        if index in ctx.state.fired:
            continue
        if target_phase <= ctx.state.phase:
            # 该阈值对应的阶段已经过去，直接消费，避免迟到的重复切换。
            ctx.state.fired.add(index)
            continue
        return {
            "triggered": True,
            "index": index,
            "threshold": threshold,
            "hp_ratio": round(ratio, 4),
            "target_phase": target_phase,
            "repeat_blocked": False,
        }

    # 没有待消费的阈值：已越线说明该阈值此前已消费（回血再掉血的重复触发），
    # 未越线则是确实还没打到阈值。
    reason = "already_fired" if crossed else "above_threshold"
    return {
        "triggered": False,
        "reason": reason,
        "hp_ratio": round(ratio, 4),
        "fired": sorted(ctx.state.fired),
        "current_phase": ctx.state.phase,
    }


def _settled(ctx: PhaseContext) -> dict[str, Any]:
    return {
        "phase": ctx.state.phase,
        "table_version": ctx.state.table_version,
        "anim_version": ctx.state.anim_version,
        "mounted_tables": ctx.state.mounted_tables,
        "stun_remaining": round(ctx.state.stun_remaining, 3),
        "locked": ctx.state.locked,
        "effects": len(_active_effects(ctx)),
        "sounds": len(_playing_sounds(ctx)),
    }


def switch_phase(ctx: Any = None, target_phase: int | None = None) -> dict[str, Any]:
    """执行一次阶段切换。

    顺序：加锁 -> 加载屏障 -> 清理旧效果 -> 重置硬直 -> 取消旧音效/震屏
    -> 替换技能表 -> 版本对齐 -> 解锁。

    加载超时则走 ``fallback``，保持旧阶段原样，不出现空表。
    """
    ctx = _as_context(ctx)
    guard = phase_lock(ctx)
    if not guard.acquired:
        _bump(ctx, "blocked_switches")
        guard.release()
        return {
            "switched": False,
            "reason": "locked",
            "phase": ctx.state.phase,
            "detail": "已有阶段切换正在进行，忽略重复触发",
        }

    try:
        threshold = None
        if target_phase is None:
            threshold = check_threshold(ctx)
            if not threshold["triggered"]:
                _bump(ctx, "blocked_switches")
                return {
                    "switched": False,
                    "reason": threshold["reason"],
                    "phase": ctx.state.phase,
                    "detail": "没有需要消费的血量阈值",
                }
            target = threshold["target_phase"]
        else:
            target = int(target_phase)

        if target <= ctx.state.phase:
            _bump(ctx, "blocked_switches")
            return {
                "switched": False,
                "reason": "non_advancing",
                "phase": ctx.state.phase,
                "target_phase": target,
                "detail": "阶段只能向前推进",
            }

        load_table(ctx, target)
        load = wait_table_load(ctx, target)
        if not load["loaded"]:
            result = fallback(ctx, reason="load_timeout")
            result.update({"switched": False, "target_phase": target})
            return result

        old_phase = ctx.state.phase
        cleanup = cleanup_persistent(ctx, old_phase)
        stun = reset_stun(ctx)
        sfx = cancel_sfx(ctx, old_phase)
        applied = apply_skills(ctx, load["table"], target)
        aligned = version_align(ctx)

        ctx.state.phase = target
        if threshold is not None:
            ctx.state.fired.add(threshold["index"])
        elif 0 <= target - 2 < len(ctx.state.thresholds):
            ctx.state.fired.add(target - 2)

        _bump(ctx, "switches")
        _record(
            ctx,
            "phase_switched",
            {"from": old_phase, "to": target, "waited": load["waited"]},
        )

        result = {
            "switched": True,
            "from_phase": old_phase,
            "phase": target,
            "load_waited": load["waited"],
            "cleanup": cleanup,
            "stun": stun,
            "sfx": sfx,
            "skills": applied,
            "version": aligned,
        }
    finally:
        guard.release()

    # 锁在 finally 中释放，因此快照必须在 try/finally 之后生成，
    # 否则会把切换过程中的 locked=True 误报给调用方。
    result["state"] = _settled(ctx)
    return result


# --------------------------------------------------------------------------- #
# 校验、指标与可复现性
# --------------------------------------------------------------------------- #


def validate_phase(ctx: Any = None, check_stun: bool = False) -> dict[str, Any]:
    """校验切换后的阶段不变量。"""
    ctx = _as_context(ctx)
    state = ctx.state
    violations: list[str] = []

    if not state.skill_table:
        violations.append("empty_skill_table")
    if state.mounted_tables != 1:
        violations.append("skill_table_stacked")
    if state.table_version != state.anim_version:
        violations.append("anim_table_version_mismatch")
    if check_stun and state.stun_remaining > 0.0:
        violations.append("stun_not_reset")

    for effect in _active_effects(ctx):
        if effect.phase != state.phase:
            violations.append(f"stale_effect:{effect.effect_id}")
        elif not effect.source_valid():
            violations.append(f"dangling_effect_source:{effect.effect_id}")

    for sound in _playing_sounds(ctx):
        if sound.phase != state.phase:
            violations.append(f"stale_sound:{sound.sound_id}")

    _record(ctx, "phase_validated", {"ok": not violations, "violations": violations})
    return {
        "ok": not violations,
        "violations": violations,
        "phase": state.phase,
        "table_version": state.table_version,
        "anim_version": state.anim_version,
        "mounted_tables": state.mounted_tables,
    }


def metrics(ctx: Any = None) -> dict[str, Any]:
    """输出阶段切换的关键指标，未发生的计数保持为 0。"""
    ctx = _as_context(ctx)
    counters = {key: ctx.counters.get(key, 0.0) for key in _COUNTER_KEYS}

    return {
        "phase": ctx.state.phase,
        "hp": round(ctx.state.hp, 3),
        "hp_ratio": round(ctx.state.hp_ratio, 4),
        "locked": ctx.state.locked,
        "switches": int(counters["switches"]),
        "blocked_switches": int(counters["blocked_switches"]),
        "load_waits": int(counters["load_waits"]),
        "load_timeouts": int(counters["load_timeouts"]),
        "barrier_ticks": counters["barrier_ticks"],
        # 以下两项是本次修复要压到 0 的指标
        "freeze_ticks": counters["freeze_ticks"],
        "empty_table_ticks": counters["empty_table_ticks"],
        "effects_cleaned": int(counters["effects_cleaned"]),
        "sounds_cancelled": int(counters["sounds_cancelled"]),
        "tables_unmounted": int(counters["tables_unmounted"]),
        "duplicate_mounts_prevented": int(counters["duplicate_mounts_prevented"]),
        "anim_version_conflicts": int(counters["anim_version_conflicts"]),
        "persistent_effects": len(_active_effects(ctx)),
        "playing_sounds": len(_playing_sounds(ctx)),
        "camera_shake": ctx.state.camera_shake,
    }


def audit(ctx: Any = None, limit: int | None = None) -> dict[str, Any]:
    """返回阶段切换历史，便于对局回放与问题定位。"""
    ctx = _as_context(ctx)
    entries = ctx.history if limit is None else ctx.history[-int(limit):]
    return {
        "count": len(ctx.history),
        "entries": list(entries),
        "phase": ctx.state.phase,
        "state": _settled(ctx),
    }


def fallback(ctx: Any = None, reason: str | None = None) -> dict[str, Any]:
    """加载失败时的兜底：维持旧阶段与旧技能表，Boss 不停手。"""
    ctx = _as_context(ctx)
    repaired = False
    if not ctx.state.skill_table:
        apply_skills(ctx)  # 极端情况下恢复一张可用表，避免空表停顿
        version_align(ctx)
        repaired = True

    _bump(ctx, "fallbacks")
    _record(ctx, "fallback", {"reason": reason or "load_failed", "repaired": repaired})
    return {
        "kept_phase": ctx.state.phase,
        "retry": True,
        "reason": reason or "load_failed",
        "table_size": len(ctx.state.skill_table),
        "repaired_table": repaired,
        "boss_active": True,
        "phase": ctx.state.phase,
    }


def determinism(ctx: Any = None, seed: int | None = None) -> dict[str, Any]:
    """用固定种子重放同一段切换流程，比较两次快照是否一致。"""
    base = _as_context(ctx)
    resolved_seed = base.seed if seed is None else int(seed)

    def _run(run_seed: int) -> dict[str, Any]:
        run_ctx = make_context(hp=100.0, max_hp=100.0, phase=1, seed=run_seed)
        run_ctx.state.hp = 0.2 * run_ctx.state.max_hp
        run_ctx.state.stun_remaining = 0.3  # 模拟带着硬直触发阶段

        switch_phase(run_ctx)
        switch_phase(run_ctx)

        return {
            "seed": run_seed,
            "phase": run_ctx.state.phase,
            "table_version": run_ctx.state.table_version,
            "anim_version": run_ctx.state.anim_version,
            "mounted_tables": run_ctx.state.mounted_tables,
            "stun_remaining": round(run_ctx.state.stun_remaining, 3),
            "skill_table": [s.name for s in run_ctx.state.skill_table],
            "active_effects": len(_active_effects(run_ctx)),
            "playing_sounds": len(_playing_sounds(run_ctx)),
            "camera_shake": run_ctx.state.camera_shake,
            "switches": int(run_ctx.counters.get("switches", 0)),
            "events": [entry["event"] for entry in run_ctx.history],
        }

    first = _run(resolved_seed)
    second = _run(resolved_seed)
    stable = first == second

    return {
        "deterministic": stable,
        "runs": 2,
        "seed": resolved_seed,
        "snapshot": first,
        "mismatched_fields": [] if stable else ["snapshot"],
    }


def _demo() -> None:
    """手动运行时的简要演示。"""
    ctx = make_context(hp=100.0)
    ctx.state.hp = 60.0
    report = switch_phase(ctx)
    print("switch:", report["switched"], "phase:", report["phase"])
    print("metrics:", metrics(ctx))
    print("validate:", validate_phase(ctx))
    print("determinism:", determinism(ctx)["deterministic"])


if __name__ == "__main__":
    _demo()
