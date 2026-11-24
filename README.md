# pairwise-boss-phase-skillhot

## 领域

游戏、图形与多媒体

## 主题

Boss 阶段切换与技能表热替换

## 说明

加载屏障、持续效果清理与阶段锁。

初始状态下 `gameplay/boss_phase_skillhot.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
