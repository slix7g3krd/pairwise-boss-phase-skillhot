import pytest
import gameplay.boss_phase_skillhot as mod

CASES = [
    ("wait_table_load", (1,)),
    ("cleanup_persistent", (1,)),
    ("reset_stun", (1,)),
    ("phase_lock", (1,)),
    ("cancel_sfx", (1,)),
    ("version_align", (1,)),
    ("check_threshold", (1,)),
    ("switch_phase", (1,)),
    ("load_table", (1,)),
    ("apply_skills", (1,)),
    ("validate_phase", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
