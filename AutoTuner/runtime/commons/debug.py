import tensordict
import types
import traceback

def strict_get(self, key, default=...):
    # 兼容 get(key) / get(key, default)
    if key in self:
        return self[key]
    # 不管 default 给没给，都报错
    raise KeyError(
        f"[STRICT TensorDict.get] Missing key: {key!r}\n"
        + "".join(traceback.format_stack(limit=20))
    )

# 保存原方法，方便恢复
_ORIG_GET = tensordict.TensorDict.get

def enable_strict_get():
    tensordict.TensorDict.get = strict_get

def disable_strict_get():
    tensordict.TensorDict.get = _ORIG_GET