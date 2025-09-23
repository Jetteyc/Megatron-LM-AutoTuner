from collections import defaultdict


def nested_dict():
    return defaultdict(nested_dict)


class NestedDict(dict):
    """A dict-like object that auto-creates nested dicts and supports merge."""

    def __getitem__(self, key):
        # 类似 defaultdict，自动创建子字典
        if key not in self:
            self[key] = NestedDict()
        return super().__getitem__(key)

    def merge(self, other):
        """Merge another NestedDict into this one (structure assumed same)."""
        for k, v in other.items():
            if isinstance(v, dict) and isinstance(self.get(k), dict):
                # 递归合并
                self[k].merge(v)
            else:
                # 覆盖值
                self[k] = v
        return self

    def to_dict(self):
        """Convert NestedDict into a normal dict recursively."""
        return {
            k: (v.to_dict() if isinstance(v, NestedDict) else v)
            for k, v in self.items()
        }
