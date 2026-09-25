"""Team Memory 的稳定公共入口。

外部调用方通常只需要从这里导入 :class:`MemoryConfig` 和
:class:`TeamMemoryService`。其余模块属于实现细节，但仍可供实验代码直接使用。
``__all__`` 用来限制通配符导入，``__version__`` 与项目包版本保持一致。
"""

from .config import MemoryConfig
from .service import TeamMemoryService

__all__ = ["MemoryConfig", "TeamMemoryService"]
__version__ = "0.1.0"
