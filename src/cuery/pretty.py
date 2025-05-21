from functools import partial

from rich import box, panel
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.logging import RichHandler
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.syntax import Syntax
from rich.text import Text

NO_BOTTOM_BOX = box.Box(
    "╭─┬╮\n"  # top
    "│ ││\n"  # head
    "├─┼┤\n"  # headrow
    "│ ││\n"  # mid
    "├─┼┤\n"  # row
    "├─┼┤\n"  # foot row
    "│ ││\n"  # foot
    # "╰─┴╯\n"  # bottom
    "    \n"  # bottom
)

DEFAULT_BOX = box.ROUNDED

panel.Panel = partial(Panel, box=DEFAULT_BOX)

__all__ = [
    "NO_BOTTOM_BOX",
    "DEFAULT_BOX",
    "Console",
    "ConsoleOptions",
    "Panel",
    "Padding",
    "Pretty",
    "RenderResult",
    "RichHandler",
    "Syntax",
    "Text",
    "Group",
]
