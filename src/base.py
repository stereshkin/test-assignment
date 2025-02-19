from rich.console import Console
from rich.layout import Layout
from abc import ABC

class BaseDisplayClass(ABC):
    def __init__(self, console: Console, layout: Layout):
        self.console = console
        self.layout = layout
