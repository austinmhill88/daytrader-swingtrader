from abc import ABC, abstractmethod
from typing import List
from src.models import Signal

class Strategy(ABC):
    name: str

    @abstractmethod
    def warmup(self, historical_df):
        pass

    @abstractmethod
    def on_bar(self, bar) -> List[Signal]:
        pass

    @abstractmethod
    def on_end_of_day(self):
        pass