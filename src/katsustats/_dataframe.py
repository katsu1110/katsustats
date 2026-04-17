from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataFrameLike(Protocol):
    def to_pandas(self) -> Any:
        ...


def _is_pandas_dataframe(obj: Any) -> bool:
    return hasattr(obj, "to_pandas")
