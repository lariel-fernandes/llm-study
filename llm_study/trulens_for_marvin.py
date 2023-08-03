from typing import Type, Generic, TypeVar

from pydantic import BaseModel
from trulens_eval import Tru

from llm_study.trulens_freeform import FreeFormApp

T = TypeVar("T", bound=BaseModel)


class TrulensForMarvin(Generic[T]):

    def __init__(self, model: Type[T], **kwargs):
        self.model = model
        self.chain = FreeFormApp(
            fn=lambda inputs: model(inputs["text"]),
            app_id=kwargs.pop("app_id", self.__repr__()),
            db=kwargs.pop("db", Tru().db),
            **kwargs
        )

    def __call__(self, text: str, **kwargs) -> T:
        return self.chain(inputs={"text": text, **kwargs})

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.model.__name__}]"


def trulens_for_marvin(**kwargs):
    def decorator(model: Type[BaseModel]):
        return TrulensForMarvin(model, **kwargs)
    return decorator
