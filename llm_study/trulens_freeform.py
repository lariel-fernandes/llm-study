"""
# Basic input output instrumentation and monitoring.
"""
import logging
from datetime import datetime
from typing import Callable, ClassVar, Sequence

from pydantic import Field
from trulens_eval.app import App
from trulens_eval.instruments import Instrument
from trulens_eval.provider_apis import Endpoint
from trulens_eval.schema import Cost
from trulens_eval.schema import RecordAppCall
from trulens_eval.util import Class
from trulens_eval.util import FunctionOrMethod

logger = logging.getLogger(__name__)


class FreeFormAppWrapper:

    def __init__(self, fn: Callable):
        self._fn = fn

    def _call(self, inputs: dict):
        return self._fn(inputs=inputs)


class FreeFormAppInstrument(Instrument):

    class Default:
        MODULES = {FreeFormAppWrapper.__module__}
        CLASSES = {FreeFormAppWrapper}
        METHODS = {"_call": lambda o: isinstance(o, FreeFormAppWrapper)}

    def __init__(self):
        super().__init__(
            root_method=FreeFormApp.call_with_record,
            modules=FreeFormAppInstrument.Default.MODULES,
            classes=FreeFormAppInstrument.Default.CLASSES,
            methods=FreeFormAppInstrument.Default.METHODS,
        )


class FreeFormApp(App):
    app: FreeFormAppWrapper

    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.of_callable(FreeFormApp.call_with_record),
        const=True
    )

    def __init__(self, fn: Callable, **kwargs):
        """
        Wrap a callable for monitoring.

        Arguments:
        - fn: A callable
        - More args in App
        - More args in AppDefinition
        - More args in WithClassInfo
        """
        super().update_forward_refs()
        app = FreeFormAppWrapper(fn)
        kwargs['app'] = app
        kwargs['root_class'] = Class.of_object(app)
        kwargs['instrument'] = FreeFormAppInstrument()
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.call_with_record(*args, **kwargs)[0]

    def call_with_record(self, inputs: dict):
        # Wrapped calls will look this up by traversing the call stack
        record: Sequence[RecordAppCall] = []

        ret = None
        error = None

        cost: Cost = Cost()

        start_time = None

        try:
            start_time = datetime.now()
            ret, cost = Endpoint.track_all_costs_tally(
                lambda: self.app._call(inputs=inputs)  # noqa
            )
            end_time = datetime.now()

        except BaseException as e:
            end_time = datetime.now()
            error = e
            logger.error(f"App raised an exception: {e}")

        assert len(record) > 0, "No information recorded in call."

        main_input = inputs
        if len(main_input) == 1:
            main_input = list(main_input.values())[0]

        ret_record_args = {"main_input": main_input}

        if ret is not None:
            ret_record_args['main_output'] = ret

        ret_record = self._post_record(
            ret_record_args, error, cost, start_time, end_time, record
        )

        return ret, ret_record
