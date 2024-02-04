from typing import List


class Callback:

    def on_epoch_start(self, trainer, **kwargs):
        return

    def on_epoch_end(self, trainer, **kwargs):
        return

    def on_train_epoch_start(self, trainer, **kwargs):
        return

    def on_train_epoch_end(self, trainer, **kwargs):
        return

    def on_valid_epoch_start(self, trainer, **kwargs):
        return

    def on_valid_epoch_end(self, trainer, **kwargs):
        return

    def on_train_step_start(self, trainer, **kwargs):
        return

    def on_train_step_end(self, trainer, **kwargs):
        return

    def on_valid_step_start(self, trainer, **kwargs):
        return

    def on_valid_step_end(self, trainer, **kwargs):
        return


class CallbackRunner:
    def __init__(self, callbacks: List[Callback], trainer):
        self.trainer = trainer
        self.callbacks = callbacks

    def __call__(self, current_state, **kwargs):
        for callback in self.callbacks:
            # print(callback, current_state.value)
            _ = getattr(callback, current_state.value)(self.trainer, **kwargs)
