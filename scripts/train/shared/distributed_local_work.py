"""Keep distributed ranks synchronized during uneven CPU/HTTP work.

The operation must not execute distributed collectives or use the trainer model.
Model forwards and weight transfers must stay on their existing synchronous path.
"""
from threading import Event, Thread


def run_local_work(label, operation, *, interval=1.0):
    import torch.distributed as distributed
    if not distributed.is_initialized() or distributed.get_world_size() == 1:
        return operation()
    if interval <= 0:
        raise ValueError('Progress interval must be positive')
    done = Event()
    outcome = {}

    def work():
        try:
            outcome['result'] = operation()
        except BaseException as error:
            outcome['error'] = error
        finally:
            done.set()

    # If a collective fails, do not block process shutdown on an HTTP request.
    thread = Thread(target=work, daemon=True, name='evaluation-local-work')
    thread.start()
    while True:
        finished = done.wait(interval)
        status = {'rank': distributed.get_rank(), 'done': finished,
                  'error': type(outcome['error']).__name__ if finished and 'error' in outcome else None}
        statuses = [None] * distributed.get_world_size()
        distributed.all_gather_object(statuses, status)
        if all(item['done'] for item in statuses):
            break
    thread.join()
    failures = [str(item['rank']) for item in statuses if item['error']]
    if failures:
        raise RuntimeError(f'Evaluation {label} failed on rank(s) {", ".join(failures)}') from outcome.get('error')
    return outcome['result']
