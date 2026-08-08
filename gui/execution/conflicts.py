"""Which steps may run at the same time.

Two jobs can coexist only if neither touches (produces or modifies) an
artifact the other uses in any way. The relations come from the step
catalog, which mirrors the workflow graph.
"""


def _touches(step):
    return set(step.produces) | set(step.modifies)


def _uses(step):
    return set(step.consumes) | _touches(step)


def conflict(a, b):
    contested = (_touches(a) & _uses(b)) | (_touches(b) & _uses(a))
    for name in sorted(contested):
        return name
    return None
