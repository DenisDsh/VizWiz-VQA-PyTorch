def vqa_accuracy(predicted, true):
    """ Approximation of VQA accuracy metric """
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    agreeing = true.gather(dim=1, index=predicted_index)
    return (agreeing * 0.33333).clamp(max=1)  # * 0.33333 is a good approximation of the VQA metric


class Tracker:

    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        return {k: list(map(list, v)) for k, v in self.data.items()}

    class ListStorage:
        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value


def get_id_from_name(name):
    import re

    n = re.search('VizWiz_(.+?)_', name)
    if n:
        split = n.group(1)

    m = re.search(('VizWiz_%s_(.+?).jpg' % split), name)
    if m:
        found = m.group(1)

    return int(found)
