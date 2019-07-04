import warnings
from torch.utils.data import DataLoader, Dataset


class WrappedDataset(Dataset):
    def __init__(self, dataset, transforms):
        self.transforms = transforms
        self.dataset = dataset

        self.is_indexable = False
        if hasattr(self.dataset, "__getitem__") and not (hasattr(self.dataset, "use_next") and self.dataset.use_next \
                                                         is True):
            self.is_indexable = True

    def __getitem__(self, index):

        if not self.is_indexable:
            item = next(self.dataset)
        else:
            item = self.dataset[index]

        item = self.transforms(**item)
        return item

    def __len__(self):
        return len(self.dataset)


class MultiThreadedDataLoader(object):
    def __init__(self, data_loader, transform, num_processes, shuffle=True, timeout=120, **kwargs):
        self.transform = transform
        self.timeout = timeout

        self.cntr = 1
        self.ds_wrapper = WrappedDataset(data_loader, transform)

        self.generator = DataLoader(self.ds_wrapper, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                                    num_workers=num_processes, pin_memory=False, drop_last=False, timeout=0)

        self.num_processes = num_processes
        self.iter = None

    def __iter__(self):
        self.iter = iter(self.generator)
        return self

    def del_iter(self):
        del self.iter

    def __next__(self):
        try:
            return next(self.iter)
        except RuntimeError:
            print("Queue is empty, None returned")
            warnings.warn("Queue is empty, None returned")
            raise StopIteration
            return

    def renew(self):
        if self.cntr > 1:
            self.generator.timeout = self.timeout
        self.cntr += 1

    def restart(self):
        pass

    def kill_iterator(self):
        pass
