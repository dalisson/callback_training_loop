class CustomList():
    def __init__(self, itens):
        self.item = itens
    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)): return self.item[idx]
        elif isinstance(idx[0], bool):
            assert len(idx) <= len(self.item)
            return [x for x, y in zip(self.item, idx) if y]
    def __getattr__(self, attr):
        itens = []
        for item in self.item:
            itens.append(getattr(item, attr))
        return itens
    def __len__(self): return len(self.item)
    def __iter__(self):return iter(self.item)
    def __delitem__(self, i): del(self.item[i])
    def __setitem__(self, i, item): self.item[i] = item