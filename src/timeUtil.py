from time import time

class execution_timer:
    
    def __init__(self)
        self.enabled = True
        self.table = {}
        self.t_start = {}
        self.t_end = {}
        return

    def start(self, name):
        t_start[name] = time()
        return

    def end(self, name):
        t_end[name] = time()
        table[name] = t_end[name] - t_start[name]
        return

    def get(self, name):
        return table[name]

    def s(self, n):
        return self.start(n)
        
    def e(self, n):
        return self.end(n)
        
    def g(self, n):
        return self.get(n)

    def clean(self):
        self.table = {}
        self.t_start = {}
        self.t_end = {}
        return

    def c(self):
        return self.clean()

    def summary(self):
        total_time = sum(self.table.values)
        for key,value in self.table.items():
            print(key+'\t\t'+str(values/total_time))

        print('total = '+str(total_time))

        
        
        
