from time import time

class execution_timer:
    
    def __init__(self, enable = False):
        self.enabled = enable
        self.table = {}
        self.t_start = {}
        self.t_end = {}
        return

    def start(self, name):
        if not self.enabled:
            return
        self.t_start[name] = time()
        return

    def end(self, name):
        if not self.enabled:
            return
        self.t_end[name] = time()
        self.table[name] = self.t_end[name] - self.t_start[name]
        return

    def get(self, name):
        if not self.enabled:
            return
        return self.table[name]

    def s(self, n):
        return self.start(n)
        
    def e(self, n):
        return self.end(n)
        
    def g(self, n):
        return self.get(n)

    def clean(self):
        if not self.enabled:
            return
        self.table = {}
        self.t_start = {}
        self.t_end = {}
        return

    def c(self):
        return self.clean()

    def summary(self):
        if not self.enabled:
            return
        total_time = sum(self.table.values())
        fraction = dict(self.table)
        fraction.update((x, y/total_time) for x, y in fraction.items()) 
        for key,value in fraction.items():
            print(key+'\t\t'+ str(value))

        print('total = '+str(total_time))
        unaccounted_time = 1-sum(fraction.values())
        print('unaccounted time = '+str(unaccounted_time))
        return

        
#debugging stuff
if __name__ == '__main__':
    from time import sleep
    t = execution_timer(True)
    t.s('sleep2')
    sleep(2)
    t.e('sleep2')
    t.s('sleep1')
    sleep(1)
    t.e('sleep1')
    sleep(1)
    t.summary()
        
        
