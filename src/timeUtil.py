from time import time

class execution_timer:
    
    def __init__(self, enable = False):
        self.enabled = enable
        self.t_start = {}
        self.t_avg = {}
        self.t_count = {}
        self.g_start = None
        self.g_end = None
        self.g_duration_avg = None
        self.g_sample_count = 0

    def global_start(self):
        if not self.enabled:
            return
        self.g_start = time()
        return

    def global_end(self):
        if not self.enabled:
            return
        self.g_end = time()
        duration = self.g_end-self.g_start
        if (self.g_duration_avg is None):
            self.g_duration_avg = duration
            self.g_sample_count = 1
        else:
            self.g_duration_avg = self.g_duration_avg*self.g_sample_count+duration
            self.g_sample_count = self.g_sample_count +1
            self.g_duration_avg = self.g_duration_avg/self.g_sample_count
        return duration
        
    def start(self, name = None):
        if not self.enabled:
            return
        if name is None:
            return self.global_start()

        self.t_start[name] = time()
        return

    def end(self, name = None):
        if not self.enabled:
            return
        if name is None:
            return self.global_end()

        duration = time()-self.t_start[name] 
        if name in self.t_avg:
            self.t_avg[name] = self.t_avg[name]*self.t_count[name]+duration
            self.t_count[name] = self.t_count[name] + 1
            self.t_avg[name] = self.t_avg[name] / self.t_count[name]
        else:
            self.t_avg[name] = duration
            self.t_count[name] = 1
            
        return duration

    def s(self, n=None):
        return self.start(n)
        
    def e(self, n=None):
        return self.end(n)
        
    def summary(self):
        if not self.enabled:
            return
        #note: sum_time is sum of all fractions not global time
        sum_time = sum(self.t_avg.values())
        #g_duration_avg is time between start() and end() averaged
        total_time = self.g_duration_avg
        #make sure we don't mess with the original copy
        fraction = dict(self.t_avg)
        fraction.update((x, y/total_time) for x, y in fraction.items()) 
        for key,value in fraction.items():
            print(key+'\t\t'+ "{0:.1f}".format(value*100)+' %')

        unaccounted_time = 1-sum_time/total_time
        print('avg frequency = '+"{0:.3f}".format(1/self.g_duration_avg)+'Hz')
        print('unaccounted time = '+"{0:.1f}".format(unaccounted_time)+' %')
        return

#sample usage        
if __name__ == '__main__':
    from time import sleep
    t = execution_timer(True)

    for i in range(1,3):
        t.s()

        t.s('sleep2')
        sleep(0.2)
        t.e('sleep2')

        t.s('sleep1')
        sleep(0.1)
        t.e('sleep1')
        #unaccounted time
        sleep(1)

        t.e()
    t.summary()
        
        
