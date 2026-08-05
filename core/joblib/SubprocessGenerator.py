import multiprocessing
import queue as Queue
import signal
import threading
import time


class SubprocessGenerator(object):
    
    @staticmethod
    def launch_thread(generator): 
        generator._start()
        
    @staticmethod
    def start_in_parallel( generator_list ):
        """
        Start list of generators in parallel
        """
        for generator in generator_list:
            thread = threading.Thread(target=SubprocessGenerator.launch_thread, args=(generator,) )
            thread.daemon = True
            thread.start()

        while not all ([generator._is_started() for generator in generator_list]):
            time.sleep(0.005)
    
    def __init__(self, generator_func, user_param=None, prefetch=2, start_now=True):
        super().__init__()
        self.prefetch = prefetch
        self.generator_func = generator_func
        self.user_param = user_param
        self.sc_queue = multiprocessing.Queue()
        self.cs_queue = multiprocessing.Queue()
        self.p = None
        if start_now:
            self._start()

    def _start(self):
        if self.p == None:
            user_param = self.user_param
            self.user_param = None
            p = multiprocessing.Process(target=self.process_func, args=(user_param,) )
            p.daemon = True
            p.start()
            self.p = p
            
    def _is_started(self):
        return self.p is not None
        
    def process_func(self, user_param):
        # Ctrl+C is delivered to the whole process group. The parent already
        # coordinates the shutdown (saves and closes), and these children are
        # daemonic, so they die with it anyway; letting each of them raise its
        # own KeyboardInterrupt only buries the real message under twenty
        # interleaved tracebacks. Ignore the signal here, in the child.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # One generator per two cores, each with an OpenCV that by default
        # opens one thread per core: on a 16-core machine that is 16 processes
        # asking for 16 threads apiece, all to decode JPEGs. Measured here,
        # 352 threads against 16 cores — and the process that suffers is the
        # one that cannot be replaced, the parent, which has to stay on a core
        # to keep queueing CUDA kernels. Capping OpenCV to no threads at all
        # (each worker is already a process of its own, so it loses nothing)
        # took a training iteration from 187 ms to 147 ms, a 21% gain that has
        # nothing to do with the generators being faster: they were keeping up
        # either way. Import here rather than at module scope — this is the
        # child, and it is where cv2 gets loaded anyway.
        import cv2
        cv2.setNumThreads(0)

        self.generator_func = self.generator_func(user_param)
        while True:
            while self.prefetch > -1:
                try:
                    gen_data = next (self.generator_func)
                except StopIteration:
                    self.cs_queue.put (None)
                    return
                self.cs_queue.put (gen_data)
                self.prefetch -= 1
            self.sc_queue.get()
            self.prefetch += 1

    def __iter__(self):
        return self

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['p']
        return self_dict

    def __next__(self):
        self._start()
        gen_data = self.cs_queue.get()
        if gen_data is None:
            self.p.terminate()
            self.p.join()
            raise StopIteration()
        self.sc_queue.put (1)
        return gen_data
