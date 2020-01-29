import queue
import subprocess
import threading


class WorkerPool(object):

    def __init__(self, num_workers):
        self.q = queue.Queue()
        self.threads = []
        self.num_workers = num_workers

        for i in range(self.num_workers):
            t = threading.Thread(target=self.worker)
            t.start()
            self.threads.append(t)


    def do_work(self, item):
        raise NotImplementedError


    def worker(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            self.do_work(item)
            self.q.task_done()


    def add_work(self, item):
        self.q.put(item)


    def join(self):
        self.q.join()

        # stop workers
        for i in range(self.num_workers):
            self.q.put(None)
        for t in self.threads:
            t.join()


class BashCommandWorkerPool(WorkerPool):

    def __init__(self, num_workers, failed_cmds_file_path=None):
        super().__init__(num_workers)
        self.failed_cmds_file_path = failed_cmds_file_path

        # Set up file to store commands that failed
        if self.failed_cmds_file_path is not None:
            self.failed_cmds_file = open(failed_cmds_file_path, 'w')
            self.failed_cmds_file_lock = threading.Lock()


    def do_work(self, cmd):
        try:
            print('Running command "{}"'.format(cmd))
            subprocess.check_call(cmd, shell=True)
            print('Finished command "{}"'.format(cmd))
        except subprocess.CalledProcessError:
            if self.failed_cmds_file_path is not None:
                with self.failed_cmds_file_lock:
                    self.failed_cmds_file.write('{}\n'.format(cmd))
                    self.failed_cmds_file.flush()


    def join(self):
        super().join()

        if self.failed_cmds_file_path is not None:
            self.failed_cmds_file.close()