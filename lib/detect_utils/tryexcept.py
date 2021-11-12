import functools
import time
import traceback
import os
def try_except(errors=(Exception,),log=None,default="default"):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self,*args, **kwargs):
                try:
                    return func(self,*args, **kwargs)
                except errors as ex:
                    err_info = str(traceback.format_exc())
                    if log != None and callable(log.debug):
                        log.debug(func.__name__,err_info)
                    elif self.log != None and callable(self.log.debug):
                        self.log.debug(func.__name__,err_info)
                    raise ValueError("error")
            return wrapper
        return decorator

def timeStamp(log=None):
    def time_func(func1):
        @functools.wraps(func1)
        def inner1(self,*args,**kwargs):
            start_time = time.time()
            result = func1(self,*args,**kwargs)
            stop_time = time.time()
            if log != None and callable(log.debug):
                log.recordTime(func1.__name__,stop_time-start_time)
            elif self.log != None and callable(self.log.debug):
                self.log.recordTime(func1.__name__,str(round(stop_time-start_time,2)))
            return result
        return inner1
    return time_func
