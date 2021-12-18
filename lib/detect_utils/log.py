import logging
import logging.handlers
import sys
import os
import datetime

# workdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# workdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# cwd = os.path.dirname(os.path.abspath(__file__))

class detlog(object):
    '''
    def __new__(cls,*args,**kw):
        if not hasattr(cls,'_instance'):
            org = super(detlog,cls)
            cls._instance = org.__new__(cls,*args,**kw)
        return cls._instance
    '''
    def __init__(self, model, filename,logID):
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        # if os.path.exists(os.path.join(workdir, 'logs')):
        #     logdir = os.path.join(workdir, 'logs', model,date + '-' + str(logID))
        # elif os.path.exists(os.path.join(cwd,'logs')):
        #     logdir = os.path.join(cwd,'logs',model,date + '-' + str(logID))
        # else:
        #     logdir = os.path.join(cwd,'..','..','logs',model,date + '-' + str(logID))

        # 直接执行 log 的位置
        # logdir = r"/usr/output_dir/logs"
        logdir = os.path.join(r"/usr/output_dir/logs", model, date + '-' + str(logID))

        if not os.path.exists(logdir):
            print(logdir)
            os.makedirs(logdir)
        
        logpath = os.path.join(logdir, filename+'.log')
        loglen = 1024*1024*1024

        self.handler = logging.handlers.RotatingFileHandler(logpath, maxBytes = loglen, backupCount = 10) # 实例化handler
        self.fmt = '%(asctime)s - %(message)s'
        #self.fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
        formatter = logging.Formatter(self.fmt)   # 实例化formatter
        self.handler.setFormatter(formatter)      # 为handler添加formatter
        self.logger = logging.getLogger('tst')    # 获取名为tst的logger
        self.logger.addHandler(self.handler)           # 为logger添加handler
        self.logger.setLevel(logging.DEBUG)

    def info(self,*msgs):
        #sysFrame = sys._getframe()
        #lineNo = sysFrame.f_back.f_lineno
        #funcName = sysFrame.f_back.f_code.co_name
        lineNo,funcName = self.get_cur_info()
        for msg in msgs:
            self.logger.debug('%s:%s - %s' % (funcName,lineNo,msg))

    def get_cur_info(self):
        try:
            raise Exception
        except Exception as e:
            #lineNo = e.__traceback__.tb_lineno
            #funcName = e.__traceback__.tb_frame.f_globals["__file__"]
            if e.__traceback__.tb_frame.f_back.f_back == None:
                lineNo = e.__traceback__.tb_frame.f_back.f_lineno
                funcName = e.__traceback__.tb_frame.f_back.f_code.co_name
            else:
                lineNo = e.__traceback__.tb_frame.f_back.f_back.f_lineno
                funcName = e.__traceback__.tb_frame.f_back.f_back.f_code.co_name
            return lineNo,funcName

    def debug(self,funcName,*msgs,depth=0):
        for msg in msgs:
            self.logger.debug('GOT ERROR: %s---->%s',funcName,msg)

    def recordTime(self,funcName,time):
        self.logger.debug('%s spend Time: %ss' % (funcName,time))
        
    def line(self,*msgs,filler="=",totLen=40):
        lineNo,funcName = self.get_cur_info()
        for msg in msgs:
            msgLen = len(msg)
            fillerLen = int((totLen - msgLen)/2) + (totLen - msgLen)%2
            lenMsg = filler * fillerLen + msg + filler * fillerLen
            self.logger.debug('%s:%s - %s' % (funcName,lineNo,lenMsg)) 
