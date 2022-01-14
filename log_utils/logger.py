#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import logging.handlers
import logging.config
import os
from utils.options import args_parser
import time

class InfoFilter(logging.Filter):
    def filter(self, record):
        if record.levelno == logging.INFO:
            return super().filter(record)
        else:
            return 0

args = args_parser()

localtime = time.localtime(time.time())
year = localtime[0]
month = localtime[1]
day = localtime[2]
hour = localtime[3]
logname = args.function+"_"+str(month)+"-"+str(day)+"_"+str(args.dataset)+"_"+str(args.num_users)+"_"+str(args.epochs)+"_"+str(args.frac)+"_"+str(args.seed)
log_dir = "/home/FedRep/logging/"+logname+"/"
attention_file = "/home/FedRep/logging/"+logname+"/attention.txt"

para_record_dir = "/home/FedRep/logging/para"

log_dict = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'standard': {
            'format': '%(asctime)s | %(process)d | %(levelname)s | %(filename)s | %(funcName)s | %(lineno)d | %(message)s'
        }
    },

    'filters': {
        'info_filter': {
            '()': InfoFilter,
        },
    },

    'handlers': {
        'loss': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': os.path.join(log_dir, 'loss.log'),
            # 'maxBytes': 1024 * 1024 * 100,
            'formatter': 'standard',
            "encoding": "utf-8",
            # 'filters': ['info_filter'],
            # 'backupCount': 50
        },
        'cfs_mtrx': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': os.path.join(log_dir, 'cfs_mtrx.log'),
            # 'maxBytes': 1024 * 1024 * 100,
            'formatter': 'standard',
            "encoding": "utf-8",
            # 'backupCount': 50
        },
        'parameter': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': os.path.join(log_dir, 'parameter.log'),
            # 'maxBytes': 1024 * 1024 * 100,
            'formatter': 'standard',
            "encoding": "utf-8",
            # 'backupCount': 50
        },
        'data': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': os.path.join(log_dir, 'data.log'),
            # 'maxBytes': 1024 * 1024 * 100,
            'formatter': 'standard',
            "encoding": "utf-8",
            # 'backupCount': 50
        },
    },

    'loggers': {
        'loss': {
            'handlers': ['loss'],
            'level': 'INFO'
        },
        'cfs_mtrx': {
            'handlers': ['cfs_mtrx'],
            'level': 'INFO'
        },
        'parameter': {
            'handlers': ['parameter'],
            'level': 'INFO'
        },
        'data': {
            'handlers': ['data'],
            'level': 'INFO'
        }
    },
}

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

if not os.path.isdir(para_record_dir):
    os.mkdir(para_record_dir)

open(log_dict['handlers']['loss']['filename'], 'a').close()
open(log_dict['handlers']['cfs_mtrx']['filename'], 'a').close()
open(log_dict['handlers']['parameter']['filename'], 'a').close()
open(log_dict['handlers']['data']['filename'], 'a').close()

logging.config.dictConfig(log_dict)

loss_logger = logging.getLogger('loss')
cfs_mtrx_logger = logging.getLogger('cfs_mtrx')
parameter_logger = logging.getLogger('parameter')
data_logger = logging.getLogger('data')