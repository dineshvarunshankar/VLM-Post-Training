import threading
import time
import unittest
from types import SimpleNamespace
from scripts.train.shared.evaluation import MedicalVLLMEvaluator

class EvaluationConcurrencyTests(unittest.TestCase):
    def test_bounded_requests_keep_order_and_seeds(self):
        evaluator=MedicalVLLMEvaluator(None,{},None,{}, {},'check',experiment={'evaluation':{'request_batch_size':2}})
        method=getattr(evaluator,'predict_many',None)
        self.assertIsNotNone(method)
        main_thread=threading.get_ident();active=0;peak=0;lock=threading.Lock();seen=[]
        def encode(trainer,row,version):
            self.assertEqual(threading.get_ident(),main_thread)
            return row
        def decode(encoded,version,seed,schema):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active);seen.append((encoded,seed,schema))
            time.sleep(.03 if encoded==0 else .01)
            with lock:active-=1
            return str(encoded)
        evaluator.encode=encode;evaluator.predict_encoded=decode
        values=method(None,list(range(5)),'v',[41,43,45,47,49],['a','b','c','d','e'])
        self.assertEqual(values,['0','1','2','3','4'])
        self.assertEqual(peak,2)
        self.assertEqual(sorted(seen),[(0,41,'a'),(1,43,'b'),(2,45,'c'),(3,47,'d'),(4,49,'e')])

    def test_invalid_request_batch_size(self):
        for value in (0,-1,True,1.5):
            with self.subTest(value=value),self.assertRaises(ValueError):
                MedicalVLLMEvaluator(None,{},None,{}, {},'check',experiment={'evaluation':{'request_batch_size':value}})
