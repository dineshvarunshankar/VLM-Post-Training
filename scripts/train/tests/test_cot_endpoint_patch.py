"""Opt-in checks against pinned vLLM source; no GPU or model weights needed.

Set VLLM_SOURCE_DIR to an unmodified vLLM 0.30.0 source tree.
"""
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from pydantic import BaseModel, Field, ValidationError
from scripts.train.rl.rollout.patch import patch_state

SOURCE = os.environ.get('VLLM_SOURCE_DIR')
PATCHES = Path(__file__).resolve().parents[1] / 'rl/rollout/patches'
PREFIX = Path('vllm/entrypoints/scale_out/token_in_token_out')

@unittest.skipUnless(SOURCE, 'Set VLLM_SOURCE_DIR to the pinned vLLM 0.30.0 source')
class CoTEndpointPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory()
        cls.root=Path(cls.directory.name)
        for name in ('protocol.py','serving.py','mm_features.py'):
            p=cls.root/PREFIX/name;p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(SOURCE)/PREFIX/name,p)
        for name in ('vllm-video-placeholder-mask.patch','vllm-task-thinking.patch'):
            patch=PATCHES/name
            assert patch_state(cls.root,patch,1)=='missing'
            subprocess.run(['patch','--batch','--forward','--fuzz=0','-p1','-d',str(cls.root),'-i',str(patch)],check=True,capture_output=True)
            assert patch_state(cls.root,patch,1)=='applied'
        # The additional patch must not invalidate the existing patch's state check.
        assert patch_state(cls.root,PATCHES/'vllm-video-placeholder-mask.patch',1)=='applied'

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_request_field_is_optional_strict_boolean(self):
        tree=ast.parse((self.root/PREFIX/'protocol.py').read_text())
        request=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GenerateRequest')
        field=next(n for n in request.body if isinstance(n,ast.AnnAssign) and n.target.id=='enable_thinking')
        cls=ast.ClassDef(name='RequestMode',bases=[ast.Name(id='BaseModel',ctx=ast.Load())],keywords=[],body=[deepcopy(field)],decorator_list=[])
        namespace={'BaseModel':BaseModel,'Field':Field}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'pinned-protocol','exec'),namespace)
        model=namespace['RequestMode']
        self.assertIsNone(model().enable_thinking)
        for value in (True,False):self.assertIs(model(enable_thinking=value).enable_thinking,value)
        for value in ('false',1,0):
            with self.assertRaises(ValidationError):model(enable_thinking=value)

    def test_endpoint_forwards_medical_and_direct_reasoning_state(self):
        tree=ast.parse((self.root/PREFIX/'serving.py').read_text())
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='generate' and ast.unparse(n.func.value)=='self.engine_client')
        fn=ast.FunctionDef(name='forward',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self'),ast.arg(arg='request')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[ast.Return(value=deepcopy(call))],decorator_list=[])
        namespace={name:None for name in ('engine_input','sampling_params','request_id','lora_request','trace_headers','data_parallel_rank','session_id')}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'pinned-serving','exec'),namespace)
        client=SimpleNamespace(engine_client=SimpleNamespace(generate=lambda *a,**kw:kw))
        for thinking in (True,False,None):
            result=namespace['forward'](client,SimpleNamespace(enable_thinking=thinking,priority=0))
            self.assertIs(result['reasoning_ended'],not thinking if thinking is not None else None)
            self.assertEqual(result['reasoning_parser_kwargs'],{'chat_template_kwargs':{'enable_thinking':thinking}} if thinking is not None else None)

    def test_pinned_engine_defers_grammar_until_native_reasoning_end(self):
        # Execute the actual pinned grammar-gating method. The reasoner supplies
        # model-native boundary detection; no GPU/grammar backend is required.
        path=Path(SOURCE)/'vllm/v1/structured_output/__init__.py'
        tree=ast.parse(path.read_text())
        method=deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_get_constraint_start'))
        method.returns=None
        for arg in method.args.args:arg.annotation=None
        namespace={'ParserEngineReasoningAdapter':type('UnusedAdapter',(),{})}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'pinned-grammar-gating','exec'),namespace)
        for family,boundary in (('qwen3',20),('gemma4',30)):
            reasoner=SimpleNamespace(is_reasoning_end=lambda ids:False,is_reasoning_end_streaming=lambda ids,delta:boundary in ids)
            manager=SimpleNamespace(enable_in_reasoning=False,_get_reasoner=lambda request:reasoner)
            gate=lambda request,tokens:namespace['_get_constraint_start'](manager,request,tokens)
            request=SimpleNamespace(structured_output_request=SimpleNamespace(reasoning_ended=False),prompt_token_ids=[1],all_token_ids=[1])
            with self.subTest(family=family):
                self.assertEqual(gate(request,[]),1)  # no grammar on next reasoning token
                self.assertEqual(gate(request,[8,9]),3)  # both remain free-form
                self.assertEqual(gate(request,[8,boundary,9]),2)  # final token constrained
                request.structured_output_request.reasoning_ended=True
                self.assertEqual(gate(request,[]),0)  # direct camera JSON from first token

if __name__=='__main__':unittest.main()
