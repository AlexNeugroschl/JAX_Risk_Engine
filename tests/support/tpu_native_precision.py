"""Probe TPU native candidates directly, without engine dispatch or fallback.

Run on one already allocated v6e VM. No cloud resources are created here.
`--phase all` isolates raw, Pallas, and fused investigations with process limits.
CPU smoke execution validates the diagnostic only, never TPU native support.
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import importlib.metadata
import io
import itertools
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import numpy as np


SHAPES = ((256,256,256), (32,32,8), (32,32,1), (33,13,7), (2,9,13,7))
F8 = ('float8_e5m2', 'float8_e4m3b11fnuz', 'float8_e4m3fn')


def atomic(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def format_spec(name):
    import jax.numpy as jnp
    from engine.precision import FORMATS
    if name in FORMATS:
        return FORMATS[name]
    dtype = jnp.dtype(getattr(jnp,name))
    info = jnp.finfo(dtype)
    return replace(FORMATS['float8_e4m3fn'], name=name, base=name, dtype=dtype,
        min_exponent=int(info.minexp), max=float(info.max), mantissa_bits=int(info.nmant))


def routes():
    pairs = [(name,name) for name in F8]
    pairs += [('float8_e5m2','float8_e4m3b11fnuz'), ('float8_e4m3b11fnuz','float8_e5m2'),
              ('float8_e5m2','float8_e4m3fn'), ('float8_e4m3fn','float8_e5m2')]
    result = [dict(lhs=a,rhs=b,algorithm=algorithm) for a,b in pairs
              for algorithm in ('ANY_F8_ANY_F8_F32','DEFAULT')]
    result += [dict(lhs=name,rhs=name,algorithm=algorithm) for name,algorithms in (
        ('bfloat16',('BF16_BF16_F32','DEFAULT')),
        ('float16',('F16_F16_F32','HIGHEST','DEFAULT')),
        ('float32',('F32_F32_F32','HIGHEST')),
        ('tensorfloat32',('TF32_TF32_F32',)),
        ('bfloat16_3x',('BF16_BF16_F32_X3',))) for algorithm in algorithms]
    return result


def requested_dtype(name):
    return 'float32' if name in ('tensorfloat32','bfloat16_3x') else name


def precision(name):
    from jax import lax
    return lax.Precision.HIGHEST if name == 'HIGHEST' else getattr(lax.DotAlgorithmPreset,name)


def matrix_operations(hlo):
    """Track SSA types in each computation, including operands of actual dots.

    A parameter dtype alone is not evidence of a low-format product. Custom calls
    and vector reductions without typed matrix operations remain unverified here.
    """
    types, found = {}, []
    for line in hlo.splitlines():
        if line.startswith('%') or line.startswith('ENTRY '):
            types = {}
        instruction = re.match(r'\s*(?:ROOT )?%([^\s=]+)\s*=\s*([a-z0-9]+)\[', line)
        if instruction:
            types[instruction[1]] = instruction[2]
        operation = re.search(r'\b(dot|convolution)\(%([^,\s]+), %([^,)\s]+)\)',line)
        if operation:
            found.append(dict(operation=operation[1], lhs=types.get(operation[2]),
                rhs=types.get(operation[3]), output=instruction[2] if instruction else None,
                text=line.strip()))
    return found


def lowering_status(route, hlo, declared):
    matrix = matrix_operations(hlo)
    short = {'float32':'f32','float16':'f16','bfloat16':'bf16',
             'float8_e5m2':'f8e5m2','float8_e4m3fn':'f8e4m3fn',
             'float8_e4m3b11fnuz':'f8e4m3b11fnuz'}
    if route['lhs'] in ('float32','tensorfloat32','bfloat16_3x'):
        return dict(status='precision_algorithm_control', matrix_operations=matrix,
                    instruction_qualification='not_established')
    wanted = (short[route['lhs']],short[route['rhs']])
    matches = [m for m in matrix if (m['lhs'],m['rhs']) == wanted]
    if matrix and len(matches) == len(matrix):
        status = 'requested_operand_types_in_optimized_matrix_operations'
    elif matrix:
        status = 'different_operand_types_in_optimized_matrix_operations'
    else:
        status = 'no_typed_matrix_operation_observed'
    return dict(status=status, declared_no_cast_support=declared, matrix_operations=matrix,
                instruction_qualification='not_established')


def values(shape, pattern, name):
    *batch,m,k,n = shape
    rng = np.random.default_rng(91)
    a = rng.uniform(-2,2,(*batch,m,k)).astype(np.float32)
    b = rng.uniform(-2,2,(*batch,k,n)).astype(np.float32)
    if pattern == 'zero':
        a.fill(0.)
    elif pattern == 'cancellation':
        a[...,1::2] = a[...,:2*(k//2):2]
        b[...,1::2,:] = -b[...,:2*(k//2):2,:]
    elif pattern == 'small_grid':
        fmt = format_spec(name)
        a = (np.rint(a*4)+np.float32(.5))*np.float32(2.**(fmt.min_exponent-fmt.mantissa_bits))
    return a,b


def host_quantize(x, name):
    from tests.support.product_boundaries import operand_reference
    if name in ('float32','bfloat16_3x'):
        return x
    fmt = format_spec(name)
    if name == 'tensorfloat32':
        q,_ = operand_reference(x,fmt,-1,1,scaling=False)
        return q.astype(np.float32)
    # Inputs are FP32; NumPy/ml_dtypes converts them directly to the selected
    # encoding. These exact buffers, not freshly generated scenarios, reach TPU.
    return x.astype(fmt.dtype)


def comparison(actual, expected, scale, *, x3=False):
    from tests.support.precision_devices import normalized_error
    error = normalized_error(actual,expected,scale)
    tolerance = 4*2.**-17 if x3 else 64*np.finfo(np.float32).eps/2
    return dict(passes=bool(np.isfinite(error) and error<=tolerance),
        normalized_error=error if np.isfinite(error) else None,tolerance=float(tolerance))


def raw_phase(device, out, *, smoke=False):
    import jax
    import jax.numpy as jnp
    import jax.experimental.pallas.tpu as pltpu
    from engine.precision import Arithmetic
    from tests.support.tpu_precision_survey import reference
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E,1)
    result = dict(complete=False,device=device.device_kind,phase='raw',routes=[],
        scope='direct JAX operations; engine capability table and emulation dispatch are not called',
        aliases='Base and _1x names share the same raw dtype kernel; fused phase tests one/two shared terms',
        fp64=dict(native_support=False,action='no software FP64 execution',guard_results=[]))
    if device.platform=='tpu':
        from engine.precision.execution import validate_product
        for compute,accumulate in (('float64','float64'),('float32','float64')):
            try:
                validate_product(compute,accumulate,'native',device=device)
                result['fp64']['guard_results'].append(dict(compute=compute,accumulate=accumulate,refused=False))
            except ValueError as error:
                result['fp64']['guard_results'].append(dict(compute=compute,accumulate=accumulate,refused=True,reason=str(error)))
    selected = routes() if not smoke else [r for r in routes() if r['lhs']=='float32']
    with jax.default_device(device):
        for route in selected:
            row = dict(**route,cases=[])
            row['declared_no_cast_support'] = info.is_matmul_supported(
                getattr(jnp,requested_dtype(route['lhs'])),getattr(jnp,requested_dtype(route['rhs'])))
            result['routes'].append(row)
            for shape in (((9,13,7),) if smoke else SHAPES):
                a,b = values(shape,'normal',route['lhs'])
                qa,qb = host_quantize(a,route['lhs']),host_quantize(b,route['rhs'])
                program = jax.jit(lambda x,y:jnp.matmul(x,y,precision=precision(route['algorithm']),
                    preferred_element_type=jnp.float32))
                shape_result = dict(shape=list(shape),compiled=False,operations=[])
                row['cases'].append(shape_result)
                label = f"raw-{len(result['routes'])-1:02d}-{len(row['cases'])-1:02d}"
                try:
                    args = (jax.device_put(qa,device),jax.device_put(qb,device))
                    lowered = program.lower(*args)
                    (out/(label+'.stablehlo.txt')).write_text(str(lowered.compiler_ir()),encoding='utf-8')
                    compiled = lowered.compile()
                    hlo = compiled.as_text()
                    (out/(label+'.hlo.txt')).write_text(hlo,encoding='utf-8')
                    shape_result.update(compiled=True,hlo=label+'.hlo.txt',
                        lowering=lowering_status(route,hlo,row['declared_no_cast_support']))
                except Exception as error:
                    shape_result.update(status='compile_rejected',error=f'{type(error).__name__}: {error}')
                    atomic(out/'summary.json',result)
                    continue
                patterns = ['normal','zero','cancellation']
                if route['lhs'].startswith(('float16','float8')):
                    patterns.append('small_grid')
                for pattern in patterns:
                    a,b = values(shape,pattern,route['lhs'])
                    qa,qb = host_quantize(a,route['lhs']),host_quantize(b,route['rhs'])
                    operation = dict(pattern=pattern,passes=False)
                    buffers = dict(qa=qa.astype(np.float32),qb=qb.astype(np.float32),
                        qa_bytes=qa.view(np.uint8),qb_bytes=qb.view(np.uint8))
                    try:
                        args = (jax.device_put(qa,device),jax.device_put(qb,device))
                        input_match = all(np.array_equal(np.ascontiguousarray(np.asarray(v)).view(np.uint8),
                                                         np.ascontiguousarray(x).view(np.uint8))
                                          for v,x in zip(args,(qa,qb)))
                        actual = np.asarray(compiled(*args),np.float64)
                        again = np.asarray(compiled(*args),np.float64)
                        expected = qa.astype(np.float64)@qb.astype(np.float64)
                        scale = np.abs(qa.astype(np.float64))@np.abs(qb.astype(np.float64))
                        if route['lhs']=='bfloat16_3x':
                            expected,scale = reference(a,b,Arithmetic('bfloat16_3x','float32'),'right',None)
                        operation.update(comparison(actual,expected,scale,x3=route['lhs']=='bfloat16_3x'))
                        operation.update(identical_operand_bytes=input_match,repeatable=np.array_equal(actual,again))
                        operation['passes'] = operation['passes'] and input_match and operation['repeatable']
                        buffers.update(actual=actual,repeated=again,expected=expected,scale=scale)
                    except Exception as error:
                        operation.update(status='execution_error',error=f'{type(error).__name__}: {error}')
                    filename = label+'-'+pattern+'.npz'
                    np.savez_compressed(out/filename,**buffers)
                    operation['buffers'] = filename
                    shape_result['operations'].append(operation)
                    atomic(out/'summary.json',result)
            print(route['lhs'],route['rhs'],route['algorithm'],
                  'compiled',sum(c['compiled'] for c in row['cases']),
                  'passed',sum(o['passes'] for c in row['cases'] for o in c['operations']),flush=True)
    result['complete'] = True
    atomic(out/'summary.json',result)


def pallas_phase(device,out,*,smoke=False):
    import jax
    from jax import lax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    result = dict(complete=False,phase='pallas',device=device.device_kind,routes=[])
    pairs = [(r['lhs'],r['rhs']) for r in routes() if r['algorithm']=='DEFAULT']
    if smoke: pairs=[('bfloat16','bfloat16')]
    for name,other in pairs:
        row = dict(lhs=name,rhs=other,compiled=False,operations=[])
        result['routes'].append(row)
        label = f'pallas-{len(result["routes"])-1:02d}'
        try:
            shape=(256,256,256)
            a,b = values(shape,'normal',name)
            qa,qb = host_quantize(a,name),host_quantize(b,other)
            def kernel(ar,br,cr):
                cr[...] = lax.dot_general(ar[...],br[...],(((1,),(0,)),((),())),
                    precision=lax.Precision.DEFAULT,preferred_element_type=jnp.float32)
            fn = pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct((256,256),jnp.float32),
                # The engine enables x64 globally; TPU index maps require i32.
                grid=(1,),in_specs=[pl.BlockSpec((256,256),lambda i:(jnp.int32(0),jnp.int32(0)))]*2,
                out_specs=pl.BlockSpec((256,256),lambda i:(jnp.int32(0),jnp.int32(0))),debug=True,
                interpret=smoke)
            capture=io.StringIO()
            try:
                with jax.default_device(device),redirect_stdout(capture):
                    args=(jax.device_put(qa,device),jax.device_put(qb,device))
                    lowered=jax.jit(fn).lower(*args)
                    compiled=lowered.compile()
            finally:
                (out/(label+'.mosaic.txt')).write_text(capture.getvalue(),encoding='utf-8')
            (out/(label+'.hlo.txt')).write_text(compiled.as_text(),encoding='utf-8')
            row.update(compiled=True,mosaic=label+'.mosaic.txt',hlo=label+'.hlo.txt',
                       instruction_qualification='Mosaic compiler IR only; physical assembly not established')
            for pattern in ('normal','zero','cancellation'):
                a,b=values(shape,pattern,name)
                qa,qb=host_quantize(a,name),host_quantize(b,other)
                with jax.default_device(device):
                    args=(jax.device_put(qa,device),jax.device_put(qb,device))
                    actual=np.asarray(compiled(*args),np.float64)
                    again=np.asarray(compiled(*args),np.float64)
                expected=qa.astype(np.float64)@qb.astype(np.float64)
                scale=np.abs(qa.astype(np.float64))@np.abs(qb.astype(np.float64))
                operation=dict(pattern=pattern,**comparison(actual,expected,scale),repeatable=np.array_equal(actual,again))
                operation['passes'] = operation['passes'] and operation['repeatable']
                filename=label+'-'+pattern+'.npz'
                np.savez_compressed(out/filename,qa=qa.astype(np.float32),qb=qb.astype(np.float32),
                    qa_bytes=qa.view(np.uint8),qb_bytes=qb.view(np.uint8),actual=actual,
                    repeated=again,expected=expected,scale=scale)
                operation['buffers']=filename
                row['operations'].append(operation)
        except Exception as error:
            row.update(status='rejected_or_failed',error=f'{type(error).__name__}: {error}')
        atomic(out/'summary.json',result)
        print('pallas',name,other,'compiled',row['compiled'],
              'passed',sum(r['passes'] for r in row['operations']),flush=True)
    result['complete']=True
    atomic(out/'summary.json',result)


def fused_phase(device,out,*,smoke=False):
    import jax
    import jax.numpy as jnp
    from engine.precision import products
    from tests.support.product_boundaries import operand_reference
    result=dict(complete=False,phase='fused',device=device.device_kind,operations=[],
        scope='explicit diagnostic candidate; production dispatch and API are unchanged')
    pairs=[(name,name) for name in F8]+[('float8_e5m2','float8_e4m3b11fnuz'),('bfloat16','bfloat16')]
    if smoke: pairs=[('bfloat16','bfloat16')]
    with jax.default_device(device):
        for (name,other),profile,rounding,shared in itertools.product(pairs,('basic','compensated'),
                ('nearest','stochastic'),('left','right')):
            fmt,held=format_spec(name),format_spec(other)
            count=1 if profile=='basic' else 2
            scaling=profile=='compensated' and name.startswith('float8')
            # Mixed representation is an explicit diagnostic choice.
            leftfmt,rightfmt=(held,fmt) if shared=='left' else (fmt,held)
            shape=(33,13,7)
            a,b=values(shape,'normal',name)
            if profile=='compensated':
                a*=np.logspace(-3,3,shape[0],dtype=np.float32)[:,None]
                b*=np.logspace(3,-2,shape[2],dtype=np.float32)[None,:]
            ua=np.random.default_rng(17).random(a.shape,dtype=np.float32)
            ub=np.random.default_rng(18).random(b.shape,dtype=np.float32)
            row=dict(lhs=leftfmt.name,rhs=rightfmt.name,profile=profile,rounding=rounding,
                shared=shared,shape=list(shape),passes=False)
            label=f'case-{len(result["operations"]):03d}'
            buffers=dict(a=a,b=b,ua=ua,ub=ub)
            try:
                def quantize(x,y,u,v):
                    qa,ka=products._terms(x,leftfmt,-1,count if shared=='left' else 1,
                        u if rounding=='stochastic' and shared=='right' else None,carrier=None,scaling=scaling)
                    qb,kb=products._terms(y,rightfmt,-2,count if shared=='right' else 1,
                        v if rounding=='stochastic' and shared=='left' else None,carrier=None,scaling=scaling)
                    return qa,qb,ka,kb
                def combine(raw,ka,kb):
                    raw=raw*products.pow2(-ka,jnp.float32)*products.pow2(-kb,jnp.float32)
                    pieces=jnp.split(raw,count,axis=-2 if shared=='left' else -1)
                    total=pieces[-1]
                    for piece in reversed(pieces[:-1]): total=total+piece
                    return total
                def fused(x,y,u,v):
                    qa,qb,ka,kb=quantize(x,y,u,v)
                    return combine(jnp.matmul(qa,qb,precision=jax.lax.Precision.DEFAULT,
                        preferred_element_type=jnp.float32),ka,kb)
                args=tuple(jax.device_put(v,device) for v in (a,b,ua,ub))
                qa,qb,ka,kb=jax.jit(quantize)(*args)
                qra,kra=operand_reference(a,leftfmt,-1,count if shared=='left' else 1,
                    ua if rounding=='stochastic' and shared=='right' else None,scaling=scaling)
                qrb,krb=operand_reference(b,rightfmt,-2,count if shared=='right' else 1,
                    ub if rounding=='stochastic' and shared=='left' else None,scaling=scaling)
                quant_ok=(np.array_equal(np.asarray(qa,np.float64),qra) and np.array_equal(np.asarray(qb,np.float64),qrb)
                    and np.array_equal(np.asarray(ka),kra) and np.array_equal(np.asarray(kb),krb))
                expected_raw=qra@qrb
                scale_raw=np.abs(qra)@np.abs(qrb)
                factors=np.exp2(-kra.astype(np.float64))*np.exp2(-krb.astype(np.float64))
                axis=-2 if shared=='left' else -1
                expected=sum(np.split(expected_raw*factors,count,axis=axis))
                scale=sum(np.split(scale_raw*factors,count,axis=axis))
                fn=jax.jit(fused).lower(*args).compile()
                hlo=fn.as_text()
                (out/(label+'.hlo.txt')).write_text(hlo,encoding='utf-8')
                actual=np.asarray(fn(*args),np.float64)
                again=np.asarray(fn(*args),np.float64)
                rawfn=jax.jit(lambda x,y:jnp.matmul(x,y,precision=jax.lax.Precision.DEFAULT,
                    preferred_element_type=jnp.float32)).lower(qa,qb).compile()
                raw=np.asarray(rawfn(qa,qb),np.float64)
                (out/(label+'.raw.hlo.txt')).write_text(rawfn.as_text(),encoding='utf-8')
                checks=dict(quantization=quant_ok,raw_product=comparison(raw,expected_raw,scale_raw)['passes'],
                    fused=comparison(actual,expected,scale)['passes'],repeatable=np.array_equal(actual,again))
                row.update(passes=bool(all(checks.values())),checks=checks,
                    fused_error=comparison(actual,expected,scale),
                    lowering=lowering_status(dict(lhs=leftfmt.name,rhs=rightfmt.name),hlo,None),hlo=label+'.hlo.txt')
                buffers.update(qa=np.asarray(qa,np.float32),qb=np.asarray(qb,np.float32),ka=np.asarray(ka),kb=np.asarray(kb),
                    reference_qa=qra,reference_qb=qrb,raw=raw,reference_raw=expected_raw,
                    actual=actual,expected=expected,repeated=again,scale=scale)
            except Exception as error:
                row.update(status='rejected_or_failed',error=f'{type(error).__name__}: {error}')
            np.savez_compressed(out/(label+'.npz'),**buffers)
            row['buffers']=label+'.npz'
            result['operations'].append(row)
            atomic(out/'summary.json',result)
        print('fused passed',sum(r['passes'] for r in result['operations']),'/',len(result['operations']),flush=True)
    result['complete']=True
    atomic(out/'summary.json',result)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform',choices=('cpu','tpu'),required=True)
    p.add_argument('--phase',choices=('all','raw','pallas','fused'),default='all')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()): p.error('output directory must be empty')
    args.out.mkdir(parents=True,exist_ok=True)
    os.environ.update(JAX_PLATFORMS='tpu,cpu' if args.platform=='tpu' else 'cpu',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    if args.phase=='all':
        steps=[]
        for phase,limit in (('raw',480),('pallas',360),('fused',480)):
            command=[sys.executable,'-u','-m','tests.support.tpu_native_precision','--platform',args.platform,
                '--phase',phase,'--out',str(args.out/phase)]+(['--smoke'] if args.smoke else [])
            with (args.out/(phase+'.log')).open('w',encoding='utf-8') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
                try: code=child.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    if os.name!='nt': os.killpg(child.pid,signal.SIGKILL)
                    else: child.kill()
                    child.wait();code=124
            steps.append(dict(phase=phase,exit_code=code))
            atomic(args.out/'steps.json',steps)
            print(phase,'exit',code,flush=True)
        atomic(args.out/'complete.json',dict(runner_finished=True,steps=steps))
        return int(any(s['exit_code']!=0 for s in steps))
    import jax
    device=jax.devices(args.platform)[0]
    if args.platform=='tpu' and ('v6' not in device.device_kind.lower() or jax.device_count()!=1):
        p.error('requires exactly one v6e TPU device')
    if jax.config.jax_disable_jit: p.error('JIT is required')
    atomic(args.out/'environment.json',dict(device=device.device_kind,platform=device.platform,
        jax=jax.__version__,jaxlib=importlib.metadata.version('jaxlib'),numpy=np.__version__,
        xla_flags=os.environ.get('XLA_FLAGS',''),smoke=args.smoke))
    {'raw':raw_phase,'pallas':pallas_phase,'fused':fused_phase}[args.phase](device,args.out,smoke=args.smoke)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
