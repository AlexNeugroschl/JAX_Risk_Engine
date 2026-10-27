"""IEEE boundary regressions for the precision grid's portable bit operations."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from engine.precision.grid import _frexp_exponent, pow2, round_to_grid, scale_exponent


@pytest.mark.parametrize("dtype,word", [(np.float32,np.uint32),(np.float64,np.uint64)])
def test_exponents_match_numpy_even_for_subnormals_and_special_values(dtype, word):
    rng=np.random.default_rng(91)
    bits=rng.integers(0,np.iinfo(word).max,size=10000,dtype=word)
    info=np.finfo(dtype)
    edges=np.array([0.,-0.,1.,-1.,info.tiny,info.max,np.inf,-np.inf,np.nan],dtype=dtype)
    subnormal=np.array([1,2,17],dtype=word).view(dtype)
    values=np.concatenate((bits.view(dtype),edges,subnormal,-subnormal))
    with np.errstate(invalid='ignore'):
        expected=np.frexp(values)[1]
    # frexp's exponent for non-finite values is unspecified in NumPy; JAX uses 0.
    expected=np.where(np.isfinite(values),expected,0)
    actual=jax.jit(_frexp_exponent)(jnp.asarray(values))
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize("dtype", [np.float32,np.float64])
def test_every_normal_power_of_two_is_exact(dtype):
    info=np.finfo(dtype)
    exponents=np.arange(info.minexp,info.maxexp,dtype=np.int32)
    expected=np.ldexp(np.ones(exponents.shape,dtype=dtype),exponents)
    np.testing.assert_array_equal(jax.jit(lambda e:pow2(e,jnp.dtype(dtype)))(exponents),expected)


@pytest.mark.parametrize("dtype", [np.float32,np.float64])
@pytest.mark.parametrize("maximum", [448.,57344.,65504.,float(jnp.finfo(jnp.bfloat16).max)])
def test_scale_selection_at_binade_and_format_boundaries(dtype,maximum):
    info=np.finfo(dtype)
    centers=np.array([0.,1.,maximum/2,maximum,info.tiny,info.max],dtype=dtype)
    with np.errstate(over='ignore',invalid='ignore'):
        values=np.concatenate((centers,np.nextafter(centers,dtype(0)),np.nextafter(centers,dtype(np.inf))))
        mantissa,exponent=np.frexp(values);top_mantissa,top_exponent=np.frexp(maximum)
        exponent=np.where(np.isfinite(values),exponent,0)
        expected=np.clip(top_exponent-exponent-(mantissa>top_mantissa),-126,126)
    actual=jax.jit(lambda x:scale_exponent(x,maximum))(values)
    np.testing.assert_array_equal(actual,expected)


def test_float64_halfway_values_do_not_round_through_float32():
    values=np.array([1+2.**-11+2.**-35,1+2.**-11-2.**-35,-1-2.**-11-2.**-35])
    rounded=jax.jit(lambda x:round_to_grid(x,10,-14))(values)
    np.testing.assert_array_equal(rounded,values.astype('float16').astype('float64'))
    assert values[0].astype('float32').astype('float16') != values[0].astype('float16')


def test_fp64_grid_does_not_need_64_bit_bitwise_operations():
    f=jax.jit(lambda x:(round_to_grid(x,3,-6),scale_exponent(jnp.abs(x),448.)))
    hlo=f.lower(jnp.array([.001,1.,1000.],dtype=jnp.float64)).as_text()
    bitwise=[line for line in hlo.splitlines() if any(op in line for op in
             ('stablehlo.and','stablehlo.shift_','stablehlo.or','stablehlo.count_leading_zeros'))]
    assert bitwise and all('i64' not in line and 'ui64' not in line for line in bitwise)
