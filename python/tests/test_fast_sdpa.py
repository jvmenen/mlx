import math
import os
import unittest
from itertools import product

import mlx.core as mx
import mlx_tests


def mlx_ref_attn(q, k, v, scale=1.0, mask=None, sinks=None):
    q_dtype = q.dtype
    q = q * mx.array(scale, q_dtype)
    n_q_heads = q.shape[-3]
    n_kv_heads = k.shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    B = q.shape[0]
    L = q.shape[2]
    kL = k.shape[2]

    if n_repeats > 1:
        q = mx.reshape(q, [B, n_kv_heads, n_repeats, L, -1])
        k = mx.expand_dims(k, 2)
        v = mx.expand_dims(v, 2)

    scores = q @ mx.swapaxes(k, -1, -2)
    is_causal = mask == "causal"
    if mask is not None:
        if is_causal:
            offset = kL - L
            q_indices = mx.arange(L) + offset
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]

        if n_repeats > 1 and mask.ndim >= 3:
            if mask.shape[-3] == 1:
                mask = mx.expand_dims(mask, -3)
            else:
                mask = mx.unflatten(mask, -3, (n_kv_heads, n_repeats))

        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask

    if sinks is not None:
        sinks = mx.expand_dims(sinks, (0, 2, 3))
        if n_repeats > 1:
            sinks = mx.unflatten(sinks, 1, (n_kv_heads, n_repeats))
        score_shape = list(scores.shape)
        score_shape[-1] = 1
        sinks = mx.broadcast_to(sinks, score_shape)
        scores = mx.concatenate([sinks, scores], axis=-1)

    scores = mx.softmax(scores, axis=-1, precise=True)
    if sinks is not None:
        scores = scores[..., 1:]

    out = scores @ v
    if n_repeats > 1:
        out = mx.reshape(out, [B, n_q_heads, L, -1])
    return out


def do_attention(f, q, k, v, scale, mask=None, transpose=False):
    if transpose:
        q_t = mx.transpose(q, (0, 2, 1, 3))
        k_t = mx.transpose(k, (0, 2, 1, 3))
        v_t = mx.transpose(v, (0, 2, 1, 3))
        o_t = f(q_t, k_t, v_t, scale=scale, mask=mask)
        return mx.transpose(o_t, (0, 2, 1, 3))
    else:
        return f(q, k, v, scale=scale, mask=mask)


def prepare_inputs(B, qL, kL, D, qH, kH, mask, transpose, dtype):
    mx.random.seed(0)

    scale = 1.0 / math.sqrt(D)
    shape_q = (B, qL, qH, D) if transpose else (B, qH, qL, D)
    shape_kv = (B, kL, kH, D) if transpose else (B, kH, kL, D)

    q = mx.random.uniform(0.0, 0.5, shape_q, dtype)
    k = mx.random.uniform(0.0, 0.5, shape_kv, dtype)
    v = mx.random.uniform(0.0, scale, shape_kv, dtype)

    if mask is not None:
        if mask == "additive":
            mask = mx.random.uniform(0.0, 0.5, (B, qH, qL, kL), dtype)
        elif mask == "bool":
            mask = mx.random.uniform(0.0, 1.0, (B, qH, qL, kL)) < 0.5

    return q, k, v, scale, mask


# SDPA for MHA (n_heads == n_kv_heads)
def mlx_primitives_sdpa(q, k, v, scale, mask=None):
    p = (q * scale) @ k.transpose(0, 1, 3, 2)
    qL = q.shape[2]
    kL = k.shape[2]
    is_causal = mask == "causal"
    if mask is not None:
        if is_causal:
            offset = kL - qL
            q_indices = mx.arange(qL) + offset
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
            p = mx.where(mask, p, mx.finfo(mx.float32).min)
        elif mask.dtype == mx.bool_:
            p = mx.where(mask, p, mx.finfo(mx.float32).min)
        else:
            p += mask
    scores = mx.softmax(p.astype(mx.float32), axis=-1).astype(p.dtype)
    return scores @ v


def mlx_ref_lse(q, k, scale=1.0, mask=None, sinks=None):
    """Reference log-sum-exp of the scaled and masked scores, in float32."""
    n_repeats = q.shape[1] // k.shape[1]
    L, kL = q.shape[2], k.shape[2]
    scores = (q.astype(mx.float32) * scale) @ mx.swapaxes(
        mx.repeat(k.astype(mx.float32), n_repeats, axis=1), -1, -2
    )
    if isinstance(mask, str):
        mask = (mx.arange(L) + (kL - L))[:, None] >= mx.arange(kL)[None]
    if mask is not None:
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, -float("inf"))
        else:
            scores = scores + mask.astype(mx.float32)
    if sinks is not None:
        sinks = mx.broadcast_to(
            sinks.astype(mx.float32)[None, :, None, None], scores.shape[:3] + (1,)
        )
        scores = mx.concatenate([sinks, scores], axis=-1)
    return mx.logsumexp(scores, axis=-1, keepdims=True)


def _f32_out_atol(qL, D):
    # Tolerance of a float32 result against a float32 reference. The NAX full
    # attention kernels (more than 8 queries, head dims 64, 96, 128, 256 and
    # 512) multiply the probabilities into V at reduced precision, about 1e-3
    # absolute for unit-normal inputs and well below one bfloat16 ulp; the
    # other kernels and the fallback are float32 throughout.
    if qL > 8 and D in (64, 96, 128, 256, 512):
        return 3e-3
    return 2e-5


def merge_attention(o_1, lse_1, o_2, lse_2):
    """Exact combination of the attention over two disjoint sets of keys.

    The outputs of scaled_dot_product_attention_with_lse are float32, so the
    result is not rounded to the input type: that is left to the caller."""
    lse = mx.logaddexp(lse_1, lse_2)
    o = mx.exp(lse_1 - lse) * o_1.astype(mx.float32)
    o = o + mx.exp(lse_2 - lse) * o_2.astype(mx.float32)
    return o, lse


class TestFastSDPA(mlx_tests.MLXTestCase):
    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU kernel path only")
    def test_sdpa_head_dim_72(self):
        B, D, qH, kH = (1, 72, 8, 2)
        for qL, kL, dtype, mask_str in product(
            (64, 65),
            (128, 127),
            (mx.float16, mx.bfloat16, mx.float32),
            (None, "additive", "bool", "causal"),
        ):
            with self.subTest(qL=qL, kL=kL, dtype=dtype, mask=mask_str):
                q, k, v, scale, mask = prepare_inputs(
                    B, qL, kL, D, qH, kH, mask_str, False, dtype
                )
                ref = mlx_ref_attn(q, k, v, scale, mask)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                )

                if dtype == mx.float32:
                    atol = 1e-5
                elif dtype == mx.bfloat16:
                    atol = 5e-3
                else:
                    atol = 3e-4
                diff = mx.abs(out - ref) - atol * mx.abs(ref)
                self.assertLessEqual(mx.max(diff).item(), atol)

    @unittest.skipIf(not mx.metal.is_available(), "Metal kernel path only")
    def test_sdpa_pad_head_dim_opt_in(self):
        if mx.default_device() != mx.gpu:
            self.skipTest("requires GPU")
        with mlx_tests.scoped_env(MLX_SDPA_PAD_HEAD_DIM="1"):
            self.test_sdpa_head_dim_72()
            self.test_sdpa_head_dim_80()
            self.test_sdpa_head_dim_72_80_sinks()

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU kernel path only")
    def test_sdpa_head_dim_80(self):
        B, D, qH, kH = (1, 80, 8, 2)
        for qL, kL, dtype, mask_str in product(
            (64, 65),
            (128, 127),
            (mx.float16, mx.bfloat16, mx.float32),
            (None, "additive", "bool", "causal"),
        ):
            with self.subTest(qL=qL, kL=kL, dtype=dtype, mask=mask_str):
                q, k, v, scale, mask = prepare_inputs(
                    B, qL, kL, D, qH, kH, mask_str, False, dtype
                )
                ref = mlx_ref_attn(q, k, v, scale, mask)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                )

                if dtype == mx.float32:
                    atol = 1e-5
                elif dtype == mx.bfloat16:
                    atol = 5e-3
                else:
                    atol = 3e-4
                diff = mx.abs(out - ref) - atol * mx.abs(ref)
                self.assertLessEqual(mx.max(diff).item(), atol)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU kernel path only")
    def test_sdpa_head_dim_72_80_sinks(self):
        B, qH, kH, qL, kL = (1, 8, 2, 64, 128)
        for D, dtype in product((72, 80), (mx.float16, mx.bfloat16)):
            with self.subTest(D=D, dtype=dtype):
                q, k, v, scale, _ = prepare_inputs(
                    B, qL, kL, D, qH, kH, None, False, dtype
                )
                sinks = 10 * mx.random.normal(shape=(qH,), dtype=dtype)
                ref = mlx_ref_attn(q, k, v, scale, sinks=sinks)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, sinks=sinks
                )

                atol = 5e-3 if dtype == mx.bfloat16 else 3e-4
                diff = mx.abs(out - ref) - atol * mx.abs(ref)
                self.assertLessEqual(mx.max(diff).item(), atol)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU kernel path only")
    def test_sdpa_head_dim_96(self):
        B, D, qH, kH = (1, 96, 8, 2)
        for qL, kL, dtype, mask_str in product(
            (64, 65),
            (128, 127),
            (mx.float16, mx.bfloat16, mx.float32),
            (None, "additive", "bool", "causal"),
        ):
            with self.subTest(qL=qL, kL=kL, dtype=dtype, mask=mask_str):
                q, k, v, scale, mask = prepare_inputs(
                    B, qL, kL, D, qH, kH, mask_str, False, dtype
                )
                ref = mlx_ref_attn(q, k, v, scale, mask)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                )

                if dtype == mx.float32:
                    atol = 1e-5
                elif dtype == mx.bfloat16:
                    atol = 5e-3
                else:
                    atol = 3e-4
                diff = mx.abs(out - ref) - atol * mx.abs(ref)
                self.assertLessEqual(mx.max(diff).item(), atol)

    @unittest.skipIf(not mx.metal.is_available(), "Metal kernel path only")
    def test_sdpa_full_d96_v64(self):
        if mx.default_device() != mx.gpu:
            self.skipTest("requires GPU")
        mx.random.seed(0)
        scale = 96**-0.5
        cells = [
            (1, 40, 40, 9, 33),
            (1, 4, 4, 64, 64),
            (2, 4, 2, 65, 97),
            (1, 4, 4, 16, 9),
            (1, 4, 2, 32, 31),
            (1, 4, 2, 33, 32),
        ]
        for dtype, (B, qH, kH, qL, kL) in product(
            (mx.float16, mx.bfloat16, mx.float32), cells
        ):
            q = (
                (0.5 * mx.random.normal((B, qL, qH, 96)))
                .astype(dtype)
                .transpose(0, 2, 1, 3)
            )
            k = (0.5 * mx.random.normal((B, kH, kL + 32, 96))).astype(dtype)[:, :, :kL]
            masks = [
                None,
                mx.arange(kL) < kL - 3,
                mx.random.normal((qL, kL)).astype(dtype),
            ]
            if qL <= kL:
                masks.append("causal")
            for dv, mask, with_sinks in product((64, 96), masks, (False, True)):
                with self.subTest(
                    dtype=dtype,
                    cell=(B, qH, kH, qL, kL),
                    dv=dv,
                    mask=type(mask).__name__,
                    sinks=with_sinks,
                ):
                    v = (0.5 * mx.random.normal((B, kH, kL + 32, dv))).astype(dtype)[
                        :, :, :kL
                    ]
                    if qL == 9:
                        # V sliced from the interleaved K/V projection.
                        v = (0.5 * mx.random.normal((B, kL, kH, 2 * dv))).astype(dtype)
                        v = v.transpose(0, 2, 1, 3)[..., dv:]
                    sinks = (
                        mx.random.normal((qH,)).astype(dtype) if with_sinks else None
                    )
                    ref = mlx_ref_attn(
                        q.astype(mx.float32),
                        k.astype(mx.float32),
                        v.astype(mx.float32),
                        scale,
                        mask,
                        sinks,
                    )
                    out = mx.fast.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        scale=scale,
                        mask=mask,
                        sinks=sinks,
                        force_fused=True,
                    )
                    self.assertEqual(out.shape, (B, qH, qL, dv))
                    tol = 5e-3 if dtype == mx.bfloat16 else 1e-3
                    self.assertTrue(mx.allclose(ref, out, atol=tol, rtol=tol))

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU kernel path only")
    def test_sdpa_full_head_dim_256(self):
        # On NAX devices, large nearly-square causal blocks take the fused
        # path; everything else takes the unfused fallback. Ragged lengths
        # exercise the kernel's unaligned pipelines, and K/V sliced out of a
        # longer preallocated cache (the way mlx-lm hands them over) exercise
        # the dispatch reading the slice past its end. All of it must be
        # correct.
        D = 256
        Nq, Nkv = 8, 2
        scale = D**-0.5
        mx.random.seed(0)
        cases = [
            # fused on NAX: aligned square, ragged square (unaligned Q and
            # K/V), aligned rectangle at the routing boundary, ragged
            # rectangle
            (2048, 2048, "causal", None),
            (2049, 2049, "causal", None),
            (2048, 2560, "causal", None),
            (2049, 2560, "causal", None),
            # fused on NAX with ragged K/V sliced out of a longer cache: the
            # rows behind the slice must not leak into the output
            (2049, 2049, "causal", 2304),
            (2048, 2500, "causal", 2560),
            (1031, 2049, "causal", 2304),
        ]
        for dtype in (mx.float32, mx.bfloat16):
            for qL, kL, mask, cache_len in cases:
                with self.subTest(
                    dtype=dtype, qL=qL, kL=kL, mask=mask, cache_len=cache_len
                ):
                    q = (5e-1 * mx.random.normal(shape=(1, Nq, qL, D))).astype(dtype)
                    if cache_len is None:
                        k = (5e-1 * mx.random.normal(shape=(1, Nkv, kL, D))).astype(
                            dtype
                        )
                        v = (5e-1 * mx.random.normal(shape=(1, Nkv, kL, D))).astype(
                            dtype
                        )
                    else:
                        # Large, finite stale rows behind the slice: any of
                        # them reaching the output is loud.
                        k_cache = 1e2 * mx.random.normal(shape=(1, Nkv, cache_len, D))
                        v_cache = 1e3 * mx.random.normal(shape=(1, Nkv, cache_len, D))
                        k_cache[..., :kL, :] = 5e-1 * mx.random.normal(
                            shape=(1, Nkv, kL, D)
                        )
                        v_cache[..., :kL, :] = 5e-1 * mx.random.normal(
                            shape=(1, Nkv, kL, D)
                        )
                        k = k_cache.astype(dtype)[..., :kL, :]
                        v = v_cache.astype(dtype)[..., :kL, :]
                    k_rep = mx.repeat(k, Nq // Nkv, axis=1)
                    v_rep = mx.repeat(v, Nq // Nkv, axis=1)
                    ref = mlx_primitives_sdpa(q, k_rep, v_rep, scale, mask=mask)
                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, mask=mask
                    )
                    self.assertEqual(out.shape, ref.shape)
                    if dtype == mx.float32:
                        # The fused shapes run through tf32 tensor ops when
                        # MLX_ENABLE_TF32 is on (the default).
                        tol = 1e-3 if qL >= 2048 else 1e-4
                    else:
                        tol = 5e-3
                    self.assertTrue(mx.allclose(ref, out, atol=tol, rtol=tol))

        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            return

        mx.random.seed(0)
        cases = (
            (1, 4, 2, 1024, 1057, 1152),
            (2, 2, 1, 1025, 1025, None),
            (1, 2, 1, 1024, 4103, 4128),
        )
        for dtype, (B, qH, kH, qL, kL, cache_len) in product(
            (mx.float16, mx.bfloat16), cases
        ):
            with self.subTest(dtype=dtype, qL=qL, kL=kL):
                q = (0.5 * mx.random.normal((B, qH, qL, 512))).astype(dtype)
                kv_len = cache_len or kL
                k_cache = (0.5 * mx.random.normal((B, kH, kv_len, 512))).astype(dtype)
                v_cache = (0.5 * mx.random.normal((B, kH, kv_len, 512))).astype(dtype)
                k = k_cache[..., :kL, :]
                v = v_cache[..., :kL, :]
                scale = 512**-0.5
                ref = mlx_ref_attn(q, k, v, scale=scale, mask="causal")
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask="causal"
                )
                tol = 1e-2 if dtype == mx.bfloat16 else 1e-3
                self.assertTrue(mx.allclose(ref, out, atol=tol, rtol=tol))

        q = mx.random.normal((1, 2, 1024, 512), mx.float16)
        k = mx.random.normal((1, 1, 1057, 512), mx.float16)
        v = mx.random.normal((1, 1, 1057, 512), mx.float16)
        scale = 512**-0.5
        mask = mx.random.uniform(shape=(1, 1, 1024, 1057)) > 0.2
        sinks = mx.random.normal((2,), mx.float16)
        fallback_cases = (
            (v, mask, None),
            (v, "causal", sinks),
            (v[..., :128], "causal", None),
        )
        for v_arg, mask_arg, sinks_arg in fallback_cases:
            ref = mlx_ref_attn(q, k, v_arg, scale=scale, mask=mask_arg, sinks=sinks_arg)
            out = mx.fast.scaled_dot_product_attention(
                q, k, v_arg, scale=scale, mask=mask_arg, sinks=sinks_arg
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-3, rtol=1e-3))

    @unittest.skipIf(not mx.metal.is_available(), "Metal kernel path only")
    def test_sdpa_full_head_dim_512_nax(self):
        if mx.default_device() != mx.gpu:
            self.skipTest("requires GPU")

        D = 512
        scale = D**-0.5
        mx.random.seed(0)

        q_probe = mx.random.normal(shape=(1, 2, 9, D)).astype(mx.bfloat16)
        k_probe = mx.random.normal(shape=(1, 1, 31, D)).astype(mx.bfloat16)
        v_probe = mx.random.normal(shape=(1, 1, 31, D)).astype(mx.bfloat16)
        try:
            mx.eval(
                mx.fast.scaled_dot_product_attention(
                    q_probe,
                    k_probe,
                    v_probe,
                    scale=scale,
                    force_fused=True,
                )
            )
        except ValueError as e:
            if "plus head dim 512 on NAX GPUs" not in str(e):
                raise
            self.skipTest("D512 full attention requires NAX")

        cases = [
            # Multiple query tiles with GQA.
            (1, 8, 2, 257, 513, None, True),
            # Multiple batches and a ragged final tile.
            (2, 4, 1, 513, 777, None, True),
            # Gemma 4 31B default route at the work threshold with sliced K/V.
            (1, 32, 16, 1024, 1025, 1056, False),
        ]
        for dtype in (mx.float16, mx.bfloat16):
            for B, Nq, Nkv, qL, kL, cache_len, force_fused in cases:
                with self.subTest(
                    dtype=dtype,
                    B=B,
                    Nq=Nq,
                    Nkv=Nkv,
                    qL=qL,
                    kL=kL,
                    cache_len=cache_len,
                ):
                    q = mx.random.normal(shape=(B, Nq, qL, D)).astype(dtype)
                    if cache_len is None:
                        k = mx.random.normal(shape=(B, Nkv, kL, D)).astype(dtype)
                        v = mx.random.normal(shape=(B, Nkv, kL, D)).astype(dtype)
                    else:
                        k = mx.random.normal(shape=(B, Nkv, cache_len, D)).astype(
                            dtype
                        )[..., :kL, :]
                        v = mx.random.normal(shape=(B, Nkv, cache_len, D)).astype(
                            dtype
                        )[..., :kL, :]

                    ref = mlx_ref_attn(q, k, v, scale, "causal")
                    out = mx.fast.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        scale=scale,
                        mask="causal",
                        force_fused=force_fused,
                    )
                    tol = 2e-3 if dtype != mx.bfloat16 else 5e-2
                    self.assertTrue(mx.allclose(ref, out, atol=tol, rtol=tol))

        q = mx.random.normal(shape=(1, 8, 33, D)).astype(mx.bfloat16)
        k = mx.random.normal(shape=(1, 2, 67, D)).astype(mx.bfloat16)
        v = mx.random.normal(shape=(1, 2, 67, D)).astype(mx.bfloat16)
        ref = mlx_ref_attn(q, k, v, scale)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, force_fused=True
        )
        self.assertTrue(mx.allclose(ref, out, atol=1e-2, rtol=1e-2))

        bool_mask = mx.random.uniform(shape=(1, 1, 33, 67)) > 0.2
        additive_mask = mx.where(bool_mask, 0.0, -1e4).astype(mx.bfloat16)
        for mask in (bool_mask, additive_mask):
            ref = mlx_ref_attn(q, k, v, scale, mask)
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=mask, force_fused=True
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-2, rtol=1e-2))

        q_same = mx.random.normal(shape=(1, 8, 33, D)).astype(mx.bfloat16)
        k_same = mx.random.normal(shape=(1, 2, 67, D)).astype(mx.bfloat16)
        v_slice = mx.random.normal(shape=(1, 2, 67, D // 4)).astype(mx.bfloat16)
        v_same = mx.tile(v_slice, (1, 1, 1, 4))
        out_same = mx.fast.scaled_dot_product_attention(
            q_same, k_same, v_same, scale=scale, force_fused=True
        )
        for d_group in range(1, 4):
            self.assertTrue(
                mx.array_equal(
                    out_same[..., : D // 4],
                    out_same[..., d_group * D // 4 : (d_group + 1) * D // 4],
                )
            )

        sinks = mx.random.normal(shape=(8,)).astype(mx.bfloat16)
        ref = mlx_ref_attn(q, k, v, scale, sinks=sinks)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, sinks=sinks, force_fused=True
        )
        self.assertTrue(mx.allclose(ref, out, atol=1e-2, rtol=1e-2))

    def test_sdpa_vector_kv_transposed_head_seq(self):
        D = 64
        Nq = 4
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))

        lengths = [43, 4096]
        for L in lengths:
            k = 5e-1 * mx.random.normal(shape=(1, L, Nkv, D))
            v = 5e-1 * mx.random.normal(shape=(1, L, Nkv, D))
            k = k.swapaxes(1, 2)
            v = v.swapaxes(1, 2)
            masks = [
                mx.array(True),
                mx.array([True] * (L - 10) + [False] * 10),
                mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
                mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            ]

            for m in masks:
                ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
                out = mx.fast.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=scale,
                    mask=m,
                )
                self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_vector(self):
        D = 64
        L = 43
        Nq = 4
        Nkv = 1
        scale = 1.0
        tol = 1e-3 if mx.cuda.is_available() else 1e-4
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=mx.full((Nq, 2, L), False),
            )

        masks = [
            None,
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            mx.random.uniform(shape=(Nq, 1, L)),
            mx.random.uniform(shape=(L, 1, Nq)).T,
            mx.log(mx.random.uniform(shape=(Nq, 1, L)) > 0.2),
            mx.log(mx.random.uniform(shape=(L, 1, Nq)).T > 0.2),
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=tol, rtol=tol))

        L = 4096
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        masks = [
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            mx.random.uniform(shape=(Nq, 1, L)),
            mx.random.uniform(shape=(L, 1, Nq)).T,
            mx.log(mx.random.uniform(shape=(Nq, 1, L)) > 0.2),
            mx.log(mx.random.uniform(shape=(L, 1, Nq)).T > 0.2),
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=tol, rtol=tol))

    def test_sdpa_vector_gqa_long(self):
        scale = 1.0
        mx.random.seed(0)
        for Nq, Nkv, D in [(32, 4, 128), (64, 8, 64), (48, 4, 128), (64, 4, 128)]:
            for B, L in [(1, 8192), (1, 8201), (2, 8192)]:
                q = 5e-1 * mx.random.normal(shape=(B, Nq, 1, D))
                k = 5e-1 * mx.random.normal(shape=(B, Nkv, L + 32, D))[:, :, :L]
                v = 5e-1 * mx.random.normal(shape=(B, Nkv, L + 32, D))[:, :, :L]
                kr = mx.repeat(k, Nq // Nkv, axis=1)
                vr = mx.repeat(v, Nq // Nkv, axis=1)
                ref = mlx_primitives_sdpa(q, kr, vr, scale)
                out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
                self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    @unittest.skipIf(not mx.metal.is_available(), "Metal kernel path only")
    def test_sdpa_vector_d96_v64(self):
        if mx.default_device() != mx.gpu:
            self.skipTest("requires GPU")
        mx.random.seed(0)
        scale = 96**-0.5
        # Include MiniCPM3 heads and GQA to reach two-pass on smaller GPUs.
        cells = [(1, 40, 40, 1, 127), (1, 40, 40, 8, 1025), (2, 4, 2, 4, 4097)]
        for dtype, (B, qH, kH, qL, kL) in product(
            (mx.float32, mx.float16, mx.bfloat16), cells
        ):
            q = mx.random.normal((B, qL, qH, 96)).astype(dtype)
            q = q.transpose(0, 2, 1, 3)
            k = mx.random.normal((B, kH, kL + 32, 96)).astype(dtype)[:, :, :kL]
            v = mx.random.normal((B, kH, kL + 32, 64)).astype(dtype)[:, :, :kL]
            masks = (
                None,
                "causal",
                mx.arange(kL) < kL - 3,
                mx.random.normal((qL, kL)).astype(dtype),
            )
            for mask in masks:
                with self.subTest(
                    dtype=dtype, cell=(B, qH, kH, qL, kL), mask=type(mask).__name__
                ):
                    # GPU float32 GEMM may use TF32; keep the reference exact.
                    with mx.stream(mx.cpu):
                        ref = mlx_ref_attn(
                            q.astype(mx.float32),
                            k.astype(mx.float32),
                            v.astype(mx.float32),
                            scale,
                            mask,
                        )
                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, mask=mask, force_fused=True
                    )
                    self.assertEqual(out.shape, (B, qH, qL, 64))
                    tol = (
                        1e-5
                        if dtype == mx.float32
                        else (3e-3 if dtype == mx.float16 else 2e-2)
                    )
                    self.assertTrue(mx.allclose(ref, out, atol=tol, rtol=tol))

    @unittest.skipIf(not mx.metal.is_available(), "Metal kernel path only")
    def test_sdpa_vector_head_dim_512(self):
        if mx.default_device() != mx.gpu:
            self.skipTest("requires GPU")
        # gemma-4 global attention: 32 query heads over 4 key/value heads.
        D = 512
        Nq, Nkv = 32, 4
        scale = D**-0.5
        mx.random.seed(0)

        with mlx_tests.scoped_env(MLX_SDPA_D512_MIN_KL="0"):
            # Test 1-pass kernel.
            for dtype in (mx.float32, mx.float16, mx.bfloat16):
                with self.subTest(L=128, dtype=dtype, threshold="off"):
                    q = mx.random.normal(shape=(1, Nq, 1, D), dtype=dtype)
                    k = mx.random.normal(shape=(1, Nkv, 128, D), dtype=dtype)
                    v = mx.random.normal(shape=(1, Nkv, 128, D), dtype=dtype)
                    ref = mlx_ref_attn(q, k, v, scale)
                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, force_fused=True
                    )
                    atol = 1e-5 if dtype == mx.float32 else 2e-2
                    self.assertTrue(mx.allclose(ref, out, atol=atol))

            # Test 2-pass kernel.
            for L, dtype in product(
                (8192, 8201), (mx.float32, mx.float16, mx.bfloat16)
            ):
                with self.subTest(L=L, dtype=dtype):
                    q = mx.random.normal(shape=(1, Nq, 1, D), dtype=dtype)
                    k = mx.random.normal(shape=(1, Nkv, L, D), dtype=dtype)
                    v = mx.random.normal(shape=(1, Nkv, L, D), dtype=dtype)
                    ref = mlx_ref_attn(q, k, v, scale)
                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, force_fused=True
                    )
                    atol = 1e-5 if dtype == mx.float32 else 2e-2
                    self.assertTrue(mx.allclose(ref, out, atol=atol))

            # Test other heads.
            for q_heads, kv_heads in ((8, 8), (32, 8)):
                with self.subTest(q_heads=q_heads, kv_heads=kv_heads):
                    q = mx.random.normal(shape=(1, q_heads, 1, D))
                    k = mx.random.normal(shape=(1, kv_heads, L, D))
                    v = mx.random.normal(shape=(1, kv_heads, L, D))
                    ref = mlx_ref_attn(q, k, v, scale)
                    out = mx.fast.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        scale=scale,
                        force_fused=True,
                    )
                    self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

            # Test batched.
            B = 2
            sinks = 10 * mx.random.normal(shape=(Nq,))
            q = mx.random.normal(shape=(B, Nq, 1, D))
            for L in (256, 8192):
                k = mx.random.normal(shape=(B, Nkv, L, D))
                v = mx.random.normal(shape=(B, Nkv, L, D))
                for s_in in (None, sinks):
                    with self.subTest(L=L, sinks=s_in is not None):
                        ref = mlx_ref_attn(q, k, v, scale, sinks=s_in)
                        out = mx.fast.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            scale=scale,
                            sinks=s_in,
                            force_fused=True,
                        )
                        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_fully_masked(self):
        Lkv = 8
        mask = mx.array(False)
        for D in [128]:
            for Lq in [1, 8, 32]:
                q = mx.random.normal(shape=(1, 4, Lq, D))
                k = mx.random.normal(shape=(1, 4, Lkv, D))
                v = mx.random.normal(shape=(1, 4, Lkv, D))

                out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1)
                self.assertFalse(mx.any(mx.isnan(out)))

    def test_sdpa_inf_score(self):
        Lkv = 8
        for D in [4, 128]:
            for Lq in [1, 8]:
                q = mx.ones(shape=(1, 4, Lq, D))
                k = mx.ones(shape=(1, 4, Lkv, D))
                v = mx.random.normal(shape=(1, 4, Lkv, D))
                k[..., 0, :] = -float("inf")
                ref = mlx_primitives_sdpa(q, k, v, scale=1, mask=None)
                out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1)
                self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_few_query(self):
        D = 64
        L = 43
        Lq = 8
        Nq = 8
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Lq, Nq, D))
        q = q.swapaxes(1, 2)
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        masks = [
            None,
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        L = 4096
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, Lq, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        masks = [
            None,
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    @unittest.skip("Different head and value dims is not enabled")
    def test_sdpa_vector_value_dims(self):
        D = 192
        V = 128
        Nq = 4
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)

        for L in [43, 128, 237, 8192]:
            q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))
            k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
            v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, V))
            ref = mlx_primitives_sdpa(q, k, v, scale)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_vector_batched(self):
        D = 64
        q = mx.random.normal(shape=(2, 1, 3, D))
        k = mx.random.normal(shape=(2, 1, 3, D))
        v = mx.random.normal(shape=(2, 1, 3, D))

        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 4, 3, D))
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 3, 4, D)).swapaxes(1, 2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        k = mx.random.normal(shape=(2, 3, 1, D)).swapaxes(1, 2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 4, 3, D))
        k = mx.random.normal(shape=(2, 3, 2, D)).swapaxes(1, 2)
        v = mx.random.normal(shape=(2, 2, 3, D))
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 4, 3, D))
        k = mx.random.normal(shape=(2, 1, 3, D))
        v = mx.random.normal(shape=(2, 1, 3, D))
        mask = 10 * mx.random.normal(shape=(1, 2, 3, 3)).swapaxes(0, 1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0)
        ref = mlx_ref_attn(q, k, v, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    @unittest.skipIf(not mx.metal.is_available(), "Metal kernel path only")
    def test_sdpa_vector_2pass_threadgroup_limit(self):
        for dtype, D, q_heads, kv_heads, q_len in (
            (mx.float16, 192, 32, 8, 8),
            (mx.float16, 256, 24, 4, 5),
            (mx.float16, 256, 32, 4, 4),
            (mx.bfloat16, 256, 24, 4, 5),
            (mx.bfloat16, 256, 32, 4, 4),
            (mx.float32, 192, 32, 8, 8),
            (mx.float32, 256, 24, 4, 5),
            (mx.float32, 256, 32, 4, 4),
        ):
            with self.subTest(dtype=dtype, D=D, q_len=q_len):
                self.assertGreater(32 * q_heads // kv_heads * q_len, 896)
                q = mx.random.normal(shape=(1, q_heads, q_len, D), dtype=dtype)
                k = mx.random.normal(shape=(1, kv_heads, 1029, D), dtype=dtype)
                v = mx.random.normal(shape=(1, kv_heads, 1029, D), dtype=dtype)
                scale = D**-0.5

                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask="causal", force_fused=True
                )
                ref = mlx_ref_attn(
                    q.astype(mx.float32),
                    k.astype(mx.float32),
                    v.astype(mx.float32),
                    scale=scale,
                    mask="causal",
                )
                atol = 1e-4 if dtype == mx.float32 else 2e-3
                if dtype == mx.bfloat16:
                    atol = 2e-2
                self.assertTrue(mx.allclose(out.astype(mx.float32), ref, atol=atol))

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU kernel path only")
    def test_sdpa_blocks_env_override(self):
        # MLX_SDPA_BLOCKS used to be applied as-is, and values that are not
        # a multiple of 32 silently corrupted the 2-pass vector output. The
        # override is now rounded up to a multiple of 32.
        D = 128
        q = mx.random.normal(shape=(1, 32, 1, D), dtype=mx.float16)
        k = mx.random.normal(shape=(1, 8, 8192, D), dtype=mx.float16)
        v = mx.random.normal(shape=(1, 8, 8192, D), dtype=mx.float16)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
        for blocks in (16, 33, 48, 100):
            with mlx_tests.scoped_env(MLX_SDPA_BLOCKS=str(blocks)):
                out = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
                self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    @unittest.skipIf(not mx.is_available(mx.gpu), "too slow on CPU")
    def test_sdpa(self):
        # fmt: off
        shapes_64 = [
            # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
            (  1,    20,    20,       64,    3,     3),
            (  1,    63,    63,       64,   24,    24),
            (  1,   129,   129,       64,   24,    24),
            (  1,   400,   400,       64,   24,    24),
            (  1,   128,   128,       64,   32,    32),
            (  1,    64,   128,       64,   32,    32),
            (  1,    65,   128,       64,   32,     8),
            (  1,    64,   127,       64,   32,     8),
            (  1,    65,   127,       64,   32,     8),
            (  1,   127,    65,       64,   32,     8),
        ]
        shapes_128 = [
            # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
            (  1,   128,   128,      128,   32,     8),
            (  1,    64,   128,      128,   32,     8),
            (  1,    65,   127,      128,   32,     8),
            (  1,   127,    65,      128,   32,     8),
        ]
        for ksl in [7, 9, 32, 63, 67, 129, 400, 2000]:
            shapes_128.append((1, 1, ksl, 128, 32, 32))
            shapes_128.append((1, 1, ksl, 128, 32, 8))
        # fmt: on

        shapes = shapes_64 + shapes_128
        dtypes = [mx.float16]
        if mx.metal.is_available():
            dtypes.append(mx.float32)
        masks = [None, "additive", "bool", "causal"]
        transposes = (False, True)

        for dtype, t, mask_str, (B, qL, kL, D, qH, kH) in product(
            dtypes, transposes, masks, shapes
        ):
            with self.subTest(
                B=B,
                qsl=qL,
                ksl=kL,
                head_dim=D,
                n_q_heads=qH,
                n_kv_heads=kH,
                mask=mask_str,
                transpose=t,
                dtype=dtype,
            ):
                q, k, v, scale, mask = prepare_inputs(
                    B, qL, kL, D, qH, kH, mask_str, t, dtype
                )

                out_ref = do_attention(mlx_ref_attn, q, k, v, scale, mask, t)

                with mlx_tests.scoped_env(MLX_ENABLE_TF32="0"):
                    out_fst = do_attention(
                        mx.fast.scaled_dot_product_attention,
                        q,
                        k,
                        v,
                        scale,
                        mask,
                        t,
                    )

                # For causal mask when qL > kL, first qL-kL rows are undefined
                # Compare only the valid portion
                if mask_str == "causal" and qL > kL:
                    offset = qL - kL
                    if t:  # transpose=True: shape is (B, qL, qH, D)
                        out_ref = out_ref[:, offset:, :, :]
                        out_fst = out_fst[:, offset:, :, :]
                    else:  # transpose=False: shape is (B, qH, qL, D)
                        out_ref = out_ref[:, :, offset:, :]
                        out_fst = out_fst[:, :, offset:, :]

                atol = 2e-5 if dtype == mx.float32 else 3e-4

                self.assertListEqual(list(out_ref.shape), list(out_fst.shape))

                diff = mx.abs(out_fst - out_ref) - atol * mx.abs(out_ref)
                self.assertLessEqual(mx.max(diff).item(), atol)

    @unittest.skipIf(not mx.is_available(mx.gpu), "too slow on CPU")
    @unittest.skipIf(mx.cuda.is_available() and "CI" in os.environ, "not enough memory")
    def test_sdpa_long_masked_sequence(self):
        # Test for int16 overflow in steel_attention_nax.h mask
        # indexing (col_pos declared as short, overflows when kL > 32767).
        D = 64
        dtype = mx.float16
        atol = 1e-3  # Slightly looser than test_sdpa due to long masked sequences

        for kL, active in [
            (8192, 1024),
            (36864, 1024),
            (49152, 1024),
            (66048, 1024),
        ]:
            with self.subTest(kL=kL, active=active):
                mx.random.seed(0)
                qH, kH, qL = 32, 16, 512
                scale = 1.0 / math.sqrt(D)

                q = mx.random.normal(shape=(1, qH, qL, D)).astype(dtype)
                k = mx.random.normal(shape=(1, kH, kL, D)).astype(dtype)
                v = mx.random.normal(shape=(1, kH, kL, D)).astype(dtype)

                # Additive mask: -1e4 for inactive, 0 for last `active` positions
                mask = mx.full((1, 1, 1, kL), -1e4, dtype=dtype)
                mask[..., kL - active :] = 0.0

                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                )
                ref = mlx_ref_attn(q, k, v, scale=scale, mask=mask)

                self.assertFalse(mx.isnan(out).any().item())
                self.assertListEqual(list(out.shape), list(ref.shape))

                diff = mx.abs(out - ref) - atol * mx.abs(ref)
                self.assertLessEqual(mx.max(diff).item(), atol)

    def test_sdpa_broadcast_mask(self):
        mask = mx.array(True)
        D = 64
        Nq = 4
        Nkv = 1
        scale = 1.0
        L = 256

        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, L, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        ref = mlx_primitives_sdpa(q, k, v, scale, mask=mask)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_noncontiguous_inputs(self):
        mask = mx.ones(shape=(4, 1, 7, 7), dtype=mx.bool_)
        mx.random.seed(0)
        q = mx.random.normal(shape=(4, 7, 32, 64)).swapaxes(1, 2)

        k = mx.random.normal(shape=(4, 7, 8, 64)).swapaxes(1, 2)
        v = mx.random.normal(shape=(4, 7, 8, 64)).swapaxes(1, 2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        ref = mlx_ref_attn(q, k, v, scale=1.0, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_vector_reversed_mask(self):
        mx.random.seed(0)
        q = mx.random.normal(shape=(1, 2, 1, 64))
        k = mx.random.normal(shape=(1, 2, 16, 64))
        v = mx.random.normal(shape=(1, 2, 16, 64))
        masks = [
            mx.random.normal(shape=(1, 2, 1, 16))[..., ::-1],
            (mx.arange(16) % 3 != 0).reshape(1, 1, 1, 16)[..., ::-1],
        ]
        for mask in masks:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask)
            ref = mlx_ref_attn(q, k, v, scale=0.125, mask=mask)
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_promote_mask(self):
        mask = mx.array(2.0, mx.bfloat16)
        D = 64
        Nq = 4
        Nkv = 1
        scale = 1.0
        L = 256

        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, L, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        ref = mlx_primitives_sdpa(q, k, v, scale, mask=mask)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_nan_bug(self):
        N = 128
        q_shape = (1, 1, N, 128)
        kv_shape = (1, 1, N, 128)
        q = mx.random.uniform(shape=q_shape)
        k = mx.random.uniform(shape=kv_shape)
        v = mx.random.uniform(shape=kv_shape)

        # Make boolean window causal mask
        linds = rinds = mx.arange(N)
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        mask = mask & (linds <= rinds + 111)

        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0)
        expected = mlx_ref_attn(q, k, v, mask=mask, scale=1.0)
        self.assertFalse(mx.isnan(out).any().item())
        self.assertLessEqual(mx.abs(out - expected).max().item(), 1e-4)

        # And an additive one
        mask = mx.log(mask)

        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0)
        expected = mlx_ref_attn(q, k, v, mask=mask, scale=1.0)
        self.assertFalse(mx.isnan(out).any().item())
        self.assertLessEqual(mx.abs(out - expected).max().item(), 1e-4)

    def test_sdpa_attention_sinks(self):
        B = 2
        N_q = N_kv = 8
        T_q = T_kv = 128
        D = 64

        q = mx.random.normal(shape=(B, N_q, T_q, D))
        k = mx.random.normal(shape=(B, N_kv, T_kv, D))
        v = mx.random.normal(shape=(B, N_kv, T_kv, D))
        scale = D**-0.5

        # sinks should promote to correct type
        sinks = mx.random.normal(shape=(N_q,))
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(
                q.astype(mx.float16),
                k.astype(mx.float16),
                v.astype(mx.float16),
                scale=scale,
                sinks=sinks,
            )

        # Wrong shapes
        sinks = mx.random.normal(shape=(N_q + 1,))
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, sinks=sinks)

        sinks = mx.random.normal(shape=())
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, sinks=sinks)

        for T_q, T_kv, N_kv, dtype in product(
            (1, 128),
            (128, 4096),
            (2, 8),
            (mx.float16, mx.float32),
        ):
            with self.subTest(T_q=T_q, T_kv=T_kv, N_kv=N_kv, dtype=dtype):
                q = mx.random.normal(shape=(B, N_q, T_q, D), dtype=dtype)
                k = mx.random.normal(shape=(B, N_kv, T_kv, D), dtype=dtype)
                v = mx.random.normal(shape=(B, N_kv, T_kv, D), dtype=dtype)
                sinks = 10 * mx.random.normal(shape=(N_q,), dtype=dtype)

                expected = mlx_ref_attn(q, k, v, scale, sinks=sinks)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, sinks=sinks
                )
                atol = 1e-5 if dtype == mx.float32 else 1e-2
                self.assertTrue(mx.allclose(out, expected, atol=atol))

    def test_sdpa_grad(self):
        # High tolerance due to cuDNN SDPA kernel requiring tf32.
        tolerance = {"rtol": 1e-2, "atol": 1e-2}

        def test_vjp(slow, fast, primals):
            cotan = mx.ones_like(primals[0])
            o1, vjp1 = mx.vjp(slow, primals, [cotan])
            o2, vjp2 = mx.vjp(fast, primals, [cotan])

            self.assertTrue(mx.allclose(o1[0], o2[0], **tolerance))
            for i in range(3):
                self.assertTrue(mx.allclose(vjp1[i], vjp2[i], **tolerance))

        def test_grad(slow, fast, args):
            g1 = mx.grad(slow)(*args)
            g2 = mx.grad(fast)(*args)

            self.assertTrue(mx.allclose(g1, g2, **tolerance))

        B, N_kv, T, D = (2, 8, 128, 64)
        scale = D**-0.5

        for N_q in (8, 32):
            q = mx.random.normal(shape=(B, N_q, T, D), dtype=mx.float16)
            k = mx.random.normal(shape=(B, N_kv, T, D), dtype=mx.float16)
            v = mx.random.normal(shape=(B, N_kv, T, D), dtype=mx.float16)

            mask_additive = mx.random.normal((B, N_q, T, T), dtype=mx.float16)
            mask_bool = mx.random.uniform(0, 1, (B, N_q, T, T), dtype=mx.float16) < 0.5

            for mask in (None, "causal", mask_additive, mask_bool):

                def sdpa_slow(q, k, v):
                    return mlx_ref_attn(q, k, v, scale=scale, mask=mask)

                def sdpa_fast(q, k, v):
                    return mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, mask=mask
                    )

                test_vjp(sdpa_slow, sdpa_fast, [q, k, v])

                def loss_slow(q, k, v):
                    return mlx_ref_attn(q, k, v, scale=scale, mask=mask).sum()

                def loss_fast(q, k, v):
                    return mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, mask=mask
                    ).sum()

                test_grad(loss_slow, loss_fast, [q, k, v])

    @unittest.skipIf(not mx.metal.is_available(), "Metal kernel path only")
    def test_sdpa_force_fused_metal(self):
        if mx.default_device() != mx.gpu:
            self.skipTest("requires GPU")
        mx.random.seed(0)

        def make_qkv(qL, kL, D, qH=8, kH=8, dtype=mx.float16):
            q = mx.random.normal((1, qH, qL, D), dtype)
            k = mx.random.normal((1, kH, kL, D), dtype)
            v = mx.random.normal((1, kH, kL, D), dtype)
            return q, k, v

        # Full attention kernel.
        for D, qL, kL, mask, dtype in product(
            (192, 256),
            (9, 16),
            (31, 512),
            (None, "causal"),
            (mx.bfloat16, mx.float32),
        ):
            with self.subTest(head_dim=D, qL=qL, kL=kL, mask=mask):
                q, k, v = make_qkv(qL, kL, D, 8, 4)
                scale = D**-0.5
                ref = mlx_ref_attn(q, k, v, scale=scale, mask=mask)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask, force_fused=True
                )
                tol = 1e-2 if dtype == mx.bfloat16 else 1e-3
                self.assertTrue(mx.allclose(ref, out, atol=tol, rtol=tol))

        # Vector attention kernel.
        for D in (192, 256, 512):
            with self.subTest(head_dim=D):
                with mlx_tests.scoped_env(MLX_SDPA_D512_MIN_KL="0"):
                    q, k, v = make_qkv(4, 16385, D, 4, 2)
                    scale = D**-0.5
                    ref = mlx_ref_attn(q, k, v, scale=scale)
                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, force_fused=True
                    )
                    self.assertTrue(mx.allclose(ref, out, atol=1e-3, rtol=1e-3))

        # No full attention fused kernels.
        with self.assertRaisesRegex(ValueError, "supports head dims"):
            q, k, v = make_qkv(16, 512, 1024)
            mx.fast.scaled_dot_product_attention(
                q, k, v, scale=1024**-0.5, force_fused=True
            )
        with self.assertRaisesRegex(
            ValueError, "query sequence to be no longer than the key sequence"
        ):
            q, k, v = make_qkv(32, 16, 64)
            mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=64**-0.5,
                mask="causal",
                force_fused=True,
            )

        # No vector attention fused kernels.
        with self.assertRaisesRegex(ValueError, "supports head dims"):
            q, k, v = make_qkv(1, 128, 72)
            mx.fast.scaled_dot_product_attention(
                q, k, v, scale=72**-0.5, force_fused=True
            )
        with self.assertRaisesRegex(ValueError, "GQA factor to be at most 32"):
            q, k, v = make_qkv(8, 128, 64, qH=8, kH=1)
            mx.fast.scaled_dot_product_attention(
                q, k, v, scale=64**-0.5, force_fused=True
            )
        with self.assertRaisesRegex(ValueError, r"requires at least \d+ keys"):
            with mlx_tests.scoped_env(MLX_SDPA_D512_MIN_KL=None):
                q, k, v = make_qkv(1, 512, 512, qH=32, kH=4)
                mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=512**-0.5, force_fused=True
                )

        # No CPU fused kernel.
        with mx.stream(mx.cpu):
            q, k, v = make_qkv(8, 128, 8)
            with self.assertRaisesRegex(ValueError, "require a GPU"):
                mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=64**-0.5, force_fused=True
                )

    @unittest.skipIf(not mx.cuda.is_available(), "CUDA kernel path only")
    def test_sdpa_force_fused_cuda(self):
        if mx.default_device() != mx.gpu:
            self.skipTest("requires GPU")

        def make_qkv(qL, kL, D, qH=8, kH=8):
            q = mx.random.normal((1, qH, qL, D), mx.float16)
            k = mx.random.normal((1, kH, kL, D), mx.float16)
            v = mx.random.normal((1, kH, kL, D), mx.float16)
            return q, k, v

        # Vector attention kernel.
        for D in (64, 96, 128):
            with self.subTest(head_dim=D):
                q, k, v = make_qkv(3, 128, D, 4, 2)
                scale = D**-0.5
                ref = mlx_ref_attn(q, k, v, scale=scale)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, force_fused=True
                )
                self.assertTrue(mx.allclose(ref, out, atol=1e-3, rtol=1e-3))

    def test_sdpa_sliced(self):
        N = 8
        D = 64
        scale = D**-0.5

        for B, T_q, T_kv, offset, mask in product(
            (1, 2, 4),
            (1, 8),
            (256, 512),
            (8, 9, 64, 79),
            (None, "causal"),
        ):
            with self.subTest(B=B, T_q=T_q, T_kv=T_kv, offset=offset, mask=mask):
                q = mx.random.normal((B, N, T_q, D), mx.float16)
                k = mx.random.normal((B, N, T_kv, D), mx.float16)
                v = mx.random.normal((B, N, T_kv, D), mx.float16)

                k = k[..., :offset, :]
                v = v[..., :offset, :]

                ref = mlx_ref_attn(q, k, v, scale=scale, mask=mask)

                for i in range(2):
                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, mask=mask
                    )
                    if B == 1:
                        tolerance = {"rtol": 1e-3, "atol": 1e-3}
                    else:
                        tolerance = {"rtol": 1e-2, "atol": 1e-2}
                    self.assertTrue(mx.allclose(ref, out, **tolerance))

    def test_sdpa_with_lse(self):
        mx.random.seed(0)
        sdpa = mx.fast.scaled_dot_product_attention
        sdpa_lse = mx.fast.scaled_dot_product_attention_with_lse
        tolerances = {
            mx.float32: (1e-4, 1e-4),
            mx.float16: (5e-3, 5e-3),
            mx.bfloat16: (3e-2, 3e-2),
        }
        # (query head dim, value head dim); 96 and 192 have dedicated kernels
        # (192 in float32 does not fit in threadgroup memory and takes the
        # fallback) and 512 needs a NAX GPU, the fallback computes the lse
        # otherwise.
        head_dims = [(64, 64), (96, 96), (96, 64), (128, 128), (192, 192), (512, 512)]
        for (D, Dv), (qL, kL), (qH, kH), mask_kind, use_sinks, dtype in product(
            head_dims,
            [(32, 100), (64, 64), (16, 16)],
            [(4, 4), (8, 2)],
            [None, "causal", "bool", "additive"],
            [False, True],
            [mx.float32, mx.float16, mx.bfloat16],
        ):
            if D == 512 and (dtype != mx.float16 or qL != 32 or use_sinks):
                continue
            if mask_kind == "causal" and qL > kL:
                continue
            with self.subTest(
                D=D,
                Dv=Dv,
                qL=qL,
                kL=kL,
                heads=(qH, kH),
                mask=mask_kind,
                sinks=use_sinks,
                dtype=dtype,
            ):
                q = mx.random.normal((2, qH, qL, D)).astype(dtype)
                k = mx.random.normal((2, kH, kL, D)).astype(dtype)
                v = mx.random.normal((2, kH, kL, Dv)).astype(dtype)
                scale = D**-0.5
                sinks = mx.random.normal((qH,)).astype(dtype) if use_sinks else None
                if mask_kind == "bool":
                    mask = mx.random.uniform(shape=(2, 1, qL, kL)) < 0.6
                    # keep at least one key per row
                    mask[..., 0] = True
                elif mask_kind == "additive":
                    mask = mx.random.normal((1, qH, qL, kL)).astype(dtype)
                else:
                    mask = mask_kind

                out, lse = sdpa_lse(q, k, v, scale=scale, mask=mask, sinks=sinks)
                atol, rtol = tolerances[dtype]
                self.assertEqual(lse.dtype, mx.float32)
                self.assertEqual(lse.shape, (2, qH, qL, 1))
                # The output is float32 as well, it is not rounded
                self.assertEqual(out.dtype, mx.float32)
                self.assertEqual(out.shape, (2, qH, qL, Dv))
                lse_ref = mlx_ref_lse(q, k, scale, mask, sinks)
                self.assertTrue(mx.allclose(lse, lse_ref, atol=atol, rtol=rtol))
                # The output is the one of the function without the lse, which
                # keeps the input type
                out_ref = sdpa(q, k, v, scale=scale, mask=mask, sinks=sinks)
                self.assertEqual(out_ref.dtype, dtype)
                self.assertTrue(mx.allclose(out, out_ref, atol=atol, rtol=rtol))
                # and it is close to a float32 computation on the same inputs
                if dtype != mx.float32:
                    ref32 = mlx_ref_attn(
                        q.astype(mx.float32),
                        k.astype(mx.float32),
                        v.astype(mx.float32),
                        scale,
                        mask,
                        None if sinks is None else sinks.astype(mx.float32),
                    )
                    self.assertTrue(mx.allclose(out, ref32, atol=1e-2, rtol=1e-2))

    def test_sdpa_with_lse_float32_output(self):
        # The output is float32 for half precision inputs on every path (fused
        # kernels, vector fallback, CPU); float32 inputs are unchanged.
        mx.random.seed(0)
        sdpa = mx.fast.scaled_dot_product_attention
        sdpa_lse = mx.fast.scaled_dot_product_attention_with_lse
        streams = [mx.gpu, mx.cpu] if mx.is_available(mx.gpu) else [mx.cpu]
        for dtype, qL, D, stream in product(
            [mx.float16, mx.bfloat16, mx.float32], [4, 33, 64], [64, 128], streams
        ):
            with self.subTest(dtype=dtype, qL=qL, D=D, stream=stream):
                q = mx.random.normal((1, 8, qL, D)).astype(dtype)
                k = mx.random.normal((1, 2, 100, D)).astype(dtype)
                v = mx.random.normal((1, 2, 100, D)).astype(dtype)
                out, lse = sdpa_lse(q, k, v, scale=D**-0.5, stream=stream)
                self.assertEqual(out.dtype, mx.float32)
                self.assertEqual(lse.dtype, mx.float32)
                plain = sdpa(q, k, v, scale=D**-0.5, stream=stream)
                self.assertEqual(plain.dtype, dtype)
                # Same accumulation, rounded once more at the end
                tol = {"atol": 1e-2, "rtol": 1e-2}
                self.assertTrue(mx.allclose(out, plain.astype(mx.float32), **tol))
                if stream == mx.gpu and qL > 8:
                    # The fused kernels round the same float32 accumulator
                    self.assertTrue(mx.array_equal(out.astype(dtype), plain))

    def test_sdpa_plain_output_unchanged(self):
        # Asking for the lse elsewhere must not change the plain function: it
        # returns the input type and the same values with or without sinks,
        # masks and head dims.
        mx.random.seed(0)
        sdpa = mx.fast.scaled_dot_product_attention
        for dtype, D, mask in product(
            [mx.float16, mx.bfloat16], [64, 128, 256], [None, "causal"]
        ):
            with self.subTest(dtype=dtype, D=D, mask=mask):
                q = mx.random.normal((1, 8, 64, D)).astype(dtype)
                k = mx.random.normal((1, 2, 128, D)).astype(dtype)
                v = mx.random.normal((1, 2, 128, D)).astype(dtype)
                a = sdpa(q, k, v, scale=D**-0.5, mask=mask)
                mx.fast.scaled_dot_product_attention_with_lse(
                    q, k, v, scale=D**-0.5, mask=mask
                )
                b = sdpa(q, k, v, scale=D**-0.5, mask=mask)
                self.assertEqual(a.dtype, dtype)
                self.assertTrue(mx.array_equal(a, b))
                ref = mlx_ref_attn(q, k, v, D**-0.5, mask)
                self.assertTrue(mx.allclose(a, ref, atol=5e-2, rtol=5e-2))

    def test_sdpa_with_lse_causal_long_queries(self):
        # With more queries than keys the first rows have no key to attend to
        mx.random.seed(0)
        sdpa_lse = mx.fast.scaled_dot_product_attention_with_lse
        for sinks in [None, mx.random.normal((4,))]:
            for qL, kL in [(40, 24), (9, 1)]:
                q = mx.random.normal((1, 4, qL, 64))
                k = mx.random.normal((1, 4, kL, 64))
                v = mx.random.normal((1, 4, kL, 64))
                _, lse = sdpa_lse(q, k, v, scale=0.125, mask="causal", sinks=sinks)
                lse_ref = mlx_ref_lse(q, k, 0.125, "causal", sinks)
                self.assertFalse(mx.any(mx.isnan(lse)).item())
                self.assertTrue(mx.allclose(lse, lse_ref, atol=1e-4, rtol=1e-4))
                if sinks is None:
                    n_empty = qL - kL
                    self.assertTrue(mx.all(lse[:, :, :n_empty] == -float("inf")))
                    self.assertTrue(mx.all(mx.isfinite(lse[:, :, n_empty:])))

    def test_sdpa_with_lse_fully_masked_rows(self):
        mx.random.seed(0)
        sdpa_lse = mx.fast.scaled_dot_product_attention_with_lse
        inf = float("inf")
        for qL in [1, 8, 9, 32]:
            for kind in ["bool", "additive"]:
                for use_sinks in [False, True]:
                    with self.subTest(qL=qL, mask=kind, sinks=use_sinks):
                        q = mx.random.normal((1, 4, qL, 128))
                        k = mx.random.normal((1, 4, 40, 128))
                        v = mx.random.normal((1, 4, 40, 128))
                        sinks = mx.random.normal((4,)) if use_sinks else None
                        # The last query of the first head sees nothing
                        keep = mx.ones((1, 4, qL, 40), dtype=mx.bool_)
                        keep[:, 0, -1, :] = False
                        if kind == "bool":
                            mask = keep
                        else:
                            mask = mx.where(keep, 0.0, -inf)
                        out, lse = sdpa_lse(q, k, v, scale=0.1, mask=mask, sinks=sinks)
                        lse_ref = mlx_ref_lse(q, k, 0.1, mask, sinks)
                        self.assertFalse(mx.any(mx.isnan(lse)).item())
                        self.assertTrue(mx.allclose(lse, lse_ref, atol=1e-4, rtol=1e-4))
                        if use_sinks:
                            # Only the sink is left
                            self.assertAlmostEqual(
                                lse[0, 0, -1, 0].item(), sinks[0].item(), places=5
                            )
                            self.assertTrue(mx.all(mx.isfinite(out)).item())
                        else:
                            self.assertEqual(lse[0, 0, -1, 0].item(), -inf)
                            self.assertTrue(mx.all(mx.isfinite(lse[:, 1:])).item())

    def test_sdpa_with_lse_noncontiguous(self):
        mx.random.seed(0)
        sdpa_lse = mx.fast.scaled_dot_product_attention_with_lse
        for qL, D in product([5, 40], [64, 128]):
            with self.subTest(qL=qL, D=D):
                # Inputs laid out as [B, T, N, D] and transposed
                q = mx.random.normal((2, qL, 8, D)).transpose(0, 2, 1, 3)
                k = mx.random.normal((2, 100, 2, D)).transpose(0, 2, 1, 3)
                v = mx.random.normal((2, 100, 2, D)).transpose(0, 2, 1, 3)
                # Keys and values sliced from a larger cache
                k_cache = mx.random.normal((2, 2, 128, D))
                v_cache = mx.random.normal((2, 2, 128, D))
                for kk, vv in [(k, v), (k_cache[:, :, :70], v_cache[:, :, :70])]:
                    out, lse = sdpa_lse(q, kk, vv, scale=0.1, mask="causal")
                    out_c, lse_c = sdpa_lse(
                        mx.contiguous(q),
                        mx.contiguous(kk),
                        mx.contiguous(vv),
                        scale=0.1,
                        mask="causal",
                    )
                    self.assertTrue(mx.allclose(out, out_c, atol=1e-5, rtol=1e-5))
                    self.assertTrue(mx.allclose(lse, lse_c, atol=1e-5, rtol=1e-5))
                    ref = mlx_ref_lse(q, kk, 0.1, "causal")
                    self.assertTrue(mx.allclose(lse, ref, atol=1e-4, rtol=1e-4))

    def test_sdpa_with_lse_cpu(self):
        mx.random.seed(0)
        sdpa_lse = mx.fast.scaled_dot_product_attention_with_lse
        q = mx.random.normal((1, 8, 20, 64))
        k = mx.random.normal((1, 2, 50, 64))
        v = mx.random.normal((1, 2, 50, 64))
        sinks = mx.random.normal((8,))
        for mask in [None, "causal"]:
            out, lse = sdpa_lse(
                q, k, v, scale=0.125, mask=mask, sinks=sinks, stream=mx.cpu
            )
            self.assertEqual(lse.dtype, mx.float32)
            self.assertEqual(lse.shape, (1, 8, 20, 1))
            self.assertTrue(
                mx.allclose(lse, mlx_ref_lse(q, k, 0.125, mask, sinks), atol=1e-5)
            )
            out_ref = mlx_ref_attn(q, k, v, 0.125, mask, sinks)
            self.assertTrue(mx.allclose(out, out_ref, atol=1e-5, rtol=1e-5))
        if mx.default_device() == mx.gpu:
            with self.assertRaises(ValueError):
                sdpa_lse(q, k, v, scale=0.125, force_fused=True, stream=mx.cpu)

    def test_sdpa_with_lse_vector(self):
        # No vector kernel writes the lse: it is computed by the fallback
        mx.random.seed(0)
        sdpa_lse = mx.fast.scaled_dot_product_attention_with_lse
        for qL, D in product([1, 4, 8], [64, 128, 192]):
            with self.subTest(qL=qL, D=D):
                q = mx.random.normal((1, 8, qL, D))
                k = mx.random.normal((1, 2, 300, D))
                v = mx.random.normal((1, 2, 300, D))
                mask = mx.random.uniform(shape=(1, 1, qL, 300)) < 0.5
                mask[..., 0] = True
                for m in [None, "causal", mask]:
                    out, lse = sdpa_lse(q, k, v, scale=D**-0.5, mask=m)
                    ref = mlx_ref_lse(q, k, D**-0.5, m)
                    self.assertTrue(mx.allclose(lse, ref, atol=1e-4, rtol=1e-4))
                    out_ref = mlx_ref_attn(q, k, v, D**-0.5, m)
                    self.assertTrue(mx.allclose(out, out_ref, atol=1e-4, rtol=1e-4))
                if mx.default_device() == mx.gpu:
                    with self.assertRaises(ValueError):
                        sdpa_lse(q, k, v, scale=D**-0.5, force_fused=True)

    def test_sdpa_with_lse_force_fused(self):
        if mx.default_device() != mx.gpu:
            self.skipTest("requires GPU")
        mx.random.seed(0)
        q = mx.random.normal((1, 8, 64, 64))
        k = mx.random.normal((1, 2, 128, 64))
        v = mx.random.normal((1, 2, 128, 64))
        mask = mx.random.uniform(shape=(1, 1, 64, 128)) < 0.5
        mask[..., 0] = True
        for m in [None, "causal", mask]:
            out, lse = mx.fast.scaled_dot_product_attention_with_lse(
                q, k, v, scale=0.125, mask=m, force_fused=True
            )
            self.assertTrue(
                mx.allclose(lse, mlx_ref_lse(q, k, 0.125, m), atol=1e-4, rtol=1e-4)
            )
        # The lse is not written for the fused kernels under differentiation
        with self.assertRaises(ValueError):
            mx.grad(
                lambda q: sum(
                    x.sum()
                    for x in mx.fast.scaled_dot_product_attention_with_lse(
                        q, k, v, scale=0.125, force_fused=True
                    )
                )
            )(q)

    def test_sdpa_with_lse_grad(self):
        mx.random.seed(0)
        sdpa_lse = mx.fast.scaled_dot_product_attention_with_lse
        weights = mx.random.normal((1, 8, 40, 1))
        cot = mx.random.normal((1, 8, 40, 64))
        mask = mx.random.uniform(shape=(1, 1, 40, 70)) < 0.6
        mask[..., 0] = True

        for m, use_sinks, (qL, kL) in product(
            [None, "causal", mask], [False, True], [(40, 70), (7, 70)]
        ):
            with self.subTest(mask=type(m).__name__, sinks=use_sinks):
                q = mx.random.normal((1, 8, qL, 64))
                k = mx.random.normal((1, 2, kL, 64))
                v = mx.random.normal((1, 2, kL, 64))
                sinks = mx.random.normal((8,)) if use_sinks else None
                mm = m[..., :qL, :kL] if isinstance(m, mx.array) else m
                w, c = weights[:, :, :qL], cot[:, :, :qL]

                def fast(q, k, v):
                    o, lse = sdpa_lse(q, k, v, scale=0.125, mask=mm, sinks=sinks)
                    return (o * c).sum() + (lse * w).sum()

                def ref(q, k, v):
                    o = mlx_ref_attn(q, k, v, 0.125, mm, sinks)
                    lse = mlx_ref_lse(q, k, 0.125, mm, sinks)
                    return (o * c).sum() + (lse * w).sum()

                g_fast = mx.grad(fast, argnums=(0, 1, 2))(q, k, v)
                g_ref = mx.grad(ref, argnums=(0, 1, 2))(q, k, v)
                for gf, gr in zip(g_fast, g_ref):
                    self.assertTrue(mx.allclose(gf, gr, atol=1e-4, rtol=1e-4))

                # A cotangent for the lse only must not be dropped
                def fast_lse(q, k, v):
                    return sdpa_lse(q, k, v, scale=0.125, mask=mm, sinks=sinks)[1].sum()

                gq = mx.grad(fast_lse)(q, k, v)
                self.assertGreater(mx.abs(gq).max().item(), 1e-3)
                gq_ref = mx.grad(lambda q: mlx_ref_lse(q, k, 0.125, mm, sinks).sum())(q)
                self.assertTrue(mx.allclose(gq, gq_ref, atol=1e-4, rtol=1e-4))

    def test_sdpa_with_lse_merge_chunks(self):
        # The use case: attention over separate sets of keys and values, e.g.
        # a prefix shared between sequences or a segmented kv cache.
        mx.random.seed(0)
        sdpa_lse = mx.fast.scaled_dot_product_attention_with_lse
        for dtype in [mx.float32, mx.float16, mx.bfloat16]:
            for D, qL, use_sinks, mask_kind in product(
                [64, 128, 256], [1, 12, 64], [False, True], [None, "bool"]
            ):
                with self.subTest(
                    dtype=dtype, D=D, qL=qL, sinks=use_sinks, mask=mask_kind
                ):
                    kL = 200
                    q = mx.random.normal((2, 8, qL, D)).astype(dtype)
                    k = mx.random.normal((2, 2, kL, D)).astype(dtype)
                    v = mx.random.normal((2, 2, kL, D)).astype(dtype)
                    sinks = mx.random.normal((8,)).astype(dtype) if use_sinks else None
                    scale = D**-0.5
                    mask = None
                    if mask_kind:
                        mask = mx.random.uniform(shape=(2, 1, qL, kL)) < 0.5
                        mask[..., 0] = True
                        # Rows without keys in the second chunk
                        mask[:, :, 0, 100:] = False

                    # The reference is float32 attention on the same inputs
                    ref = mlx_ref_attn(
                        q.astype(mx.float32),
                        k.astype(mx.float32),
                        v.astype(mx.float32),
                        scale,
                        mask,
                        None if sinks is None else sinks.astype(mx.float32),
                    )
                    atol = _f32_out_atol(qL, D)

                    # Two chunks at an arbitrary boundary. The sinks are one
                    # more key and belong to a single chunk.
                    for split in [1, 37, 100, 199]:
                        sl_1 = slice(0, split)
                        sl_2 = slice(split, kL)
                        m_1 = None if mask is None else mask[..., sl_1]
                        m_2 = None if mask is None else mask[..., sl_2]
                        o_1, lse_1 = sdpa_lse(
                            q, k[:, :, sl_1], v[:, :, sl_1], scale=scale,
                            mask=m_1, sinks=sinks,
                        )  # fmt: skip
                        o_2, lse_2 = sdpa_lse(
                            q, k[:, :, sl_2], v[:, :, sl_2], scale=scale, mask=m_2
                        )
                        self.assertEqual(o_1.dtype, mx.float32)
                        o, lse = merge_attention(o_1, lse_1, o_2, lse_2)
                        # Nothing was rounded: the merge matches float32
                        self.assertTrue(mx.allclose(o, ref, atol=atol, rtol=1e-4))
                        # A single rounding at the end differs from the rounded
                        # reference in a few elements at most, rounding the
                        # chunks first as well in many more. Only where the
                        # kernel is exact: the NAX kernels' own error is above
                        # the rounding of half precision.
                        if dtype != mx.float32 and atol < 1e-3:
                            ref_r = ref.astype(dtype)
                            once = (o.astype(dtype) != ref_r).astype(mx.float32).mean()
                            o_r, _ = merge_attention(
                                o_1.astype(dtype), lse_1, o_2.astype(dtype), lse_2
                            )
                            twice = (
                                (o_r.astype(dtype) != ref_r).astype(mx.float32).mean()
                            )
                            self.assertLess(once.item(), 0.05)
                            self.assertLessEqual(once.item(), twice.item())
                        lse_ref = mlx_ref_lse(q, k, scale, mask, sinks)
                        self.assertTrue(mx.allclose(lse, lse_ref, atol=1e-3, rtol=1e-3))

    def test_sdpa_with_lse_merge_causal_chunks(self):
        # Causal attention over a cache split in a prefix and a new segment
        mx.random.seed(0)
        sdpa_lse = mx.fast.scaled_dot_product_attention_with_lse
        prefix, new, D = 150, 50, 128
        q = mx.random.normal((1, 8, new, D))
        k = mx.random.normal((1, 8, prefix + new, D))
        v = mx.random.normal((1, 8, prefix + new, D))
        ref = mlx_ref_attn(q, k, v, D**-0.5, "causal")
        # Every query attends to the whole prefix, the new segment is causal
        o_1, lse_1 = sdpa_lse(q, k[:, :, :prefix], v[:, :, :prefix], scale=D**-0.5)
        o_2, lse_2 = sdpa_lse(
            q, k[:, :, prefix:], v[:, :, prefix:], scale=D**-0.5, mask="causal"
        )
        o, _ = merge_attention(o_1, lse_1, o_2, lse_2)
        self.assertTrue(mx.allclose(o, ref, atol=1e-5, rtol=1e-4))

        # The first query of a later chunk of a ring has nothing to attend to
        # in a chunk of future keys: its lse is -inf and drops out of the merge
        o_f, lse_f = sdpa_lse(
            q,
            k[:, :, prefix:],
            v[:, :, prefix:],
            scale=D**-0.5,
            mask=mx.tril(mx.ones((new, new), dtype=mx.bool_), k=-1),
        )
        self.assertEqual(lse_f[0, 0, 0, 0].item(), -float("inf"))
        o, lse = merge_attention(o_1, lse_1, o_f, lse_f)
        self.assertFalse(mx.any(mx.isnan(o)).item())
        self.assertTrue(mx.allclose(o[:, :, 0], o_1[:, :, 0], atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner(failfast=True)
