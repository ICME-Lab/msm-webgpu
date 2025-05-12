# ICME WebGPU MSM

An implementation of the [cuZK paper](https://eprint.iacr.org/2022/1321.pdf) in Rust for Multi-Scalar Multiplication (MSM) over the BN254 curve for WebGPU.

For an introduction to ZK proving using WebGPU, see [this recent post](https://blog.zksecurity.xyz/posts/webgpu/) by zkSecurity.

This work is built upon the existing [ZPrize 2023 submission](https://github.com/td-kwj-zp2023/webgpu-msm-bls12-377) by Tal Derei and Koh Wei Jie. Their [documentation](https://hackmd.io/HNH0DcSqSka4hAaIfJNHEA) will largely serve as a documentation for this implementation, too. 

Further reads on the optimisations applied:
- [Optimizing Montgomery Multiplication in WebAssembly](https://baincapitalcrypto.com/optimizing-montgomery-multiplication-in-webassembly/) by Koh Wei Jie.
- [Optimizing Barrett Reduction: Tighter Bounds Eliminate Redundant Subtractions](https://blog.zksecurity.xyz/posts/barrett-tighter-bound/) by Suneal Gong.
- [Signed Bucket Indexes for Multi-Scalar Multiplication (MSM)](https://hackmd.io/@drouyang/signed-bucket-index) by drouyang.eth.

## Test


For $2^{16}$ MSMs:
```
wasm-pack test --chrome --test test_webgpu_msm_cuzk_16
```

For $2^{17}$ MSMs:
```
wasm-pack test --chrome --test test_webgpu_msm_cuzk_17
```

For $2^{18}$ MSMs:
```
wasm-pack test --chrome --test test_webgpu_msm_cuzk_18
```

For $2^{19}$ MSMs:
```
wasm-pack test --chrome --test test_webgpu_msm_cuzk_19
```

For $2^{20}$ MSMs:
```
wasm-pack test --chrome --test test_webgpu_msm_cuzk_20
```


## Future work

- Implement cuzk on other curves.
- Implement cuzk on other libraries other than `halo2curves`, such as `arkworks`.
- Explore the trade-off between the running
time and the extra storage space needed by parallel Pippenger algorithm on webGPU as the paper [Elastic MSM](https://eprint.iacr.org/2024/057.pdf) suggests.


