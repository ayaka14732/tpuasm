# 版本兼容性

本页按 tpuasm 版本列出支持的组合。所有版本都只支持 Linux x86-64，并需要支持 C++17 的 g++ 编译随包分发的原生桥接。

## 使用旧版本 libtpu

tpuasm 的新版本可能不再支持旧的 libtpu 构建。使用的 libtpu 不在“未发布”一节时，在下面已发布的版本中找到仍支持它的最新版本，固定安装该版本，例如：

```sh
python -m pip install 'tpuasm==0.1.0'
```

tpuasm 只删除已随某个发布版本发出的 libtpu 构建，所以本页登记过的每个构建都有可以安装的 tpuasm 版本。删除写在[更新日志](changelog.md)的“移除”一节中。

## JAX

编译来源捕获会改写 JAX 的两个 Pallas lowering 函数。tpuasm 核对已安装 JAX 中这两个函数源码字节的 SHA-256，源码相同即可接受，不限定 JAX 版本号，详见[TC 编译来源映射](design/tc_source_mapping.md#jax-lowering)。下表中的 JAX 是该版本测试通过的版本，也是摘要对应的上游 commit。只做汇编和反汇编时不使用 JAX。

## 支持的组合

表中“编解码”指汇编与反汇编，“来源”指 `compiler_source_mapping` 的编译来源捕获。“—”表示该构建没有这个目标的后端。

### 未发布

测试所用 JAX：`0.12.0.dev20260926+886d2370c1`（commit `886d2370c1c959d210f522e352e3ddc6bcff7d6c`），jaxlib `0.11.2`。

| Python | libtpu | GNU build-id | TPU v4 TC | TPU v4 BCS | TPU v6e TC |
|---|---|---|---|---|---|
| CPython 3.14t | `0.0.48` | `3310a7c8c137cd515c7a2ba1ce2ea38c` | 编解码、来源 | 编解码 | — |
| CPython 3.14t | `0.0.49` | `97e27df7268da25ab03e455e30dd86b0` | 编解码、来源 | 编解码 | 编解码、来源 |

各构建登记的来源 hook 不同：0.0.48 只登记发射、替换、BF16 合并和 store 注释 hook，load 合并、DMA 展开、MXU prep 改写等 hook 只在 0.0.49 登记，所以同一 kernel 在 0.0.48 上有来源的指令更少。

### 0.1.0

测试所用 JAX：`0.12.0.dev20260926+886d2370c1`（commit `886d2370c1c959d210f522e352e3ddc6bcff7d6c`），jaxlib `0.11.2`。

| Python | libtpu | GNU build-id | TPU v4 TC | TPU v4 BCS | TPU v6e TC |
|---|---|---|---|---|---|
| CPython 3.14t | `0.0.48.dev20260912+nightly` | `825044f87748f172b7db935904b3f754` | 编解码、来源 | 编解码 | — |
| CPython 3.14t | `0.0.48` | `3310a7c8c137cd515c7a2ba1ce2ea38c` | 编解码、来源 | 编解码 | — |
| CPython 3.14t | `0.0.49` | `97e27df7268da25ab03e455e30dd86b0` | 编解码、来源 | 编解码 | 编解码、来源 |

0.0.48 系列只登记发射、替换、BF16 合并和 store 注释 hook，load 合并 hook 只在 0.0.49 登记。
