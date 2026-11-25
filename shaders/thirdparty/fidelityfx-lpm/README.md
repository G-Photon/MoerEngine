# FidelityFX LPM

Unmodified AMD FidelityFX LPM headers, distributed under the MIT license embedded
in both files.

- Repository: https://github.com/GPUOpen-Effects/FidelityFX-LPM
- Commit: `ed6ecd5b8963d2ec24603809b90ecfa00a1c3614`
- Original directory: `ffx-lpm/`
- `ffx_a.h` SHA256: `1a25c6fe1b6278b8f8ec34f11ba280d93fc8ad6b025ce4703fa50a61a9126daa`
- `ffx_lpm.h` SHA256: `7a7c0f376c89bcecd51fe8a5d8ba3b34ec59841e9915cb5b3e16ee87e9448c4d`

Moer uses CPU `LpmSetup` and FP32 shader `LpmFilter`, with the same
`LPM_CONFIG_709_709` preset and no additional shoulder contrast. The input and
output are linear Rec.709; the existing sRGB render target supplies display
encoding. This integration does not enable an HDR display swapchain.

LPM's exposure-range parameter is the number of stops between `hdrMax` and the
input's 18% middle gray. It shapes the tone curve and is separate from Moer's
existing exposure EV, which is applied once before LPM. The default vendor
settings map 0.18 gray to 0.18 and 256 gray to 1, before display encoding.
