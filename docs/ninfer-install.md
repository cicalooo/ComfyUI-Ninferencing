# NInfer CLI install

The pack runs whatever `ninfer` it finds (see Discovery in the main README). Install one matching your GPU.

## Linux, RTX 30/40

Prebuilt sm_86 binaries, no CUDA compile:

```bash
curl -fsSL https://github.com/cicalooo/ninfer-linux/releases/latest/download/install.sh | bash
```

Installs to `/opt/ninfer/ninfer` (RTX 40 uses the same archive). Details, runtime libraries, and a user-prefix install: [cicalooo/ninfer-linux](https://github.com/cicalooo/ninfer-linux).

## Other GPUs

RTX 50 and RTX PRO 6000 are not in that archive. Build or pull [ninfer-all](https://github.com/iamwavecut/ninfer-all), then point the pack at it with `NINFER_EXECUTABLE`. See [rtx-6000.md](rtx-6000.md) for RTX PRO 6000 Blackwell specifics.

## Windows

Put `ninfer.exe` on `PATH`, or at `C:\ninfer\ninfer.exe`, or set `NINFER_EXECUTABLE`.
