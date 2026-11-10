# Radio Telescope RFI Mitigation — Data Pipeline

Code I wrote as an undergraduate research assistant at the **Experimental Astronomy Lab (Florida Space Institute / UCF)** from 2024 to 2025. The lab studies how radio telescopes can keep working when they share spectrum with cell networks (5G) and other transmitters. My part was the software that captures raw telescope data from an FPGA, checks it, correlates it and plots it, so the team could study and filter radio-frequency interference (RFI).

I'm a co-author on the lab's abstract for the NRDZ (National Radio Dynamic Zone) workshop at Green Bank: *"High-dynamic Range Radio Astronomy Systems, Interference Mitigation Strategies, and an Experimental Setup for Radio Dynamic Zones"* (Roshi, Armas, Westcott, **Dellinger**, Burrett, Liu, Werthimer). The abstract and poster are in [`Documentation/`](Documentation/).

## The system

```
 Telescope + reference antenna (4 inputs, 10–3400 MHz RF front end)
                  │
        RFSoC 4x2 FPGA: 14-bit ADCs, up to ~120 MHz bandwidth
                  │  UDP packets over 100 GbE
                  ▼
   Acquisition PC  ──►  Acquisition/FPGA        capture packets to disk
                   ──►  Processing/Validation   check FPGA packet counters for dropped data
                   ──►  Acquisition/CarseData   CPU correlator (auto/cross spectra)
                   ──►  Processing/Gpu-Acceleration   CUDA/cuFFT port of the correlator
                   ──►  Processing/Filtering    median filtering to flag RFI
                   ──►  Processing/Plotting     spectra, waterfalls, cross-correlation plots
```

## What I built

| Area | What it does | Stack |
|---|---|---|
| `Acquisition/FPGA/` | Receives UDP packets from the RFSoC board, does FFTs and averages them, writes timestamped auto-spectra for all 4 ADC channels, and plots RFI | Python, NumPy, SciPy, sockets |
| `Processing/Validation/CheckCount.py` | Reads the packet counters the FPGA embeds across multi-file recordings to find dropped or out-of-order packets | Python, NumPy |
| `Acquisition/CarseData/` | `carseoption.cpp` (all mine): file-processing options and per-ADC-channel output. `corr+carse.c`: my changes to the lab's existing C correlator for our data format | C, C++, FFTW |
| `Processing/Gpu-Acceleration/` | A port of the correlator to the GPU. Includes cuFFT tests, a block-processing correlator (`block_gpu_complex.cu`) and `CudaRunner.py`, which builds and benchmarks the variants | CUDA, cuFFT, Python |
| `Processing/Filtering/Median-Filtering/` | Median and moving-average filtering over time and frequency to flag RFI in the recorded spectra | Python, SciPy |
| `Processing/Plotting/` | `DLITE/`: my first-year analysis of data from the DLITE station in Malabar, FL. `carse/`: second-year analysis of data we recorded in the lab | Python, Matplotlib, Astropy |

## Credit

This was team research, so here is what isn't mine:
- The base software correlator (`corr_cpu_complex.c`, and its header in the CUDA versions) was written by Randall Wayth in 2008. I adapted it for this project and ported it to the GPU.
- `Processing/Plotting/DLITE/dliteTools.py` was provided by the lab.
- The hardware design and research direction come from Dr. D. Anish Roshi and the EAL team.

## Status

This is research code, not a packaged library. Paths point to the lab's acquisition machine (`/mnt/carsedat/`), and the raw `.dat` recordings are not included because they're tens of GB. The GPU correlator is a work in progress.
