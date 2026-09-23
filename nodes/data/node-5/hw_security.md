# Hardware Security & Post-Quantum Cryptoprocessors

## 1. Post-Quantum Lattice Cryptography Hardware Architectures
Modern post-quantum cryptography standards—specifically ML-KEM (formerly Kyber) and ML-DSA (formerly Dilithium)—are founded on the hardness of Module Learning With Errors (M-LWE) and Module Short Integer Solution (M-SIS) problems over the polynomial ring:
$$R_q = \mathbb{Z}_q[X] / (X^{256} + 1)$$
where $q = 3329$ for ML-KEM and $q = 8380417$ for ML-DSA. The performance bottleneck in hardware implementations lies in polynomial multiplication over $R_q$, which scales quadratically $O(n^2)$ using schoolbook algorithms.

## 2. Number Theoretic Transform (NTT) Pipelines
To accelerate polynomial multiplication to $O(n \log n)$, high-throughput cryptographic engines implement the Number Theoretic Transform (NTT).
- **Butterfly Units**: Architectures employ Cooley-Tukey (decimation-in-time) and Gentleman-Sande (decimation-in-frequency) butterflies configured with dual-port block RAMs.
- **PipeNTT Architecture**: Pipelined NTT implementations interleave butterfly computation stages with conflict-free memory addressing logic to achieve continuous streaming throughput across polynomial coefficient vectors without stall cycles.
- **Twiddle Factor Management**: Precomputed twiddle factors $\omega^k \pmod q$ are stored in dedicated ROM banks and accessed sequentially using bit-reversed addressing counters.

## 3. Fast Modular Reduction Datapaths
Modular reduction forms the critical path delay in arithmetic logic units:
- **Montgomery Reduction**: Translates division-based reduction into word-level multiplications and bit-shifts, requiring Montgomery domain conversions.
- **Plantard Modular Reduction**: Eliminates dividend adjustments by pre-multiplying constants, reducing DSP block usage and combinational path depth on FPGA fabric.
- **Barrett Reduction**: Precomputes approximation multipliers $\mu \approx \lfloor 2^k / q \rfloor$, bounded to integer widths suitable for native FPGA DSP48E2 slices.

## 4. Side-Channel Attack (SCA) Vectors on FPGA Systems
Unmanned Aerial Vehicle (UAV) secure telemetry links and embedded FPGA accelerators are vulnerable to physical emission analysis:
- **Correlation Power Analysis (CPA)**: Measures dynamic power consumption fluctuations across clock transitions to extract Hamming distance or Hamming weight models of secret polynomial coefficients.
- **Electromagnetic Analysis (EMA)**: Localized near-field probes capture high-frequency radiation leakage from arithmetic datapaths, bypassing coarse board-level decoupling capacitors.
- **Fault Injection Attacks (FIA)**: Clock glitching or voltage starvation during SHAKE-128 / SHAKE-256 sponge permutation rounds to force faulty signature states that leak private keys.

## 5. Hardware Countermeasures and Masking Schemes
Protecting cryptoprocessors on evaluation boards such as AMD-Xilinx ZCU102 and ZCU104 requires layered defense architectures:
- **First-Order and Higher-Order Masking**: Sensitive variables $x$ are split into random shares $x = x_1 \oplus x_2$ (Boolean) or $x = (x_1 + x_2) \pmod q$ (Arithmetic), ensuring instantaneous power leakage is statistically independent of secret data.
- **Mask Conversion Gadgets**: High-speed, provably secure Arithmetic-to-Boolean (A2B) and Boolean-to-Arithmetic (B2A) conversion networks prevent leakage during non-linear operations.
- **Bus Scrambling & AXI4 Security Wrappers**: Inter-module AXI4 interconnects incorporate randomized address shuffling and cryptographic bus obfuscation to mitigate physical probe tapping.
- **Randomized Execution & Dummy Operations**: Arithmetic datapaths inject jitter into instruction schedules and execute dummy NTT operations to decorrelate power consumption traces from algorithmic execution time.
