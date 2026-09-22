# Computer Vision & Feature Extraction Pipelines

## 1. Differential Optical Flow and Motion Estimation
Optical flow quantifies 2D motion vector fields between consecutive video frames $I(x, y, t)$ and $I(x, y, t + \delta t)$ based on the brightness constancy constraint equation:
$$I_x u + I_y v + I_t = 0$$
where $I_x, I_y$ are spatial intensity gradients, $I_t$ is temporal gradient, and $(u, v)$ represent horizontal and vertical velocity components.
- **Lucas-Kanade Method**: Assumes motion constancy within an $n \times n$ local neighborhood window $\Omega$. It solves an overdetermined system via weighted least-squares:
  $$\begin{bmatrix} u \\ v \end{bmatrix} = (A^T A)^{-1} A^T b$$
  Singular values of $A^T A$ reveal aperture problem ambiguities in textureless regions.
- **Horn-Schunck Method**: Formulates a global energy functional combining the brightness constancy constraint with a global smoothness regularizer:
  $$E = \iint \left( (I_x u + I_y v + I_t)^2 + \alpha^2 \left( \|\nabla u\|^2 + \|\nabla v\|^2 \right) \right) dx dy$$
  The variational problem is iteratively solved over pixel grids using Gauss-Seidel relaxation.

## 2. Multi-Scale Blob and Corner Detection
- **Laplacian of Gaussian (LoG)**: Evaluates multi-scale blob detection by convolving an image with scale-normalized Mexican-hat filters:
  $$\nabla^2_{\text{norm}} L(x, y; \sigma) = \sigma^2 (L_{xx} + L_{yy})$$
  Extrema across 3D scale-space $(x, y, \sigma)$ identify invariant circular structural centers.
- **Difference of Gaussians (DoG)**: Computes an efficient approximation of the scale-normalized Laplacian by subtracting blurred images at adjacent octave scales: $D(x, y; \sigma) = L(x, y; k\sigma) - L(x, y; \sigma)$.
- **Harris Corner Detector**: Evaluates the second-moment autocorrelation matrix:
  $$M = \sum_{(x,y) \in W} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$
  Corners are localized using the response metric $R = \det(M) - k \cdot (\text{trace}(M))^2$ without computing explicit matrix eigenvalues.

## 3. Boundary Localization and Edge Processing
The Canny edge detection pipeline guarantees single-pixel edge responses while minimizing spurious responses:
1. **Gaussian Filtering**: Suppresses high-frequency sensor noise.
2. **Gradient Estimation**: Applies directional Sobel or Scharr operators to derive magnitude $G = \sqrt{I_x^2 + I_y^2}$ and orientation $\theta = \arctan(I_y / I_x)$.
3. **Non-Maximum Suppression (NMS)**: Compares pixel gradient magnitudes along directional normals ($0^\circ, 45^\circ, 90^\circ, 135^\circ$), suppressing non-peak intensities.
4. **Hysteresis Thresholding**: Double-threshold filtering ($T_{\text{high}}, T_{\text{low}}$) retains strong edges and conditionally admits connected weak edge segments.

## 4. Statistical and Structural Texture Descriptors
- **Gray-Level Co-occurrence Matrix (GLCM)**: Analyzes second-order joint conditional probability densities $P(i, j | d, \theta)$ of pixel pairs separated by distance $d$ along angle $\theta$:
  - *Contrast*: $\sum_{i,j} |i - j|^2 P(i,j)$ measures local variations.
  - *Homogeneity*: $\sum_{i,j} \frac{P(i,j)}{1 + |i - j|}$ detects diagonal matrix concentration.
  - *Energy / ASM*: $\sum_{i,j} P(i,j)^2$ captures organizational uniformity.
- **Local Binary Patterns (LBP)**: Encodes local micro-textural structures by thresholding circular neighborhoods $P$ at radius $R$ against the center pixel, generating rotation-invariant and uniform texture codes.
