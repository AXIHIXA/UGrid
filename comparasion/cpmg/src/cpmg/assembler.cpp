#include "cpmg/assembler.h"


namespace cpmg
{

int assemble2DPoissonProblem(
        const pybind11::array_t<bool> & boundaryMask,
        const pybind11::array_t<Float> & laplacian,
        const pybind11::array_t<Float> & boundaryValue,
        std::vector<Float> & val,
        std::vector<int> & col,
        std::vector<int> & ptr,
        std::vector<Float> & rhs
)
{
    static constexpr int kDim = 2;

    auto m = boundaryMask.unchecked<kDim>();
    auto f = laplacian.unchecked<kDim>();
    auto b = boundaryValue.unchecked<kDim>();

//    // Assert all three NumPy arrays are square matrices of the same size.
//    // Note: assert works only in Debug builds.
//    assert(m.ndim() == kDim and f.ndim() == kDim and b.ndim() == kDim and
//           m.shape(0) == m.shape(1) and f.shape(0) == f.shape(1) and b.shape(0) == b.shape(1) and
//           m.shape(0) == f.shape(0) and f.shape(0) == b.shape(0) and
//           m.shape(1) == f.shape(1) and f.shape(1) == b.shape(1));

    auto n = static_cast<int>(m.shape(0));
    int n2 = n * n;  // Number of points in the grid.

    // We use 5-point stencil, so the matrix will have at most n2 * 5 nonzero elements.
    ptr.clear();
    ptr.reserve(n2 + 1);
    ptr.push_back(0);

    col.clear();
    col.reserve(n2 * 5);

    val.clear();
    val.reserve(n2 * 5);

    rhs.resize(n2);

    for (int i = 0, k = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j, ++k)
        {
            // i, j: The 2D row, column coordinates.
            // k:    The 1D index of position (i, j).

            if (m(i, j))
            {
                // Boundary point. Use Dirichlet condition.
                col.emplace_back(k);
                val.emplace_back(1.0);

                rhs[k] = b(i, j);
            }
            else
            {
                // Interior point. Use 5-point finite difference stencil.
                col.emplace_back(k - n);
                val.emplace_back(1.0);

                col.emplace_back(k - 1);
                val.emplace_back(1.0);

                col.emplace_back(k);
                val.emplace_back(-4.0);

                col.emplace_back(k + 1);
                val.emplace_back(1.0);

                col.emplace_back(k + n);
                val.emplace_back(1.0);

                rhs[k] = f(i, j);
            }

            ptr.emplace_back(static_cast<int>(col.size()));
        }
    }

    return n2;
}

}  // namespace cpmg
