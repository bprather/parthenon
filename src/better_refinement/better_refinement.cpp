//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#include <algorithm>
#include <exception>
#include <memory>
#include <utility>

#include "amr_criteria.hpp"
#include "better_refinement.hpp"
#include "interface/StateDescriptor.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

namespace parthenon {
namespace BetterRefinement {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto ref = std::make_shared<StateDescriptor>("Refinement");
  Params& params = ref->AllParams();

  int numcrit = 0;
  while(true) {
    std::string block_name = "Refinement" + std::to_string(numcrit);
    if (!pin->DoesBlockExist(block_name)) {
      break;
    }
    std::string method = pin->GetOrAddString(block_name, "method", "PLEASE SPECIFY method");
    ref->amr_criteria.push_back(
      AMRCriteria::MakeAMRCriteria(method, pin, block_name)
    );
    numcrit++;
  }
  return std::move(ref);
}


int CheckAllRefinement(Container<Real>& rc) {
  MeshBlock *pmb = rc.pmy_block;
  int delta_level = -1;
  for (auto &pkg : pmb->packages) {
    auto& desc = pkg.second;
    int package_delta_level = -1;
    // call package specific function, if set
    if (desc->CheckRefinement != nullptr) {
        package_delta_level = std::max(package_delta_level, desc->CheckRefinement(rc));
        if (package_delta_level == 1) {
          delta_level = 1;
          break;
        }
    }
    // call parthenon criteria that were registered
    for (auto & amr : desc->amr_criteria) {
      int temp_delta = (*amr)(rc);
      if (temp_delta == 0) {
        package_delta_level = 0;
      } else if (temp_delta == 1) {
        if (rc.pmy_block->loc.level < amr->max_level) {
          delta_level = 1;
        } else {
          package_delta_level = 0;
        }
        break;
      }
    }
    delta_level = std::max(delta_level, package_delta_level);
  }

  return delta_level;
}

int FirstDerivative(Variable<Real>& q,
                    const Real refine_criteria, const Real derefine_criteria) {
  Real maxd = 0.0;
  const int dim1 = q.GetDim1();
  const int dim2 = q.GetDim2();
  const int dim3 = q.GetDim3();
  int klo=0, khi=1, jlo=0, jhi=1, ilo=0, ihi=1;
  if (dim3 > 1) {
    klo = 1;
    khi = dim3-1;
  }
  if (dim2 > 1) {
    jlo = 1;
    jhi = dim2-1;
  }
  if (dim1 > 1) {
    ilo = 1;
    ihi = dim1-1;
  }
  for (int k=klo; k<khi; k++) {
    for (int j=jlo; j<jhi; j++) {
      for (int i=ilo; i<ihi; i++) {
        Real scale = std::abs(q(k,j,i));
        Real d = 0.5*std::abs((q(k,j,i+1)-q(k,j,i-1)))/(scale+1.e-16);
        maxd = (d > maxd ? d : maxd);
        if (dim2 > 1) {
          d = 0.5*std::abs((q(k,j+1,i)-q(k,j-1,i)))/(scale+1.e-16);
          maxd = (d > maxd ? d : maxd);
        }
        if (dim3 > 1) {
          d = 0.5*std::abs((q(k+1,j,i) - q(k-1,j,i)))/(scale+1.e-16);
          maxd = (d > maxd ? d : maxd);
        }
      }
    }
  }
  if (maxd > refine_criteria) return 1;
  if (maxd < derefine_criteria) return -1;
  return 0;;
}

} // namespace BetterRefinement
} // namespace parthenon
