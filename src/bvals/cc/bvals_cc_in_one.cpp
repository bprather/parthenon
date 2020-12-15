//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#include <memory>
#include <sstream> // debug
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "bvals/bvals_interfaces.hpp"
#include "bvals_cc_in_one.hpp"
#include "config.hpp"
#include "interface/meshblock_data_iterator.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"

namespace parthenon {

namespace cell_centered_bvars {

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesSetSame(int ox, int &s, int &e,
//                                                   const IndexRange &bounds)
//  \brief Calculate indices for SetBoundary routines for buffers on the same
//  level

void CalcIndicesSetSame(int ox, int &s, int &e, const IndexRange &bounds) {
  if (ox == 0) {
    s = bounds.s;
    e = bounds.e;
  } else if (ox > 0) {
    s = bounds.e + 1;
    e = bounds.e + NGHOST;
  } else {
    s = bounds.s - NGHOST;
    e = bounds.s - 1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesSetFomCoarser(const int &ox, int
//! &s, int &e,
//                                                         const IndexRange
//                                                         &bounds, const
//                                                         std::int64_t &lx,
//                                                         const int &cng,
//                                                         const bool
//                                                         include_dim)
//  \brief Calculate indices for SetBoundary routines for buffers from coarser
//  levels

void CalcIndicesSetFromCoarser(const int &ox, int &s, int &e,
                               const IndexRange &bounds, const std::int64_t &lx,
                               const int &cng, const bool include_dim) {
  if (ox == 0) {
    s = bounds.s;
    e = bounds.e;
    if (include_dim) {
      if ((lx & 1LL) == 0LL) {
        e += cng;
      } else {
        s -= cng;
      }
    }
  } else if (ox > 0) {
    s = bounds.e + 1;
    e = bounds.e + cng;
  } else {
    s = bounds.s - cng;
    e = bounds.s - 1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesSetFromFiner(int &si, int &ei, int
//! &sj,
//                                                        int &ej, int &sk, int
//                                                        &ek, const
//                                                        NeighborBlock &nb,
//                                                        MeshBlock *pmb)
//  \brief Calculate indices for SetBoundary routines for buffers from finer
//  levels

void CalcIndicesSetFromFiner(int &si, int &ei, int &sj, int &ej, int &sk,
                             int &ek, const NeighborBlock &nb, MeshBlock *pmb) {
  IndexDomain interior = IndexDomain::interior;
  const IndexShape &cellbounds = pmb->cellbounds;
  if (nb.ni.ox1 == 0) {
    si = cellbounds.is(interior);
    ei = cellbounds.ie(interior);
    if (nb.ni.fi1 == 1)
      si += pmb->block_size.nx1 / 2;
    else
      ei -= pmb->block_size.nx1 / 2;
  } else if (nb.ni.ox1 > 0) {
    si = cellbounds.ie(interior) + 1;
    ei = cellbounds.ie(interior) + NGHOST;
  } else {
    si = cellbounds.is(interior) - NGHOST;
    ei = cellbounds.is(interior) - 1;
  }

  if (nb.ni.ox2 == 0) {
    sj = cellbounds.js(interior);
    ej = cellbounds.je(interior);
    if (pmb->block_size.nx2 > 1) {
      if (nb.ni.ox1 != 0) {
        if (nb.ni.fi1 == 1)
          sj += pmb->block_size.nx2 / 2;
        else
          ej -= pmb->block_size.nx2 / 2;
      } else {
        if (nb.ni.fi2 == 1)
          sj += pmb->block_size.nx2 / 2;
        else
          ej -= pmb->block_size.nx2 / 2;
      }
    }
  } else if (nb.ni.ox2 > 0) {
    sj = cellbounds.je(interior) + 1;
    ej = cellbounds.je(interior) + NGHOST;
  } else {
    sj = cellbounds.js(interior) - NGHOST;
    ej = cellbounds.js(interior) - 1;
  }

  if (nb.ni.ox3 == 0) {
    sk = cellbounds.ks(interior);
    ek = cellbounds.ke(interior);
    if (pmb->block_size.nx3 > 1) {
      if (nb.ni.ox1 != 0 && nb.ni.ox2 != 0) {
        if (nb.ni.fi1 == 1)
          sk += pmb->block_size.nx3 / 2;
        else
          ek -= pmb->block_size.nx3 / 2;
      } else {
        if (nb.ni.fi2 == 1)
          sk += pmb->block_size.nx3 / 2;
        else
          ek -= pmb->block_size.nx3 / 2;
      }
    }
  } else if (nb.ni.ox3 > 0) {
    sk = cellbounds.ke(interior) + 1;
    ek = cellbounds.ke(interior) + NGHOST;
  } else {
    sk = cellbounds.ks(interior) - NGHOST;
    ek = cellbounds.ks(interior) - 1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesLoadSame(int ox, int &s, int &e,
//                                                    const IndexRange &bounds)
//  \brief Calculate indices for LoadBoundary routines for buffers on the same
//  level
//         and to coarser.

void CalcIndicesLoadSame(int ox, int &s, int &e, const IndexRange &bounds) {
  if (ox == 0) {
    s = bounds.s;
    e = bounds.e;
  } else if (ox > 0) {
    s = bounds.e - NGHOST + 1;
    e = bounds.e;
  } else {
    s = bounds.s;
    e = bounds.s + NGHOST - 1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesLoadToFiner(int &si, int &ei, int
//! &sj,
//                                                       int &ej, int &sk, int
//                                                       &ek, const
//                                                       NeighborBlock &nb,
//                                                       MeshBlock *pmb)
//  \brief Calculate indices for LoadBoundary routines for buffers to finer
//  levels

void CalcIndicesLoadToFiner(int &si, int &ei, int &sj, int &ej, int &sk,
                            int &ek, const NeighborBlock &nb, MeshBlock *pmb) {
  int cn = pmb->cnghost - 1;

  IndexDomain interior = IndexDomain::interior;
  const IndexShape &cellbounds = pmb->cellbounds;
  si = (nb.ni.ox1 > 0) ? (cellbounds.ie(interior) - cn)
                       : cellbounds.is(interior);
  ei = (nb.ni.ox1 < 0) ? (cellbounds.is(interior) + cn)
                       : cellbounds.ie(interior);
  sj = (nb.ni.ox2 > 0) ? (cellbounds.je(interior) - cn)
                       : cellbounds.js(interior);
  ej = (nb.ni.ox2 < 0) ? (cellbounds.js(interior) + cn)
                       : cellbounds.je(interior);
  sk = (nb.ni.ox3 > 0) ? (cellbounds.ke(interior) - cn)
                       : cellbounds.ks(interior);
  ek = (nb.ni.ox3 < 0) ? (cellbounds.ks(interior) + cn)
                       : cellbounds.ke(interior);

  // send the data first and later prolongate on the target block
  // need to add edges for faces, add corners for edges
  if (nb.ni.ox1 == 0) {
    if (nb.ni.fi1 == 1)
      si += pmb->block_size.nx1 / 2 - pmb->cnghost;
    else
      ei -= pmb->block_size.nx1 / 2 - pmb->cnghost;
  }
  if (nb.ni.ox2 == 0 && pmb->block_size.nx2 > 1) {
    if (nb.ni.ox1 != 0) {
      if (nb.ni.fi1 == 1)
        sj += pmb->block_size.nx2 / 2 - pmb->cnghost;
      else
        ej -= pmb->block_size.nx2 / 2 - pmb->cnghost;
    } else {
      if (nb.ni.fi2 == 1)
        sj += pmb->block_size.nx2 / 2 - pmb->cnghost;
      else
        ej -= pmb->block_size.nx2 / 2 - pmb->cnghost;
    }
  }
  if (nb.ni.ox3 == 0 && pmb->block_size.nx3 > 1) {
    if (nb.ni.ox1 != 0 && nb.ni.ox2 != 0) {
      if (nb.ni.fi1 == 1)
        sk += pmb->block_size.nx3 / 2 - pmb->cnghost;
      else
        ek -= pmb->block_size.nx3 / 2 - pmb->cnghost;
    } else {
      if (nb.ni.fi2 == 1)
        sk += pmb->block_size.nx3 / 2 - pmb->cnghost;
      else
        ek -= pmb->block_size.nx3 / 2 - pmb->cnghost;
    }
  }
}

// send boundary buffers with MeshBlockPack support
// TODO(pgrete) should probaly be moved to the bvals or interface folders
TaskStatus SendBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_SendBoundaryBuffers_MeshData");

  auto var_pack =
      md->PackVariables(std::vector<MetadataFlag>({Metadata::FillGhost}));
  const auto Nv = var_pack.GetDim(4);

  Kokkos::Profiling::pushRegion("Reset boundaries");

  int buffers_used = 0;
  // reset buffer and count required buffers
  for (int block = 0; block < md->NumBlocks(); block++) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();

    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsSet(parthenon::Metadata::FillGhost)) {
        v->resetBoundary();
      }
    }

    // TODO(?) currently this only works for a single "Variable" per container.
    // Need to update the buffer sizes so that it matches the packed Variables.
    MeshBlockDataIterator<Real> it(
        rc, std::vector<MetadataFlag>{Metadata::FillGhost});
    auto &vars = it.vars;
    assert(vars.size() == 1);
    auto *bd_var_ = vars[0]->vbvar->GetPBdVar();

    for (int n = 0; n < pmb->pbval->nneighbor; n++) {
      parthenon::NeighborBlock &nb = pmb->pbval->neighbor[n];
      if (bd_var_->sflag[nb.bufid] == parthenon::BoundaryStatus::completed)
        continue;
      buffers_used += 1;
    }
  }
  Kokkos::Profiling::popRegion(); // Reset boundaries

  auto boundary_info = md->GetSendBuffers();
  bool is_allocated = boundary_info.is_allocated(); // debug

  // debug
  auto boundary_info_old_h = md->send_buffers_h;

  if (!is_allocated) {
    // if (true) {
    Kokkos::Profiling::pushRegion("Create bndinfo array");

    boundary_info = BufferCache_t("send_boundary_info", buffers_used);
    auto boundary_info_h = Kokkos::create_mirror(boundary_info);

    // now fill the buffer information
    int b = 0; // buffer index
    for (auto block = 0; block < md->NumBlocks(); block++) {
      auto &rc = md->GetBlockData(block);
      auto pmb = rc->GetBlockPointer();

      // TODO(?) currently this only works for a single "Variable" per
      // container. Need to update the buffer sizes so that it matches the
      // packed Variables.
      MeshBlockDataIterator<Real> it(
          rc, std::vector<MetadataFlag>{Metadata::FillGhost});
      auto &vars = it.vars;
      assert(vars.size() == 1);
      auto *bd_var_ = vars[0]->vbvar->GetPBdVar();

      int mylevel = pmb->loc.level;
      for (int n = 0; n < pmb->pbval->nneighbor; n++) {
        parthenon::NeighborBlock &nb = pmb->pbval->neighbor[n];
        if (bd_var_->sflag[nb.bufid] == parthenon::BoundaryStatus::completed)
          continue;

        auto &si = boundary_info_h(b).si;
        auto &ei = boundary_info_h(b).ei;
        auto &sj = boundary_info_h(b).sj;
        auto &ej = boundary_info_h(b).ej;
        auto &sk = boundary_info_h(b).sk;
        auto &ek = boundary_info_h(b).ek;

        IndexDomain interior = IndexDomain::interior;
        auto &var_cc = vars[0]->data;
        if (nb.snb.level == mylevel) {
          const parthenon::IndexShape &cellbounds = pmb->cellbounds;
          CalcIndicesLoadSame(nb.ni.ox1, si, ei,
                              cellbounds.GetBoundsI(interior));
          CalcIndicesLoadSame(nb.ni.ox2, sj, ej,
                              cellbounds.GetBoundsJ(interior));
          CalcIndicesLoadSame(nb.ni.ox3, sk, ek,
                              cellbounds.GetBoundsK(interior));
          boundary_info_h(b).var = var_cc.Get<4>();

        } else if (nb.snb.level < mylevel) {
          const IndexShape &c_cellbounds = pmb->c_cellbounds;
          // "Same" logic is the same for loading to a coarse buffer, just using
          // c_cellbounds
          CalcIndicesLoadSame(nb.ni.ox1, si, ei,
                              c_cellbounds.GetBoundsI(interior));
          CalcIndicesLoadSame(nb.ni.ox2, sj, ej,
                              c_cellbounds.GetBoundsJ(interior));
          CalcIndicesLoadSame(nb.ni.ox3, sk, ek,
                              c_cellbounds.GetBoundsK(interior));

          auto &coarse_buf = vars[0]->vbvar->coarse_buf;
          pmb->pmr->RestrictCellCenteredValues(var_cc, coarse_buf, 0, Nv - 1,
                                               si, ei, sj, ej, sk, ek);

          boundary_info_h(b).var = coarse_buf.Get<4>();

        } else {
          CalcIndicesLoadToFiner(si, ei, sj, ej, sk, ek, nb, pmb.get());
          boundary_info_h(b).var = var_cc.Get<4>();
        }
        // on the same process fill the target buffer directly
        if (nb.snb.rank == parthenon::Globals::my_rank) {
          auto target_block = pmb->pmy_mesh->FindMeshBlock(nb.snb.gid);
          // TODO(?) again hardcoded 0 index for single Variable
          boundary_info_h(b).buf =
              target_block->pbval->bvars[0]->GetPBdVar()->recv[nb.targetid];
        } else {
          boundary_info_h(b).buf = bd_var_->send[nb.bufid];
        }
        b++;
      }
    }

    // debug
    if (is_allocated) {
      auto check_int = [](const std::string &s, int a, int b) {
        if (a != b) {
          std::stringstream ss;
          ss << s << " not equal! " << a << " " << b << std::endl;
          PARTHENON_THROW(ss);
        }
      };
      auto bi_old = boundary_info_old_h;
      auto bi_new = boundary_info_h;
      check_int("extents", bi_old.extent(0), bi_new.extent(0));
      for (int b = 0; b < bi_old.extent_int(0); b++) {
        check_int("si at b=" + std::to_string(b), bi_old(b).si, bi_new(b).si);
        check_int("ei at b=" + std::to_string(b), bi_old(b).ei, bi_new(b).ei);
        check_int("sj at b=" + std::to_string(b), bi_old(b).sj, bi_new(b).sj);
        check_int("ej at b=" + std::to_string(b), bi_old(b).ej, bi_new(b).ej);
        check_int("sk at b=" + std::to_string(b), bi_old(b).sk, bi_new(b).sk);
        check_int("ek at b=" + std::to_string(b), bi_old(b).ek, bi_new(b).ek);
        // Explicit view comparitors not supported by Kokkos.
        // Instead we compare size and pointer locations.
        if ((bi_old(b).buf.size() != bi_new(b).buf.size()) ||
            (bi_old(b).buf.data() != bi_new(b).buf.data())) {
          std::stringstream ss;
          ss << "buf at b=" << b << " not equal!" << std::endl;
          PARTHENON_THROW(ss);
        }
        if ((bi_old(b).var.size() != bi_new(b).var.size()) ||
            (bi_old(b).var.data() != bi_new(b).var.data())) {
          std::stringstream ss;
          ss << "var at b=" << b << " not equal!" << std::endl;
          PARTHENON_THROW(ss);
        }
      }
    }

    Kokkos::deep_copy(boundary_info, boundary_info_h);
    if (!is_allocated) {
      md->send_buffers_h = boundary_info_h; // debug
      md->SetSendBuffers(boundary_info);
    }

    Kokkos::Profiling::popRegion(); // Create bndinfo array
  }

  Kokkos::parallel_for(
      "SendBoundaryBuffers",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), buffers_used,
                           Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        const int &si = boundary_info(b).si;
        const int &ei = boundary_info(b).ei;
        const int &sj = boundary_info(b).sj;
        const int &ej = boundary_info(b).ej;
        const int &sk = boundary_info(b).sk;
        const int &ek = boundary_info(b).ek;
        const int Ni = ei + 1 - si;
        const int Nj = ej + 1 - sj;
        const int Nk = ek + 1 - sk;
        const int NvNkNj = Nv * Nk * Nj;
        const int NkNj = Nk * Nj;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange<>(team_member, NvNkNj), [&](const int idx) {
              const int v = idx / NkNj;
              int k = (idx - v * NkNj) / Nj;
              int j = idx - v * NkNj - k * Nj;
              k += sk;
              j += sj;

              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, si, ei + 1),
                  [&](const int i) {
                    boundary_info(b).buf(
                        i - si + Ni * (j - sj + Nj * (k - sk + Nk * v))) =
                        boundary_info(b).var(v, k, j, i);
                  });
            });
      });

  Kokkos::fence();
  Kokkos::Profiling::pushRegion("Set complete and/or start sending via MPI");
  for (int block = 0; block < md->NumBlocks(); block++) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();

    MeshBlockDataIterator<Real> it(
        rc, std::vector<MetadataFlag>{Metadata::FillGhost});
    auto &vars = it.vars;
    assert(vars.size() == 1);

    int mylevel = pmb->loc.level;
    for (int n = 0; n < pmb->pbval->nneighbor; n++) {
      parthenon::NeighborBlock &nb = pmb->pbval->neighbor[n];
      // TODO(?) currently this only works for a single "Variable" per
      // container. Need to update the buffer sizes so that it matches the
      // packed Variables.
      auto *bd_var_ = vars[0]->vbvar->GetPBdVar();
      if (bd_var_->sflag[nb.bufid] == parthenon::BoundaryStatus::completed)
        continue;

      // on the same rank the data has been directly copied to the target buffer
      if (nb.snb.rank == parthenon::Globals::my_rank) {
        // TODO(?) check performance of FindMeshBlock. Could be caching from
        // call above.
        auto target_block = pmb->pmy_mesh->FindMeshBlock(nb.snb.gid);
        target_block->pbval->bvars[0]->GetPBdVar()->flag[nb.targetid] =
            parthenon::BoundaryStatus::arrived;
      } else {
#ifdef MPI_PARALLEL
        MPI_Start(&(bd_var_->req_send[nb.bufid]));
#endif
      }

      bd_var_->sflag[nb.bufid] = parthenon::BoundaryStatus::completed;
    }
  }
  Kokkos::Profiling::popRegion(); // Set complete and/or start sending via MPI

  Kokkos::Profiling::popRegion(); // Task_SendBoundaryBuffers_MeshData
  // TODO(?) reintroduce sparse logic (or merge with above)
  return TaskStatus::complete;
}

TaskStatus ReceiveBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_ReceiveBoundaryBuffers_MeshData");
  bool ret = true;
  for (int i = 0; i < md->NumBlocks(); i++) {
    auto &rc = md->GetBlockData(i);
    auto task_status = rc->ReceiveBoundaryBuffers();
    if (task_status == TaskStatus::incomplete) {
      ret = false;
    }
  }

  // TODO(?) reintroduce sparse logic (or merge with above)
  Kokkos::Profiling::popRegion(); // Task_ReceiveBoundaryBuffers_MeshData
  if (ret)
    return TaskStatus::complete;
  return TaskStatus::incomplete;
}

// set boundaries from buffers with MeshBlockPack support
// TODO(pgrete) should probaly be moved to the bvals or interface folders
TaskStatus SetBoundaries(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_SetBoundaries_MeshData");

  IndexDomain interior = IndexDomain::interior;

  // first calculate the number of active buffers
  int buffers_used = 0;
  for (int block = 0; block < md->NumBlocks(); block++) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();
    for (int n = 0; n < pmb->pbval->nneighbor; n++) {
      buffers_used += 1;
    }
  }

  auto boundary_info = md->GetSetBuffers();
  if (!boundary_info.is_allocated()) {
    Kokkos::Profiling::pushRegion("Create bndinfo array");

    boundary_info = BufferCache_t("set_boundary_info", buffers_used);
    auto boundary_info_h = Kokkos::create_mirror_view(boundary_info);
    // now fill the buffer info
    int b = 0;
    for (int block = 0; block < md->NumBlocks(); block++) {
      auto &rc = md->GetBlockData(block);
      auto pmb = rc->GetBlockPointer();

      // TODO(?) currently this only works for a single "Variable" per
      // container. Need to update the buffer sizes so that it matches the
      // packed Variables.
      MeshBlockDataIterator<Real> it(
          rc, std::vector<MetadataFlag>{Metadata::FillGhost});
      auto &vars = it.vars;
      assert(vars.size() == 1);
      auto *bd_var_ = vars[0]->vbvar->GetPBdVar();

      int mylevel = pmb->loc.level;
      for (int n = 0; n < pmb->pbval->nneighbor; n++) {
        parthenon::NeighborBlock &nb = pmb->pbval->neighbor[n];

        auto &si = boundary_info_h(b).si;
        auto &ei = boundary_info_h(b).ei;
        auto &sj = boundary_info_h(b).sj;
        auto &ej = boundary_info_h(b).ej;
        auto &sk = boundary_info_h(b).sk;
        auto &ek = boundary_info_h(b).ek;

        if (nb.snb.level == mylevel) {
          const parthenon::IndexShape &cellbounds = pmb->cellbounds;
          CalcIndicesSetSame(nb.ni.ox1, si, ei,
                             cellbounds.GetBoundsI(interior));
          CalcIndicesSetSame(nb.ni.ox2, sj, ej,
                             cellbounds.GetBoundsJ(interior));
          CalcIndicesSetSame(nb.ni.ox3, sk, ek,
                             cellbounds.GetBoundsK(interior));
          boundary_info_h(b).var = vars[0]->data.Get<4>();
        } else if (nb.snb.level < mylevel) {
          const IndexShape &c_cellbounds = pmb->c_cellbounds;
          const auto &cng = pmb->cnghost;
          CalcIndicesSetFromCoarser(nb.ni.ox1, si, ei,
                                    c_cellbounds.GetBoundsI(interior),
                                    pmb->loc.lx1, cng, true);
          CalcIndicesSetFromCoarser(nb.ni.ox2, sj, ej,
                                    c_cellbounds.GetBoundsJ(interior),
                                    pmb->loc.lx2, cng, pmb->block_size.nx2 > 1);
          CalcIndicesSetFromCoarser(nb.ni.ox3, sk, ek,
                                    c_cellbounds.GetBoundsK(interior),
                                    pmb->loc.lx3, cng, pmb->block_size.nx3 > 1);

          boundary_info_h(b).var = vars[0]->vbvar->coarse_buf.Get<4>();
        } else {
          CalcIndicesSetFromFiner(si, ei, sj, ej, sk, ek, nb, pmb.get());
          boundary_info_h(b).var = vars[0]->data.Get<4>();
        }
        boundary_info_h(b).buf = bd_var_->recv[nb.bufid];
        // safe to set completed here as the kernel updating all buffers is
        // called immediately afterwards
        bd_var_->flag[nb.bufid] = parthenon::BoundaryStatus::completed;
        b++;
      }
    }
    Kokkos::deep_copy(boundary_info, boundary_info_h);
    md->SetSetBuffers(boundary_info);

    Kokkos::Profiling::popRegion(); // Create bndinfo array
  }
  // TODO(pgrete) this var pack can probably go. Currently only needed for 4th
  // dim.
  auto var_pack =
      md->PackVariables(std::vector<MetadataFlag>({Metadata::FillGhost}));
  const auto Nv = var_pack.GetDim(4);

  Kokkos::parallel_for(
      "SetBoundaries",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), buffers_used,
                           Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        // TODO(pgrete) profile perf implication of using reference.
        // Test in two jobs indicted a 10% difference, but were also run on
        // diff. nodes
        const int &si = boundary_info(b).si;
        const int &ei = boundary_info(b).ei;
        const int &sj = boundary_info(b).sj;
        const int &ej = boundary_info(b).ej;
        const int &sk = boundary_info(b).sk;
        const int &ek = boundary_info(b).ek;

        const int Ni = ei + 1 - si;
        const int Nj = ej + 1 - sj;
        const int Nk = ek + 1 - sk;

        const int NvNkNj = Nv * Nk * Nj;
        const int NkNj = Nk * Nj;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange<>(team_member, NvNkNj), [&](const int idx) {
              const int v = idx / NkNj;
              int k = (idx - v * NkNj) / Nj;
              int j = idx - v * NkNj - k * Nj;
              k += sk;
              j += sj;

              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, si, ei + 1),
                  [&](const int i) {
                    boundary_info(b).var(v, k, j, i) = boundary_info(b).buf(
                        i - si + Ni * (j - sj + Nj * (k - sk + Nk * v)));
                  });
            });
      });

  Kokkos::Profiling::popRegion(); // Task_SetBoundaries_MeshData
  // TODO(?) reintroduce sparse logic (or merge with above)
  return TaskStatus::complete;
}

} // namespace cell_centered_bvars
} // namespace parthenon