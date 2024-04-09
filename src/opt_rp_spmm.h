#ifndef __OPT_RP_SPMM_H__
#define __OPT_RP_SPMM_H__

#include <mpi.h>

struct opt_rp_spmm
{
    int    nproc, my_rank;      // Number of processes and rank of this process in comm
    int    glb_n;               // Global number of columns of B and C
    int    A_nrow;              // Number of local A matrix rows
    int    B_local_nrow;        // Number of local B matrix rows
    int    B_remote_nrow;       // Total number of B rows to receive on this process
    int    *B_sdispls;          // Size nproc, the first B row to send to proc p is the B_sdispls[p]-th row on this proc
    int    *B_scnts;            // Size nproc, the number of B rows to send to proc p from this proc is B_scnts[p]
    int    *B_sridxs;           // Size TBD, the i-th B row to send to proc p is the B_sridxs[B_sdispls[p] + i]-th row of B_local
    int    *B_rdispls;          // Size nproc, the first B row to recv from proc p is the B_rdispls[p]-th row on this proc
    int    *B_rcnts;            // Size nproc, the number of B rows to recv from proc p on this proc is B_rcnts[p]
    int    *A_diag_rowptr;      // Size A_nrow + 1, local A diagonal block CSR row pointer
    int    *A_diag_colidx;      // Size A_diag_nnz (== A_diag_rowptr[A_nrow]), local A diagonal block CSR column index
    int    *A_offd_rowptr;      // Size A_nrow + 1, local A off-diagonal block CSR row pointer
    int    *A_offd_colidx;      // Size A_offd_nnz (== A_offd_rowptr[A_nrow]), local A off-diagonal block CSR column index
    double *A_diag_val;         // Size A_diag_nnz, local A diagonal block CSR value
    double *A_offd_val;         // Size A_offd_nnz, local A off-diagonal block CSR value
    void   *mkl_A_diag;         // MKL sparse matrix handle, diagonal block of A
    void   *mkl_A_offd;         // MKL sparse matrix handle, off-diagonal block of A
    MPI_Comm comm;

    // Statistic info
    int    n_exec;              // Number of times rp_spmm_exec() is called
    double t_init;              // Time (s) for opt_rp_spmm_init()
    double t_pack;              // Time (s) for packing B send buffer
    double t_comm;              // Time (s) for send and receive B rows
    double t_unpack;            // Time (s) for unpacking B recv buffer
    double t_spmm;              // Time (s) for local SpMM
    double t_exec;              // Time (s) for opt_rp_spmm_exec()
};
typedef struct opt_rp_spmm  opt_rp_spmm_s;
typedef struct opt_rp_spmm *opt_rp_spmm_p;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize a opt_rp_spmm struct for a 1D row-parallel SpMM
// Input parameters:
//   A_{srow, nrow} : Starting row and number of rows of local A
//   A_rowptr       : Size A_nrow + 1, local A matrix CSR row pointer
//   A_colidx       : Local A matrix CSR column index
//   A_val          : Local A matrix CSR value
//   B_row_displs   : Size nproc + 1, indices of the first row of B on each process
//   glb_n          : Global number of columns of B and C
//   comm           : MPI communicator for all processes, will not be duplicated, do not free it
// Output parameter:
//   opt_rp_spmm : Pointer to an initialized opt_rp_spmm struct
// Note: A_colidx[i] and A_val[i] will be accessed for 0 <= i < A_rowptr[A_srow + A_nrow] - A_rowptr[A_srow]
void opt_rp_spmm_init(
    const int A_srow, const int A_nrow, const int *A_rowptr, const int *A_colidx, 
    const double *A_val, const int *B_row_displs, const int glb_n, MPI_Comm comm, 
    opt_rp_spmm_p *opt_rp_spmm
);

// Free an opt_rp_spmm struct
void opt_rp_spmm_free(opt_rp_spmm_p *opt_rp_spmm);

// Compute C := A * B
// Input parameters:
//   opt_rp_spmm : Pointer to an initialized opt_rp_spmm struct
//   BC_layout   : Layout of B and C, 0 for row-major, 1 for column-major
//   B_local     : Size >= ldB * glb_n (col-major) or opt_rp_spmm->B_local_nrow * ldB (row-major), local B matrix
//   ldB         : Leading dimension of B, >= opt_rp_spmm->B_local_nrow (col-major) or glb_n (row-major)
//   ldC         : Leading dimension of C, >= opt_rp_spmm->A_nrow (col-major) or glb_n (row-major)
// Output parameter:
//   C_local : Size >= ldC * glb_n (col-major) or opt_rp_spmm->A_nrow * ldC (row-major), local C matrix
void opt_rp_spmm_exec(
    opt_rp_spmm_p opt_rp_spmm, const int BC_layout, const double *B_local, const int ldB,
    double *C_local, const int ldC
);

// Compute y := A * x
// Input parameters:
//   opt_rp_spmm : Pointer to an initialized opt_rp_spmm struct
//   x_local     : Size opt_rp_spmm->B_local_nrow, local x vector
// Output parameter:
//   y_local : Size opt_rp_spmm->A_nrow, local y vector
void opt_rp_spmv_exec(opt_rp_spmm_p opt_rp_spmm, const double *x_local, double *y_local);

// Print statistic info of an opt_rp_spmm struct
void opt_rp_spmm_print_stat(opt_rp_spmm_p opt_rp_spmm);

// Clear statistic info of an opt_rp_spmm struct
void opt_rp_spmm_clear_stat(opt_rp_spmm_p opt_rp_spmm);

#ifdef __cplusplus
}
#endif

#endif
