#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include <omp.h>
#include <mpi.h>

#include <mkl.h>

#include "utils.h"
#include "opt_rp_spmm.h"

// Initialize a opt_rp_spmm struct for a 1D row-parallel SpMM
void opt_rp_spmm_init(
    const int A_srow, const int A_nrow, const int *A_rowptr, const int *A_colidx, 
    const double *A_val, const int *B_row_displs, const int glb_n, MPI_Comm comm, 
    opt_rp_spmm_p *opt_rp_spmm
)
{
    opt_rp_spmm_p opt_rp_spmm_ = (opt_rp_spmm_p) malloc(sizeof(opt_rp_spmm_s));
    memset(opt_rp_spmm_, 0, sizeof(opt_rp_spmm_s));
    *opt_rp_spmm = opt_rp_spmm_;

    double st = get_wtime_sec();

    int nproc, my_rank;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &my_rank);
    opt_rp_spmm_->nproc   = nproc;
    opt_rp_spmm_->my_rank = my_rank;
    opt_rp_spmm_->A_nrow  = A_nrow;
    opt_rp_spmm_->glb_n   = glb_n;
    opt_rp_spmm_->comm    = comm;

    // 1. Mark all non-empty columns of local A and count the nnz of A_diag/offd 
    int A_nnz = A_rowptr[A_nrow] - A_rowptr[0];
    int A_nnz_sidx = A_rowptr[0];
    int glb_k = B_row_displs[nproc];
    int *A_col_flag = (int *) malloc(sizeof(int) * glb_k);
    ASSERT_PRINTF(A_col_flag != NULL, "Failed to allocate allocate work memory\n");
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < glb_k; i++) A_col_flag[i] = -1;
    int A_diag_nnz = 0, A_offd_nnz = 0;
    int B_local_srow = B_row_displs[my_rank];
    int B_local_erow = B_row_displs[my_rank + 1];
    int B_local_nrow = B_local_erow - B_local_srow;
    for (int i = 0; i < A_nnz; i++)
    {
        int col = A_colidx[i];
        if (B_local_srow <= col && col < B_local_erow)
        {
            A_diag_nnz++;
        } else {
            A_offd_nnz++;
            A_col_flag[col] = 1;
        }
    }
    opt_rp_spmm_->B_local_nrow = B_local_nrow;

    // 2. Re-index the non-empty columns of off-diagonal block of A
    int B_remote_nrow = 0;
    for (int i = 0; i < glb_k; i++)
    {
        if (A_col_flag[i] == 1)
        {
            A_col_flag[i] = B_remote_nrow;
            B_remote_nrow++;
        }
    }
    opt_rp_spmm_->B_remote_nrow = B_remote_nrow;

    // 3. Separate the diagonal and off-diagonal blocks of A
    int    *A_diag_rowptr = (int *)    malloc(sizeof(int)    * (A_nrow + 1));
    int    *A_diag_colidx = (int *)    malloc(sizeof(int)    * A_diag_nnz);
    double *A_diag_val    = (double *) malloc(sizeof(double) * A_diag_nnz);
    int    *A_offd_rowptr = (int *)    malloc(sizeof(int)    * (A_nrow + 1));
    int    *A_offd_colidx = (int *)    malloc(sizeof(int)    * A_offd_nnz);
    double *A_offd_val    = (double *) malloc(sizeof(double) * A_offd_nnz);
    ASSERT_PRINTF(
        A_diag_rowptr != NULL && A_diag_colidx != NULL && A_diag_val != NULL &&
        A_offd_rowptr != NULL && A_offd_colidx != NULL && A_offd_val != NULL,
        "Failed to allocate work memory\n"
    );
    memset(A_diag_rowptr, 0, sizeof(int) * (A_nrow + 1));
    memset(A_offd_rowptr, 0, sizeof(int) * (A_nrow + 1));
    A_diag_nnz = 0;
    A_offd_nnz = 0;
    for (int irow = 0; irow < A_nrow; irow++)
    {
        int idx_s = A_rowptr[irow] - A_nnz_sidx;
        int idx_e = A_rowptr[irow + 1] - A_nnz_sidx;
        for (int idx = idx_s; idx < idx_e; idx++)
        {
            int col = A_colidx[idx];
            if (B_local_srow <= col && col < B_local_erow)
            {
                A_diag_rowptr[irow + 1]++;
                A_diag_colidx[A_diag_nnz] = col - B_local_srow;  // Also need re-indexing for diagonal block
                A_diag_val[A_diag_nnz] = A_val[idx];
                A_diag_nnz++;
            } else {
                A_offd_rowptr[irow + 1]++;
                A_offd_colidx[A_offd_nnz] = A_col_flag[col];
                A_offd_val[A_offd_nnz] = A_val[idx];
                A_offd_nnz++;
            }
        }
        A_diag_rowptr[irow + 1] += A_diag_rowptr[irow];
        A_offd_rowptr[irow + 1] += A_offd_rowptr[irow];
    }  // End of irow loop
    opt_rp_spmm_->A_diag_rowptr = A_diag_rowptr;
    opt_rp_spmm_->A_diag_colidx = A_diag_colidx;
    opt_rp_spmm_->A_diag_val    = A_diag_val;
    opt_rp_spmm_->A_offd_rowptr = A_offd_rowptr;
    opt_rp_spmm_->A_offd_colidx = A_offd_colidx;
    opt_rp_spmm_->A_offd_val    = A_offd_val;

    // 4. Create MKL CSR descriptors for A_diag and A_offd
    struct matrix_descr mkl_descA;
    mkl_descA.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_descA.mode = SPARSE_FILL_MODE_FULL;
    mkl_descA.diag = SPARSE_DIAG_NON_UNIT;
    sparse_matrix_t mkl_A_diag = NULL, mkl_A_offd = NULL;
    if (B_local_nrow > 0)
    {
        mkl_sparse_d_create_csr(
            &mkl_A_diag, SPARSE_INDEX_BASE_ZERO, A_nrow, B_local_nrow, 
            A_diag_rowptr, A_diag_rowptr + 1, A_diag_colidx, A_diag_val
        );
        if (glb_n > 1) mkl_sparse_set_mm_hint(mkl_A_diag, SPARSE_OPERATION_NON_TRANSPOSE, mkl_descA, SPARSE_LAYOUT_ROW_MAJOR, glb_n, 10);
        else mkl_sparse_set_mv_hint(mkl_A_diag, SPARSE_OPERATION_NON_TRANSPOSE, mkl_descA, 10);
    }
    if (B_remote_nrow > 0)
    {
        mkl_sparse_d_create_csr(
            &mkl_A_offd, SPARSE_INDEX_BASE_ZERO, A_nrow, B_remote_nrow, 
            A_offd_rowptr, A_offd_rowptr + 1, A_offd_colidx, A_offd_val
        );
        if (glb_n > 1) mkl_sparse_set_mm_hint(mkl_A_offd, SPARSE_OPERATION_NON_TRANSPOSE, mkl_descA, SPARSE_LAYOUT_ROW_MAJOR, glb_n, 10);
        else mkl_sparse_set_mv_hint(mkl_A_offd, SPARSE_OPERATION_NON_TRANSPOSE, mkl_descA, 10);
    }
    opt_rp_spmm_->mkl_A_diag = (void *) mkl_A_diag;
    opt_rp_spmm_->mkl_A_offd = (void *) mkl_A_offd;

    // 5. Find the rows of B corresponding to the off-diagonal block of A
    int *B_rcnts   = (int *) malloc(sizeof(int) * nproc);
    int *B_rdispls = (int *) malloc(sizeof(int) * (nproc + 1));
    int *B_rridxs  = (int *) malloc(sizeof(int) * B_remote_nrow);
    ASSERT_PRINTF(B_rcnts != NULL && B_rdispls != NULL && B_rridxs != NULL, "Failed to allocate work memory\n");
    memset(B_rcnts, 0, sizeof(int) * nproc);
    // The number of B rows to recv from proc p on this proc is B_rcnts[p]
    for (int p = 0; p < nproc; p++)
    {
        if (p == my_rank) continue;  // We don't really need this, keep it for logical completeness
        for (int irow = B_row_displs[p]; irow < B_row_displs[p + 1]; irow++)
            if (A_col_flag[irow] != -1) B_rcnts[p]++;
    }
    B_rdispls[0] = 0;  // The first B row to recv from proc p is the B_rdispls[p]-th row on this proc
    for (int p = 0; p < nproc; p++)
        B_rdispls[p + 1] = B_rdispls[p] + B_rcnts[p];
    ASSERT_PRINTF(B_rdispls[nproc] == B_remote_nrow, "B_rdispls[nproc] != B_remote_nrow\n");
    for (int p = 0; p < nproc; p++)
    {
        if (p == my_rank) continue;  // We don't really need this, keep it for logical completeness
        for (int irow = B_row_displs[p]; irow < B_row_displs[p + 1]; irow++)
        {
            if (A_col_flag[irow] == -1) continue;
            // The B_rdispls[p]-th row to be received is the irow-th row of global B
            B_rridxs[B_rdispls[p]] = irow;
            B_rdispls[p]++;
        }
    }
    // Reconstrcut the original B_rdispls
    B_rdispls[0] = 0;
    for (int p = 0; p < nproc; p++)
        B_rdispls[p + 1] = B_rdispls[p] + B_rcnts[p];
    opt_rp_spmm_->B_rcnts   = B_rcnts;
    opt_rp_spmm_->B_rdispls = B_rdispls;

    // 6. Get the indices of B rows this process needs to send
    int *B_scnts   = (int *) malloc(sizeof(int) * nproc);
    int *B_sdispls = (int *) malloc(sizeof(int) * (nproc + 1));
    ASSERT_PRINTF(B_scnts != NULL && B_sdispls != NULL, "Failed to allocate work memory\n");
    // The number of B rows to send to proc p from this proc is B_scnts[p]
    MPI_Alltoall(B_rcnts, 1, MPI_INT, B_scnts, 1, MPI_INT, comm);
    B_sdispls[0] = 0;  // The first B row to send to proc p is the B_sdispls[p]-th row on this proc
    for (int p = 0; p < nproc; p++)
        B_sdispls[p + 1] = B_sdispls[p] + B_scnts[p];
    int *B_sridxs = (int *) malloc(sizeof(int) * B_sdispls[nproc]);
    ASSERT_PRINTF(B_sridxs != NULL, "Failed to allocate work memory\n");
    MPI_Alltoallv(
        B_rridxs, B_rcnts, B_rdispls, MPI_INT,
        B_sridxs, B_scnts, B_sdispls, MPI_INT, comm
    );
    opt_rp_spmm_->B_scnts   = B_scnts;
    opt_rp_spmm_->B_sdispls = B_sdispls;
    opt_rp_spmm_->B_sridxs  = B_sridxs;

    // 7. Convert the indices of B rows to be sent and received to local indices
    // for (int i = 0; i < B_rdispls[nproc]; i++) B_rridxs[i] = A_col_flag[B_rridxs[i]];  // Unncecessary, since we know B_rridxs[i] should == i
    for (int i = 0; i < B_sdispls[nproc]; i++) B_sridxs[i] -= B_local_srow;

    free(A_col_flag);
    free(B_rridxs);

    double et = get_wtime_sec();
    opt_rp_spmm_->t_init = et - st;
}

// Free an opt_rp_spmm struct
void opt_rp_spmm_free(opt_rp_spmm_p *opt_rp_spmm)
{
    opt_rp_spmm_p opt_rp_spmm_ = *opt_rp_spmm;
    if (opt_rp_spmm_ == NULL) return;
    free(opt_rp_spmm_->B_sdispls);
    free(opt_rp_spmm_->B_scnts);
    free(opt_rp_spmm_->B_sridxs);
    free(opt_rp_spmm_->B_rdispls);
    free(opt_rp_spmm_->B_rcnts);
    free(opt_rp_spmm_->A_diag_rowptr);
    free(opt_rp_spmm_->A_diag_colidx);
    free(opt_rp_spmm_->A_diag_val);
    free(opt_rp_spmm_->A_offd_rowptr);
    free(opt_rp_spmm_->A_offd_colidx);
    free(opt_rp_spmm_->A_offd_val);
    if (opt_rp_spmm_->mkl_A_diag != NULL) mkl_sparse_destroy(opt_rp_spmm_->mkl_A_diag);
    if (opt_rp_spmm_->mkl_A_offd != NULL) mkl_sparse_destroy(opt_rp_spmm_->mkl_A_offd);
    *opt_rp_spmm = NULL;
}

// Compute C := A * B
void opt_rp_spmm_exec(
    opt_rp_spmm_p opt_rp_spmm, const int BC_layout, const double *B_local, const int ldB,
    double *C_local, const int ldC
)
{
    if (opt_rp_spmm == NULL) return;
    int my_rank    = opt_rp_spmm->my_rank;
    int nproc      = opt_rp_spmm->nproc;
    int glb_n      = opt_rp_spmm->glb_n;
    int *B_scnts   = opt_rp_spmm->B_scnts;
    int *B_sdispls = opt_rp_spmm->B_sdispls;
    int *B_sridxs  = opt_rp_spmm->B_sridxs;
    int *B_rcnts   = opt_rp_spmm->B_rcnts;
    int *B_rdispls = opt_rp_spmm->B_rdispls;

    double st, et;
    double exec_s = get_wtime_sec();

    // 1. Allocate work memory
    int n_send = 0, n_recv = 0;
    const int B_local_nrow  = opt_rp_spmm->B_local_nrow;
    const int B_remote_nrow = opt_rp_spmm->B_remote_nrow;
    const int B_sbuf_nrow   = opt_rp_spmm->B_sdispls[nproc];
    double *B_sbuf = (double *) malloc(sizeof(double) * B_sbuf_nrow   * glb_n);
    double *B_rbuf = (double *) malloc(sizeof(double) * B_remote_nrow * glb_n);
    MPI_Request *B_sreqs = (MPI_Request *) malloc(sizeof(MPI_Request) * nproc);
    MPI_Request *B_rreqs = (MPI_Request *) malloc(sizeof(MPI_Request) * nproc);
    ASSERT_PRINTF(
        B_sbuf != NULL && B_rbuf != NULL && B_sreqs != NULL && B_rreqs != NULL, 
        "Failed to allocate work memory\n"
    );

    // 2. Post all Irecv
    st = get_wtime_sec();
    for (int shift = 1; shift < nproc; shift++)
    {
        int p = (my_rank + shift) % nproc;
        if (B_rcnts[p] == 0) continue;
        // The rows to recv from each proc are stored contiguously in B_rbuf, so the offset
        // of the 1st row to recv from proc p is B_rdispls[p] * glb_n
        size_t recv_offset = (size_t) B_rdispls[p] * (size_t) glb_n;
        int recv_cnt = B_rcnts[p] * glb_n;
        MPI_Irecv(
            B_rbuf + recv_offset, recv_cnt, MPI_DOUBLE, 
            p, p, opt_rp_spmm->comm, B_rreqs + n_recv
        );
        n_recv++;
    }
    et = get_wtime_sec();
    opt_rp_spmm->t_comm += et - st;

    // 3. Pack B send buffer for each proc and post Isend
    st = get_wtime_sec();
    for (int shift = 1; shift < nproc; shift++)
    {
        int p = (my_rank + nproc - shift) % nproc;

        // The first row to send to proc p is the B_sdispls[p]-th row to send on this proc
        int p_srow = B_sdispls[p];
        // The number of row to send to proc p from this proc is B_scnts[p]
        int p_nrow = B_scnts[p];
        if ((p == my_rank) || (p_nrow == 0)) continue;
        // The i-th row to send to proc p is the B_sridxs[B_sdispls[p] + i]-th row of B_local
        int *p_B_sridxs = B_sridxs + p_srow;
        // The rows to send to each proc are stored contiguously in B_sbuf, so the offset
        // of the 1st row to send to proc p is B_sdispls[p] * glb_n
        size_t send_offset = (size_t) B_sdispls[p] * (size_t) glb_n;
        double *p_B_sbuf = B_sbuf + send_offset;
        
        if (BC_layout == 0)
        {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < p_nrow; i++)
            {
                size_t src_offset = (size_t) p_B_sridxs[i] * (size_t) ldB;
                size_t dst_offset = (size_t) i * (size_t) glb_n;
                const double *src = B_local + src_offset;
                double *dst = p_B_sbuf + dst_offset;
                memcpy(dst, src, sizeof(double) * glb_n);
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < glb_n; j++)
            {
                size_t src_offset = (size_t) j * (size_t) ldB;
                size_t dst_offset = (size_t) j * (size_t) p_nrow;
                const double *src_j = B_local + src_offset;
                double *dst_j = p_B_sbuf + dst_offset;
                #pragma omp simd
                for (int i = 0; i < p_nrow; i++)
                    dst_j[i] = src_j[p_B_sridxs[i]];
            }
        }  // End of "if (BC_layout == 0)"

        int send_cnt = B_scnts[p] * glb_n;
        MPI_Isend(
            B_sbuf + send_offset, send_cnt, MPI_DOUBLE, 
            p, my_rank, opt_rp_spmm->comm, B_sreqs + n_send
        );
        n_send++;
    }  // End of p loop
    et = get_wtime_sec();
    opt_rp_spmm->t_pack += et - st;

    // 4. Compute C = A_diag * B_diag
    st = get_wtime_sec();
    const double d_one = 1.0, d_zero = 0.0;
    struct matrix_descr mkl_descA;
    mkl_descA.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_descA.mode = SPARSE_FILL_MODE_FULL;
    mkl_descA.diag = SPARSE_DIAG_NON_UNIT;
    sparse_layout_t layout = (BC_layout == 0) ? SPARSE_LAYOUT_ROW_MAJOR : SPARSE_LAYOUT_COLUMN_MAJOR;
    if (B_local_nrow > 0)
    {
        sparse_matrix_t mkl_A_diag = (sparse_matrix_t) opt_rp_spmm->mkl_A_diag;
        mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE, d_one, mkl_A_diag, mkl_descA, 
            layout, B_local, glb_n, ldB, d_zero, C_local, ldC 
        );
    } else {
        if (BC_layout == 0)
        {
            for (int i = 0; i < opt_rp_spmm->A_nrow; i++)
                memset(C_local + i * ldC, 0, sizeof(double) * glb_n);
        } else {
            for (int i = 0; i < glb_n; i++)
                memset(C_local + i * ldC, 0, sizeof(double) * opt_rp_spmm->A_nrow);
        }
    }
    et = get_wtime_sec();
    opt_rp_spmm->t_spmm += et - st;

    // 5. Wait for all Isend and Irecv
    st = get_wtime_sec();
    MPI_Waitall(n_send, B_sreqs, MPI_STATUSES_IGNORE);
    MPI_Waitall(n_recv, B_rreqs, MPI_STATUSES_IGNORE);
    et = get_wtime_sec();
    opt_rp_spmm->t_comm += et - st;

    // 6. Unpack B receive buffer if B_local is col-major
    st = get_wtime_sec();
    double *B_remote = B_rbuf;
    if (BC_layout != 0)
    {
        B_remote = (double *) malloc(sizeof(double) * B_remote_nrow * glb_n);
        ASSERT_PRINTF(B_remote != NULL, "Failed to allocate work memory\n");
        for (int p = 0; p < nproc; p++)
        {
            // The first row received from proc p is the B_rdispls[p]-th row on this proc
            int p_srow = B_rdispls[p];
            // The number of rows received from proc p on this proc is B_rcnts[p]
            int p_nrow = B_rcnts[p];
            if ((p == my_rank) || (p_nrow == 0)) continue;
            // The rows received from each proc are stored contiguously in B_rbuf, so the offset
            // of the 1st row received from proc p is B_rdispls[p] * glb_n
            size_t recv_offset = (size_t) B_rdispls[p] * (size_t) glb_n;
            double *p_B_rbuf = B_rbuf + recv_offset;

            #pragma omp parallel for schedule(static)
            for (int j = 0; j < glb_n; j++)
            {
                size_t src_offset = (size_t) j * (size_t) p_nrow;
                size_t dst_offset = (size_t) j * (size_t) B_remote_nrow + (size_t) p_srow;
                double *src_j = p_B_rbuf + src_offset;
                double *dst_j = B_remote + dst_offset;
                memcpy(dst_j, src_j, sizeof(double) * p_nrow);
            }
        }  // End of p loop
    }  // End of "if (BC_layout != 0)"
    et = get_wtime_sec();
    opt_rp_spmm->t_unpack += et - st;

    // 7. Compute C += A_offd * B_offd
    st = get_wtime_sec();
    if (B_remote_nrow > 0)
    {
        int ldBr = (BC_layout == 0) ? glb_n : B_remote_nrow;
        sparse_matrix_t mkl_A_offd = (sparse_matrix_t) opt_rp_spmm->mkl_A_offd;
        mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE, d_one, mkl_A_offd, mkl_descA, 
            layout, B_remote, glb_n, ldBr, d_one, C_local, ldC
        );
    }
    et = get_wtime_sec();
    opt_rp_spmm->t_spmm += et - st;

    // 8. Free work memory
    free(B_sbuf);
    free(B_rbuf);
    if (BC_layout != 0) free(B_remote);
    free(B_sreqs);
    free(B_rreqs);

    double exec_e = get_wtime_sec();
    opt_rp_spmm->t_exec += exec_e - exec_s;
    opt_rp_spmm->n_exec++;
}

// Compute y := A * x
void opt_rp_spmv_exec(opt_rp_spmm_p opt_rp_spmm, const double *x_local, double *y_local)
{
    if (opt_rp_spmm == NULL) return;
    int my_rank    = opt_rp_spmm->my_rank;
    int nproc      = opt_rp_spmm->nproc;
    int *B_scnts   = opt_rp_spmm->B_scnts;
    int *B_sdispls = opt_rp_spmm->B_sdispls;
    int *B_sridxs  = opt_rp_spmm->B_sridxs;
    int *B_rcnts   = opt_rp_spmm->B_rcnts;
    int *B_rdispls = opt_rp_spmm->B_rdispls;

    double st, et;
    double exec_s = get_wtime_sec();

    // 1. Allocate work memory
    int n_send = 0, n_recv = 0;
    const int B_local_nrow  = opt_rp_spmm->B_local_nrow;
    const int B_remote_nrow = opt_rp_spmm->B_remote_nrow;
    const int B_sbuf_nrow   = opt_rp_spmm->B_sdispls[nproc];
    double *x_sbuf   = (double *) malloc(sizeof(double) * B_sbuf_nrow);
    double *x_remote = (double *) malloc(sizeof(double) * B_remote_nrow);
    MPI_Request *x_sreqs = (MPI_Request *) malloc(sizeof(MPI_Request) * nproc);
    MPI_Request *x_rreqs = (MPI_Request *) malloc(sizeof(MPI_Request) * nproc);
    ASSERT_PRINTF(
        x_sbuf != NULL && x_remote != NULL && x_sreqs != NULL && x_rreqs != NULL, 
        "Failed to allocate work memory\n"
    );

    // 2. Post all Irecv
    st = get_wtime_sec();
    for (int shift = 1; shift < nproc; shift++)
    {
        int p = (my_rank + shift) % nproc;
        if (B_rcnts[p] == 0) continue;
        MPI_Irecv(
            x_remote + B_rdispls[p], B_rcnts[p], MPI_DOUBLE, 
            p, p, opt_rp_spmm->comm, x_rreqs + n_recv
        );
        n_recv++;
    }
    et = get_wtime_sec();
    opt_rp_spmm->t_comm += et - st;

    // 3. Pack B send buffer for each proc and post Isend
    st = get_wtime_sec();
    for (int shift = 1; shift < nproc; shift++)
    {
        int p = (my_rank + nproc - shift) % nproc;
        if (B_scnts[p] == 0) continue;
        int *p_x_sridxs = B_sridxs + B_sdispls[p];
        double *p_x_sbuf = x_sbuf + B_sdispls[p];
        #pragma omp simd
        for (int i = 0; i < B_scnts[p]; i++) p_x_sbuf[i] = x_local[p_x_sridxs[i]];
        MPI_Isend(
            x_sbuf + B_sdispls[p], B_scnts[p], MPI_DOUBLE, 
            p, my_rank, opt_rp_spmm->comm, x_sreqs + n_send
        );
        n_send++;
    }  // End of p loop
    et = get_wtime_sec();
    opt_rp_spmm->t_pack += et - st;

    // 4. Compute C = A_diag * B_diag
    st = get_wtime_sec();
    const double d_one = 1.0, d_zero = 0.0;
    struct matrix_descr mkl_descA;
    mkl_descA.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_descA.mode = SPARSE_FILL_MODE_FULL;
    mkl_descA.diag = SPARSE_DIAG_NON_UNIT;
    if (B_local_nrow > 0)
    {
        sparse_matrix_t mkl_A_diag = (sparse_matrix_t) opt_rp_spmm->mkl_A_diag;
        mkl_sparse_d_mv(
            SPARSE_OPERATION_NON_TRANSPOSE, d_one, mkl_A_diag, 
            mkl_descA, x_local, d_zero, y_local
        );
    } else {
        for (int i = 0; i < opt_rp_spmm->A_nrow; i++) y_local[i] = 0.0;
    }
    et = get_wtime_sec();
    opt_rp_spmm->t_spmm += et - st;

    // 5. Wait for all Isend and Irecv
    st = get_wtime_sec();
    MPI_Waitall(n_send, x_sreqs, MPI_STATUSES_IGNORE);
    MPI_Waitall(n_recv, x_rreqs, MPI_STATUSES_IGNORE);
    et = get_wtime_sec();
    opt_rp_spmm->t_comm += et - st;

    // 6. Compute C += A_offd * B_offd
    st = get_wtime_sec();
    if (B_remote_nrow > 0)
    {
        sparse_matrix_t mkl_A_offd = (sparse_matrix_t) opt_rp_spmm->mkl_A_offd;
        mkl_sparse_d_mv(
            SPARSE_OPERATION_NON_TRANSPOSE, d_one, mkl_A_offd,
            mkl_descA, x_remote, d_one, y_local
        );
    }
    et = get_wtime_sec();
    opt_rp_spmm->t_spmm += et - st;

    // 8. Free work memory
    free(x_sbuf);
    free(x_remote);
    free(x_sreqs);
    free(x_rreqs);

    double exec_e = get_wtime_sec();
    opt_rp_spmm->t_exec += exec_e - exec_s;
    opt_rp_spmm->n_exec++;
}

// Print statistic info of opt_rp_spmm_p
void opt_rp_spmm_print_stat(opt_rp_spmm_p opt_rp_spmm)
{
    if (opt_rp_spmm == NULL) return;
    int my_rank = opt_rp_spmm->my_rank;
    int n_exec  = opt_rp_spmm->n_exec;
    if (n_exec == 0) return;
    size_t B_recv = (size_t) opt_rp_spmm->B_remote_nrow * (size_t) opt_rp_spmm->glb_n; 
    size_t B_recv_max = 0, B_recv_sum = 0;
    double t_raw[6], t_max[6], t_avg[6];
    t_raw[0] = opt_rp_spmm->t_init;
    t_raw[1] = opt_rp_spmm->t_pack;
    t_raw[2] = opt_rp_spmm->t_comm;
    t_raw[3] = opt_rp_spmm->t_unpack;
    t_raw[4] = opt_rp_spmm->t_spmm;
    t_raw[5] = opt_rp_spmm->t_exec;
    MPI_Reduce(&B_recv, &B_recv_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, opt_rp_spmm->comm);
    MPI_Reduce(&B_recv, &B_recv_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, opt_rp_spmm->comm);
    MPI_Reduce(&t_raw[0], &t_max[0], 6, MPI_DOUBLE, MPI_MAX, 0, opt_rp_spmm->comm);
    MPI_Reduce(&t_raw[0], &t_avg[0], 6, MPI_DOUBLE, MPI_SUM, 0, opt_rp_spmm->comm);
    for (int i = 1; i <= 5; i++)
    {
        t_max[i] = t_max[i] / n_exec;
        t_avg[i] = t_avg[i] / (n_exec * opt_rp_spmm->nproc);
    }
    if (my_rank == 0)
    {
        printf("opt_rp_spmm_init() time = %.2f s\n", t_max[0]);
        printf("Total number of communicated B matrix elements        = %zu\n", B_recv_sum);
        printf("Max number of received B matrix elements on a process = %zu\n", B_recv_max);
        printf("-------------------- Runtime (s) --------------------\n");
        printf("                                     avg         max\n");
        printf("Pack B matrix send buffer         %6.3f      %6.3f\n", t_avg[1], t_max[1]);
        printf("Wait for receiving B matrix       %6.3f      %6.3f\n", t_avg[2], t_max[2]);
        printf("Unpack received B matrix buffer   %6.3f      %6.3f\n", t_avg[3], t_max[3]);
        printf("Local SpMM                        %6.3f      %6.3f\n", t_avg[4], t_max[4]);
        printf("Total opt_rp_spmm_exec()          %6.3f      %6.3f\n", t_avg[5], t_max[5]);
        printf("\n");
        fflush(stdout);
    }
}

// Clear statistic info of opt_rp_spmm_p
void opt_rp_spmm_clear_stat(opt_rp_spmm_p opt_rp_spmm)
{
    if (opt_rp_spmm == NULL) return;
    opt_rp_spmm->n_exec   = 0;
    opt_rp_spmm->t_pack   = 0.0;
    opt_rp_spmm->t_comm   = 0.0;
    opt_rp_spmm->t_unpack = 0.0;
    opt_rp_spmm->t_spmm   = 0.0;
    opt_rp_spmm->t_exec   = 0.0;
}
