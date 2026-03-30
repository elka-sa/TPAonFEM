
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy as sp
import time
import copy
from ansys.mapdl import reader as pymapdl_reader
from ansys.dpf import core as dpf
import psutil
import threading
import pypardiso as pp
import scipy as sc
import joblib
import pyvista as pv
from sklearn.cluster import AgglomerativeClustering, KMeans
import ansys.math.core.math as pymath
from sparse_dot_mkl import pardisoinit, pardiso



#%% Monitoring functions
# Globals (must exist somewhere in your module)
memory_thread = None
monitoring = False
peak_memory = 0.0
elapsed_time = 0.0


def _format_hms(seconds):
    """
    PURPOSE:
        Convert a duration given in seconds into a formatted string "HH:MM:SS.s".

    INPUTS:
        seconds (float):
            Duration in seconds.

    OUTPUTS:
        formatted_time (str):
            Duration formatted as "HH:MM:SS.s".

    SIDE EFFECTS:
        None

    NOTES:
        - Uses floor division for hours/minutes and keeps 1 decimal place for seconds.

    EXAMPLES:
        >>> _format_hms(3661.2)
        '01:01:01.2'
    """
    hours = seconds // 3600
    minutes = (seconds - 3600 * hours) // 60
    secs = seconds - 3600 * hours - 60 * minutes
    return f"{hours:02.0f}:{minutes:02.0f}:{secs:04.1f}"


def monitor_time_memory(interval=1e-4):
    """
    PURPOSE:
        Monitor elapsed time and RSS memory usage of the current process in a loop.
        Tracks peak memory increase (MB) relative to the start of monitoring.

    INPUTS:
        interval (float, optional):
            Sleep time between checks (seconds). Default = 1e-4.

    OUTPUTS:
        None

    SIDE EFFECTS:
        - Updates global variables:
            elapsed_time (float): seconds since monitoring started
            peak_memory  (float): peak RSS delta in MB above baseline
        - Reads process RSS via psutil.

    NOTES:
        - This function is typically run in a background thread.
        - Peak memory is measured as (current RSS - baseline RSS) in MB.

    EXAMPLES:
        >>> monitoring = True
        >>> monitor_time_memory(interval=0.01)  # typically run in a thread
    """
    global peak_memory, elapsed_time, monitoring

    process = psutil.Process(os.getpid())
    start_time = time.time()
    baseline_mb = process.memory_info().rss / 1024 / 1024

    while monitoring:
        current_mb = process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current_mb - baseline_mb)
        elapsed_time = time.time() - start_time
        time.sleep(interval)


def thread_monitoring(start=True, print_res=True):
    """
    PURPOSE:
        Start or stop a background thread that monitors execution time and memory usage.

    INPUTS:
        start (bool, optional):
            If True: start monitoring. If False: stop monitoring. Default = True.
        print_res (bool, optional):
            If True, print results when stopping. Default = True.

    OUTPUTS:
        If start is True:
            None
        If start is False:
            elapsed_time (float):
                Total elapsed time in seconds.
            peak_memory (float):
                Peak RSS memory increase in MB above baseline.

    SIDE EFFECTS:
        - When starting:
            sets globals (monitoring=True, peak_memory=0.0, elapsed_time=0.0)
            starts a daemon thread
        - When stopping:
            sets monitoring=False and joins the thread
            optionally prints timing/memory summary

    NOTES:
        - Requires module globals: memory_thread, monitoring, peak_memory, elapsed_time.

    EXAMPLES:
        >>> thread_monitoring(start=True)
        >>> # ... run work ...
        >>> elapsed_time, peak_memory = thread_monitoring(start=False)
    """
    global memory_thread, monitoring, peak_memory, elapsed_time

    if start:
        peak_memory = 0.0
        elapsed_time = 0.0
        monitoring = True
        memory_thread = threading.Thread(target=monitor_time_memory, daemon=True)
        memory_thread.start()
        return None

    monitoring = False
    if memory_thread is not None:
        memory_thread.join()

    if print_res:
        print(f"Execution time: {_format_hms(elapsed_time)} (hh:mm:ss.)")
        print(f"Peak memory usage: {peak_memory:.2f} MB\n")

    return elapsed_time, peak_memory


def remaining_time(el_time, el_op, rem_op):
    """
    PURPOSE:
        Estimate and print remaining time for an operation assuming constant time per operation.

    INPUTS:
        el_time (float):
            Elapsed time so far (seconds).
        el_op (int | float):
            Number of operations completed so far.
        rem_op (int | float):
            Number of remaining operations.

    OUTPUTS:
        None

    SIDE EFFECTS:
        - Prints an estimated remaining time string to stdout.

    NOTES:
        - Assumes average time per operation remains constant.
        - If el_op == 0, this will raise a ZeroDivisionError (same behavior as current code).

    EXAMPLES:
        >>> remaining_time(el_time=10.0, el_op=5, rem_op=20)
    """
    remaining_seconds = (el_time / el_op) * rem_op
    print(f"\tRemaining time of operation: {_format_hms(remaining_seconds)} (hh:mm:ss.). ")
    
#%% Solver and analysis functions
def inverse_sparse_matrix(A, method='cg', tol=1e-5, maxiter=100):
    """
    PURPOSE:
        Compute (an explicit representation of) the inverse of a sparse square matrix A
        by solving A x = e_i for each canonical basis vector e_i.

        This function also detects and temporarily removes empty rows/columns
        (rows/cols with all zeros) before inversion, then maps the inverse back to the
        original indexing.

    INPUTS:
        A (scipy.sparse.csc_matrix | scipy.sparse.csr_matrix):
            Sparse square matrix to invert.

        method (str, optional):
            Algorithm used to compute the inverse:
                - 'cg'         : Conjugate Gradient (requires SPD matrix in theory)
                - 'gmres'      : GMRES (more general, works for nonsymmetric)
                - 'spsolve'    : Direct sparse solve per column (factorization may repeat)
                - 'LinearOperator': Return a lazy inverse operator instead of an explicit matrix
                - 'inv'        : SciPy sparse inverse (may densify / be expensive)
                - 'pardiso'    : Factorize once using pypardiso and solve for each column
            Default = 'cg'.

        tol (float, optional):
            Iterative solver tolerance (for cg/gmres). Default = 1e-5.

        maxiter (int, optional):
            Maximum iterations (for cg/gmres). Default = 100.

    OUTPUTS:
        A_inv (scipy.sparse.csc_matrix | scipy.sparse.linalg.LinearOperator):
            - If method in {'cg','gmres','spsolve','inv','pardiso'}:
                Returns a sparse matrix intended to represent A^{-1}.
            - If method == 'LinearOperator':
                Returns a LinearOperator that applies A^{-1} to a vector.

    SIDE EFFECTS:
        - Prints periodic remaining-time estimates via remaining_time(...) (every ~5 s).
        - Uses external solver objects (pypardiso factorization) depending on method.

    NOTES:
        - Computing an explicit inverse is usually expensive and numerically delicate.
          Often you should solve A x = b directly instead of forming A^{-1}.
        - The “empty row/col removal” step builds a selector matrix nnz_mat that
          extracts the non-empty DOFs; inversion is performed on the reduced matrix.
        - For 'cg' to be mathematically valid, A should be symmetric positive definite.

    EXAMPLES:
        >>> A_inv = inverse_sparse_matrix(A, method='pardiso')
        >>> x = A_inv @ b
    """

    # --- 1) Detect empty rows/cols (all zeros) to avoid singular / ill-posed inversion ---
    # rows_nnz[j] is True if column j has any nonzero entries (A.getnnz(axis=0)).
    rows_nnz = A.getnnz(axis=0) != 0

    # cols_nnz[i] is True if row i has any nonzero entries (A.getnnz(axis=1)).
    cols_nnz = A.getnnz(axis=1) != 0

    # nnz is True only for indices that are non-empty both as row and as column.
    # This is a conservative choice: keep DOFs that participate in the system.
    nnz = rows_nnz * cols_nnz

    # --- 2) Build a "selection / embedding" matrix nnz_mat ---
    # nnz_mat is initially a diagonal matrix with 1 on "kept" entries, 0 otherwise.
    # Shape initially: (N, N)
    nnz_mat = sp.sparse.diags(nnz.astype(int), format='csc')

    # Then we slice columns to remove the zero columns, yielding shape (N, n_kept)
    # This acts like an embedding from reduced space to full space.
    nnz_mat = nnz_mat[:, nnz]

    # --- 3) Reduce A to the non-empty subspace ---
    # Reduced A has shape (n_kept, n_kept)
    A = nnz_mat.T @ A @ nnz_mat

    # Dimension of reduced problem
    n = A.shape[0]

    # Identity matrix in reduced space; columns are the e_i right-hand sides
    I = sp.sparse.eye(n, format='csc')

    # --- 4) Progress tracking setup (prints remaining time every t_check seconds) ---
    t0 = time.time()   # global start time
    t1 = time.time()   # last time we printed a progress estimate
    t_check = 5        # seconds between progress prints

    # --- 5) Solve A x = e_i for each i; stack x vectors into columns of A^{-1} ---
    if method == 'cg':
        A_inv_cols = []
        for i in range(n):
            # Extract i-th canonical basis vector as dense 1D array
            e = I[:, i].toarray().ravel()

            # Solve A x = e using Conjugate Gradient
            x, info = sp.sparse.linalg.cg(A, e, tol=tol, maxiter=maxiter)

            # Store solution as sparse column (csc_matrix of shape (n, 1))
            A_inv_cols.append(sp.sparse.csc_matrix(x))

            # Periodically print remaining time estimate
            if time.time() - t1 >= t_check:
                el_time = time.time() - t0
                el_op = i + 1          # completed columns
                rem_op = n - i         # remaining columns (note: off-by-one in original kept)
                remaining_time(el_time, el_op, rem_op)
                t1 = time.time()

        # We stacked columns as a list of column-vectors; vstack makes (n*n,1) style,
        # so we transpose at the end to get (n, n).
        A_inv = sp.sparse.vstack(A_inv_cols).T

    elif method == 'gmres':
        A_inv_cols = []
        for i in range(n):
            e = I[:, i].toarray().ravel()

            # GMRES is suitable for more general matrices than CG
            x, info = sp.sparse.linalg.gmres(A, e, tol=tol, maxiter=maxiter)
            A_inv_cols.append(sp.sparse.csc_matrix(x))

            if time.time() - t1 >= t_check:
                el_time = time.time() - t0
                el_op = i + 1
                rem_op = n - i
                remaining_time(el_time, el_op, rem_op)
                t1 = time.time()

        A_inv = sp.sparse.vstack(A_inv_cols).T

    elif method == 'spsolve':
        A_inv_cols = []
        for i in range(n):
            e = I[:, i].toarray().ravel()

            # Direct solve for one RHS (SciPy chooses a factorization internally)
            x = sp.sparse.linalg.spsolve(A, e)
            A_inv_cols.append(sp.sparse.csc_matrix(x))

            if time.time() - t1 >= t_check:
                el_time = time.time() - t0
                el_op = i + 1
                rem_op = n - i
                remaining_time(el_time, el_op, rem_op)
                t1 = time.time()

        A_inv = sp.sparse.vstack(A_inv_cols).T

    elif method == 'LinearOperator':
        # This returns a lazy operator that computes A^{-1} x without building A^{-1}.
        # NOTE: original code uses pp.spsolve(...) here; we keep that behavior.
        n = A.shape[0]
        A_inv = sp.sparse.linalg.LinearOperator((n, n), matvec=lambda x: pp.spsolve(A, x))

    elif method == 'inv':
        # SciPy sparse inverse (may be expensive; can fill-in heavily)
        A_inv = sp.sparse.linalg.inv(A)

    elif method == 'pardiso':
        # Factorize A once, then solve repeatedly (fast if many RHS).
        # pp.factorized(A) returns a callable solve(rhs)->x.
        solve = pp.factorized(A)

        A_inv_cols = []
        for i in range(n):
            e = I[:, i].toarray().ravel()
            x = solve(e)
            A_inv_cols.append(sp.sparse.csc_matrix(x))

            if time.time() - t1 >= t_check:
                el_time = time.time() - t0
                el_op = i + 1
                rem_op = n - i
                remaining_time(el_time, el_op, rem_op)
                t1 = time.time()

        # If reduced matrix size is 0, return an empty sparse matrix
        if len(A_inv_cols) != 0:
            A_inv = sp.sparse.vstack(A_inv_cols).T
        else:
            A_inv = sp.sparse.csc_matrix((0, 0))

        # Map inverse back to the original full space:
        # A_full^{-1} = nnz_mat * A_reduced^{-1} * nnz_mat^T
        A_inv = nnz_mat @ A_inv @ nnz_mat.T

    return A_inv


def compute_modal_basis(K, M, constraint_mat=None, n_modes=10, freq0=0.0, which='LM', ndim=1, sparse=False,
                        hermitian=True, compute_K_inv=False, tol_static=0.0, alpha_M=0.0, shapes=[], return_eigenvects=False,
                        vibro_acoustic_shapes=[], vibro_acoustic_ndim=[1, 3]):
    """
    PURPOSE:
        Compute a modal basis (eigenfrequencies and mode shapes) of a mechanical system
        defined by stiffness K and mass M, solving the generalized eigenproblem:

            K * modes = omega2 * M * modes

        The returned eigenfrequencies are:
            eigenfreq = sqrt(omega2) / (2*pi)

    INPUTS:
        K (scipy.sparse matrix):
            Stiffness matrix (square). Typically sparse.

        M (scipy.sparse matrix):
            Mass matrix (square). Typically sparse.

        constraint_mat (scipy.sparse matrix | None, optional):
            If provided, maps computed eigenvectors through:
                modes = constraint_mat.T @ modes
            Default = None.

        n_modes (int | list, optional):
            - If int: number of modes to compute.
            - If list [lim_down, lim_up]: compute up to lim_up and slice results.
            Default = 10.

        freq0 (float, optional):
            Shift (sigma) used by sparse eigensolver to target eigenvalues near freq0.
            Default = 0.0.

        which (str, optional):
            Which eigenvalues to target in sparse eigensolver (eigsh). Default = 'LM'.

        ndim (int, optional):
            Dimensionality used by modes_flat_to_3d(...) reshaping. Default = 1.

        sparse (bool, optional):
            If True, returns modes as sparse csc_matrix (where applicable). Default = False.

        hermitian (bool, optional):
            If True, treat problem as symmetric/hermitian and use eigsh/eigh.
            If False, use general eig and compute/normalize left eigenvectors.
            Default = True.

        compute_K_inv (bool, optional):
            If True, uses dense eigh(K, M) and truncates modes. Also computes omegas_inv (unused here).
            Default = False.

        tol_static (float, optional):
            Present in signature but not used in the provided code. Default = 0.0.

        alpha_M (float, optional):
            Stabilization term for mass matrix: solves with (M + alpha_M * I).
            Helps make M positive definite for eigensolvers. Default = 0.0.

        shapes (list, optional):
            If provided, uses modes_flat_to_3d_2(..., shapes=shapes). Default = [].

        return_eigenvects (bool, optional):
            If True, returns raw eigenvectors (dense or sparse) additionally. Default = False.

        vibro_acoustic_shapes (list, optional):
            If provided (non-empty), split modes into two parts (structure/acoustic).
            Default = [].

        vibro_acoustic_ndim (list, optional):
            ndims used in the vibro-acoustic split reshape. Default = [1, 3].

    OUTPUTS:
        eigenfreq (np.ndarray):
            Natural frequencies (Hz). Complex if eigenvalues are complex/negative.

        modes (np.ndarray | scipy.sparse.csc_matrix | list):
            Mode shapes in flattened form (columns correspond to modes). May be:
              - a matrix (n_dof, n_modes), or
              - a list of two matrices for vibro-acoustic splitting.

        modes_ndim (np.ndarray | list):
            Reshaped modes (output of modes_flat_to_3d*). May be array or list for vibro-acoustic.

        If hermitian == False:
            returns (eigenfreq, modes, modes_ndim)

        If return_eigenvects == True (and hermitian == True):
            returns (eigenfreq, modes, modes_ndim, eigenvects)

    SIDE EFFECTS:
        - Converts sparse matrices to dense arrays in some branches (toarray()).

    NOTES:
        - In the non-hermitian branch, left eigenvectors are used to normalize modes with M.
        - If K.shape[0] == 0, returns (inf frequency, empty modes, None).
        - If n_modes >= K.shape[0]-1, code may compute full dense spectrum (expensive).
        - `tol_static` is unused in the provided snippet (kept for API compatibility).

    EXAMPLES:
        >>> f, Phi, Phi_nd = compute_modal_basis(K, M, n_modes=20, sparse=True)
    """

    # --- Handle n_modes being either int or [lim_down, lim_up] slicing request ---
    if type(n_modes) == list:
        lim_down, lim_up = n_modes[0], n_modes[1]
        n_modes = lim_up
    else:
        lim_down, lim_up = None, None

    # --- Optional dense solve branch (called "compute_K_inv" in your code) ---
    if compute_K_inv:
        # Dense generalized Hermitian eigenproblem solve:
        # omegas here are actually omega^2 (eigenvalues lambda)
        omegas, modes = sp.linalg.eigh(K.toarray(), b=M.toarray())

        # This would represent 1/omega^2 (used for pseudo-inverse ideas)
        omegas_inv = omegas**(-1)
        omegas_inv[omegas_inv > 1e3] = 0.0

        # Truncate to requested modes
        omegas = omegas[:n_modes]
        modes = modes[:, :n_modes]

    else:
        # --- Main eigen-solve path ---
        if n_modes < K.shape[0] - 1:
            # Stabilize M (help positive definiteness)
            alphaI = alpha_M * sp.sparse.eye(M.shape[0], format='csc')

            if hermitian == True:
                # Force symmetry (numerical cleanup)
                K = 0.5 * (K.T + K)
                M = 0.5 * (M.T + M)

                # Sparse symmetric generalized eigenproblem near sigma=freq0:
                # Returns k eigenpairs
                omegas, modes = sp.sparse.linalg.eigsh(
                    K, M=(M + alphaI), k=n_modes, sigma=freq0, which=which
                )

            else:
                # Dense general (non-Hermitian) generalized eigenproblem:
                # left=True returns left eigenvectors too.
                omegas, modes_L, modes = sp.linalg.eig(
                    K.toarray(), b=(M + alphaI).toarray(), left=True
                )

                # Normalize right eigenvectors using left eigenvectors and M:
                # normal_fact[j] = (phi_L_j^H M phi_R_j)
                normal_fact = np.diag(modes_L.T.conj() @ M @ modes)
                normal_fact = np.where(normal_fact == 0, 1.0, normal_fact)
                modes = modes / normal_fact

                # Sort by eigenvalue magnitude (actually by raw value)
                order_omega = np.argsort(omegas)
                modes = modes[:, order_omega]
                modes_L = modes_L[:, order_omega]
                omegas = omegas[order_omega]

                # Keep only first n_modes
                modes = modes[:, :n_modes]
                modes_L = modes_L[:, :n_modes]
                omegas = omegas[:n_modes]

        elif K.shape[0] != 0:
            # If asked for many modes, compute full dense spectrum
            alphaI = alpha_M * sp.sparse.eye(M.shape[0])

            if hermitian == True:
                # Force symmetry and solve dense Hermitian generalized eigenproblem
                K = 0.5 * (K.T + K)
                M = 0.5 * (M.T + M)
                omegas, modes = sp.linalg.eigh(K.toarray(), b=(M + alphaI).toarray())

            else:
                omegas, modes_L, modes = sp.linalg.eig(
                    K.toarray(), b=(M + alphaI).toarray(), left=True
                )

                # Normalize and sort as above
                normal_fact = np.diag(modes_L.T.conj() @ M @ modes)
                normal_fact = np.where(normal_fact == 0, 1.0, normal_fact)
                modes = modes / normal_fact

                order_omega = np.argsort(omegas)
                modes = modes[:, order_omega]
                modes_L = modes_L[:, order_omega]
                omegas = omegas[order_omega]

        else:
            # Edge case: empty system
            return (np.array([np.inf]), sp.sparse.csc_matrix((0, 0)), None)

    # --- Convert eigenvalues omega^2 to eigenfrequencies (Hz) ---
    # If eigenvalues are complex or negative, use complex sqrt
    if np.any(omegas.imag) or np.any(omegas.real < 0):
        eigenfreq = np.sqrt(omegas, dtype=complex) / (2 * np.pi)
    else:
        eigenfreq = np.sqrt(omegas.real) / (2 * np.pi)

    # If modes are purely real, drop tiny imaginary part
    if ~np.any(modes.imag):
        modes = modes.real

    # Optional: store raw eigenvectors before constraints/reshaping
    if return_eigenvects == True:
        if sparse == True:
            eigenvects = sp.sparse.csc_matrix(modes)
        else:
            eigenvects = 1.0 * modes

    # Apply constraint mapping (e.g., reduce/expand DOFs)
    if constraint_mat != None:
        modes = constraint_mat.T @ modes

    # Reshape modes into (something like) [n_nodes, n_modes, ndim] depending on helper
    if len(shapes) == 0:
        modes_ndim = modes_flat_to_3d(modes, ndim=ndim)
    else:
        modes, modes_ndim = modes_flat_to_3d_2(modes, ndim=ndim, shapes=shapes)

    # Vibro-acoustic split: treat one vector as concatenation of two physics fields
    if len(vibro_acoustic_shapes) != 0:
        modes_ndim = [None, None]

        # Split flattened modes into two blocks
        modes = [
            modes[:vibro_acoustic_shapes[0].shape[0]],
            modes[vibro_acoustic_shapes[0].shape[0]:]
        ]

        # Reshape each block separately
        modes_ndim[0] = modes_flat_to_3d(modes[0], ndim=vibro_acoustic_ndim[0])
        modes_ndim[1] = modes_flat_to_3d(modes[1], ndim=vibro_acoustic_ndim[1])

    # Convert to sparse if requested
    if sparse == True:
        if len(vibro_acoustic_shapes) == 0:
            if len(shapes) == 0:
                modes = sp.sparse.csc_matrix(modes)
            else:
                modes = [sp.sparse.csc_matrix(modes[i]) for i in range(0, len(modes))]
        else:
            modes = [sp.sparse.csc_matrix(modes[0]), sp.sparse.csc_matrix(modes[1])]

    # If user requested slicing (n_modes given as [lim_down, lim_up])
    if type(lim_down) != type(None):
        eigenfreq = eigenfreq[lim_down:lim_up]
        modes = modes[:, lim_down:lim_up]
        modes_ndim = modes_ndim[:, lim_down:lim_up, :]

    # Output selection consistent with your original return logic
    if hermitian == False:
        return eigenfreq, modes, modes_ndim
    elif return_eigenvects == True:
        return eigenfreq, modes, modes_ndim, eigenvects
    else:
        return eigenfreq, modes.tocsc(), modes_ndim


def compute_constraint_modes(K, Bii_dict_comp, Bbb_dict_comp, LUKii=None, va=False):
    """
    PURPOSE:
        Compute constraint (static) modes for substructuring / CMS coupling.

        For each interface 'intf', it forms:
            Kii = Bii * K * Bii^T
            Kib = Bii * K * Bbb_intf^T

        and solves:
            Kii * Psi_ib = - Kib

        producing Psi_ib as a matrix whose columns correspond to boundary/interface DOFs.

    INPUTS:
        K (scipy.sparse matrix):
            Global stiffness matrix of the component.

        Bii_dict_comp (scipy.sparse matrix):
            Boolean/selection matrix extracting internal DOFs ('i') from the full DOF set.

        Bbb_dict_comp (dict):
            Dictionary mapping interface name -> selection matrix extracting boundary DOFs ('b')
            for that interface from the full DOF set.

        LUKii (callable | None, optional):
            If None, factorize Kii using pp.factorized(Kii).
            If provided, assumed to be a solver such that LUKii(rhs) solves Kii x = rhs.
            Default = None.

        va (bool, optional):
            Controls the shape of empty interface return matrix.
            If va is True, returns (n_i, n_b) for empty interface, else (n_i, 0).
            Default = False.

    OUTPUTS:
        Psi_ib0_dict_comp (dict):
            Dictionary mapping interface name -> constraint modes Psi_ib0 as sparse csc_matrix.

        LUKii (callable):
            Factorized solver for Kii (reused across interfaces).

    SIDE EFFECTS:
        - Performs sparse factorization if LUKii is None (can be expensive).

    NOTES:
        - If an interface has no boundary DOFs (empty selection matrix),
          Kib0 is treated as empty and Psi is returned as appropriately sized empty matrix.
        - The code builds Psi by solving each RHS column separately.

    EXAMPLES:
        >>> Psi_dict, LUKii = compute_constraint_modes(K, Bii, Bbb_dict)
    """

    # If no internal-factorization provided, build it once.
    if LUKii == None:
        # Extract the internal-internal stiffness block Kii
        Kii = Bii_dict_comp @ K @ Bii_dict_comp.T

        # Factorize Kii once: returns function solve(rhs) -> x
        LUKii = pp.factorized(Kii)

    Psi_ib0_dict_comp = {}

    # Loop over each interface block
    for intf in Bbb_dict_comp.keys():

        # --- Build Kib0 = K_{i b(intf)} ---
        if len(Bbb_dict_comp[intf].data) == 0:
            # If the boundary selector is empty, create an empty Kib0 with correct shape.
            # Here we multiply by a (n_b, 0) empty matrix to preserve dimensions.
            Kib0 = Bii_dict_comp @ K @ sp.sparse.csc_matrix((Bbb_dict_comp[intf].shape[1], 0))
        else:
            # Normal case: Kib0 is internal vs boundary coupling block
            Kib0 = Bii_dict_comp @ K @ Bbb_dict_comp[intf].T

        # --- Solve Kii * psi = -Kib0[:, j] for each boundary DOF column j ---
        Psi_ib0 = []
        for i in range(Kib0.shape[1]):
            if Bii_dict_comp.shape[0] != 0:
                # Convert RHS column to dense vector for the solver callable
                psi_ib0 = LUKii((-Kib0[:, i]).toarray().ravel())
            else:
                # No internal DOFs -> return zeros
                psi_ib0 = np.zeros(Kib0.shape[0])

            Psi_ib0.append(psi_ib0)

        # Stack solutions into a matrix (n_i, n_b).
        # If no columns, return a carefully sized empty matrix.
        if len(Psi_ib0) != 0:
            Psi_ib0_dict_comp[intf] = sp.sparse.csc_matrix(np.vstack(Psi_ib0)).T
        else:
            # If vibro-acoustic (va) you preserve boundary size, else return 0 columns
            Psi_ib0_dict_comp[intf] = (
                sp.sparse.csc_matrix((Bii_dict_comp.shape[0], Bbb_dict_comp[intf].shape[0]))
                if va else
                sp.sparse.csc_matrix((Bii_dict_comp.shape[0], 0))
            )

    return Psi_ib0_dict_comp, LUKii


def compute_attachment_modes(K, Bbb_dict_comp, Phi, omega2_arr, LUK_nnz=[], compute_Kr_bb0=False):
    """
    PURPOSE:
        Compute attachment modes (also interpretable as residual flexibility contributions)
        at interfaces for a component, using:
            - constrained static solutions (to remove rigid body singularity)
            - subtraction of modal contribution Phi * Omega^{-1} * Phi_b^T

        For each interface 'intf', it computes a matrix Gr_b0 (full DOFs response) and
        optionally Kr_bb0 = (Gr_bb0)^{-1} at the boundary.

    INPUTS:
        K (scipy.sparse matrix):
            Component stiffness matrix.

        Bbb_dict_comp (dict):
            Dictionary mapping interface name -> boundary selection matrix.

        Phi (np.ndarray | scipy.sparse matrix):
            Modal matrix (flattened DOFs x n_modes).

        omega2_arr (np.ndarray):
            Eigenvalues (omega^2) associated with Phi (same ordering).

        LUK_nnz (list, optional):
            If empty, the function will:
                - build nnz_mat (constraints)
                - factorize constrained stiffness
            If provided, expected to be [LUK, nnz_mat] where:
                LUK(rhs) solves constrained K system,
                nnz_mat maps constrained DOFs to full DOFs.
            Default = [].

        compute_Kr_bb0 (bool, optional):
            If True, computes inverse of Gr_bb0 via inverse_sparse_matrix(..., 'pardiso').
            Default = False.

    OUTPUTS:
        Gr_b0_dict_comp (dict):
            Dictionary interface -> Gr_b0 matrix (full DOFs x boundary DOFs).

        Kr_bb0_dict_comp (dict):
            Dictionary interface -> Kr_bb0 matrix (boundary DOFs x boundary DOFs)
            (either computed inverse or placeholder sparse matrix).

        (LUK, nnz_mat) (tuple):
            Solver and constraint mapping used, for reuse in later calls.

    SIDE EFFECTS:
        - If LUK_nnz is empty, uses randomness to pick constrained DOFs.
        - Factorizes a constrained stiffness matrix (expensive).
        - May compute explicit sparse inverses if compute_Kr_bb0 is True.

    NOTES:
        - The random constraints are intended to remove rigid-body singularity in K.
          For reproducibility you may want to seed RNG outside (not changed here).
        - n_RB_modes is inferred as the first index where omega2_arr > 1.0.
          This assumes low omega^2 correspond to rigid-body modes.

    EXAMPLES:
        >>> Gr_dict, Kr_dict, cache = compute_attachment_modes(K, Bbb, Phi, omega2)
    """

    # --- If no cached solver, build constraints and factorize constrained stiffness ---
    if len(LUK_nnz) == 0:
        # Number of rigid-body modes inferred from omega^2 threshold.
        # n_RB_modes = first index where omega2_arr > 1.0
        n_RB_modes = np.where(omega2_arr > 1.0)[0].min()

        # nnz: boolean mask of DOFs to keep in the constrained system
        nnz = np.ones(K.shape[0]).astype(bool)

        # Randomly select DOFs to constrain (set to False) equal to number of RB modes
        pos_False = np.sort(np.random.randint(0, K.shape[0], n_RB_modes))

        # Apply constraints (remove these DOFs from the reduced system)
        nnz[pos_False] = False

        # Build nnz_mat (embedding from reduced DOFs to full DOFs), shape (N, n_kept)
        nnz_mat = sp.sparse.diags(nnz.astype(int), format='csc')
        nnz_mat = nnz_mat[:, nnz]

        # Factorize constrained stiffness once
        LUK = pp.factorized(nnz_mat.T @ K @ nnz_mat)

    else:
        # Reuse cached factorization and mapping
        LUK, nnz_mat = LUK_nnz

    Gr_b0_dict_comp = {}
    Kr_bb0_dict_comp = {}

    # I is a reduced-space identity embedded in constrained coordinates:
    # I has shape (n_kept, N) because nnz_mat.T is (n_kept, N).
    I = nnz_mat.T @ sp.sparse.eye(K.shape[0], format='csc')

    for intf in Bbb_dict_comp.keys():

        # --- Build RHS unit vectors at the interface (in reduced coordinates) ---
        if len(Bbb_dict_comp[intf].data) == 0:
            # Empty interface => empty RHS
            Ib0 = I @ sp.sparse.csc_matrix((Bbb_dict_comp[intf].shape[1], 0))
            Phi_b0 = sp.sparse.csc_matrix((0, Phi.shape[1]))
        else:
            # Ib0 columns correspond to unit forces at each boundary DOF, mapped into reduced space
            Ib0 = I @ Bbb_dict_comp[intf].T

            # Modal shapes restricted to the boundary interface:
            # Phi_b0 has shape (n_b, n_modes)
            Phi_b0 = Bbb_dict_comp[intf] @ Phi

        # --- Solve constrained static system for each RHS column ---
        K_b0_inv = []
        for i in range(Ib0.shape[1]):
            # Solve (nnz_mat.T K nnz_mat) * x = -Ib0[:, i]
            k_b0_inv = LUK((-Ib0[:, i]).toarray().ravel())
            K_b0_inv.append(k_b0_inv)

        # Stack solutions and map back to full space:
        # full_solution = nnz_mat * x
        if len(K_b0_inv) != 0:
            K_b0_inv = nnz_mat @ sp.sparse.csc_matrix(np.vstack(K_b0_inv)).T
        else:
            K_b0_inv = sp.sparse.csc_matrix((K.shape[0], 0))

        # --- Subtract modal contribution to get attachment/residual term ---
        # Omega2_inv is diagonal with 1/omega^2 where omega^2 > 1e-5 (avoid divide-by-0)
        Omega2_inv = sp.sparse.diags(
            np.divide(1, omega2_arr, where=omega2_arr > 1e-5, out=np.zeros_like(omega2_arr)),
            format='csc'
        )

        # Gr_b0 := static constrained response - modal response part
        # Shape: (N, n_b)
        Gr_b0_dict_comp[intf] = K_b0_inv - Phi @ Omega2_inv @ Phi_b0.T

        # --- Compute boundary-only block and optionally its inverse ---
        if len(Bbb_dict_comp[intf].data) == 0:
            Gr_bb0 = sp.sparse.csc_matrix((0, 0))
            Kr_bb0 = sp.sparse.csc_matrix((0, 0))
        else:
            # Restrict Gr_b0 to boundary: Gr_bb0 = Bbb * Gr_b0
            Gr_bb0 = Bbb_dict_comp[intf] @ Gr_b0_dict_comp[intf]

            if compute_Kr_bb0:
                # Inverse of interface residual flexibility matrix
                Kr_bb0 = inverse_sparse_matrix(Gr_bb0, method='pardiso')
            else:
                # Placeholder (keeps shapes consistent without extra compute)
                Kr_bb0 = sp.sparse.csc_matrix(Gr_bb0.shape)

        Kr_bb0_dict_comp[intf] = Kr_bb0

    return Gr_b0_dict_comp, Kr_bb0_dict_comp, (LUK, nnz_mat)


def transient_response(K, D, M, f, DELTAt, n_inc, sparse=True, dtype=complex, solver='EXP', al_solver='mkl_pardiso'):
    """
    PURPOSE:
        Compute transient (time-domain) response of a second-order linear dynamic system:

            M u_ddot + D u_dot + K u = f(t)

        by rewriting it into a first-order state-space form and integrating in time.

        The state vector is stored as:
            eta = [v; u]
        where v is velocity-like and u is displacement-like (based on your block matrices).

    INPUTS:
        K, D, M (scipy.sparse matrices):
            Stiffness, damping, and mass matrices (square, same size).

        f (np.ndarray):
            Force vector(s):
                - shape (n_dof,) for constant/impulse-like forcing
                - shape (n_dof, n_steps) for time-varying forcing

        DELTAt (float):
            Time step size (seconds).

        n_inc (int):
            Number of increments (time steps). Total stored steps = n_inc + 1.

        sparse (bool, optional):
            Present in signature; current code always uses sparse operators internally.
            Default = True.

        dtype (type, optional):
            dtype of state vector storage (complex or float). Default = complex.

        solver (str, optional):
            Time integration scheme:
                - 'EXP'     : Forward Euler (explicit)
                - 'IMP'     : Backward Euler (implicit)
                - 'EXP-IMP' : Averaged scheme (semi-implicit)
            Default = 'EXP'.

        al_solver (str, optional):
            Linear algebra backend for implicit solves:
                - 'mkl_pardiso'
                - 'pypardiso'
            Default = 'mkl_pardiso'.

    OUTPUTS:
        eta (np.ndarray):
            Displacement history (only the u part is returned), shape (n_dof, n_steps).

        stamps (np.ndarray):
            Time stamps (seconds), length n_steps.

    SIDE EFFECTS:
        - Prints progress each time step (including max state magnitude).
        - For 'IMP' with mkl_pardiso: initializes and frees Pardiso each step.

    NOTES:
        - The explicit solver builds an approximate M^{-1} via a reduced pseudo-inverse
          to handle potentially singular mass matrices (zero rows/cols).
        - The implicit solver solves (A - dt B) eta_{k+1} = A eta_k + dt C f_{k+1}
          depending on forcing handling branch.
        - The function returns only the displacement part u = eta[n_dof:, :].

    EXAMPLES:
        >>> u_t, t = transient_response(K, D, M, f, 1e-4, 1000, solver='IMP')
    """

    # --- Build first-order state-space matrices ---
    # Identity of size n_dof
    I = sp.sparse.eye(K.shape[0])

    # Zero matrix of same shape as K
    O = sp.sparse.csc_matrix(K.shape)

    # A is block diagonal [M, I]
    # This encodes M*v_dot + ... in first block and u_dot - v = 0 in second block (conceptually)
    A = sp.sparse.block_diag([M, I], format='csc')

    # B is the system coupling matrix in first-order form
    B = sp.sparse.bmat([[-D, -K],
                        [ I, None]], format='csc')

    # C maps input forcing into the first block equation (acceleration/velocity equation)
    C = sp.sparse.bmat([[I],
                        [O]], format='csc')

    # --- Explicit solver (Forward Euler): eta_{k+1} = A^{-1} ((A + dt B) eta_k + dt C f_k) ---
    if solver == 'EXP':
        # Build an approximate inverse for A:
        # Ainv = block_diag([M_inv, I]) because A is block_diag([M, I]).
        #
        # Here you compute a pseudo-inverse for a reduced mass matrix
        # to handle possible empty rows/cols in M (singular M).
        vect_red = M.getnnz(axis=0).astype(bool)  # keep DOFs with nonzero mass column
        mat_red = sp.sparse.diags(vect_red.astype(int), format='csc')[vect_red, :]  # selector

        M_red = mat_red @ M @ mat_red.T  # reduced mass block
        M_red_inv = sp.sparse.csc_matrix(np.linalg.pinv(M_red.todense()))  # pseudo-inverse dense
        M_inv = mat_red.T @ M_red_inv @ mat_red  # embed back to full size
        Ainv = sp.sparse.block_diag([M_inv, I], format='csc')

        # Time vector (n_inc+1 points)
        time_vector = np.arange(0, DELTAt * (n_inc + 1), DELTAt)

        # State history: eta has 2*n_dof rows (v and u) and n_steps columns
        eta = np.zeros((2 * K.shape[1], time_vector.shape[0]), dtype=dtype)

        # Working vectors for iterative update
        eta_k1 = np.zeros(eta.shape[0])
        eta_k = np.zeros(eta.shape[0])

        stamps = time_vector

        print('Solving using the explicit solver.')

        for k in range(0, len(time_vector) - 1):

            # Build the RHS F = (A + dt B) eta_k + dt C f_k (forcing handling)
            if len(f.shape) == 1:
                # Force given as a single vector (applied only at first step in your code)
                if k == 0:
                    F = (A + DELTAt * B) @ eta_k + DELTAt * C @ f
                else:
                    F = (A + DELTAt * B) @ eta_k
            else:
                # Force provided for each time step
                F = (A + DELTAt * B) @ eta_k + DELTAt * C @ f[:, k]

            # Explicit step: eta_{k+1} = Ainv * F
            eta_k1 = Ainv @ F
            eta[:, k + 1] = eta_k1
            eta_k = 1 * eta_k1  # copy

            print(str(k + 1) + ' | t = ' + f"{time_vector[k]:.2e}" + ' [s] | ' + f"{max(eta_k):.2e}")

    # --- Implicit solver (Backward Euler): (A - dt B) eta_{k+1} = A eta_k + dt C f_{k+1} ---
    elif solver == 'IMP':
        print('Solving using the implicit solver.')

        time_vector = np.arange(0, DELTAt * (n_inc + 1), DELTAt)
        eta = np.zeros((2 * K.shape[1], time_vector.shape[0]), dtype=dtype)
        stamps = time_vector

        for k in range(0, len(time_vector) - 1):
            # Build RHS F = A eta_k + dt C f_{k+1} (your code uses special case at k=0)
            if len(f.shape) == 1:
                if k == 0:
                    F = A @ eta[:, k] + DELTAt * C @ f
                else:
                    F = A @ eta[:, k]
            else:
                F = A @ eta[:, k] + DELTAt * C @ f[:, k + 1]

            # Solve linear system for eta_{k+1}
            if al_solver == 'mkl_pardiso':
                # mtype=11 => real unsymmetric matrix solver
                mtype = 11
                pt, iparm = pardisoinit(mtype=mtype)

                # Solve (A - dt B) * eta_{k+1} = F
                eta[:, k + 1] = pardiso(
                    (A - DELTAt * B).real.astype(float).tocsr(),
                    F.real.astype(float),
                    pt, mtype, iparm
                )[0].ravel()

                # Release pardiso internal memory
                pardiso((A - DELTAt * B).real.astype(float).tocsr(), F, pt, mtype, iparm, phase=-1)

            elif al_solver == 'pypardiso':
                eta[:, k + 1] = pp.spsolve((A - DELTAt * B).astype(float), F.astype(float))

            print(str(k + 1) + ' | t = ' + f"{time_vector[k]:.2e}" + ' [s] | ' + f"{max(eta[:, k + 1]):.2e}")

    # --- Averaged (semi-implicit) scheme ---
    elif solver == 'EXP-IMP':
        print('Solving using the averaged solver.')

        time_vector = np.arange(0, DELTAt * (n_inc + 1), DELTAt)
        eta = np.zeros((2 * K.shape[1], time_vector.shape[0]), dtype=dtype)
        stamps = time_vector

        for k in range(0, len(time_vector) - 1):
            if len(f.shape) == 1:
                if k == 0:
                    F = (A + 1 / 2 * DELTAt * B) @ eta[:, k] + DELTAt * C @ f
                else:
                    F = (A + 1 / 2 * DELTAt * B) @ eta[:, k]
            else:
                F = (A + 1 / 2 * DELTAt * B) @ eta[:, k] + DELTAt * C @ f[:, k]

            eta[:, k + 1] = sp.sparse.linalg.spsolve(A - 1 / 2 * DELTAt * B, F)
            print(str(k + 1) + ' | t = ' + f"{time_vector[k]:.2e}" + ' [s] | ' + f"{max(eta[:, k + 1]):.2e}")

    # Return only displacements u (second half of the state)
    eta = eta[K.shape[0]:, :]

    return eta, stamps


def harmonic_response_2(K, D, M, f, freqs, sparse=False, dtype=complex, solver='spsolve'):
    """
    PURPOSE:
        Compute the harmonic (steady-state) response of a linear dynamic system
        at a set of excitation frequencies by solving:

            (K + j*omega*D - omega^2*M) * eta = f

        for each omega = 2*pi*freq.

    INPUTS:
        K, D, M (scipy.sparse matrix | np.ndarray):
            Stiffness, damping, mass matrices (square, same size).

        f (np.ndarray):
            Force vector(s):
                - shape (n_dof,) for same RHS at all frequencies
                - shape (n_dof, n_freq) for frequency-dependent forcing

        freqs (np.ndarray):
            Frequencies in Hz.

        sparse (bool, optional):
            If True, uses sparse solvers; if False, uses dense np.linalg.solve.
            Default = False.

        dtype (type, optional):
            complex for damped/complex response, float for undamped real solve.
            Default = complex.

        solver (str, optional):
            Sparse solver backend:
                - 'spsolve'
                - 'pypardiso'
                - 'ansys'
                - 'mkl_pardiso'
            Default = 'spsolve'.

    OUTPUTS:
        eta (np.ndarray):
            Frequency response, shape (n_dof, n_freq).

    SIDE EFFECTS:
        - Prints per-frequency progress and timing.
        - If using ANSYS / Pardiso, interacts with external solver state.

    NOTES:
        - For complex solves with solvers that only accept real systems, the code
          forms an equivalent real block system:
              [Re(Z) -Im(Z); Im(Z) Re(Z)] [Re(x); Im(x)] = [Re(f); Im(f)]
        - Timing prints are kept exactly in spirit of your original code.

    EXAMPLES:
        >>> eta = harmonic_response_2(K, D, M, f, freqs, sparse=True, solver='mkl_pardiso')
    """

    omegas = 2 * np.pi * freqs  # angular frequency vector
    eta = np.zeros((K.shape[1], freqs.shape[0]), dtype=dtype)

    t0 = time.time()
    t1 = time.time()
    t_check = 5  # seconds between remaining-time checks

    for i in range(0, len(freqs)):
        t2 = time.time()
        print(f"\nSolving at f = {freqs[i].round(2)} Hz. ", end='', flush=True)

        # Build dynamic stiffness Z(omega)
        if dtype == complex:
            Z = K + 1j * omegas[i] * D - (omegas[i])**2 * M
            is_complex = True
        elif dtype == float:
            Z = K - (omegas[i])**2 * M
            is_complex = False

        if sparse:
            # --- Sparse solver path ---
            if solver == 'spsolve':
                if len(f.shape) == 1:
                    eta[:, i] = sp.sparse.linalg.spsolve(Z, f)
                else:
                    eta[:, i] = sp.sparse.linalg.spsolve(Z, f[:, i])

            elif solver == 'pypardiso':
                # pypardiso is real-only -> convert complex system to real block form if needed
                if is_complex:
                    Z_p = sp.sparse.bmat([[Z.real, -Z.imag],
                                          [Z.imag,  Z.real]], format='csc')

                    if len(f.shape) == 1:
                        f_p = np.concatenate([f.real, f.imag])
                        eta_p = pp.spsolve(Z_p, f_p)
                    else:
                        f_p = np.vstack((f.real, f.imag))
                        eta_p = pp.spsolve(Z_p, f_p[:, i])

                    eta[:, i] = eta_p[:f.real.shape[0]] + 1j * eta_p[f.real.shape[0]:]
                else:
                    if len(f.shape) == 1:
                        eta[:, i] = pp.spsolve(Z.real, f.real)
                    else:
                        eta[:, i] = pp.spsolve(Z.real, f.real[:, i])

            elif solver == 'ansys':
                # External ANSYS solver through AnsMath
                mm = get_AnsMathObject()

                if is_complex:
                    A = mm.matrix(Z.astype(complex))
                    solver_obj = mm.factorize(A)
                    b = mm.set_vec(f.astype(complex)) if len(f.shape) == 1 else mm.set_vec(f[:, i].astype(complex))
                    x = solver_obj.solve(b)
                    eta[:, i] = x.asarray(dtype=np.complex128)
                else:
                    A = mm.matrix(Z.real)
                    solver_obj = mm.factorize(A, algo='DSP')
                    b = mm.set_vec(f.real) if len(f.shape) == 1 else mm.set_vec(f[:, i].real)
                    x = solver_obj.solve(b)
                    eta[:, i] = x.asarray(dtype=np.complex128)

                    # Free ANSYS temps
                    mm.free(b, x)
                    mm.free(solver_obj, A)

            elif solver == 'mkl_pardiso':
                # MKL Pardiso can solve real or complex systems depending on mtype
                if is_complex:
                    mtype = 13  # complex unsymmetric
                    pt, iparm = pardisoinit(mtype=mtype)

                    if len(f.shape) == 1:
                        eta[:, i] = pardiso(Z.tocsr(), f.astype(complex), pt, mtype, iparm)[0].ravel()
                    else:
                        eta[:, i] = pardiso(Z.tocsr(), f[:, i].astype(complex), pt, mtype, iparm)[0].ravel()
                else:
                    mtype = 11  # real unsymmetric
                    pt, iparm = pardisoinit(mtype=mtype)

                    if len(f.shape) == 1:
                        eta[:, i] = pardiso(Z.real.tocsr(), f.real.astype(float), pt, mtype, iparm)[0].ravel()
                    else:
                        eta[:, i] = pardiso(Z.real.tocsr(), f[:, i].real.astype(float), pt, mtype, iparm)[0].ravel()

                # Release internal memory
                pardiso(Z.tocsr(), f, pt, mtype, iparm, phase=-1)

        else:
            # --- Dense solver path ---
            if len(f.shape) == 1:
                sol = np.linalg.solve(Z, f)
                eta[:, i] = sol.real if dtype == float else sol
            else:
                sol = np.linalg.solve(Z, f[:, i])
                if len(sol.shape) != 1:
                    eta[:, i] = sol[:, 0].real if dtype == float else sol[:, 0]
                else:
                    eta[:, i] = sol.real if dtype == float else sol

        # Print per-frequency elapsed time
        elapsed_time = time.time() - t2
        elapsed_hours = elapsed_time // 3600
        rem = elapsed_time - 3600 * elapsed_hours
        elapsed_minutes = rem // 60
        elapsed_seconds = rem - 60 * elapsed_minutes
        print(f"Solved in {elapsed_hours:02.0f}:{elapsed_minutes:02.0f}:{elapsed_seconds:04.1f} (hh:mm:ss.). ")

        # Print remaining time every t_check seconds
        if time.time() - t1 >= t_check:
            el_time = time.time() - t0
            el_op = i + 1
            rem_op = len(freqs) - i - 1
            remaining_time(el_time, el_op, rem_op)
            t1 = time.time()

    return eta


def get_AnsMathObject():
    """
    PURPOSE:
        Return a live (cached) AnsMath session object, reusing an existing one if possible.
        If the cached object is missing or invalid, creates a new AnsMath session.

    INPUTS:
        None

    OUTPUTS:
        mm (pymath.AnsMath):
            A valid AnsMath session object.

    SIDE EFFECTS:
        - Reads/writes global variable _mm.
        - Creates new AnsMath sessions if needed.

    NOTES:
        - Calls _mm.status() as a liveness ping; if it fails, the session is recreated.

    EXAMPLES:
        >>> mm = get_AnsMathObject()
        >>> mm.status()
    """
    global _mm

    # If no session exists, create one.
    if _mm is None:
        _mm = pymath.AnsMath()
        return _mm

    # If session exists, verify it is still alive.
    try:
        _mm.status()
    except Exception:
        # Recreate on failure
        _mm = pymath.AnsMath()

    return _mm


#%% Operations on the nodes and elements
def mapping_matrix(dofref):
    """
    PURPOSE:
        Build a sparse mapping (permutation) matrix that reorders DOFs according to `dofref`.

        The intent is: given a reference table `dofref` describing DOFs, create a matrix
        that maps a vector/matrix stored in one ordering ("internal") to a sorted ordering
        ("user"), or vice versa depending on how it's applied. This is tailored for the
        dofref vector from PyANSYS.

    INPUTS:
        dofref (np.ndarray):
            Array of shape (N, >=2). The first two columns define a sorting key
            (typically something like [node_id, dof_id] or [entity_id, component_id]).

    OUTPUTS:
        mapping (scipy.sparse.csc_matrix):
            Sparse permutation matrix of shape (N, N).
            Multiplying `mapping @ x` reorders `x` according to the lexicographic
            sorting of dofref by column 1 then column 0 (as implemented).

    SIDE EFFECTS:
        None

    NOTES:
        - np.lexsort uses the *last key as primary*. With keys (dofref[:,0], dofref[:,1]),
          the primary sort is dofref[:,1], secondary is dofref[:,0].
        - The matrix is built by reordering rows of the identity matrix.

    EXAMPLES:
        >>> P = mapping_matrix(dofref)
        >>> x_sorted = P @ x
    """

    # Sort indices using lexicographic ordering:
    # IMPORTANT: np.lexsort((secondary, primary)) => primary key is last.
    # Here primary = dofref[:,1], secondary = dofref[:,0].
    sorted_indices = np.lexsort((dofref[:, 0], dofref[:, 1]))

    # Start with identity: each row picks one element.
    mapping = sp.sparse.eye(sorted_indices.shape[0], format='csc')

    # Reorder rows to encode the permutation.
    # If x is an (N,) vector, mapping @ x returns x in the sorted order.
    mapping = mapping[sorted_indices, :]

    return mapping

def find_connected_nodes_by_element(mesh, node_ids):
    """
    PURPOSE:
        Given a mesh object and a set of node IDs, find mesh elements that include
        those nodes and group the node IDs by element.

        The function returns a list of [element, node_ids_in_that_element] pairs,
        restricted to elements that contain at least 3 of the queried nodes
        (enough nodes to define a face/patch in many element types).

    INPUTS:
        mesh (object):
            Mesh-like object expected to provide:
              - mesh.nodes.node_by_id(node_id).nodal_connectivity  (elements connected to node)
              - mesh.elements[el_id] returning an element object
              - element.nodes list of node objects with .id

        node_ids (array-like):
            List/array of node IDs to analyze.

    OUTPUTS:
        element_list (list):
            List of entries: [element, node_ids_in_element]
            where node_ids_in_element are node IDs from `node_ids` that belong to that element.
            Only elements with >= 3 selected nodes are included.

        node_ids (array-like):
            Returns the same `node_ids` input (unchanged). (Kept for API consistency.)

    SIDE EFFECTS:
        None

    NOTES:
        - The function first collects all elements incident to the nodes (`nodal_connectivity`),
          then filters by membership.
        - Membership test `x in node_ids` can be slow if node_ids is a list. If you ever
          optimize, convert to a set outside. (Not changed here.)

    EXAMPLES:
        >>> element_list, node_ids = find_connected_nodes_by_element(mesh, node_ids=[1,2,3,4])
    """

    element_ids = []

    # 1) For each node, gather all connected elements
    for node_id in node_ids:
        # nodal_connectivity is assumed to be a list of element IDs incident to this node
        for el_id in mesh.nodes.node_by_id(node_id).nodal_connectivity:
            element_ids.append(el_id)

    # 2) Unique element IDs (elements touched by at least one selected node)
    element_ids = np.unique(np.array(element_ids))

    # 3) For each element, find which of its nodes are in node_ids
    element_list = []
    for el_id in element_ids:
        element = mesh.elements[el_id]
        element_nodes = element.nodes

        # Collect node IDs of this element
        element_node_ids = np.array([element_nodes[i].id for i in range(0, len(element_nodes))])

        # Filter to those node IDs that are part of the user selection
        node_ids_in_element = np.array(
            [element_node_id for element_node_id in element_node_ids if element_node_id in node_ids]
        )

        # Keep only if at least 3 nodes are present (face definition, etc.)
        if len(node_ids_in_element) >= 3:
            element_list.append([element, node_ids_in_element])

    return element_list, node_ids



def normal_of_element_face(element, node_ids):
    """
    PURPOSE:
        Compute an outward normal vector and area for a face of a finite element,
        where the face is specified by a list of node IDs belonging to the element.

        Supported element types (by node count):
          - HEX8  : 8-node hexahedron
          - WEDGE6: 6-node wedge/prism
          - Shell quad: 4-node surface element

    INPUTS:
        element (object):
            Element-like object expected to have:
              - element.nodes: list of node objects
              - node.id, node.coordinates

        node_ids (array-like):
            Node IDs defining the face. Ordering is not assumed; the function reorders
            based on the element’s internal connectivity.

    OUTPUTS:
        normal (np.ndarray):
            Normal vector (3,), intended to be unit-length for solid faces.
            For shell elements, normalization is not enforced in the provided code.

        area (float):
            Face area (units depend on mesh coordinate units).

    SIDE EFFECTS:
        None

    NOTES:
        - The sign of the normal is determined via hard-coded connectivity lookup tables
          (positive_normal / negative_normal). If the connectivity pattern is not found,
          the function returns a zero normal.
        - Face area is computed by splitting quad faces into two triangles and summing.
        - For unsupported element types, returns normal=[0,0,0], area=0.

    EXAMPLES:
        >>> n, a = normal_of_element_face(element, node_ids=[10,11,12,13])
    """

    # -------------------------
    # HEX8: 8-node hexahedron
    # -------------------------
    if len(element.nodes) == 8:
        # Extract node IDs and coordinates in element's native ordering (1..8)
        element_nodes = [element.nodes[i] for i in range(0, 8)]
        element_node_ids = np.array([element_nodes[i].id for i in range(0, 8)])
        element_nodes_coords = np.array([element_nodes[i].coordinates for i in range(0, 8)])

        # Connectivity indices are 1..8 (ANSYS-style numbering)
        node_connectivity = np.arange(1, 9, 1)

        # Map the provided node_ids to their connectivity numbers within the element.
        # For each requested node_id, find its position in element_node_ids, then read
        # the corresponding connectivity number (1..8).
        node_con = np.array([node_connectivity[element_node_ids == node_id][0] for node_id in node_ids])

        # Sort by connectivity so we have a consistent local face ordering
        order_node_con = np.argsort(node_con)

        node_con = node_con[order_node_con]
        node_ids = np.array(node_ids)[order_node_con]

        # Reorder coordinates to match the face-node connectivity ordering
        element_nodes_coords = element_nodes_coords[node_con - 1, :]

        # conn_id: use first 3 connectivity numbers to decide outward sign
        conn_id = ''.join(node_con[:3].astype(str))

        # Lookup tables: if conn_id is in positive_normal -> keep normal,
        # if in negative_normal -> flip normal, else -> zero normal.
        positive_normal = ['125', '126', '132', '142', '143', '154', '158', '165', '184', 
                           '213', '214', '236', '237', '243', '251', '261', '265', '276', 
                           '314', '321', '324', '347', '348', '362', '372', '376', '387', 
                           '415', '418', '421', '431', '432', '458', '473', '483', '487', 
                           '512', '516', '526', '541', '567', '568', '578', '581', '584', 
                           '612', '623', '627', '637', '651', '652', '675', '678', '685', 
                           '723', '734', '738', '748', '756', '762', '763', '785', '786', 
                           '815', '834', '841', '845', '856', '857', '867', '873', '874']
        
        negative_normal = ['123', '124', '134', '145', '148', '152', '156', '162', '185',
                           '215', '216', '231', '234', '241', '256', '263', '267', '273',
                           '312', '326', '327', '341', '342', '367', '374', '378', '384',
                           '412', '413', '423', '437', '438', '451', '478', '481', '485',
                           '514', '518', '521', '548', '561', '562', '576', '586', '587',
                           '615', '621', '625', '632', '657', '658', '672', '673', '687',
                           '726', '732', '736', '743', '758', '765', '768', '783', '784', 
                           '814', '837', '843', '847', '851', '854', '865', '875', '876']
        

        # Compute a candidate normal using two edge vectors from first three points
        v1 = element_nodes_coords[1, :] - element_nodes_coords[0, :]
        v2 = element_nodes_coords[2, :] - element_nodes_coords[0, :]

        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        normal = normal.round(6)

        # Enforce outward orientation via lookup tables
        if conn_id in positive_normal:
            normal = 1.0 * normal
        elif conn_id in negative_normal:
            normal = -1.0 * normal
        else:
            normal = 0.0 * normal

        # ---- Area computation (face-type dependent) ----
        area = 0.0
        con_id_area = ''.join(node_con.astype(str))

        # Faces for HEX8 can be quads; these groups decide how to split into triangles
        area_type_1 = ['1234', '5678']
        area_type_2 = ['2367', '1256', '1458', '3478']

        if con_id_area in area_type_1:
            # Split quad (1,2,3,4) into triangles (1,2,4) and (3,2,4) using cross products.
            v1 = element_nodes_coords[1, :] - element_nodes_coords[0, :]
            v2 = element_nodes_coords[3, :] - element_nodes_coords[0, :]
            v3 = element_nodes_coords[1, :] - element_nodes_coords[2, :]
            v4 = element_nodes_coords[3, :] - element_nodes_coords[2, :]

            area = 0.5 * (np.linalg.norm(np.cross(v2, v1)) + np.linalg.norm(np.cross(v4, v3)))

        elif con_id_area in area_type_2:
            # Another quad layout: use triangles (1,2,3) and (4,2,3) in local arrangement.
            v1 = element_nodes_coords[1, :] - element_nodes_coords[0, :]
            v2 = element_nodes_coords[2, :] - element_nodes_coords[0, :]
            v3 = element_nodes_coords[1, :] - element_nodes_coords[3, :]
            v4 = element_nodes_coords[2, :] - element_nodes_coords[3, :]

            area = 0.5 * (np.linalg.norm(np.cross(v2, v1)) + np.linalg.norm(np.cross(v4, v3)))

    # -------------------------
    # WEDGE6: 6-node wedge/prism
    # -------------------------
    elif len(element.nodes) == 6:
        element_nodes = [element.nodes[i] for i in range(0, 6)]
        element_node_ids = np.array([element_nodes[i].id for i in range(0, 6)])
        element_nodes_coords = np.array([element_nodes[i].coordinates for i in range(0, 6)])

        node_connectivity = np.arange(1, 7, 1)

        node_con = np.array([node_connectivity[element_node_ids == node_id][0] for node_id in node_ids])
        order_node_con = np.argsort(node_con)

        node_con = node_con[order_node_con]
        node_ids = np.array(node_ids)[order_node_con]
        element_nodes_coords = element_nodes_coords[node_con - 1, :]

        conn_id = ''.join(node_con[:3].astype(str))

        positive_normal = ['132', '213', '321', '456', '564', '645', '236', '362', '623', '235', '352', '523', '265', '652',
                           '526', '365', '653', '536', '125', '251', '512', '124', '241', '412', '154', '541', '415', '254',
                           '542', '425', '163', '316', '631', '143', '314', '431', '146', '614', '461', '346', '634', '463']
        
        negative_normal = ['123', '231', '312', '465', '546', '654', '263', '326', '632', '253', '325', '532', '256', '625', 
                           '562', '356', '635', '563', '152', '215', '521', '142', '214', '421', '145', '514', '451', '245',
                           '524', '452', '136', '361', '613', '134', '341', '413', '164', '641', '416', '364', '643', '436']

        v1 = element_nodes_coords[1, :] - element_nodes_coords[0, :]
        v2 = element_nodes_coords[2, :] - element_nodes_coords[0, :]

        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        normal = normal.round(6)

        if conn_id in positive_normal:
            normal = 1.0 * normal
        elif conn_id in negative_normal:
            normal = -1.0 * normal
        else:
            normal = 0.0 * normal

        area = 0.0
        con_id_area = ''.join(node_con.astype(str))

        # WEDGE has triangular faces (3 nodes) and quad faces (4 nodes)
        area_type_1 = ['123', '456']          # triangular faces
        area_type_2 = ['2356', '1245', '1346']  # quad faces

        if con_id_area in area_type_1:
            v1 = element_nodes_coords[1, :] - element_nodes_coords[0, :]
            v2 = element_nodes_coords[2, :] - element_nodes_coords[0, :]
            area = 0.5 * np.linalg.norm(np.cross(v2, v1))

        elif con_id_area in area_type_2:
            v1 = element_nodes_coords[1, :] - element_nodes_coords[0, :]
            v2 = element_nodes_coords[2, :] - element_nodes_coords[0, :]
            v3 = element_nodes_coords[1, :] - element_nodes_coords[3, :]
            v4 = element_nodes_coords[2, :] - element_nodes_coords[3, :]
            area = 0.5 * (np.linalg.norm(np.cross(v2, v1)) + np.linalg.norm(np.cross(v4, v3)))

    # -------------------------
    # Shell quad: 4-node surface element
    # -------------------------
    elif len(element.nodes) == 4:
        element_nodes = [element.nodes[i] for i in range(0, 4)]
        element_nodes_coords = np.array([element_nodes[i].coordinates for i in range(0, 4)])

        # Split quad into two triangles and sum their areas
        v1 = element_nodes_coords[1, :] - element_nodes_coords[0, :]
        v2 = element_nodes_coords[3, :] - element_nodes_coords[0, :]
        v3 = element_nodes_coords[1, :] - element_nodes_coords[2, :]
        v4 = element_nodes_coords[3, :] - element_nodes_coords[2, :]

        area = 0.5 * (np.linalg.norm(np.cross(v2, v1)) + np.linalg.norm(np.cross(v4, v3)))

        # Normal: here the code uses a particular cross product choice.
        # (Not normalized; could be scaled by area.)
        normal = np.cross(-v3, -v1)

    else:
        normal = np.array([0.0, 0.0, 0.0])
        area = 0.0

    return normal, area

def nodal_normal(mesh, node_list_ids=[], filt_node_ids=[]):
    """
    PURPOSE:
        Compute nodal normal vectors (and representative areas) for a set of nodes,
        by accumulating face normals of connected element faces.

        The algorithm:
          1) Find elements that contain groups of selected nodes (>=3 per element).
          2) For each element-face group, compute face normal and area.
          3) Accumulate (normal * area) per node.
          4) Normalize the accumulated normals per node.
          5) Output arrays aligned with node_list_ids or filt_node_ids.

    INPUTS:
        mesh (object):
            Mesh object used by find_connected_nodes_by_element(...) and normal_of_element_face(...).

        node_list_ids (list, optional):
            Node IDs for which to compute normals. If empty, behavior falls back to using
            find_connected_nodes_by_element(mesh, node_list_ids) (which will return nothing).
            Default = [].

        filt_node_ids (list, optional):
            If provided, output normals/areas aligned to this list; nodes missing from the
            computed dictionary get zeros.
            Default = [].

    OUTPUTS:
        normals (np.ndarray):
            Array of shape (N, 3) containing unit normals per node (or zero).

        areas (np.ndarray):
            Array of shape (N,) containing average face area contribution per node (or zero).

    SIDE EFFECTS:
        None

    NOTES:
        - Uses area-weighted averaging: sum(normal*area) then normalize.
        - The averaged `area` returned is the mean area over contributions (area_sum/count).

    EXAMPLES:
        >>> normals, areas = nodal_normal(mesh, node_list_ids=[1,2,3,4])
    """

    # If node_list_ids is provided, use it directly; otherwise this function still calls
    # find_connected_nodes_by_element with an empty list (likely returning empty).
    if len(node_list_ids) != 0:
        element_list, _ = find_connected_nodes_by_element(mesh, node_list_ids)
    else:
        element_list, node_list_ids = find_connected_nodes_by_element(mesh, node_list_ids)

    node_normal_dict = {}

    # Accumulate area-weighted normals for each node
    for i in range(0, len(element_list)):
        element = element_list[i][0]
        node_ids = element_list[i][1]

        normal, area = normal_of_element_face(element, node_ids)

        for node_id in node_ids:
            if node_id not in node_normal_dict.keys():
                node_normal_dict[node_id] = {'normal': normal * area, 'area': area, 'count': 1}
            else:
                node_normal_dict[node_id]['normal'] += normal * area
                node_normal_dict[node_id]['area'] += area
                node_normal_dict[node_id]['count'] += 1

    # Normalize each node normal and average the area
    for node_id in node_normal_dict.keys():
        if np.linalg.norm(node_normal_dict[node_id]['normal']) != 0:
            node_normal_dict[node_id]['normal'] /= np.linalg.norm(node_normal_dict[node_id]['normal'])
            node_normal_dict[node_id]['area'] /= (node_normal_dict[node_id]['count'])
        else:
            node_normal_dict[node_id]['normal'] *= 0.0
            node_normal_dict[node_id]['area'] *= 0.0

    empty_vector = np.array([0.0, 0.0, 0.0])
    empty_area = 0.0

    # Build output arrays aligned with requested node ordering
    if len(filt_node_ids) == 0:
        normals = np.array([node_normal_dict[node_id]['normal'] for node_id in node_list_ids])
        areas = np.array([node_normal_dict[node_id]['area'] for node_id in node_list_ids])
    else:
        normals = np.array([
            node_normal_dict[node_id]['normal'] if node_id in node_normal_dict.keys() else empty_vector
            for node_id in filt_node_ids
        ])
        areas = np.array([
            node_normal_dict[node_id]['area'] if node_id in node_normal_dict.keys() else empty_area
            for node_id in filt_node_ids
        ])

    return normals, areas


def normal_matrix(normals, ndim=3):
    """
    PURPOSE:
        Convert a set of normal vectors into a sparse block matrix N_mat that can be
        used to apply component-wise normal weighting to stacked DOF vectors.

        For example, if you have a displacement vector ordered as:
            [u_x; u_y; u_z] (stacked by component),
        then N_mat can act as a matrix that multiplies each component by the
        corresponding normal component per node.

    INPUTS:
        normals (np.ndarray):
            Array of shape (N, 3) typically. Each row is a normal vector at a node.

        ndim (int, optional):
            Number of components/DOFs per node to build blocks for.
            Default = 3.

    OUTPUTS:
        N_mat (scipy.sparse.csc_matrix):
            Sparse block matrix of shape (ndim*N, N)?? (depends on bmat layout).
            As implemented, it builds a vertical block of diagonal matrices:
                [[diag(nx)],
                 [diag(ny)],
                 [diag(nz)]]
            which results in shape (ndim*N, N).

    SIDE EFFECTS:
        None

    NOTES:
        - If ndim > normals.shape[1], missing components are filled with zero blocks.
        - This is a "stacked-components" operator (vertical stacking), not a full (ndim*N, ndim*N)
          block-diagonal matrix.

    EXAMPLES:
        >>> N = normal_matrix(normals, ndim=3)
        >>> normal_component = N.T @ u_stacked   # depending on your vector conventions
    """

    block = []

    # For each component i, create a diagonal matrix with normals[:, i]
    for i in range(0, ndim):
        if i < normals.shape[1]:
            block.append([sp.sparse.diags(normals[:, i])])
        else:
            # If no such component exists, append a zero block of matching size
            block.append([sp.sparse.csc_matrix((normals.shape[0], normals.shape[0]))])

    # Stack these diagonal blocks vertically
    N_mat = sp.sparse.bmat(block, format='csc')

    return N_mat

def cluster_indices(coords, radius=None, n_clusters=None, centers=None):
    """
    PURPOSE:
        Cluster a set of coordinate points either:
          - by a distance threshold `radius` using agglomerative clustering, or
          - by user-provided initial `centers` using KMeans.

        The function outputs:
          - indices belonging to each cluster,
          - the centroid of each cluster,
          - the index of the point closest to each centroid.

    INPUTS:
        coords (np.ndarray):
            Array of shape (N, d) containing N points in d dimensions.

        radius (float, optional):
            If provided, use AgglomerativeClustering with distance_threshold=radius
            and n_clusters=None (clusters determined automatically).

        n_clusters (int, optional):
            Number of clusters. Used by KMeans (when centers are given) and potentially
            by Agglomerative if you choose to extend logic. (In current code, radius path
            forces n_clusters=None).

        centers (np.ndarray, optional):
            If provided, use KMeans with init=centers and n_init=1.

    OUTPUTS:
        clusters (list):
            List of clusters, each cluster is a list of point indices.

        centroids (np.ndarray):
            Array of shape (n_clusters_found, d), mean coordinate of each cluster.

        index_centroids (list):
            List of indices (one per cluster): the point index closest to the centroid.

    SIDE EFFECTS:
        None

    NOTES:
        - In the radius case, clusters are produced by complete-linkage hierarchical clustering.
        - In the centers case, KMeans is run once with the provided initial centers.

    EXAMPLES:
        >>> clusters, centroids, idx = cluster_indices(coords, radius=0.1)
        >>> clusters, centroids, idx = cluster_indices(coords, centers=init_centers)
    """

    if radius is not None:
        # Case 1: hierarchical clustering where clusters are merged until max distance <= radius.
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=radius,
            linkage="complete",
            metric="euclidean"
        )

    elif centers is not None:
        # Case 2: KMeans with fixed initialization centers (no random restarts)
        n_clusters = centers.shape[0]
        model = KMeans(
            n_clusters=n_clusters,
            init=centers,
            n_init=1,
            random_state=0
        )

    # Fit and extract labels for each point
    labels = model.fit(coords).labels_

    # Group indices by label
    clusters = []
    for lbl in np.unique(labels):
        if lbl == -1:
            # Noise label would appear in DBSCAN-type models, not expected here,
            # but kept as defensive code.
            continue
        clusters.append(list(np.where(labels == lbl)[0]))

    # Compute centroid per cluster (mean of coordinates)
    centroids = np.vstack([coords[cluster].mean(axis=0) for cluster in clusters])

    # Find the representative index: point closest to the centroid
    index_centroids = [
        cluster[np.argmin(np.linalg.norm(coords[cluster] - centroids[i], axis=1))]
        for i, cluster in enumerate(clusters)
    ]

    return clusters, centroids, index_centroids


def rotation_matrix(axis, theta):
    """
    PURPOSE:
        Compute the 3x3 rotation matrix that rotates vectors by angle `theta`
        (radians) around a given rotation axis using Rodrigues' rotation formula.

    INPUTS:
        axis (array-like):
            3-element vector defining the rotation axis. Will be normalized.
            If axis is zero, function prints a warning and proceeds (matrix may contain NaNs).

        theta (float):
            Rotation angle in radians.

    OUTPUTS:
        R (np.ndarray):
            3x3 rotation matrix.

    SIDE EFFECTS:
        - Prints a warning if axis has zero norm.

    NOTES:
        - Uses Rodrigues' formula:
            R = I*cos(theta) + (1-cos(theta))*(a a^T) + sin(theta)*[a]_x
          where a is the unit axis and [a]_x is the cross-product matrix.

    EXAMPLES:
        >>> R = rotation_matrix([0,0,1], np.pi/2)
    """

    axis = np.asarray(axis, dtype=float)

    # Guard against zero axis; code prints warning but still continues.
    if np.linalg.norm(axis) == 0:
        print("Warning: Axis vector cannot be zero.")
    else:
        axis = axis / np.linalg.norm(axis)

    n_x, n_y, n_z = axis

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    one_minus_cos = 1 - cos_theta

    # Rodrigues rotation matrix expanded component-wise
    R = np.array([
        [n_x**2 * one_minus_cos + cos_theta,
         n_x * n_y * one_minus_cos - n_z * sin_theta,
         n_x * n_z * one_minus_cos + n_y * sin_theta],

        [n_x * n_y * one_minus_cos + n_z * sin_theta,
         n_y**2 * one_minus_cos + cos_theta,
         n_y * n_z * one_minus_cos - n_x * sin_theta],

        [n_x * n_z * one_minus_cos - n_y * sin_theta,
         n_y * n_z * one_minus_cos + n_x * sin_theta,
         n_z**2 * one_minus_cos + cos_theta]
    ])

    return R

def modes_flat_to_3d(modes_flat, ndim=3, dtype=float):
    """
    PURPOSE:
        Convert mode shapes stored in a flat (stacked-components) format into a 3D array:

            modes_3d[node_index, mode_index, component_index]

        Assumes `modes_flat` is ordered by components in blocks:
            [comp0(all nodes), comp1(all nodes), ..., comp(ndim-1)(all nodes)]

    INPUTS:
        modes_flat (np.ndarray):
            Either:
              - shape (ndim*n_nodes, n_modes) for multiple modes, or
              - shape (ndim*n_nodes,) for a single mode vector.

        ndim (int, optional):
            Number of components per node (e.g., 3 for x,y,z). Default = 3.

        dtype (type, optional):
            Output dtype. Default = float.

    OUTPUTS:
        modes_3d (np.ndarray):
            Array of shape (n_nodes, n_modes, ndim).

    SIDE EFFECTS:
        None

    NOTES:
        - n_nodes is inferred as modes_flat.shape[0] // ndim.
        - If the input is interleaved per node (x1,y1,z1,x2,y2,z2,...) this function
          will NOT interpret it correctly.

    EXAMPLES:
        >>> Phi_3d = modes_flat_to_3d(Phi, ndim=3)
    """

    n_rows = modes_flat.shape[0] // ndim

    # Case: multiple modes matrix (2D)
    if len(modes_flat.shape) != 1:
        modes_3d = np.zeros((n_rows, modes_flat.shape[1], ndim), dtype=dtype)

        for i in range(0, modes_flat.shape[1]):
            # For each component j, slice its block (j*n_rows:(j+1)*n_rows)
            # Then stack components into shape (n_rows, ndim)
            modes_3d[:, i, :] = np.array(
                [modes_flat[j * n_rows:(j + 1) * n_rows, i] for j in range(0, ndim)]
            ).T

    # Case: single mode vector (1D)
    else:
        modes_3d = np.zeros((n_rows, 1, ndim), dtype=dtype)
        modes_3d[:, 0, :] = np.array(
            [modes_flat[j * n_rows:(j + 1) * n_rows] for j in range(0, ndim)]
        ).T

    return modes_3d


def modes_flat_to_3d_2(modes_flat, ndim=3, dtype=float, shapes=[]):
    """
    PURPOSE:
        Split a flat modes matrix into multiple sub-blocks (given by `shapes`),
        reshape each block with modes_flat_to_3d(...), then vertically stack them.

        This is useful when different substructures or meshes are concatenated
        into a single DOF vector and you want per-substructure reshaping.

    INPUTS:
        modes_flat (np.ndarray):
            Flat modes array, typically (sum(shapes), n_modes).

        ndim (int, optional):
            Components per node for each sub-block. Default = 3.

        dtype (type, optional):
            Output dtype. Default = float.

        shapes (list, optional):
            List of integer lengths describing how to split rows of modes_flat.
            Default = [].

    OUTPUTS:
        modes (list):
            List of modes sub-block arrays, each of shape (shapes[i], n_modes).

        modes_3d (np.ndarray):
            Vertically stacked 3D modes from each block.

    SIDE EFFECTS:
        None

    NOTES:
        - The function assumes each block uses the same `ndim`.
        - The output modes_3d is stacked along the node dimension.

    EXAMPLES:
        >>> modes_blocks, Phi_3d = modes_flat_to_3d_2(Phi, ndim=3, shapes=[n1, n2])
    """

    # Build cumulative split indices
    pos_1 = [0]
    for i in range(0, len(shapes)):
        pos_1.append(pos_1[i] + shapes[i])

    # Slice each block from the flat array
    modes = [modes_flat[pos_1[i]:pos_1[i + 1], :] for i in range(0, len(pos_1) - 1)]

    # Reshape each block and vertically stack
    modes_3d = np.vstack([modes_flat_to_3d(modes[i], ndim=ndim) for i in range(0, len(modes))])

    return modes, modes_3d

def modes_flat_to_3d_3(modes_flat, ndim=[3], dtype=float, shapes=[]):
    """
    PURPOSE:
        Split a flat modes array into multiple blocks (defined by `shapes`), where each block
        may have its own `ndim` (components per node). Each block is reshaped into 3D form.
        Blocks with smaller component dimension are padded with zeros so all blocks can be
        vertically stacked.

    INPUTS:
        modes_flat (np.ndarray):
            Flat modes array, typically 1D or 2D, whose rows represent DOFs.

        ndim (list, optional):
            List of ndims per block, same length as `shapes`.
            Default = [3].

        dtype (type, optional):
            Output dtype. Default = float.

        shapes (list, optional):
            List of integer sizes defining where to split modes_flat along rows.

    OUTPUTS:
        modes_3d (np.ndarray):
            Single stacked 3D array with unified last-dimension size equal to max(ndim).

    SIDE EFFECTS:
        None

    NOTES:
        - Pads with zeros in the last dimension for blocks with fewer components.
        - Useful for vibro-acoustic concatenations.

    EXAMPLES:
        >>> Phi_3d = modes_flat_to_3d_3(Phi, ndim=[3,1], shapes=[n_struct, n_acoustic])
    """

    pos_1 = [0]
    for i in range(0, len(shapes)):
        pos_1.append(pos_1[i] + shapes[i])

    # Reshape each block separately with its own ndim
    modes_3d_list = [
        modes_flat_to_3d(modes_flat[pos_1[i]:pos_1[i + 1]], ndim=ndim[i], dtype=dtype)
        for i in range(0, len(shapes))
    ]

    # Determine the maximum component dimension among blocks
    modes_3d_shapes = [modes_3d.shape[2] for modes_3d in modes_3d_list]
    max_3d_shape = max(modes_3d_shapes)

    # Pad blocks with fewer components so all have shape (..., max_3d_shape)
    modes_3d = []
    for i in range(0, len(modes_3d_list)):
        dif_dim = max_3d_shape - modes_3d_shapes[i]

        if dif_dim == 0:
            modes_3d.append(modes_3d_list[i])
        else:
            ap_mode_sh = list(modes_3d_list[i].shape)
            ap_mode_sh[2] = dif_dim

            reshaped_mode = np.dstack((modes_3d_list[i], np.zeros(ap_mode_sh)))
            modes_3d.append(reshaped_mode)

    # Stack all blocks along node dimension
    modes_3d = np.vstack(modes_3d)

    return modes_3d

def assembly_nodal_connectivity(component_matrices, level_comp):
    """
    PURPOSE:
        Build the assembly-level nodal connectivity by concatenating component connectivities,
        offsetting node indices so that each component's nodes occupy a distinct index range.

        This is typically needed when creating a combined mesh representation from multiple
        subcomponents (e.g., for visualization or global assembly operations).

    INPUTS:
        component_matrices (dict):
            Expected keys:
              - 'connectivity': dict[comp] -> (cells_array, cell_types)
              - 'coords'      : dict[comp] -> points array of shape (n_nodes_comp, dim)

        level_comp (list):
            Ordered list of component identifiers to assemble.

    OUTPUTS:
        nodal_connectivity (tuple):
            (cells_array, cell_types_array) where:
              - cells_array is the concatenated VTK-style connectivity array with corrected offsets
              - cell_types_array is concatenated cell type array

    SIDE EFFECTS:
        None

    NOTES:
        - Offsets are applied only to the node indices, not to the per-cell node-count entries.
        - The code assumes the VTK "cells array" format: [n, i0, i1, ... , n, j0, ...].

    EXAMPLES:
        >>> conn = assembly_nodal_connectivity(component_matrices, level_comp=['A','B'])
    """

    el_connectivity = component_matrices['connectivity']
    coords = component_matrices['coords']

    cells_array = []
    cell_types_array = []
    points = []
    offset = 0

    for i, comp in enumerate(level_comp):
        # Load component connectivity and points
        c_arr, cel_typ = el_connectivity[comp]
        pnt = coords[comp]

        if i > 0:
            # Update node offset by the number of nodes in the previous component
            offset += points[i - 1].shape[0]

            c_arr_2 = c_arr.copy()

            # Walk through the VTK cell array.
            # At positions where a new cell starts, c_arr[j] is the number of nodes in the cell.
            # The next_position variable jumps to the next cell header.
            next_position = 0
            for j in range(0, len(c_arr_2)):
                if j == next_position:
                    # Move to next cell header: skip over [n_nodes + that many indices]
                    next_position += c_arr[j] + 1
                else:
                    # This is a node index; shift it by offset
                    c_arr_2[j] += offset

            cells_array.append(c_arr_2)

        else:
            cells_array.append(c_arr)

        cell_types_array.append(cel_typ)
        points.append(pnt)

    if len(level_comp) != 0:
        cells_array = np.hstack(cells_array)
        cell_types_array = np.concatenate(cell_types_array)
        points = np.vstack(points)

    nodal_connectivity = (cells_array, cell_types_array)

    return nodal_connectivity


def delete_DOF_j_contribution(mat, sub_mats_j):
    """
    PURPOSE:
        Suppress/replace a contribution associated with a DOF block "j" by:
          1) removing a number of rows from the bottom of `mat`, and
          2) appending a new block built from `sub_mats_j`.

        This is typically used in assembly operations where certain DOF blocks must be
        overwritten or constrained.

    INPUTS:
        mat (scipy.sparse matrix):
            Original matrix to modify.

        sub_mats_j (list):
            List of sparse matrices intended to be combined as a block row.
            The number of rows removed from `mat` is sub_mats_j[0].shape[0].

    OUTPUTS:
        mat (scipy.sparse.csc_matrix):
            Modified matrix with the bottom rows replaced/extended by the block row.

    SIDE EFFECTS:
        None

    NOTES:
        - This function assumes that the rows to remove correspond exactly to the
          size of sub_mats_j[0].shape[0].
        - If sub_mats_j[0] has zero rows, the function does nothing.

    EXAMPLES:
        >>> mat2 = delete_DOF_j_contribution(mat, [Aij, Ajj])
    """

    # Only do work if the replacement block has nonzero row count
    if sub_mats_j[0].shape[0] != 0:
        # Remove last block of rows from mat
        mat = mat[:-sub_mats_j[0].shape[0], :]

        # Append the new block row at the bottom:
        # sp.sparse.bmat([sub_mats_j]) builds a single block row matrix
        # then bmat([[mat], [that_block_row]]) stacks vertically.
        mat = sp.sparse.bmat(
            [[mat],
             [sp.sparse.bmat([sub_mats_j], format='csc')]],
            format='csc'
        )

    return mat

def suppress_zero_row_cols(Mat, Mat_pres, T):
    """
    PURPOSE:
        Remove (suppress) rows and columns that are entirely zero from a list of matrices,
        and consistently apply the same reduction to associated "prescribed" matrices and
        a transformation matrix T.

        The reduction mask is built from Mat[0] by keeping only rows with nonzero entries.

    INPUTS:
        Mat (list):
            List of square sparse matrices (same shape). Mat[0] is used to decide which
            rows/cols to keep.

        Mat_pres (list):
            List of sparse matrices that are reduced only on the left side:
                Mat_pres[i] := const_mat @ Mat_pres[i]

        T (scipy.sparse matrix):
            Transformation matrix updated as:
                T := T @ const_mat.T

    OUTPUTS:
        Mat (list):
            Reduced matrices: const_mat @ mat @ const_mat.T for each mat in Mat.

        Mat_pres (list):
            Reduced prescribed matrices: const_mat @ mat for each mat in Mat_pres.

        T (scipy.sparse matrix):
            Updated transformation matrix.

    SIDE EFFECTS:
        None

    NOTES:
        - const_mat is a row-selector matrix formed from the identity matrix. It keeps
          the DOFs whose corresponding rows in Mat[0] are not all zero.
        - Using Mat[0].getnnz(axis=1) means the criterion is based on nonzero *rows*.
          Because the matrix is presumed to be symmetric in many mechanical contexts,
          removing rows implies removing the corresponding columns.

    EXAMPLES:
        >>> Mat_red, Mat_pres_red, T_red = suppress_zero_row_cols(Mat, Mat_pres, T)
    """

    # Build a selector matrix that keeps only rows with at least one nonzero in Mat[0].
    # Mat[0].getnnz(axis=1) gives number of nonzeros per row.
    keep_rows = Mat[0].getnnz(axis=1).astype(bool)

    # const_mat is a reduced identity: shape (n_kept, n_full)
    const_mat = sp.sparse.eye(Mat[0].shape[0], format='csc')[keep_rows, :]

    # Reduce each square matrix by removing the same rows and columns
    Mat = [const_mat @ mat @ const_mat.T for mat in Mat]

    # Reduce the "prescribed" matrices only on the left (remove rows)
    Mat_pres = [const_mat @ mat for mat in Mat_pres]

    # Update transformation matrix: consistent with removing DOFs in the reduced space
    T = T @ const_mat.T

    return Mat, Mat_pres, T

#%% Functions for coupling models
def constraint_matrix(interface_1, interface_2, matching=True, atol=1e-3,
    sparse=False, ndim=(1, 1), constraint_matrices=None, coupling_dof=([], [])):
    """
    PURPOSE:
        Create the constraint matrices that couple two interfaces (two node sets) belonging to
        two domains/substructures.

        Depending on `matching`, the function builds constraints in one of three ways:

        1) matching == 'NODE-NODE'
           - Assumes interfaces are conformal (matching meshes)
           - Matches nodes by proximity within tolerance `atol` using a KDTree
           - Builds boundary selection matrices B1bb and B2bb such that:
                 B1bb * u1 + B2bb * u2 = 0
             where B2bb has a negative sign.

        2) matching == 'MPC'
           - Builds Multi-Point Constraints by clustering interface nodes in domain 1 and
             mapping clusters to domain 2.
           - Introduces auxiliary "reference" nodes (a new "component") for each cluster and
             springs (Kref) connecting set nodes to reference nodes.
           - Returns tuples B1bb and B2bb containing selection matrices and auxiliary data.

        3) matching == 'NON-CONFORMAL'
           - For non-conformal meshes, computes interpolation weights from domain 2 nodes to
             domain 1 interface nodes.
           - Uses either bilinear interpolation on a triangle (3 nearest non-collinear nodes)
             or linear interpolation on a segment (2 nearest nodes if collinear).

    INPUTS:
        interface_1 (tuple):
            (nodes_1, coords_1, nodes_12_1, coords_12_1)
            where:
              - nodes_1      : array of all node IDs in domain 1
              - coords_1     : coordinates for all nodes_1 (not always used here)
              - nodes_12_1   : node IDs belonging to the interface subset (domain 1)
              - coords_12_1  : coordinates of interface nodes_12_1

        interface_2 (tuple):
            (nodes_2, coords_2, nodes_12_2, coords_12_2) (same structure for domain 2)

        matching (str | bool):
            Expected values:
              - 'NODE-NODE'
              - 'MPC'
              - 'NON-CONFORMAL'
            NOTE: In your original code, the NON-CONFORMAL branch is written as:
                elif 'NON-CONFORMAL':
            which is ALWAYS True (because non-empty strings are truthy).
            This is preserved in spirit but flagged in NOTES below.

        atol (float, optional):
            Tolerance / pinball radius used for node matching or clustering. Default = 1e-3.

        sparse (bool, optional):
            Present in signature, not used explicitly in the provided logic. Default = False.

        ndim (tuple, optional):
            (ndim_domain1, ndim_domain2)
            Number of DOFs per node in each domain. Used to block-diagonalize constraint matrices.

        constraint_matrices (tuple | list | None, optional):
            (C1, C2) constraint reduction matrices for domain 1 and domain 2.
            Used in NODE-NODE and MPC branches to suppress constrained DOFs and reduce vectors.
            REQUIRED by your current NODE-NODE and MPC code paths (must not be None there).

        coupling_dof (tuple, optional):
            (dof_list1, dof_list2) lists selecting which DOF components to couple.
            Example: for ndim=3, coupling_dof=([2],[2]) might couple only z DOF (0-based indexing).

    OUTPUTS:
        B1bb:
            - If matching == 'NODE-NODE': sparse matrix selecting boundary DOFs in domain 1
            - If matching == 'MPC'      : tuple (comp_B1bb, ref_B1bb, MPC_B1bb, Kref1, coords_ref1, av_mat1, vtk_conn)
            - If matching is NON-CONFORMAL: weights matrix (each interface node in 1 expressed as combination of nodes in 2)

        B2bb:
            - Analogous to B1bb but for domain 2 (negative sign to enforce continuity)

        diag_1i (np.ndarray):
            Integer vector (0/1) indicating interior DOFs of domain 1 after coupling selection/reduction.

        diag_2i (np.ndarray):
            Same for domain 2.

        nodes_12_1_final (np.ndarray):
            Interface nodes actually coupled on domain 1.

        nodes_12_2_final (np.ndarray):
            Interface nodes actually coupled on domain 2.

    SIDE EFFECTS:
        - Uses KDTree queries (SciPy).
        - Uses clustering models (AgglomerativeClustering, KMeans) in MPC branch.
        - Uses randomness ONLY indirectly if your clustering or earlier code uses it (not here).

    NOTES:
        - CRITICAL: The original line `elif 'NON-CONFORMAL':` is logically incorrect if you meant:
              elif matching == 'NON-CONFORMAL':
          because the current code makes that branch always execute whenever earlier branches fail.
          I keep the structure but strongly recommend fixing it in your source.

        - In NODE-NODE and MPC branches, `constraint_matrices` is used and must be provided.

        - The matrices B1bb/B2bb are typically used as:
              B1bb * u1 + B2bb * u2 = 0
          meaning they assemble continuity constraints.

    EXAMPLES:
        >>> B1bb, B2bb, d1i, d2i, n1, n2 = constraint_matrix(intf1, intf2, matching='NODE-NODE', atol=1e-4,
        ...                                                  ndim=(3,3), constraint_matrices=(C1, C2))
    """

    # Unpack interface tuples
    nodes_1, coords_1, nodes_12_1, coords_12_1 = interface_1
    nodes_2, coords_2, nodes_12_2, coords_12_2 = interface_2

    # Initialize outputs that must exist for all branches (safety)
    B1bb = None
    B2bb = None
    diag_1i = None
    diag_2i = None
    nodes_12_1_final = np.array([], dtype=nodes_12_1.dtype if hasattr(nodes_12_1, "dtype") else int)
    nodes_12_2_final = np.array([], dtype=nodes_12_2.dtype if hasattr(nodes_12_2, "dtype") else int)

    if matching == 'NODE-NODE':
        """
        NODE-NODE coupling:
            - Use KDTree to match nodes_12_1 to nodes_12_2 within atol.
            - Enforce 1-to-1 pairing by preventing reuse of interface_2 nodes.
            - Build selection matrices for boundary nodes (with sign on domain 2).
        """

        # --- STEP 1: Build a KDTree for fast nearest-neighbor lookups on interface 2 coords
        tree = sc.spatial.cKDTree(coords_12_2)

        # --- STEP 2: For each node in interface 1, find nearest neighbor in interface 2 (within atol)
        distances, indices = tree.query(coords_12_1, distance_upper_bound=atol)

        # Track which interface_2 nodes have already been assigned to ensure 1-to-1 mapping
        matched_2 = set()

        # Storage for final matched nodes and their indices in the FULL node arrays (nodes_1, nodes_2)
        nodes_12_1_final = []
        nodes_12_2_final = []
        nodes_12_1_final_indices = []
        nodes_12_2_final_indices = []

        # --- STEP 3: Select valid unique matches
        for i, (dist, idx_2) in enumerate(zip(distances, indices)):
            # KDTree convention: if no neighbor within bound, it returns idx=len(data) and dist=inf
            # We require: dist < atol, idx_2 is valid, and not already matched.
            if dist < atol and idx_2 not in matched_2 and idx_2 < len(nodes_12_2):
                matched_2.add(idx_2)

                # Matched node IDs
                node_1 = nodes_12_1[i]
                node_2 = nodes_12_2[idx_2]

                nodes_12_1_final.append(node_1)
                nodes_12_2_final.append(node_2)

                # Indices in the FULL domain arrays
                idx_1_full = np.where(nodes_1 == node_1)[0][0]
                idx_2_full = np.where(nodes_2 == node_2)[0][0]

                nodes_12_1_final_indices.append(idx_1_full)
                nodes_12_2_final_indices.append(idx_2_full)

        # Convert lists to arrays (important later for boolean masks)
        nodes_12_1_final = np.array(nodes_12_1_final)
        nodes_12_2_final = np.array(nodes_12_2_final)
        nodes_12_1_final_indices = np.array(nodes_12_1_final_indices, dtype=int)
        nodes_12_2_final_indices = np.array(nodes_12_2_final_indices, dtype=int)

        # --- STEP 4: Build boolean masks for boundary nodes in each domain
        # diag_1b[k] True if nodes_1[k] is among the matched boundary nodes
        diag_1b = np.array([node in nodes_12_1_final for node in nodes_1]).astype(bool)
        diag_2b = np.array([node in nodes_12_2_final for node in nodes_2]).astype(bool)

        # Interior masks: everything not boundary
        diag_1i = ~diag_1b
        diag_2i = ~diag_2b

        # --- STEP 5: Build boundary selection matrices
        # Start with diagonal selector matrices; later we will extract only active rows and reorder.
        B1bb = sp.sparse.diags(diag_1b.astype(int), format='csc')           # +1 on boundary rows of domain 1
        B2bb = sp.sparse.diags(-(diag_2b.astype(int)), format='csc')        # -1 on boundary rows of domain 2

        # Convert interior masks to integer (later used as diag vectors)
        diag_1i = diag_1i.astype(int)
        diag_2i = diag_2i.astype(int)

        # --- STEP 6: Reorder/select boundary rows to enforce consistent pairing order
        # Reorder domain 2 boundary rows so they correspond to the matched ordering of interface pairs
        B2bb = B2bb[nodes_12_2_final_indices, :]

        # Extract only the boundary rows from domain 1 in the order they appear in nodes_1
        B1bb = B1bb[diag_1b == 1, :]

        # --- STEP 7: Extend to multiple DOFs per node (block-diagonal repetition)
        B1bb = sp.sparse.block_diag([B1bb for _ in range(0, ndim[0])], format='csc')
        B2bb = sp.sparse.block_diag([B2bb for _ in range(0, ndim[1])], format='csc')

        # --- STEP 8: Couple only selected DOF components (if provided)
        dof_list1, dof_list2 = coupling_dof

        # Domain 1 DOF selection
        if dof_list1:
            # Take only blocks whose component index is in dof_list1
            B1bb = sp.sparse.bmat(
                [[B1bb[(i * (B1bb.shape[0] // ndim[0])):(((i + 1) * B1bb.shape[0]) // ndim[0])]]
                 for i in range(0, ndim[0]) if i in dof_list1],
                format='csc'
            )
            # Interior indicator: for non-coupled DOFs, set to ones (treated as interior/not constrained)
            diag_1i = np.concatenate([diag_1i if i in dof_list1 else np.ones(diag_1i.shape[0]) for i in range(0, ndim[0])])
        else:
            diag_1i = np.concatenate([diag_1i for _ in range(0, ndim[0])])

        # Domain 2 DOF selection
        if dof_list2:
            B2bb = sp.sparse.bmat(
                [[B2bb[(i * (B2bb.shape[0] // ndim[1])):(((i + 1) * B2bb.shape[0]) // ndim[1])]]
                 for i in range(0, ndim[1]) if i in dof_list2],
                format='csc'
            )
            diag_2i = np.concatenate([diag_2i if i in dof_list2 else np.ones(diag_2i.shape[0]) for i in range(0, ndim[1])])
        else:
            diag_2i = np.concatenate([diag_2i for _ in range(0, ndim[1])])

        # --- STEP 9: Apply constraint reduction matrices (removes constrained DOFs)
        # These matrices are used as:
        #   reduced_B = B @ C.T
        #   reduced_diag = C @ diag
        # Meaning: C maps full DOF vector -> reduced DOF vector.
        if constraint_matrices is None:
            raise ValueError("constraint_matrices must be provided for matching == 'NODE-NODE'.")

        B1bb = B1bb @ constraint_matrices[0].T
        diag_1i = constraint_matrices[0] @ diag_1i

        B2bb = B2bb @ constraint_matrices[1].T
        diag_2i = constraint_matrices[1] @ diag_2i

    elif matching == 'MPC':
        """
        MPC coupling:
            - Cluster interface_1 nodes using pinball radius atol.
            - For each cluster, create a new set of "contact nodes" and one "reference node" per cluster.
            - Connect each contact node to its reference node with a spring (Kref).
            - Build selection matrices:
                comp_B*bb: selects original component boundary nodes participating in MPC
                ref_B*bb : selects "contact nodes" in the new reference component
                MPC_B*bb : selects the "reference nodes" that couple across the two sides
            - Return tuples B1bb, B2bb containing these matrices and auxiliary geometry.
        """

        def flatten_list_array(list1):
            # Flatten nested list-of-lists into a 1D numpy array
            return np.array([x for sublist in [list1[i] for i in range(0, len(list1))] for x in sublist])

        def sparse_matrix_ones_at_indices(matrix_shape, indices, num_nodes_list=[]):
            """
            Create a sparse matrix where each row i has ones at positions given by indices[i].

            If num_nodes_list is provided, weights become (1/(num_nodes_list[i] + 1)).
            This is used to distribute influence across nodes (including a ref node idea).
            """
            matrix = sp.sparse.lil_matrix(matrix_shape)
            for i, ind_set in enumerate(indices):
                if len(num_nodes_list) == 0:
                    matrix[i, ind_set] = 1
                else:
                    matrix[i, ind_set] = 1.0 / (num_nodes_list[i] + 1)
            return matrix.tocsc()

        # ---- STEP 1: Create clusters in interface 1
        # First do radius-based clustering to get initial centers
        _, centroids_set1, _ = cluster_indices(coords_12_1, radius=atol)
        # Then do KMeans with those centers for stable/explicit clustering
        ind_set1, centroids_set1, _ = cluster_indices(coords_12_1, centers=centroids_set1)
        # Cluster interface 2 using the same centers from interface 1
        ind_set2, centroids_set2, _ = cluster_indices(coords_12_2, centers=centroids_set1)

        # ---- STEP 2: Map interface indices to component indices (indices in nodes_1 and nodes_2 arrays)
        ind_comp_sets1 = [
            [np.where(nodes_1 == nodes_12_1[ind_set1[i][j]])[0][0] for j in range(0, len(ind_set1[i]))]
            for i in range(0, len(ind_set1))
        ]
        ind_comp_sets2 = [
            [np.where(nodes_2 == nodes_12_2[ind_set2[i][j]])[0][0] for j in range(0, len(ind_set2[i]))]
            for i in range(0, len(ind_set2))
        ]

        # ---- STEP 3: Selection matrices selecting the original component boundary nodes in each domain
        indices1 = flatten_list_array(ind_comp_sets1)
        indices2 = flatten_list_array(ind_comp_sets2)

        comp_B1bb = sparse_matrix_ones_at_indices((indices1.shape[0], nodes_1.shape[0]), indices1)
        comp_B2bb = sparse_matrix_ones_at_indices((indices2.shape[0], nodes_2.shape[0]), indices2)

        # ---- STEP 4: Create the new "contact nodes" and "reference nodes" indexing inside REF components
        ind_comp_refs1 = [[i for _ in range(0, len(ind_set1[i]))] for i in range(0, len(ind_set1))]
        ind_comp_refs2 = [[i for _ in range(0, len(ind_set2[i]))] for i in range(0, len(ind_set2))]
        indices1ref = flatten_list_array(ind_comp_refs1)
        indices2ref = flatten_list_array(ind_comp_refs2)

        # Contact nodes in REF components: identity (each original selected node gets a copy/contact node)
        node_set_contact1 = sp.sparse.eye(comp_B1bb.shape[0], format='csc')
        node_set_contact2 = sp.sparse.eye(comp_B2bb.shape[0], format='csc')

        # Reference nodes in REF components: each contact node points to a cluster reference id
        node_ref_contact1 = sparse_matrix_ones_at_indices((indices1ref.shape[0], len(ind_comp_refs1)), indices1ref)
        node_ref_contact2 = sparse_matrix_ones_at_indices((indices2ref.shape[0], len(ind_comp_refs2)), indices2ref)

        # ---- STEP 5: Spring stiffness connections between contact nodes and ref nodes
        # This constructs a stiffness matrix for the artificial REF component:
        #   [ Kcc  Kcr ]
        #   [ Krc  Krr ]
        # where Kcc = I*k, Kcr = -k*incidence, Krr = k*diag(degree)
        k = 1e8  # N/m (very stiff springs)
        ref_diag_comp1 = sp.sparse.diags([len(ind_comp) for ind_comp in ind_comp_refs1])
        ref_diag_comp2 = sp.sparse.diags([len(ind_comp) for ind_comp in ind_comp_refs2])

        Kref1 = k * sp.sparse.bmat(
            [[node_set_contact1, -node_ref_contact1],
             [-node_ref_contact1.T, ref_diag_comp1]],
            format='csc'
        )
        Kref2 = k * sp.sparse.bmat(
            [[node_set_contact2, -node_ref_contact2],
             [-node_ref_contact2.T, ref_diag_comp2]],
            format='csc'
        )

        # ---- STEP 6: Selection matrices from the new REF components
        # ref_B*bb selects contact nodes in REF (negative sign aligns with constraint convention)
        ref_B1bb = -sp.sparse.bmat([[node_set_contact1, sp.sparse.csc_matrix(node_ref_contact1.shape)]], format='csc')
        ref_B2bb = -sp.sparse.bmat([[node_set_contact2, sp.sparse.csc_matrix(node_ref_contact2.shape)]], format='csc')

        # MPC_B*bb selects the reference nodes to couple across domains
        MPC_B1bb = sp.sparse.bmat(
            [[sp.sparse.csc_matrix(node_ref_contact1.T.shape), sp.sparse.eye(node_ref_contact1.shape[1])]],
            format='csc'
        )
        MPC_B2bb = -sp.sparse.bmat(
            [[sp.sparse.csc_matrix(node_ref_contact2.T.shape), sp.sparse.eye(node_ref_contact2.shape[1])]],
            format='csc'
        )

        # ---- STEP 7: Internal node indicators: mark coupled nodes as boundary (0), rest interior (1)
        diag_1i = np.ones(nodes_1.shape[0])
        diag_1i[indices1] = 0

        diag_2i = np.ones(nodes_2.shape[0])
        diag_2i[indices2] = 0

        # ---- STEP 8: Nodes actually connected (these are the participating boundary nodes)
        nodes_12_1_final = nodes_1[indices1]
        nodes_12_2_final = nodes_2[indices2]

        # ---- STEP 8.1: VTK-format connectivities (currently unused -> placeholders)
        el_con1, el_type1 = [], []
        el_con2, el_type2 = [], []

        # ---- STEP 9: Extend to multiple DOFs per node (block-diagonal repetition)
        # Domain 1
        comp_B1bb = sp.sparse.block_diag([comp_B1bb for _ in range(0, ndim[0])], format='csc')
        ref_B1bb = sp.sparse.block_diag([ref_B1bb for _ in range(0, ndim[0])], format='csc')
        MPC_B1bb = sp.sparse.block_diag([MPC_B1bb for _ in range(0, ndim[0])], format='csc')
        Kref1 = sp.sparse.block_diag([Kref1 for _ in range(0, ndim[0])], format='csc')

        # Domain 2
        comp_B2bb = sp.sparse.block_diag([comp_B2bb for _ in range(0, ndim[1])], format='csc')
        ref_B2bb = sp.sparse.block_diag([ref_B2bb for _ in range(0, ndim[1])], format='csc')
        MPC_B2bb = sp.sparse.block_diag([MPC_B2bb for _ in range(0, ndim[1])], format='csc')
        Kref2 = sp.sparse.block_diag([Kref2 for _ in range(0, ndim[1])], format='csc')

        # ---- STEP 10: Couple only selected DOF components if requested
        dof_list1, dof_list2 = coupling_dof

        if dof_list1:
            comp_B1bb = sp.sparse.bmat(
                [[comp_B1bb[(i * (comp_B1bb.shape[0] // ndim[0])):(((i + 1) * comp_B1bb.shape[0]) // ndim[0])]]
                 for i in range(0, ndim[0]) if i in dof_list1],
                format='csc'
            )
            ref_B1bb = sp.sparse.bmat(
                [[ref_B1bb[(i * (ref_B1bb.shape[0] // ndim[0])):(((i + 1) * ref_B1bb.shape[0]) // ndim[0])]]
                 for i in range(0, ndim[0]) if i in dof_list1],
                format='csc'
            )
            MPC_B1bb = sp.sparse.bmat(
                [[MPC_B1bb[(i * (MPC_B1bb.shape[0] // ndim[0])):(((i + 1) * MPC_B1bb.shape[0]) // ndim[0])]]
                 for i in range(0, ndim[0]) if i in dof_list1],
                format='csc'
            )
            diag_1i = np.concatenate([diag_1i if i in dof_list1 else np.ones(diag_1i.shape[0]) for i in range(0, ndim[0])])
        else:
            diag_1i = np.concatenate([diag_1i for _ in range(0, ndim[0])])

        if dof_list2:
            comp_B2bb = sp.sparse.bmat(
                [[comp_B2bb[(i * (comp_B2bb.shape[0] // ndim[1])):(((i + 1) * comp_B2bb.shape[0]) // ndim[1])]]
                 for i in range(0, ndim[1]) if i in dof_list2],
                format='csc'
            )
            ref_B2bb = sp.sparse.bmat(
                [[ref_B2bb[(i * (ref_B2bb.shape[0] // ndim[1])):(((i + 1) * ref_B2bb.shape[0]) // ndim[1])]]
                 for i in range(0, ndim[1]) if i in dof_list2],
                format='csc'
            )
            MPC_B2bb = sp.sparse.bmat(
                [[MPC_B2bb[(i * (MPC_B2bb.shape[0] // ndim[1])):(((i + 1) * MPC_B2bb.shape[0]) // ndim[1])]]
                 for i in range(0, ndim[1]) if i in dof_list2],
                format='csc'
            )
            diag_2i = np.concatenate([diag_2i if i in dof_list2 else np.ones(diag_2i.shape[0]) for i in range(0, ndim[1])])
        else:
            diag_2i = np.concatenate([diag_2i for _ in range(0, ndim[1])])

        # ---- STEP 11: Reduce DOFs using constraint matrices
        if constraint_matrices is None:
            raise ValueError("constraint_matrices must be provided for matching == 'MPC'.")

        comp_B1bb = comp_B1bb @ constraint_matrices[0].T
        diag_1i = constraint_matrices[0] @ diag_1i

        comp_B2bb = comp_B2bb @ constraint_matrices[1].T
        diag_2i = constraint_matrices[1] @ diag_2i

        # ---- STEP 12: Coordinates for new REF elements:
        # First stack all contact-node coordinates (original interface nodes grouped),
        # then append cluster centroids (reference nodes).
        coords_ref1 = np.vstack([np.vstack([coords_12_1[ind_set1[i]] for i in range(0, len(ind_set1))]), centroids_set1])
        coords_ref2 = np.vstack([np.vstack([coords_12_2[ind_set2[i]] for i in range(0, len(ind_set2))]), centroids_set2])

        # ---- STEP 13: Averaging matrices used for nodal normals (cluster -> ref averaging)
        av_mat1 = node_ref_contact1.copy()
        av_mat2 = node_ref_contact2.copy()

        # ---- STEP 15: Package results
        B1bb = (comp_B1bb, ref_B1bb, MPC_B1bb, Kref1, coords_ref1, av_mat1, (el_con1, el_type1))
        B2bb = (comp_B2bb, ref_B2bb, MPC_B2bb, Kref2, coords_ref2, av_mat2, (el_con2, el_type2))

    elif matching == 'NON-CONFORMAL':
        """
        NON-CONFORMAL coupling:
            - For each node in interface set 1, find 3 nearest nodes in interface set 2.
            - If those 3 are not collinear, build a planar triangle basis and compute
              barycentric-like interpolation weights (here implemented via linear shape function solve).
            - If collinear, fall back to linear interpolation using 2 nearest nodes.

        IMPORTANT BUG NOTE:
            This branch condition is written as `elif 'NON-CONFORMAL':` which is ALWAYS True.
            If you intended to select this method by string, it should be:
                elif matching == 'NON-CONFORMAL':
            Keeping your current structure as provided, but this should be fixed in your source.
        """

        # We build weights matrices:
        #   weights_1: maps interface nodes in set 1 to full nodes_1 indices (identity rows)
        #   weights_2: maps interface nodes in set 1 to full nodes_2 indices (interpolation weights)
        weights_1 = sp.sparse.csc_matrix((nodes_12_1.shape[0], nodes_1.shape[0]))
        weights_2 = sp.sparse.csc_matrix((nodes_12_1.shape[0], nodes_2.shape[0]))

        for i in range(0, len(nodes_12_1)):
            # Distances from node i in interface_1 to all interface_2 nodes
            dist = np.linalg.norm(coords_12_2 - coords_12_1[i, :], axis=1)

            # Prepare arrays for nearest nodes
            near_indices = np.zeros(3, dtype=int)
            near_dist = np.zeros(3, dtype=float)
            near_coords = np.zeros((4, 3), dtype=float)

            # Row 0 is the query point (interface_1 node coordinate)
            near_coords[0, :] = coords_12_1[i, :]

            bilinear_interpolation = True

            # Pick 3 nearest distinct nodes (greedy), but detect collinearity by triangle normal
            for j in range(0, 3):
                near_id = np.argmin(dist)
                near_indices[j] = near_id
                near_dist[j] = dist[near_id]
                near_coords[j + 1, :] = coords_12_2[near_id, :]

                if j == 1:
                    # v1 = vector from nearest #1 to nearest #2
                    v1 = near_coords[2, :] - near_coords[1, :]
                    v1_norm = np.linalg.norm(v1)
                elif j == 2:
                    # v2 = vector from nearest #1 to nearest #3
                    v2 = near_coords[3, :] - near_coords[1, :]
                    v2_norm = np.linalg.norm(v2)

                    # Triangle normal magnitude indicates collinearity
                    normal_tri = np.cross(v1 / v1_norm, v2 / v2_norm)
                    normal_tri_norm = np.linalg.norm(normal_tri)

                    # If ~0, points are collinear -> use only 2 nearest and do linear interpolation
                    next_node = normal_tri_norm <= 1e-5
                    if next_node:
                        bilinear_interpolation = False
                        near_indices = near_indices[:2]
                        near_dist = near_dist[:2]
                        near_coords = near_coords[:3, :]

                # Prevent selecting the same node again
                dist[near_id] = np.inf

            # Transform coordinates to a local basis:
            #   - centered at centroid of nearest nodes
            #   - rotated so the triangle normal aligns with z (for 2D interpolation)
            if bilinear_interpolation:
                # Center around triangle centroid (of the 3 nearest nodes)
                near_coords -= np.mean(near_coords[1:, :], axis=0)

                # Rotate so triangle normal aligns with z-axis
                v3 = np.array([0, 0, 1])
                n = np.cross(normal_tri, v3)      # rotation axis
                theta = np.arccos(normal_tri @ v3)  # rotation angle

                R = rotation_matrix(n, theta)
                rot_near_coords = (R @ near_coords.T).T

                # Build linear shape functions on triangle: phi_i(x,y) = a + b x + c y
                x = rot_near_coords[1:, 0]
                y = rot_near_coords[1:, 1]

                matrix_bilinear = np.array([
                    [1, x[0], y[0]],
                    [1, x[1], y[1]],
                    [1, x[2], y[2]]
                ])

                # Solve for coefficients so that phi_i(node_j) = delta_ij
                coefs1 = np.linalg.solve(matrix_bilinear, np.array([1, 0, 0]))
                coefs2 = np.linalg.solve(matrix_bilinear, np.array([0, 1, 0]))
                coefs3 = np.linalg.solve(matrix_bilinear, np.array([0, 0, 1]))

                # Evaluate shape functions at the query point (row 0)
                w1 = coefs1[0] + coefs1[1:] @ rot_near_coords[0, :2]
                w2 = coefs2[0] + coefs2[1:] @ rot_near_coords[0, :2]
                w3 = coefs3[0] + coefs3[1:] @ rot_near_coords[0, :2]
                w = np.array([w1, w2, w3])

                # weights_1: identity mapping of interface_1 node to its full index in nodes_1
                idx_nodes_1 = np.where(nodes_1 == nodes_12_1[i])[0][0]
                weights_1[i, idx_nodes_1] = 1.0

                # weights_2: interpolation mapping to nodes_2 full indices
                for j, idx in enumerate(near_indices):
                    idx_nodes_2 = np.where(nodes_2 == nodes_12_2[idx])[0][0]
                    if w[j] != 0.0:
                        weights_2[i, idx_nodes_2] = w[j]

            else:
                # Linear interpolation using 2 nearest nodes:
                # Align segment direction with x-axis and solve phi_i(x) = a + b x
                near_coords -= np.mean(near_coords[1:, :], axis=0)

                v3 = np.array([1, 0, 0])
                n = np.cross(v1 / v1_norm, v3)
                theta = np.arccos((v1 / v1_norm) @ v3)

                R = rotation_matrix(n, theta)
                rot_near_coords = (R @ near_coords.T).T

                x = rot_near_coords[1:, 0]
                matrix_linear = np.array([
                    [1, x[0]],
                    [1, x[1]]
                ])

                coefs1 = np.linalg.solve(matrix_linear, np.array([1, 0]))
                coefs2 = np.linalg.solve(matrix_linear, np.array([0, 1]))

                w1 = coefs1[0] + coefs1[1:] @ rot_near_coords[0, :1]
                w2 = coefs2[0] + coefs2[1:] @ rot_near_coords[0, :1]
                w = np.array([w1, w2])

                idx_nodes_1 = np.where(nodes_1 == nodes_12_1[i])[0][0]
                weights_1[i, idx_nodes_1] = 1.0

                for j, idx in enumerate(near_indices):
                    idx_nodes_2 = np.where(nodes_2 == nodes_12_2[idx])[0][0]
                    if w[j] != 0.0:
                        weights_2[i, idx_nodes_2] = w[j]

        # Determine which full nodes are on the boundary (appear in weights matrices)
        diag_1b = np.array(weights_1.sum(axis=0))[0, :].astype(bool)
        diag_2b = np.array(weights_2.sum(axis=0))[0, :].astype(bool)

        # Interior nodes are those not participating in coupling
        diag_1i = (~diag_1b).astype(int)
        diag_2i = (~diag_2b).astype(int)

        # Constraint matrices for non-conformal coupling
        B1bb = weights_1.copy()
        B2bb = -weights_2.copy()

        # IMPORTANT: define final connected node lists for consistent return
        # Domain 1: all interface nodes in set1 are "used" as constraints rows
        nodes_12_1_final = nodes_12_1.copy()
        # Domain 2: the participating nodes are those with nonzero column sum in weights_2
        nodes_12_2_final = nodes_2[diag_2b]

    else:
        # NOTE: In the provided code, this else is effectively unreachable due to `elif 'NON-CONFORMAL'`.
        print('Matching method wrongly selected: available options are NODE-NODE, MPC and NON-CONFORMAL.')

    return B1bb, B2bb, diag_1i, diag_2i, np.array(nodes_12_1_final), np.array(nodes_12_2_final)
    
def assembly_constraint_matrix(B_dict, shapes_dom, find_overlapping=False, norm_values=False):
    """
    PURPOSE:
        Assemble a global constraint matrix B for a multi-domain system from pairwise constraint
        blocks stored in B_dict.

        Each entry in B_dict represents a constraint equation row-block coupling two domains i and j:
            B_row = [ 0 ... Bij ... Bji ... 0 ]
        so that:
            Bij * u_i + Bji * u_j = 0

        Optionally attempts to detect overlapping DOFs across different constraint rows in the
        same domain column and merge them (find_overlapping=True).

    INPUTS:
        B_dict (dict):
            Keys: arbitrary position identifiers
            Values: (Bij, Bji, ij_ind)
                - Bij (sparse matrix): constraint block acting on domain i DOFs
                - Bji (sparse matrix): constraint block acting on domain j DOFs
                - ij_ind (tuple): (i, j) domain indices (1-based indexing in your code)

        shapes_dom (list):
            List of number of DOFs per domain (column sizes).

        find_overlapping (bool, optional):
            If True, tries to identify constraints that act on the same DOFs and merges them
            (experimental / fragile). Default = False.

        norm_values (bool, optional):
            If True, normalizes constraint row values when merging overlaps:
              - rows with multiple +1 blocks divided by number of + blocks
              - rows with multiple -1 blocks divided by number of - blocks
            Default = False.

    OUTPUTS:
        B_sp (scipy.sparse.csc_matrix):
            Assembled global constraint matrix (block assembled).

        B (np.ndarray of sparse matrices):
            Block matrix representation with shape (#constraints, #domains),
            where each entry is a sparse matrix.

    SIDE EFFECTS:
        - If find_overlapping=True, uses dense conversions `.todense()` / `.toarray()` in places.

    NOTES:
        - Domain indices in ij_ind are assumed to be 1-based (comp_i - 1 indexing).
        - The overlap detection method can be expensive and may be numerically brittle.

    EXAMPLES:
        >>> B_sp, B_blocks = assembly_constraint_matrix(B_dict, shapes_dom)
    """

    positions = list(B_dict.keys())
    n = len(shapes_dom)

    # Initialize block container B (rows = constraints, cols = domains)
    if len(positions) == 0:
        if shapes_dom:
            # No constraints: return a single empty-row block structure
            B = np.array([[sp.sparse.csc_matrix((0, shape)) for shape in shapes_dom]])
        else:
            B = np.array([[sp.sparse.csc_matrix((0, 0))]])
    else:
        B = np.array([[None] * n] * len(positions))

    # Fill blocks for each constraint row
    for i in range(len(positions)):
        Bij, Bji, ij_ind = B_dict[positions[i]]
        comp_i = ij_ind[0]
        comp_j = ij_ind[1]

        # Place Bij in domain i column, Bji in domain j column
        B[i, comp_i - 1] = Bij
        B[i, comp_j - 1] = Bji

    # Optionally merge overlapping DOFs (experimental)
    if find_overlapping:
        B = copy.deepcopy(B)

        if B.shape[0]:
            for col_ref in range(B.shape[1]):
                for row_ref in range(B.shape[0]):
                    ref = B[row_ref, col_ref]

                    if ref is not None:
                        ref_nz = ref.getnnz(axis=0)

                        for i in range(row_ref + 1, B.shape[0]):
                            comp = B[i, col_ref]

                            if comp is not None:
                                comp_nz = comp.getnnz(axis=0)

                                # prod_nz True where both ref and comp act on same DOF columns
                                prod_nz = (ref_nz * comp_nz).astype(bool)

                                if np.any(prod_nz):
                                    # Locate which ROWS in those overlapping columns are nonzero
                                    rows_ref_nz = np.where(ref[:, prod_nz].todense() != 0)[0]
                                    rows_comp_nz = np.where(comp[:, prod_nz].todense() != 0)[0]

                                    # For remaining columns (domains), add corresponding rows from comp into ref row
                                    for j in range(col_ref + 1, B.shape[1]):
                                        if B[row_ref, j] is not None:
                                            if B[i, j] is not None:
                                                B[row_ref, j][rows_ref_nz, :] += B[i, j][rows_comp_nz, :]
                                        else:
                                            if B[i, j] is not None:
                                                B[row_ref, j] = sp.sparse.csc_matrix((ref.shape[0], B[i, j].shape[1]))
                                                B[row_ref, j][rows_ref_nz, :] = B[i, j][rows_comp_nz, :]

                                        # Remove overlapped rows from the compared constraint row i
                                        if B[i, j] is not None:
                                            if B[i, col_ref].shape[0] == B[i, j].shape[0]:
                                                B[i, col_ref] = sp.sparse.csc_matrix(
                                                    np.delete(B[i, col_ref].toarray(), rows_comp_nz, axis=0)
                                                )
                                            B[i, j] = sp.sparse.csc_matrix(
                                                np.delete(B[i, j].toarray(), rows_comp_nz, axis=0)
                                            )

            if norm_values:
                # Determine sign pattern (+1 or -1 blocks)
                pos_neg_mat = np.zeros(B.shape)
                for i in range(0, B.shape[0]):
                    for j in range(0, B.shape[1]):
                        if B[i, j] is not None:
                            if B[i, j].max() == 1:
                                pos_neg_mat[i, j] = 1
                            elif B[i, j].min() == -1:
                                pos_neg_mat[i, j] = -1

                # Count positive/negative blocks per constraint row
                pos_div = (pos_neg_mat > 0).sum(axis=1)
                neg_div = (pos_neg_mat < 0).sum(axis=1)

                # Normalize
                for i in range(0, B.shape[0]):
                    for j in range(0, B.shape[1]):
                        if pos_neg_mat[i][j] == 1:
                            B[i, j] = 1 / pos_div[i] * B[i, j]
                        elif pos_neg_mat[i][j] == -1:
                            B[i, j] = 1 / neg_div[i] * B[i, j]

    # Replace None slots with correctly-sized empty sparse matrices
    if len(positions) != 0:
        row_sizes = np.array([None] * len(positions))
        col_sizes = np.array(shapes_dom)

        # Infer row/col sizes from existing blocks
        for i in range(0, B.shape[0]):
            for j in range(0, B.shape[1]):
                if B[i, j] is not None:
                    row_sizes[i], col_sizes[j] = B[i, j].shape

        # Fill None blocks
        for i in range(0, B.shape[0]):
            for j in range(0, B.shape[1]):
                if B[i, j] is None:
                    B[i, j] = sp.sparse.csc_matrix((row_sizes[i], col_sizes[j]))
                else:
                    B[i, j] = sp.sparse.csc_matrix(B[i, j])

    # Assemble into a single sparse matrix
    if B.shape[0]:
        B_sp = sp.sparse.bmat(B, format='csc')
    else:
        B_sp = sp.sparse.csc_matrix((0, 0))

    return B_sp, B

def assembly_localization_matrix(diag_L_dict, B_dict, shapes, constraint_matrices=None):
    """
    PURPOSE:
        Build the assembly localization matrix L and related helper matrices for a multi-domain system.

        L is constructed to represent:
            - Interior DOF selection (Bii)
            - Boundary/equilibrium constraints (B_eq)

        In your implementation:
            L = [[Bii],
                 [abs(B_assembly)]]^T

        where:
          - Bii selects all interior DOFs across all domains
          - B_assembly is the global continuity constraint matrix between domains
          - B_eq is the equilibrium-style constraint matrix (here approximated as abs(B))

    INPUTS:
        diag_L_dict (dict):
            Keys: identifiers
            Values: (diag_ij, diag_ji, ij_ind)
                diag_ij and diag_ji are 0/1 vectors indicating interior DOFs for each domain
                for the interface between domains i and j (1-based indexing in ij_ind).

        B_dict (dict):
            Dictionary of pairwise constraint matrices used by assembly_constraint_matrix.

        shapes (list):
            DOF sizes of each domain (columns for block assembly).

        constraint_matrices (optional):
            Present for signature compatibility; not used in this function as provided.

    OUTPUTS:
        L (scipy.sparse.csc_matrix):
            Global localization matrix (transposed at the end).

        B (scipy.sparse.csc_matrix):
            Global continuity constraint matrix.

        B_eq (scipy.sparse.csc_matrix):
            Equilibrium matrix used in localization (here abs(B)).

        Bii (scipy.sparse.csc_matrix):
            Block-diagonal selection matrix for interior DOFs.

        L_vect (list):
            List form of localization blocks per domain.

        B_vect, B_eq_vect, B_eq_list, Bii_vect, B_list, Bbb_vect:
            Various per-domain block decompositions used later in your workflow.

    SIDE EFFECTS:
        None

    NOTES:
        - Your code uses B_eq = abs(B) because the overlap resolution method is disabled/commented.
        - diag vectors are combined multiplicatively across all interfaces.

    EXAMPLES:
        >>> L, B, B_eq, Bii, *rest = assembly_localization_matrix(diag_L_dict, B_dict, shapes)
    """

    positions = list(diag_L_dict.keys())
    n = len(shapes)

    # Build continuity constraint matrix B
    B, B_list = assembly_constraint_matrix(B_dict, shapes, find_overlapping=False, norm_values=False)

    # Build equilibrium matrix (currently abs-based approximation)
    B_eq, B_eq_list = np.abs(B), np.abs(B_list)

    # Build combined interior indicator per domain
    diag = [np.ones(shape) for shape in shapes]
    for i in range(0, len(positions)):
        diag_ij, diag_ji, ij_ind = diag_L_dict[positions[i]]
        comp_i = ij_ind[0]
        comp_j = ij_ind[1]

        # Multiply masks so any boundary marking (0) persists
        diag[comp_i - 1] *= diag_ij
        diag[comp_j - 1] *= diag_ji

    # Interior selection matrices per domain: keep rows where diag==1
    if diag:
        Bii_vect = [sp.sparse.diags(diag[i], format='csc')[diag[i] == 1, :] for i in range(0, n)]
    else:
        Bii_vect = [sp.sparse.csc_matrix((0, 0))]

    Bii = sp.sparse.block_diag(Bii_vect, format='csc')

    # Boundary selection matrices per domain (complement of interior)
    diag_b = [(~(diag[i].astype(bool))).astype(int) for i in range(0, n)]
    Bbb_vect = [sp.sparse.diags(diag_b[i], format='csc')[diag_b[i] == 1, :] for i in range(0, n)]

    # Compute cumulative positions of each domain block in the assembled B columns
    pos = [0]
    for i in range(0, len(Bii_vect)):
        pos.append(pos[i] + Bii_vect[i].shape[1])

    # Split B_eq and B into per-domain column blocks
    B_eq_vect = [B_eq[:, pos[i]:pos[i + 1]] for i in range(0, len(Bii_vect))]
    B_vect = [B[:, pos[i]:pos[i + 1]] for i in range(0, len(Bii_vect))]

    # Global localization matrix (Boolean-style, assembled then transposed)
    L = sp.sparse.bmat([[Bii],
                        [B_eq]], format='csc').T

    # Per-domain localization blocks
    L_vect = [sp.sparse.bmat([[Bii_vect[i]],
                              [B_eq_vect[i]]], format='csc') for i in range(0, len(Bii_vect))]

    return L, B, B_eq, Bii, L_vect, B_vect, B_eq_vect, B_eq_list, Bii_vect, B_list, Bbb_vect


def assembly_coupling_matrix(B_dict, shapes):
    """
    PURPOSE:
        Build the vibro-acoustic coupling matrices between structural and acoustic subdomains
        using per-interface localization matrices and geometric operators (normals and areas).

        Each interface entry in B_dict provides:
            Bij: structure localization (select structural interface DOFs)
            Nij: normal operator/matrix
            Aij: area operator/matrix
            Bji: acoustic localization (select acoustic interface DOFs)
            ij_ind: (i_struct, j_acoustic) indices (1-based)

        The coupling matrix in block-list form is constructed as:
            C_list_sf = B_list_s^T * N_diag_sf * A_diag_sf * B_list_f

        Additional convenience outputs:
            C_list_va_s = N_diag_sf * A_diag_sf * B_list_f
            C_list_va_f = A_diag_sf * N_diag_sf^T * B_list_s

    INPUTS:
        B_dict (dict):
            Keys: interface identifiers
            Values: (Bij, Nij, Aij, Bji, ij_ind)

        shapes (tuple):
            (shapes_s, shapes_f)
            where shapes_s is list of DOF sizes per structural subdomain,
            and shapes_f is list of DOF sizes per acoustic subdomain.

    OUTPUTS:
        C_list_sf:
            Block-list coupling matrix from structural DOFs to acoustic DOFs (assembled as array product).

        B_list_s, B_list_f:
            Block-lists of localization matrices for structural and acoustic domains.

        N_diag_sf, A_diag_sf:
            Block-diagonal arrays (as numpy 2D arrays of sparse matrices) for normals and areas.

        C_list_va_s, C_list_va_f:
            Convenience coupling forms used in vibro-acoustic formulations.

    SIDE EFFECTS:
        None

    NOTES:
        - This function constructs block arrays (numpy arrays of sparse matrices) and multiplies them
          using numpy's @ which performs array-matrix multiplication with object dtype.
        - There is a TODO in your code about detecting repeated nodes and dividing areas accordingly.
          It is intentionally not implemented here (kept as-is).

    EXAMPLES:
        >>> C_list_sf, B_s, B_f, N, A, C_va_s, C_va_f = assembly_coupling_matrix(B_dict, (shapes_s, shapes_f))
    """

    positions = list(B_dict.keys())
    shapes_s, shapes_f = shapes

    shapes_s = [0] if shapes_s == [] else shapes_s
    shapes_f = [0] if shapes_f == [] else shapes_f

    n_s, n_f = len(shapes_s), len(shapes_f)

    # Initialize block lists (rows = interfaces, cols = domains)
    B_list_s = np.array([[None] * n_s] * max(1, len(positions)))
    B_list_f = np.array([[None] * n_f] * max(1, len(positions)))

    # Per-interface normal and area operators (kept as separate lists then diagonalized)
    N_list_sf = np.array([sp.sparse.csc_matrix((0, 0))] * max(1, len(positions)))
    A_list_sf = np.array([sp.sparse.csc_matrix((0, 0))] * max(1, len(positions)))

    # Populate blocks from B_dict
    for i in range(0, len(positions)):
        Bij, Nij, Aij, Bji, ij_ind = B_dict[positions[i]]
        comp_i = ij_ind[0]
        comp_j = ij_ind[1]

        B_list_s[i, comp_i - 1] = Bij
        N_list_sf[i] = Nij
        A_list_sf[i] = Aij
        B_list_f[i, comp_j - 1] = Bji

    # Replace None entries in B_list_s with correct-size empty matrices
    rows_s = np.zeros(max(1, len(positions)), dtype=int)
    cols_s = np.zeros(n_s, dtype=int)

    for i in range(0, B_list_s.shape[0]):
        for j in range(0, B_list_s.shape[1]):
            rows_s[i] = B_list_s[i, j].shape[0] if B_list_s[i, j] is not None else np.max((rows_s[i], 0))
            cols_s[j] = B_list_s[i, j].shape[1] if B_list_s[i, j] is not None else shapes_s[j]

    for i in range(0, B_list_s.shape[0]):
        for j in range(0, B_list_s.shape[1]):
            if B_list_s[i, j] is None:
                B_list_s[i, j] = sp.sparse.csc_matrix((rows_s[i], cols_s[j]))

    # Replace None entries in B_list_f similarly
    rows_f = np.zeros(max(1, len(positions)), dtype=int)
    cols_f = np.zeros(n_f, dtype=int)

    for i in range(0, B_list_f.shape[0]):
        for j in range(0, B_list_f.shape[1]):
            rows_f[i] = B_list_f[i, j].shape[0] if B_list_f[i, j] is not None else np.max((rows_f[i], 0))
            cols_f[j] = B_list_f[i, j].shape[1] if B_list_f[i, j] is not None else shapes_f[j]

    for i in range(0, B_list_f.shape[0]):
        for j in range(0, B_list_f.shape[1]):
            if B_list_f[i, j] is None:
                B_list_f[i, j] = sp.sparse.csc_matrix((rows_f[i], cols_f[j]))

    # Transpose B_list_s into shape (n_s, n_interfaces)
    B_list_sT = np.array([[None] * max(1, len(positions))] * n_s)
    for i in range(0, B_list_sT.shape[0]):
        for j in range(0, B_list_sT.shape[1]):
            B_list_sT[i, j] = B_list_s[j, i].T

    # Diagonalize A_list_sf and N_list_sf (block diagonal in interface space)
    A_diag_sf = np.diag(A_list_sf)
    N_diag_sf = np.diag(N_list_sf)
    N_diag_sfT = np.diag([N.T for N in N_list_sf])

    # Fill off-diagonal entries with empty matrices of correct shapes
    for i in range(0, A_diag_sf.shape[0]):
        for j in range(0, A_diag_sf.shape[1]):
            if i != j:
                A_diag_sf[i, j] = sp.sparse.csc_matrix((A_list_sf[i].shape[0], A_list_sf[j].shape[1]))
                N_diag_sf[i, j] = sp.sparse.csc_matrix((N_list_sf[i].shape[0], N_list_sf[j].shape[1]))
                N_diag_sfT[i, j] = sp.sparse.csc_matrix((N_list_sf[i].shape[1], N_list_sf[j].shape[0]))

    # Detect repetition of structural and acoustic nodes: count repetitions and divide A_diag_sf accordingly
    # """Insert the code here"""
    # (Left intentionally as in your original code.)

    # Surface coupling block-list matrix (structure -> acoustic)
    C_list_sf = B_list_sT @ N_diag_sf @ A_diag_sf @ B_list_f

    # Convenience forms for vibro-acoustic coupling
    C_list_va_s = N_diag_sf @ A_diag_sf @ B_list_f
    C_list_va_f = A_diag_sf @ N_diag_sfT @ B_list_s

    return C_list_sf, B_list_s, B_list_f, N_diag_sf, A_diag_sf, C_list_va_s, C_list_va_f



#%% Importing/exporting data

def import_physical_matrices_from_full(folder_full, save_data=True, load_data=False):
    """
    PURPOSE:
        Read an ANSYS binary FULL file and extract the global stiffness (K), mass (M),
        and damping (D) matrices. The function also:
          - Symmetrizes the matrices returned by the solver
          - Builds a mapping matrix from solver (internal) DOF ordering to user DOF ordering
          - Applies the mapping to obtain K/M/D in user ordering
          - Builds a constraint reduction matrix to remove constrained/zero rows
          - Returns the reduced ("constrained space") matrices and the mapping matrices

    INPUTS:
        folder_full (str):
            Path to the folder containing "file.full".
            The FULL file is expected at:
                folder_full + "file.full"

    OUTPUTS:
        k_const (scipy.sparse.csc_matrix):
            Stiffness matrix reduced to the constrained DOF space.

        m_const (scipy.sparse.csc_matrix):
            Mass matrix reduced to the constrained DOF space.

        d_const (scipy.sparse.csc_matrix):
            Damping matrix reduced to the constrained DOF space.
            If solver damping is not present, returns a zero sparse matrix
            of the same shape as k_const.

        mapping_back (scipy.sparse.csc_matrix):
            Permutation/mapping matrix that maps solver internal DOF ordering to user DOF ordering.
            (Name kept from your code: "mapping_back".)

        constraint_mat (scipy.sparse.csc_matrix):
            Constraint reduction matrix selecting only DOFs whose rows in K are nonzero.
            It acts like a row/column selector:
                k_const = constraint_mat @ k @ constraint_mat.T

    SIDE EFFECTS:
        - Prints a folder identifier: folder_full.split('/')[-2]
        - Reads binary FULL file via pymapdl_reader

    NOTES:
        - Symmetrization:
            The solver may store only the upper triangle. The code mirrors it to build full symmetric matrices:
                A = A + triu(A,1).T
          This preserves the diagonal and fills the lower triangle.

        - Constraint suppression:
            Rows that are entirely zero in K are assumed to correspond to constrained DOFs
            (or DOFs eliminated in the solver). Those DOFs are removed by `constraint_mat`.

        - Mapping:
            dofref returned by full.load_km() encodes solver internal ↔ user ordering.
            `mapping_matrix(dofref)` builds a permutation matrix consistent with your earlier definition.

    EXAMPLES:
        >>> Kc, Mc, Dc, P, C = import_physical_matrices_from_full("/path/to/run/")
        >>> # Solve in constrained space:
        >>> # (Kc - w^2 Mc) x = f
    """

    # Print parent folder name (helpful for tracking which run is being loaded)
    # folder_full typically ends with ".../<case>/", so [-2] is "<case>"
    print(folder_full.split('/')[-2])
    
    # --- STEP 0: Load dictionary ---
    if load_data and os.path.isfile(os.path.join(folder_full, 'matrix_data.joblib')):
        data = joblib.load(folder_full + "matrix_data.joblib")
        k_const, m_const, d_const, mapping_back, constraint_mat = data['K'], data['M'], data['D'], data['mapping_back'], data['constraint_mat']
    
    else:
        # --- STEP 1: Read ANSYS FULL file ---
        # Reads "file.full" from the given folder.
        full = pymapdl_reader.read_binary(folder_full + "file.full")
    
        # Extract:
        #   dofref   : DOF reference map between solver ordering and user ordering
        #   k_solver : stiffness matrix in solver internal DOF ordering
        #   m_solver : mass matrix in solver internal DOF ordering
        #   d_solver : damping matrix in solver internal DOF ordering (may be None)
        dofref, k_solver, m_solver, d_solver = full.load_km()
    
        # --- STEP 2: Symmetrize matrices ---
        # The FULL file may store only the upper triangle. This reconstructs symmetry by mirroring.
        # triu(A,1) extracts strictly upper triangle; transpose places it in strictly lower triangle.
        k_solver += sp.sparse.triu(k_solver, 1).T
        m_solver += sp.sparse.triu(m_solver, 1).T
        if d_solver is not None:
            d_solver += sp.sparse.triu(d_solver, 1).T
    
        # --- STEP 3: Build mapping matrix from solver internal -> user ordering ---
        # dofref is used to build a permutation. In your conventions:
        #   mapping_back @ A_solver @ mapping_back.T -> A_user
        mapping_back = mapping_matrix(dofref)
    
        # --- STEP 4: Map solver matrices into user coordinates/order ---
        # These transformations reorder rows/cols consistently.
        # (Your inline comment is correct: this is a proper similarity transformation for reordering.)
        k = mapping_back @ k_solver @ mapping_back.T
        m = mapping_back @ m_solver @ mapping_back.T
        if d_solver is not None:
            d = mapping_back @ d_solver @ mapping_back.T
    
        # --- STEP 5: Build constraint reduction matrix ---
        # Identify DOFs whose stiffness rows are not completely zero.
        # k.getnnz(axis=1) == 0 means "row has no nonzeros".
        # We keep only rows where nnz != 0.
        diag = ~(k.getnnz(axis=1) == 0)
    
        # constraint_mat is a row-selector matrix built from identity:
        # shape: (n_kept, n_full)
        # selecting only rows where diag==True.
        constraint_mat = sp.sparse.eye(diag.shape[0], format='csc')[diag, :]
    
        # --- STEP 6: Transform matrices into constrained space ---
        # Apply the selector on both sides: remove constrained DOFs from rows and columns.
        k_const = constraint_mat @ k @ constraint_mat.T
        m_const = constraint_mat @ m @ constraint_mat.T
    
        if d_solver is not None:
            d_const = constraint_mat @ d @ constraint_mat.T
        else:
            # If no damping, return a zero matrix of appropriate shape
            d_const = sp.sparse.csc_matrix(k_const.shape)
        
        # ---- STEP 7: Save data in a joblib file
        if save_data:
            data = {'K': k_const, 'M': m_const, 'D': d_const,
                    'mapping_back': mapping_back, 'constraint_mat': constraint_mat}
            joblib.dump(data, folder_full + "matrix_data.joblib", compress=3)
    
    
    return k_const, m_const, d_const, mapping_back, constraint_mat


def import_nodal_coordinates_from_rst(
    folder, named_selection='', compute_normals=False,
    filt_node_ids=[], element_connectivity=False, load_data=False
):
    """
    PURPOSE:
        Read nodal coordinates from an ANSYS result file (.rst) using DPF and optionally:
          - restrict the output to a named selection
          - filter the resulting nodes by a user-provided list of node IDs
          - compute nodal normals and areas for the selected nodes
          - extract element connectivity in VTK UnstructuredGrid-style format

        This version also supports a lightweight job-file cache (`mesh_data.joblib`) to
        avoid reloading and re-parsing the `.rst` file when the same selection/options
        are requested repeatedly.

    INPUTS:
        folder (str):
            Path to folder containing "file.rst". The RST file must exist at:
                folder + "file.rst"

        named_selection (str, optional):
            If non-empty, only nodes in this named selection are returned.
            If the named selection does not exist, prints available selections.
            Default = '' (meaning: return all mesh nodes).

        compute_normals (bool, optional):
            If True, compute nodal normals and areas for the selected nodes using:
                nodal_normal(mesh, node_list_ids=..., filt_node_ids=...)
            Default = False.

        filt_node_ids (list | np.ndarray, optional):
            If provided (non-empty) and named_selection is used, further restrict the nodes
            to those whose IDs are in filt_node_ids.
            Also passed to nodal_normal to ensure output aligns with filt_node_ids if requested.
            Default = [].

        element_connectivity (bool, optional):
            If True, also extract connectivity arrays (cells, celltypes) suitable for VTK
            unstructured grids. Default = False.

        load_data (bool, optional):
            If True, attempt to serve the request purely from `mesh_data.joblib` cache,
            using `named_selection` as the cache key. If cache is missing or incomplete,
            fall back to parsing the `.rst` file and refresh the cache. Default = False.

    OUTPUTS:
        Depending on flags, returns one of:

        (1) element_connectivity == False and compute_normals == False:
            node_ids (np.ndarray), coords (np.ndarray)

        (2) element_connectivity == False and compute_normals == True:
            node_ids (np.ndarray), coords (np.ndarray), normals (np.ndarray), areas (np.ndarray)

        (3) element_connectivity == True and compute_normals == False:
            node_ids (np.ndarray), coords (np.ndarray), connectivity (tuple)

        (4) element_connectivity == True and compute_normals == True:
            node_ids (np.ndarray), coords (np.ndarray), normals (np.ndarray), areas (np.ndarray), connectivity (tuple)

        Where:
            node_ids: shape (N,)
            coords: shape (N,3)
            normals: shape (N,3)
            areas: shape (N,)
            connectivity: (cells, celltypes)
                - cells is a flattened VTK "cells array": [n0, i0, i1, ..., n1, j0, j1, ...]
                - celltypes is an array of VTK cell type integers

    SIDE EFFECTS:
        - Prints messages if named selection is not available.
        - Loads DPF model and mesh (can be time/memory heavy for large models).
        - Creates/updates `mesh_data.joblib` in `folder`.

    NOTES ON THE CACHE (`mesh_data.joblib`):
        - Cache key is exactly `named_selection` (including '' for the "full mesh" case).
        - The cache currently stores *one* entry per named_selection, but that entry may or
          may not include 'normals', 'areas', and 'connectivity' depending on what was
          requested when it was generated.
        - If `load_data=True` and the cache entry exists, the function returns cached data
          without reading the `.rst`.
        - If the cache entry exists but misses a requested field (e.g. you ask for
          connectivity but it was cached without it), the current logic will *not* detect
          partial cache completeness. (If you want, you can later harden this by checking
          for required keys before early-return.)

    """
    # =========================================================================
    # STEP 0) Load/initialize the cache container (mesh_data.joblib)
    # =========================================================================
    cache_path = os.path.join(folder, 'mesh_data.joblib')

    # If present, load the existing cache dict; otherwise initialize an empty cache.
    if os.path.isfile(cache_path):
        data = joblib.load(cache_path)
    else:
        data = {}

    # =========================================================================
    # STEP 1) Fast path: serve from cache if requested and available
    # =========================================================================
    # `load_data` means: "try to avoid reading the .rst and use the joblib cache instead".
    # We only consider the cache entry keyed by `named_selection`.
    loaded = False
    if load_data and data.get(named_selection):
        # Unpack cached payload according to the flags (connectivity / normals)
        if element_connectivity:
            if compute_normals:
                fields = ['node_ids', 'coords','normals','areas','connectivity']
                if np.all(np.array([type(data[named_selection].get(field)) for field in fields]) != type(None)):
                    node_ids = data[named_selection]['node_ids']
                    coords = data[named_selection]['coords']
                    normals = data[named_selection]['normals']
                    areas = data[named_selection]['areas']
                    connectivity = data[named_selection]['connectivity']
                    loaded=True
            else:
                fields = ['node_ids', 'coords','connectivity']
                if np.all(np.array([type(data[named_selection].get(field)) for field in fields]) != type(None)):
                    node_ids = data[named_selection]['node_ids']
                    coords = data[named_selection]['coords']
                    connectivity = data[named_selection]['connectivity']
                    loaded = True
        else:
            if compute_normals:
                fields = ['node_ids', 'coords','normals','areas']
                if np.all(np.array([type(data[named_selection].get(field)) for field in fields]) != type(None)):
                    node_ids = data[named_selection]['node_ids']
                    coords = data[named_selection]['coords']
                    normals = data[named_selection]['normals']
                    areas = data[named_selection]['areas']
                    loaded=True
            else:
                fields = ['node_ids', 'coords']
                if np.all(np.array([type(data[named_selection].get(field)) for field in fields]) != type(None)):
                    node_ids = data[named_selection]['node_ids']
                    coords = data[named_selection]['coords']
                    loaded=True

    if loaded is False:
        # =========================================================================
        # STEP 2) Slow path: read mesh from .rst via DPF
        # =========================================================================
        # This is the expensive step; everything below is derived from the DPF mesh.
        model = dpf.Model(folder + 'file.rst')
        mesh = model.metadata.meshed_region
        nodes = mesh.nodes

        # =========================================================================
        # STEP 3) Select nodes: either by named selection or entire mesh
        # =========================================================================
        if named_selection != '':
            # ---- 3A) Named selection path ----

            # Validate that selection exists; if not, print the available ones.
            if named_selection not in mesh.available_named_selections:
                print('The named selection ' + named_selection + ' is not available. The available selections are:')
                for ns in mesh.available_named_selections:
                    print('\t-' + ns)

            # Node IDs from the named selection scoping
            node_ids = np.array(mesh.named_selection(named_selection).ids)

            # All mesh node IDs and indices (used for mapping selection IDs -> coordinates)
            all_node_ids = np.array(nodes.scoping.ids)
            indices_all_ids = np.arange(0, all_node_ids.shape[0])

            # ---- 3B) Extract coordinates for the selection ----
            # We scan selection IDs and find matching nodes in the mesh node list.
            coords = []
            new_node_ids = []
            for i in range(0, len(node_ids)):
                position = all_node_ids == node_ids[i]
                if np.any(position):
                    index = indices_all_ids[position][0]
                    coords.append(nodes[index].coordinates)
                    new_node_ids.append(node_ids[i])

            node_ids = np.array(new_node_ids)
            coords = np.array(coords)

            # =========================================================================
            # STEP 4) Optional node-ID filter within the named selection
            # =========================================================================
            # `initial_node_ids` preserves the selection IDs for normal computation,
            # even if we later restrict nodes for the returned arrays.
            if len(filt_node_ids) != 0:
                initial_node_ids = (1.0 * node_ids).astype(int)

                # Keep only IDs explicitly listed in filt_node_ids
                is_filtered = np.array([True if node_id in filt_node_ids else False for node_id in node_ids])
                node_ids = node_ids[is_filtered]
                coords = coords[is_filtered]
            else:
                initial_node_ids = (1.0 * node_ids).astype(int)

            # =========================================================================
            # STEP 5) Optional normals/areas for the named selection
            # =========================================================================
            if compute_normals:
                # Compute normals/areas for the selection.
                # filt_node_ids is passed so nodal_normal can align output to the user filter.
                normals, areas = nodal_normal(mesh, node_list_ids=initial_node_ids, filt_node_ids=filt_node_ids)

        else:
            # ---- 3C) Full mesh path (no named selection) ----
            node_ids = np.zeros(len(mesh.nodes))
            coords = np.zeros((len(node_ids), 3))

            for i in range(0, len(nodes)):
                node = nodes[i]
                node_ids[i] = node.id
                coords[i, :] = node.coordinates

            # =========================================================================
            # STEP 6) Optional element connectivity (VTK-style) for full mesh
            # =========================================================================
            if element_connectivity:
                # Map ANSYS node IDs -> local (0..N-1) indices for VTK connectivity
                id_to_index = {nid: i for i, nid in enumerate(node_ids)}

                cells = []
                celltypes = []

                # Heuristic: VTK cell type inferred only from node count in the element.
                VTK_dict = {
                    1: 1,    # VTK_VERTEX
                    2: 3,    # VTK_LINE
                    3: 5,    # VTK_TRIANGLE
                    4: 10,   # VTK_QUAD (NOTE: could also be VTK_TETRA depending on topology)
                    5: 14,   # VTK_PYRAMID
                    6: 13,   # VTK_WEDGE
                    8: 12,   # VTK_HEXAHEDRON
                    10: 24,  # VTK_QUADRATIC_TETRA
                    13: 27,  # VTK_QUADRATIC_PYRAMID
                    15: 26,  # VTK_QUADRATIC_WEDGE
                    20: 25,  # VTK_QUADRATIC_HEXAHEDRON
                }

                # Encode each element as a VTK "cells array" segment
                for elem in mesh.elements:
                    node_ids_con = elem.node_ids

                    # Convert global ANSYS node IDs to local indices
                    indices = [id_to_index[nid] for nid in node_ids_con]

                    # VTK cells array format: [n_nodes, idx0, idx1, ...]
                    cells.append([len(indices)] + indices)

                    # VTK cell type inferred from node count
                    celltypes.append(VTK_dict[len(indices)])

                cells = np.hstack(cells).astype(np.int32)
                celltypes = np.array(celltypes)
                connectivity = (cells, celltypes)

        # =========================================================================
        # STEP 7) Update cache: store whatever was computed under this named_selection
        # =========================================================================
        # NOTE: key is `named_selection` ('' is valid and represents the full mesh case).
        if element_connectivity:
            if compute_normals:
                aux_dict = {
                   'node_ids': node_ids, 'coords': coords,
                   'normals': normals, 'areas': areas,
                   'connectivity': connectivity
                   }
            else:
                aux_dict = {
                    'node_ids': node_ids, 'coords': coords,
                    'connectivity': connectivity
                }

        else:
            if compute_normals:
                aux_dict = {
                    'node_ids': node_ids, 'coords': coords,
                    'normals': normals, 'areas': areas
                }

            else:
                aux_dict = {'node_ids': node_ids, 'coords': coords}
                
        # Update the fields maintaining extra variables if stored
        if data.get(named_selection):
            for key, val in aux_dict.items():
                data[named_selection][key] = val
        else:
            data[named_selection] = {key: val for key, val in aux_dict.items()}
                

        # Persist cache to disk (compress=3 is a good size/speed trade-off)
        joblib.dump(data, cache_path, compress=3)

    # =========================================================================
    # STEP 8) Return results in the exact format requested by the flags
    # =========================================================================
    if element_connectivity:
        if compute_normals:
            return node_ids, coords, normals, areas, connectivity
        else:
            return node_ids, coords, connectivity
    else:
        if compute_normals:
            return node_ids, coords, normals, areas
        else:
            return node_ids, coords


def load_solution_data(job_folder, job_name):
    """
    PURPOSE:
        Load the solution data stored in the job folder with the given job name.
        
    INPUTS:
        job_folder (str): route of the job folder
        job_name (str): name of the result job to load
    
    OUTPUTS:
        X (dict): dictionary containing the solution information
        COUP_info (dict): dictionary containing the coupling information
    
    
    """
    print(f'Loading SOLUTION_{job_name}')
    X = joblib.load(f'{job_folder}/SOLUTION_{job_name}')
    
    print(f'Loading COUP_INFO_{job_name}')
    Coup_info =  joblib.load(f'{job_folder}/COUP_INFO_{job_name}')
    
    return X, Coup_info

#%% Dynamic Substructuring, CMS and TPA functions
def build_ds_models_vibroacoustic(subdomains, interface_pairs, modal_CB_va=False, save_info=(False, '', '')):
    """
    PURPOSE:
        Build all data structures required for Dynamic Substructuring (DS) of vibro-acoustic systems.

        This function:
          1) Loads per-subdomain physical matrices (K, M, D) from ANSYS *.full files and nodal data from *.rst files.
          2) Applies solver/user DOF mappings and constraint reduction (via import_physical_matrices_from_full).
          3) Classifies interfaces into:
                - structure-structure (S-S),
                - acoustic-acoustic  (F-F),
                - structure-acoustic (F-S) vibro-acoustic couplings.
          4) Builds constraint matrices for each interface via `constraint_matrix(...)`:
                - NODE-NODE matching: standard continuity constraints,
                - MPC matching: creates extra "reference" components (star/spring MPC components),
                - NON-CONFORMAL: interpolation-based constraints (from constraint_matrix).
          5) Assembles localization matrices (Bii, Bbb, B lists) for structural and acoustic subsystems.
          6) Assembles vibro-acoustic coupling blocks (C_sf and related operators) using normals and areas.
          7) Optionally computes fixed-interface inverses of stiffness sub-blocks (for CMS / static condensation)
             using PARDISO through `inverse_sparse_matrix(..., method='pardiso')`.
          8) Returns:
                - `component_matrices`: per-component matrices and mesh info,
                - `coupling_info`: per-component localization operators + per-interface connectivity metadata.

    INPUTS:
        subdomains (dict):
            Maps each subdomain name (e.g., "S1", "F2") to:
                (folder, ndim, tol_rigid_modes, condition, density, load_dictionary)

            Where:
                folder (str): path to the run directory containing "file.full" and "file.rst"
                ndim (int): DOF per node for this domain (e.g., 3 for structure, 1 for acoustics)
                tol_rigid_modes (float): tolerance used later to classify rigid-body modes (stored but not used here)
                condition (str): e.g. "FIXED" to compute fixed-interface K inverse; anything else -> no inverse
                density (float): used ONLY for acoustic domains (keys starting with 'F') to normalize K/M/D
                load_dictionary (bool): used to load a dictionary contaiining K,M,D instead of reading the full file

        interface_pairs (dict):
            Dictionary keyed by interface identifier `idx` with entries of the form:
                (
                  (int1, int2),                # named selections in ANSYS (strings)
                  (sub1, sub2),                # subdomain names ("S1","S2") or ("F1","S1") etc.
                  (idx1, idx2),                # position indices (stored/unused in this function body)
                  (dof_list1, dof_list2),      # lists of DOF indices to couple (possibly empty)
                  (matching, atol),            # matching method and tolerance/pinball radius
                  connected                    # bool: False -> treat as disconnected interface
                )

            IMPORTANT:
                - For vibro-acoustic interfaces, the code assumes (sub1 starts with 'F', sub2 starts with 'S')
                  and uses that ordering to compute normals/areas on the acoustic side.

        modal_CB_va (bool, optional):
            If True, when computing FIXED-interface inverse for STRUCTURAL domains, the interior set is
            augmented with vibro-acoustic interface DOFs (Bva) before inversion.
            This is used when the Craig-Bampton (or similar) internal basis should "see" the VA DOFs.
            Default: False.

        save_info (tuple, optional):
            (save_dicts, folder_name, model_name)
                save_dicts (bool): if True, dump dictionaries with joblib
                folder_name (str): output folder
                model_name (str): suffix used for file names
            Default: (False, '', '')

    OUTPUTS:
        component_matrices (dict):
            Dictionary of per-component data stored in dictionaries:
                component_matrices['K'][comp]               -> sparse stiffness matrix
                component_matrices['M'][comp]               -> sparse mass matrix
                component_matrices['D'][comp]               -> sparse damping matrix
                component_matrices['constraint_mat'][comp]  -> constraint reduction matrix
                component_matrices['nodes'][comp]           -> node IDs (np.ndarray)
                component_matrices['coords'][comp]          -> node coordinates (N,3)
                component_matrices['nodal_normals'][comp]   -> area-weighted normals stored at mesh nodes (N,3)
                component_matrices['connectivity'][comp]    -> VTK connectivity (cells, celltypes)
                component_matrices['ndim'][comp]            -> ndim
                component_matrices['Kinv'][comp]            -> computed K inverse for fixed-interface, or [] if not computed
                component_matrices['shapes'][comp]          -> full DOF count before constraint reduction (constraint_mat.shape[1])
                component_matrices['constraint_nodes'][comp]-> per-node counter of how many interfaces constrain that node

        coupling_info (dict):
            A dictionary grouping all DS coupling operators:
                'Bii'   : per-component interior selector matrices
                'Bbb'   : per-component per-interface equilibrium/local boundary matrices (list form)
                'B'     : per-component per-interface continuity/local boundary matrices (list form)
                'Aco'   : per-component per-interface compatibility sign matrices (+I/-I per side)
                'Aeq'   : per-component per-interface equilibrium averaging matrices (0.5*I per side)
                'Bva'   : per-component per-VA-interface localization matrices
                'Csf'   : per-component coupling blocks between structural/acoustic domains
                'Cva'   : per-component per-VA-interface coupling blocks in VA reduced spaces
                'components'        : list of all component names (including created REF components)
                'interfaces'        : list of original interface keys
                'interface_shapes'  : dict of interface sizes (used later for block assembly)
                'connectivity'      : dict describing graph connectivity between components and through which interface keys

    NOTES / CAVEATS (kept as code-relevant facts):
        - MPC matching adds two new components per interface: "{sub}_REF-{idx}" for each side.
          These reference components contain spring-star elements and are appended to the model.
        - Nodal normals are computed ONLY for acoustic side of F-S interfaces (sub1).
          They are stored in the original acoustic component's nodal_normals array as area-weighted normals.
        - Constraint matrices are always applied through `constraint_mats[subX]` so B matrices operate
          in the constrained DOF space consistent with K/M/D.
        - Density normalization for acoustic domains: K, M, D are divided by density (as provided in subdomains).

    """

    # -------------------------------------------------------------------------
    # 0) Split components by type and split interfaces by type
    # -------------------------------------------------------------------------

    # Domain name list preserves input dict order
    domain_names = list(subdomains.keys())

    # Structural vs acoustic domains are identified by their key prefix
    domain_names_s = [domain_name for domain_name in domain_names if 'S' in domain_name]
    domain_names_f = [domain_name for domain_name in domain_names if 'F' in domain_name]

    num_domains_s = len(domain_names_s)

    # Classify interfaces into S-S, F-F, and F-S
    interface_keys_s, interface_keys_f, interface_keys_sf = [], [], []
    interface_keys = list(interface_pairs.keys())

    for key in interface_keys:
        sub1, sub2 = interface_pairs[key][1]  # (subdomain1, subdomain2)
        if sub1[0] == 'S' and sub2[0] == 'S':
            interface_keys_s.append(key)
        elif sub1[0] == 'F' and sub2[0] == 'F':
            interface_keys_f.append(key)
        elif (sub1[0] == 'F' and sub2[0] == 'S'):
            # For VA, the acoustic side is expected to be the first (F), structure second (S)
            interface_keys_sf.append(key)

    # -------------------------------------------------------------------------
    # 1) Load per-component matrices and mesh information
    # -------------------------------------------------------------------------

    component_matrices = {}     # temporary: list per component, later repacked into dict-of-dicts
    constraint_mats = {}        # constraint reduction matrices per component
    nodes_list = {}             # node IDs per component
    coords_list = {}            # node coords per component
    connectivity_list = {}      # VTK connectivity per component
    conditions = {}             # (condition, tol_rigid_modes) per component
    component_keys = {}         # mapping component name -> "column index" used in assembled lists
    idx_node_mapping = {}       # mapping component name -> dict(node_id -> local node index)

    idx0 = 0  # global counter to assign positions

    for key in domain_names:
        folder, ndim, tol_rigid_modes, condition, density, load_dictionary = subdomains[key]

        # Load K, M, D reduced by solver constraints (see import_physical_matrices_from_full)
        K, M, D, _, constraint_mat = import_physical_matrices_from_full(folder, load_data=load_dictionary)

        # Acoustic matrices are normalized by density (as provided by your model convention)
        if key[0] == 'F':
            K, M, D = K / density, M / density, D / density

        # Load nodes, coordinates, and element connectivity (VTK format)
        nodes, coords, nodal_connectivity = import_nodal_coordinates_from_rst(
            folder, '', element_connectivity=True, load_data=load_dictionary
        )

        # Initialize nodal normals and constraint counters
        # nodal_normals will be filled later only for acoustic sides of VA interfaces
        nodal_normals = np.zeros(coords.shape)
        constraint_nodes = np.zeros(nodes.shape)

        # Store temporary per-component pack (list form for mutability)
        component_matrices[key] = [
            K, M, D, constraint_mat, nodes, coords,
            nodal_normals, nodal_connectivity, ndim, constraint_nodes
        ]

        constraint_mats[key] = constraint_mat
        nodes_list[key] = nodes.astype(int)
        coords_list[key] = coords
        connectivity_list[key] = nodal_connectivity
        conditions[key] = (condition, tol_rigid_modes)

        # Map node_id -> index for fast writes later (normals and constraint counters)
        idx_node_mapping[key] = {nodes_list[key][i]: i for i in range(0, nodes.shape[0])}

        # component_keys is used to index into assembled vectors/matrices later.
        # Your convention: structural domains are indexed 0..(n_s-1), acoustic domains shifted by -n_s.
        # This produces a contiguous indexing inside each "S" and "F" subsystem list.
        if key in domain_names_f:
            component_keys[key] = idx0 - num_domains_s
        else:
            component_keys[key] = idx0
        idx0 += 1

        # Sanity check: constraint_mat columns should match ndim * n_nodes.
        # constraint_mat.shape[1] is the *full* DOF count before constraint reduction.
        # This check catches inconsistent partitioning or unexpected DOFs in exported matrices.
        if constraint_mat.shape[1] != ndim * nodes.shape[0]:
            if ndim * nodes.shape[0] > constraint_mat.shape[1]:
                print(f'ERROR: The model {key} has more nodes than expected. Check the division of the subdomains in ANSYS.')
                print(f'Number of expected nodes: {constraint_mat.shape[1]//ndim}, Number of nodes: {nodes.shape[0]}')
            elif ndim * nodes.shape[0] < constraint_mat.shape[1]:
                print(f'ERROR: The model {key} has less nodes than expected. Check the division of the subdomains in ANSYS.')

    # Reduced DOF sizes per component (rows of constraint_mat)
    shapes_red_s = [component_matrices[comp][3].shape[0] for comp in domain_names_s]
    shapes_red_f = [component_matrices[comp][3].shape[0] for comp in domain_names_f]

    # -------------------------------------------------------------------------
    # 2) Build interface constraint matrices: B and L dictionaries (struct, acoustic, VA)
    # -------------------------------------------------------------------------

    B_s_dict, L_s_dict = {}, {}
    B_f_dict, L_f_dict = {}, {}
    B_va_dict = {}

    # Dictionaries mapping interface key -> sequential index in assembled B_list arrays
    int_keys_bb_s = {-1: 0}   # initialize with dummy entry (-1) so downstream code can assume dict exists
    int_keys_bb_f = {-1: 0}
    interface_shapes = {-1: [0]}  # store interface DOF sizes (later used by assembly or diagnostics)
    int_keys_va = {}

    idx_bb_s, idx_bb_f, idx_va = (0, 0, 0)

    # connectivity graph: which component connects to which component and through which interface keys
    connectivity = {}

    for idx in interface_pairs.keys():
        (int1, int2), (sub1, sub2), (idx1, idx2), (dof_list1, dof_list2), (matching, atol), connected, load_dict_rst = interface_pairs[idx]

        # Retrieve interface node sets from named selections for both subdomains
        nodes_i, coords_i = import_nodal_coordinates_from_rst(subdomains[sub1][0], int1, load_data=load_dict_rst)
        nodes_j, coords_j = import_nodal_coordinates_from_rst(subdomains[sub2][0], int2, load_data=load_dict_rst)

        # If explicitly disconnected, wipe interface sets (so constraint_matrix returns empties)
        if connected is False:
            nodes_i, nodes_j = np.array([]), np.array([])
            coords_i, coords_j = np.empty((0, 3)), np.empty((0, 3))

        # Build local constraint matrices (B1, B2) and interior indicators (diag1, diag2)
        # IMPORTANT: constraint_matrices are applied inside constraint_matrix so B1/B2 operate in reduced DOFs.
        B1, B2, diag1, diag2, nodes_i, nodes_j = constraint_matrix(
            (nodes_list[sub1], coords_list[sub1], nodes_i, coords_i),
            (nodes_list[sub2], coords_list[sub2], nodes_j, coords_j),
            matching=matching,
            sparse=True,
            ndim=(subdomains[sub1][1], subdomains[sub2][1]),
            coupling_dof=(dof_list1, dof_list2),
            constraint_matrices=[constraint_mats[sub1], constraint_mats[sub2]],
            atol=atol
        )

        # In MPC case, B1 and B2 are tuples with additional info; we unpack later.
        print(f'Processing connection between {int1} and {int2}.')

        if matching == 'MPC':
            # Unpack MPC outputs:
            #   B?       : selects set nodes in original component
            #   ref_B?   : selects the created "contact/ref" nodes in the reference component
            #   MPC_B?   : selects the reference nodes for coupling between REF components
            #   Kref?    : stiffness matrix of the MPC star component (springs)
            #   coords_ref? : coordinates for the reference component
            #   av_mat?  : averaging matrix used later to average normals / areas
            #   nodal_conn? : VTK connectivity for the reference star (may be empty in your current generator)
            B1, ref_B1, MPC_B1, Kref1, coords_ref1, av_mat1, nodal_conn1 = B1
            B2, ref_B2, MPC_B2, Kref2, coords_ref2, av_mat2, nodal_conn2 = B2

            # Diagonals for REF components are initialized to zero -> all DOFs treated as "boundary" for DS purposes
            diag_ref1 = np.zeros(ref_B1.shape[1])
            diag_ref2 = np.zeros(ref_B2.shape[1])

            # Names for the newly created reference components
            ref_sub1, ref_sub2 = f'{sub1}_REF-{idx}', f'{sub2}_REF-{idx}'

            # The code uses synthetic interface keys for the extra DS couplings created by MPC:
            # idx01: original sub1 <-> ref_sub1
            # idx02: original sub2 <-> ref_sub2
            # idx03: ref_sub1 <-> ref_sub2
            idx01, idx02, idx03 = int(f'{idx}01'), int(f'{idx}02'), int(f'{idx}')

        # ---------------------------------------------------------------------
        # 2A) Vibro-acoustic interfaces (F-S)
        # ---------------------------------------------------------------------
        if idx in interface_keys_sf:

            if matching == 'MPC':
                """
                MPC on vibro-acoustic interface is handled as a two-stage coupling:

                Stage 1: Dynamic substructuring coupling between original domains and their REF components:
                    sub1 (acoustic) <-> ref_sub1 (acoustic REF)
                    sub2 (structure) <-> ref_sub2 (struct REF)

                Stage 2: Vibro-acoustic coupling between REF components:
                    ref_sub1 (acoustic REF) <-> ref_sub2 (struct REF)

                For Stage 2, nodal normals must be averaged over the cluster sets and areas summed so that the
                reference coupling has the physically correct net normal/area per reference point.
                """

                # Compute nodal normals and areas on acoustic side (sub1) for the interface nodes.
                # filt_node_ids=nodes_i makes nodal_normal return arrays aligned with nodes_i (including missing nodes as zeros)
                nodes_i2, coords_i, normals_i, areas_i = import_nodal_coordinates_from_rst(
                    subdomains[sub1][0], int1, compute_normals=True, filt_node_ids=nodes_i, load_data=load_dict_rst
                )

                # Reorder normals/areas to match the ordering used by constraint_matrix output nodes_i
                index_i = [np.where(nodes_i2 == node)[0][0] for node in nodes_i]
                nodes_i, coords_i, normals_i, areas_i = nodes_i2[index_i], coords_i[index_i], normals_i[index_i], areas_i[index_i]

                # Determine new indices for the added REF components inside each subsystem (F and S)
                keys_f = [el for el in list(component_keys.keys()) if el[0] == 'F']
                max_dom_f = max([component_keys[key] for key in keys_f])

                keys_s = [el for el in list(component_keys.keys()) if el[0] == 'S']
                max_dom_s = max([component_keys[key] for key in keys_s])

                # Assign indices to REF components
                component_keys[ref_sub1] = max_dom_f + 1
                component_keys[ref_sub2] = max_dom_s + 1

                # REF components: condition/tol are set to "almost rigid" placeholders
                conditions[ref_sub1] = ('', 1e-10)
                conditions[ref_sub2] = ('', 1e-10)

                # Append reduced shapes for REF components
                shape_ref1 = Kref1.shape[1]
                shape_ref2 = Kref2.shape[1]
                shapes_red_f.append(shape_ref1)
                shapes_red_s.append(shape_ref2)

                # Add REF components to the domain lists
                domain_names.append(ref_sub1)
                domain_names.append(ref_sub2)
                domain_names_f.append(ref_sub1)
                domain_names_s.append(ref_sub2)

                # Store area-weighted normals on the original acoustic component nodes:
                # component_matrices[sub1][6] is the per-node normal accumulator (N,3).
                component_normals = component_matrices[sub1][6]
                for i in range(0, len(nodes_i)):
                    if nodes_i[i] in idx_node_mapping[sub1]:
                        index_normal = idx_node_mapping[sub1][nodes_i[i]]
                        component_normals[index_normal, :] = areas_i[i] * normals_i[i, :]
                component_matrices[sub1][6] = component_normals

                # ---- Stage 1: DS coupling original -> REF (acoustic and structural separately)
                B_f_dict[idx01] = (B1, ref_B1, (component_keys[sub1] + 1, component_keys[ref_sub1] + 1))
                B_s_dict[idx02] = (B2, ref_B2, (component_keys[sub2] + 1, component_keys[ref_sub2] + 1))

                L_f_dict[idx01] = (diag1, diag_ref1, (component_keys[sub1] + 1, component_keys[ref_sub1] + 1))
                L_s_dict[idx02] = (diag2, diag_ref2, (component_keys[sub2] + 1, component_keys[ref_sub2] + 1))

                # ---- Stage 2: VA coupling REF <-> REF
                # Average normals within each cluster and sum areas:
                # av_mat1 is built so that each "reference point" is connected to a cluster of contact nodes.
                # normals_av, areas_av are per-reference-node quantities.
                normals_av = av_mat1.T @ normals_i
                normals_av = np.reshape(1.0 / np.linalg.norm(normals_av, axis=1), (normals_av.shape[0], 1)) * normals_av
                areas_av = av_mat1.T @ areas_i

                # For visualization/storage: create a nodal_normals field for REF1.
                nodal_normals_ref1 = np.zeros(coords_ref1.shape)
                nodal_normals_ref1[-areas_av.shape[0]:] = np.reshape(areas_av, (areas_av.shape[0], 1)) * normals_av

                # Build the normal matrix for structure side DOF dimension (ndim of sub2)
                N12 = normal_matrix(normals_av, ndim=subdomains[sub2][1])

                # Area diagonal matrix
                A12 = sp.sparse.diags(areas_av)

                # Store vibro-acoustic coupling:
                # Convention in your code: (B_struct, N, A, B_acoustic, (struct_idx, acoustic_idx))
                # Here, MPC_B2 belongs to the STRUCT REF side, MPC_B1 to the ACOUST REF side.
                B_va_dict[f'{idx}'] = (
                    np.abs(MPC_B2), N12, A12, np.abs(MPC_B1),
                    (component_keys[ref_sub2] + 1, component_keys[ref_sub1] + 1)
                )

                # For DS localization: REF components are treated as fully "boundary" here (diag_ref = zeros),
                # but you store them in L dict so that assembly_localization_matrix sees them.
                L_s_dict[f'{idx}'] = (diag_ref2, diag_ref2, (component_keys[ref_sub2] + 1, component_keys[ref_sub2] + 1))
                L_f_dict[f'{idx}'] = (diag_ref1, diag_ref1, (component_keys[ref_sub1] + 1, component_keys[ref_sub1] + 1))

                # Store interface sizes for later reference
                interface_shapes[idx01] = [B1.shape[0]]
                interface_shapes[idx02] = [B2.shape[0]]
                interface_shapes[idx] = [MPC_B2.shape[0], MPC_B1.shape[0]]

                # Remove dummy entries now that we have at least one real interface
                if -1 in int_keys_bb_s.keys():
                    del int_keys_bb_s[-1]
                if -1 in int_keys_bb_f.keys():
                    del int_keys_bb_f[-1]

                # Index bookkeeping for assembled lists
                int_keys_bb_f[idx01] = idx_bb_f
                int_keys_bb_s[idx02] = idx_bb_s
                int_keys_va[idx] = idx_va
                idx_bb_s += 1
                idx_bb_f += 1
                idx_va += 1

                # ---- Create and register REF component matrices
                # REF components are spring networks: Kref is nonzero, M/D are zeros, constraint is identity.
                # Node IDs are artificially created by offsetting with 1e6.
                # NOTE: the line `np.arange(component_matrices[sub1][4][0], ...)` is kept as-is;
                # it assumes `component_matrices[subX][4][0]` can be used as a seed node id.
                K_ref1 = Kref1.copy()
                M_ref1 = sp.sparse.csc_matrix(Kref1.shape)
                D_ref1 = sp.sparse.csc_matrix(Kref1.shape)
                cmat_ref1 = sp.sparse.eye(Kref1.shape[0], format='csc')

                ndim_ref1 = component_matrices[sub1][8]
                nodes_ref1 = 1e6 + np.arange(
                    component_matrices[sub1][4][0],
                    component_matrices[sub1][4][0] + shape_ref1 // ndim_ref1,
                    1
                )
                nodal_conn_ref1 = nodal_conn1
                constr_nod_ref1 = 2 * np.ones(nodes_ref1.shape[0])

                component_matrices[ref_sub1] = [
                    K_ref1, M_ref1, D_ref1, cmat_ref1,
                    nodes_ref1, coords_ref1, nodal_normals_ref1,
                    nodal_conn_ref1, ndim_ref1, constr_nod_ref1
                ]

                K_ref2 = Kref2.copy()
                M_ref2 = sp.sparse.csc_matrix(Kref2.shape)
                D_ref2 = sp.sparse.csc_matrix(Kref2.shape)
                cmat_ref2 = sp.sparse.eye(Kref2.shape[0], format='csc')

                ndim_ref2 = component_matrices[sub2][8]
                nodes_ref2 = 1e6 + np.arange(
                    component_matrices[sub2][4][0],
                    component_matrices[sub2][4][0] + shape_ref2 // ndim_ref2,
                    1
                )
                nodal_conn_ref2 = nodal_conn2
                constr_nod_ref2 = 2 * np.ones(nodes_ref2.shape[0])

                # NOTE: REF2 normals are zeros in your original code for VA MPC case.
                nodal_normals_ref2 = np.zeros((shape_ref2 // ndim_ref2, 3))

                component_matrices[ref_sub2] = [
                    K_ref2, M_ref2, D_ref2, cmat_ref2,
                    nodes_ref2, coords_ref2, nodal_normals_ref2,
                    nodal_conn_ref2, ndim_ref2, constr_nod_ref2
                ]

            else:
                # Standard NODE-NODE VA interface (no MPC):
                # compute normals/areas on acoustic side and build N and A.
                nodes_i, coords_i, normals_i, areas_i = import_nodal_coordinates_from_rst(
                    subdomains[sub1][0], int1, compute_normals=True, filt_node_ids=nodes_i, load_data=load_dict_rst
                )

                # Store area-weighted normals on acoustic component
                component_normals = component_matrices[sub1][6]
                for i in range(0, len(nodes_i)):
                    if nodes_i[i] in idx_node_mapping[sub1]:
                        index_normal = idx_node_mapping[sub1][nodes_i[i]]
                        component_normals[index_normal, :] = areas_i[i] * normals_i[i, :]
                component_matrices[sub1][6] = component_normals

                # Normal and area operators
                N12 = normal_matrix(normals_i, ndim=subdomains[sub2][1])
                A12 = sp.sparse.diags(areas_i)

                # Store VA coupling with your convention:
                # (B_struct, N, A, B_acoustic, (struct_idx, acoustic_idx))
                B_va_dict[f'{idx}'] = (
                    np.abs(B2), N12, A12, np.abs(B1),
                    (component_keys[sub2] + 1, component_keys[sub1] + 1)
                )

                # Store DS localization placeholders for VA interfaces
                L_s_dict[f'{idx}'] = (diag2, diag2, (component_keys[sub2] + 1, component_keys[sub2] + 1))
                L_f_dict[f'{idx}'] = (diag1, diag1, (component_keys[sub1] + 1, component_keys[sub1] + 1))

                interface_shapes[idx] = [B2.shape[0], B1.shape[0]]

                int_keys_va[idx] = idx_va
                idx_va += 1

        # ---------------------------------------------------------------------
        # 2B) Structural interfaces (S-S)
        # ---------------------------------------------------------------------
        elif idx in interface_keys_s:
            if matching == 'MPC':
                # Add two REF components on the structural subsystem and split coupling into 3 interactions:
                # sub1 <-> ref_sub1, sub2 <-> ref_sub2, ref_sub1 <-> ref_sub2 (MPC_B1/MPC_B2)

                keys_s = [el for el in list(component_keys.keys()) if el[0] == 'S']
                max_dom_s = max([component_keys[key] for key in keys_s])

                component_keys[ref_sub1] = max_dom_s + 1
                component_keys[ref_sub2] = max_dom_s + 2

                conditions[ref_sub1] = ('', 1e-10)
                conditions[ref_sub2] = ('', 1e-10)

                shape_ref1 = ref_B1.shape[1]
                shape_ref2 = ref_B2.shape[1]
                shapes_red_s.append(shape_ref1)
                shapes_red_s.append(shape_ref2)

                # Build REF matrices (spring stars)
                K_ref1 = Kref1.copy()
                M_ref1 = sp.sparse.csc_matrix(Kref1.shape)
                D_ref1 = sp.sparse.csc_matrix(Kref1.shape)
                cmat_ref1 = sp.sparse.eye(Kref1.shape[0], format='csc')
                ndim_ref1 = component_matrices[sub1][8]
                nodes_ref1 = component_matrices[sub1][4][:shape_ref1 // ndim_ref1] + 1e6
                nodal_normals_ref1 = np.zeros((shape_ref1 // ndim_ref1, 3))
                nodal_conn_ref1, constr_nod_ref1 = nodal_conn1, 2 * np.ones(nodes_ref1.shape[0])
                component_matrices[ref_sub1] = [
                    K_ref1, M_ref1, D_ref1, cmat_ref1,
                    nodes_ref1, coords_ref1, nodal_normals_ref1,
                    nodal_conn_ref1, ndim_ref1, constr_nod_ref1
                ]

                K_ref2 = Kref2.copy()
                M_ref2 = sp.sparse.csc_matrix(Kref2.shape)
                D_ref2 = sp.sparse.csc_matrix(Kref2.shape)
                cmat_ref2 = sp.sparse.eye(Kref2.shape[0], format='csc')
                ndim_ref2 = component_matrices[sub2][8]
                nodes_ref2 = component_matrices[sub2][4][:shape_ref2 // ndim_ref2] + 1e6
                nodal_normals_ref2 = np.zeros((shape_ref2 // ndim_ref2, 3))
                nodal_conn_ref2, constr_nod_ref2 = nodal_conn2, 2 * np.ones(nodes_ref2.shape[0])
                component_matrices[ref_sub2] = [
                    K_ref2, M_ref2, D_ref2, cmat_ref2,
                    nodes_ref2, coords_ref2, nodal_normals_ref2,
                    nodal_conn_ref2, ndim_ref2, constr_nod_ref2
                ]

                # Register REF components
                domain_names.append(ref_sub1)
                domain_names.append(ref_sub2)
                domain_names_s.append(ref_sub1)
                domain_names_s.append(ref_sub2)

                # Couplings (3 edges)
                B_s_dict[f'{idx01}'] = (B1, ref_B1, (component_keys[sub1] + 1, component_keys[ref_sub1] + 1))
                B_s_dict[f'{idx02}'] = (B2, ref_B2, (component_keys[sub2] + 1, component_keys[ref_sub2] + 1))
                B_s_dict[f'{idx03}'] = (MPC_B1, MPC_B2, (component_keys[ref_sub1] + 1, component_keys[ref_sub2] + 1))

                L_s_dict[f'{idx01}'] = (diag1, diag_ref1, (component_keys[sub1] + 1, component_keys[ref_sub1] + 1))
                L_s_dict[f'{idx02}'] = (diag2, diag_ref2, (component_keys[sub2] + 1, component_keys[ref_sub2] + 1))
                L_s_dict[f'{idx03}'] = (diag_ref1, diag_ref2, (component_keys[ref_sub1] + 1, component_keys[ref_sub2] + 1))

                interface_shapes[idx01] = [B1.shape[0]]
                interface_shapes[idx02] = [B2.shape[0]]
                interface_shapes[idx03] = [MPC_B1.shape[0]]

                if -1 in int_keys_bb_s.keys():
                    del int_keys_bb_s[-1]

                int_keys_bb_s[idx01] = idx_bb_s
                int_keys_bb_s[idx02] = idx_bb_s + 1
                int_keys_bb_s[idx03] = idx_bb_s + 2
                idx_bb_s += 3

            else:
                # Standard structure-structure interface
                B_s_dict[f'{idx}'] = (B1, B2, (component_keys[sub1] + 1, component_keys[sub2] + 1))
                L_s_dict[f'{idx}'] = (diag1, diag2, (component_keys[sub1] + 1, component_keys[sub2] + 1))
                interface_shapes[idx] = [B1.shape[0]]

                if -1 in int_keys_bb_s.keys():
                    del int_keys_bb_s[-1]

                int_keys_bb_s[idx] = idx_bb_s
                idx_bb_s += 1

        # ---------------------------------------------------------------------
        # 2C) Acoustic interfaces (F-F)
        # ---------------------------------------------------------------------
        elif idx in interface_keys_f:
            if matching == 'MPC':
                keys_f = [el for el in list(component_keys.keys()) if el[0] == 'F']
                max_dom_f = max([component_keys[key] for key in keys_f])

                component_keys[ref_sub1] = max_dom_f + 1
                component_keys[ref_sub2] = max_dom_f + 2

                conditions[ref_sub1] = ('', 1e-10)
                conditions[ref_sub2] = ('', 1e-10)

                shape_ref1 = ref_B1.shape[1]
                shape_ref2 = ref_B2.shape[1]
                shapes_red_f.append(shape_ref1)
                shapes_red_f.append(shape_ref2)

                # Build REF matrices
                K_ref1 = Kref1.copy()
                M_ref1 = sp.sparse.csc_matrix(Kref1.shape)
                D_ref1 = sp.sparse.csc_matrix(Kref1.shape)
                cmat_ref1 = sp.sparse.eye(Kref1.shape[0], format='csc')
                ndim_ref1 = component_matrices[sub1][8]
                nodes_ref1 = component_matrices[sub1][4][:shape_ref1 // ndim_ref1] + 1e6
                nodal_normals_ref1 = np.zeros((shape_ref1 // ndim_ref1, 3))
                nodal_conn_ref1, constr_nod_ref1 = nodal_conn1, 2 * np.ones(nodes_ref1.shape[0])
                component_matrices[ref_sub1] = [
                    K_ref1, M_ref1, D_ref1, cmat_ref1,
                    nodes_ref1, coords_ref1, nodal_normals_ref1,
                    nodal_conn_ref1, ndim_ref1, constr_nod_ref1
                ]

                K_ref2 = Kref2.copy()
                M_ref2 = sp.sparse.csc_matrix(Kref2.shape)
                D_ref2 = sp.sparse.csc_matrix(Kref2.shape)
                cmat_ref2 = sp.sparse.eye(Kref2.shape[0], format='csc')
                ndim_ref2 = component_matrices[sub2][8]
                nodes_ref2 = component_matrices[sub2][4][:shape_ref2 // ndim_ref2] + 1e6
                nodal_normals_ref2 = np.zeros((shape_ref2 // ndim_ref2, 3))
                nodal_conn_ref2, constr_nod_ref2 = nodal_conn2, 2 * np.ones(nodes_ref2.shape[0])
                component_matrices[ref_sub2] = [
                    K_ref2, M_ref2, D_ref2, cmat_ref2,
                    nodes_ref2, coords_ref2, nodal_normals_ref2,
                    nodal_conn_ref2, ndim_ref2, constr_nod_ref2
                ]

                domain_names.append(ref_sub1)
                domain_names.append(ref_sub2)
                domain_names_f.append(ref_sub1)
                domain_names_f.append(ref_sub2)

                B_f_dict[f'{idx01}'] = (B1, ref_B1, (component_keys[sub1] + 1, component_keys[ref_sub1] + 1))
                B_f_dict[f'{idx02}'] = (B2, ref_B2, (component_keys[sub2] + 1, component_keys[ref_sub2] + 1))
                B_f_dict[f'{idx03}'] = (MPC_B1, MPC_B2, (component_keys[ref_sub1] + 1, component_keys[ref_sub2] + 1))

                L_f_dict[f'{idx01}'] = (diag1, diag_ref1, (component_keys[sub1] + 1, component_keys[ref_sub1] + 1))
                L_f_dict[f'{idx02}'] = (diag2, diag_ref2, (component_keys[sub2] + 1, component_keys[ref_sub2] + 1))
                L_f_dict[f'{idx03}'] = (diag_ref1, diag_ref2, (component_keys[ref_sub1] + 1, component_keys[ref_sub2] + 1))

                interface_shapes[idx01] = [B1.shape[0]]
                interface_shapes[idx02] = [B2.shape[0]]
                interface_shapes[idx03] = [MPC_B1.shape[0]]

                if -1 in int_keys_bb_f.keys():
                    del int_keys_bb_f[-1]

                int_keys_bb_f[idx01] = idx_bb_f
                int_keys_bb_f[idx02] = idx_bb_f + 1
                int_keys_bb_f[idx03] = idx_bb_f + 2
                idx_bb_f += 3

            else:
                B_f_dict[f'{idx}'] = (B1, B2, (component_keys[sub1] + 1, component_keys[sub2] + 1))
                L_f_dict[f'{idx}'] = (diag1, diag2, (component_keys[sub1] + 1, component_keys[sub2] + 1))
                interface_shapes[idx] = [B1.shape[0]]

                if -1 in int_keys_bb_f.keys():
                    del int_keys_bb_f[-1]

                int_keys_bb_f[idx] = idx_bb_f
                idx_bb_f += 1

        # ---------------------------------------------------------------------
        # 2D) Connectivity graph bookkeeping + store "constraint_nodes" counters
        # ---------------------------------------------------------------------

        # Connectivity graph encodes the DS graph. MPC introduces REF nodes, so the graph differs.
        if matching != 'MPC':
            if connectivity.get(sub1) is None:
                connectivity[sub1] = {sub2: [idx]}
            else:
                connectivity[sub1].setdefault(sub2, []).append(idx)

            if connectivity.get(sub2) is None:
                connectivity[sub2] = {sub1: [idx]}
            else:
                connectivity[sub2].setdefault(sub1, []).append(idx)

        else:
            # Original -> its REF
            connectivity.setdefault(sub1, {}).setdefault(ref_sub1, []).append(idx01)
            connectivity.setdefault(sub2, {}).setdefault(ref_sub2, []).append(idx02)

            # REF nodes connect both to original and to the opposite REF
            connectivity[ref_sub1] = {sub1: [idx01], ref_sub2: [idx03]}
            connectivity[ref_sub2] = {sub2: [idx02], ref_sub1: [idx03]}

        # Store per-node constraint counts (how many interfaces touch each node)
        component_constr = component_matrices[sub1][9]
        for i in range(0, len(nodes_i)):
            if int(nodes_i[i]) in idx_node_mapping[sub1]:
                index = idx_node_mapping[sub1][int(nodes_i[i])]
                component_constr[index] += 1
        component_matrices[sub1][9] = component_constr

        component_constr = component_matrices[sub2][9]
        for i in range(0, len(nodes_j)):
            if int(nodes_j[i]) in idx_node_mapping[sub2]:
                index = idx_node_mapping[sub2][int(nodes_j[i])]
                component_constr[index] += 1
        component_matrices[sub2][9] = component_constr

    # -------------------------------------------------------------------------
    # 3) Assemble localization matrices for each subsystem
    # -------------------------------------------------------------------------

    # Structural subsystem localization matrices
    _, _, _, _, _, _, B_eq_vect_s, B_eq_list_s, Bii_vect_s, B_list_s, _ = assembly_localization_matrix(
        L_s_dict, B_s_dict, shapes_red_s
    )

    # Acoustic subsystem localization matrices
    _, _, _, _, _, _, B_eq_vect_f, B_eq_list_f, Bii_vect_f, B_list_f, _ = assembly_localization_matrix(
        L_f_dict, B_f_dict, shapes_red_f
    )

    # Vibro-acoustic coupling matrices (list/block form)
    C_list_sf, B_list_va_s, B_list_va_f, _, _, C_list_va_s, C_list_va_f = assembly_coupling_matrix(
        B_va_dict, (shapes_red_s, shapes_red_f)
    )

    # -------------------------------------------------------------------------
    # 4) Repack into per-component dictionaries and compute K inverses if requested
    # -------------------------------------------------------------------------

    Bii_dict_s, B_eq_dict_s, B_eq_list_dict_s, B_list_dict_s, B_list_dict_va_s = {}, {}, {}, {}, {}
    C_sf_dict_s, C_va_dict_s = {}, {}
    A_co_dict_s, A_eq_dict_s = {}, {}

    Bii_dict_f, B_eq_dict_f, B_eq_list_dict_f, B_list_dict_f, B_list_dict_va_f = {}, {}, {}, {}, {}
    C_sf_dict_f, C_va_dict_f = {}, {}
    A_co_dict_f, A_eq_dict_f = {}, {}

    K_comp, M_comp, D_comp = {}, {}, {}
    constraint_mat_comp, nodes_comp, coords_comp = {}, {}, {}
    nodal_normals_comp, connectivity_comp = {}, {}
    ndim_comp, const_nodes_comp, shapes_comp, K_inv_comp = {}, {}, {}, {}

    # Track whether an interface has already been paired for A_co sign convention (+I/-I on opposite side)
    paired_int_bb_s = {int_key: False for int_key in int_keys_bb_s.keys()}
    paired_int_bb_f = {int_key: False for int_key in int_keys_bb_f.keys()}

    for key in component_matrices.keys():
        K, M, D, constraint_mat, nodes, coords, nodal_normals, nodal_connectivity, ndim, constraint_nodes = component_matrices[key]
        condition, tol_rigid_modes = conditions[key]

        # ---------------------- Structural side dictionaries -------------------
        if key in domain_names_s:
            Bii_dict_s[key] = Bii_vect_s[component_keys[key]]
            B_eq_dict_s[key] = B_eq_vect_s[component_keys[key]]

            # Per-interface localized matrices stored as dict-of-dicts
            B_eq_list_dict_s[key] = {
                int_key: B_eq_list_s[int_keys_bb_s[int_key], component_keys[key]]
                for int_key in int_keys_bb_s.keys()
            }
            B_list_dict_s[key] = {
                int_key: B_list_s[int_keys_bb_s[int_key], component_keys[key]]
                for int_key in int_keys_bb_s.keys()
            }
            B_list_dict_va_s[key] = {
                int_key: B_list_va_s[int_keys_va[int_key], component_keys[key]]
                for int_key in int_keys_va.keys()
            }

            # Structural coupling blocks to each acoustic domain
            C_sf_dict_s[key] = {
                comp_key: C_list_sf[component_keys[key], component_keys[comp_key]]
                for comp_key in domain_names_f
            }

            # VA reduced-space blocks (acoustic->struct mapping in your naming)
            C_va_dict_f[key] = {
                int_key: C_list_va_f[int_keys_va[int_key], component_keys[key]]
                for int_key in int_keys_va.keys()
            }

            # Build A_co and A_eq matrices per interface:
            # - A_co: compatibility sign (+I on one side, -I on the paired side)
            # - A_eq: equilibrium averaging (0.5 I on each participating side)
            for int_key in int_keys_bb_s.keys():
                A_eq_dict_s.setdefault(key, {})[int_key] = None
                A_co_dict_s.setdefault(key, {})[int_key] = None

                pair_sign = -1 if paired_int_bb_s[int_key] else 1

                if len(B_eq_list_dict_s[key][int_key].data) != 0:
                    A_co_dict_s[key][int_key] = pair_sign * sp.sparse.eye(
                        B_eq_list_dict_s[key][int_key].shape[0], format='csc'
                    )
                    A_eq_dict_s[key][int_key] = 0.5 * sp.sparse.eye(
                        B_eq_list_dict_s[key][int_key].shape[0], format='csc'
                    )
                    paired_int_bb_s[int_key] = True
                else:
                    sh = B_eq_list_dict_s[key][int_key].shape[0]
                    A_co_dict_s[key][int_key] = sp.sparse.csc_matrix((sh, sh))
                    A_eq_dict_s[key][int_key] = sp.sparse.csc_matrix((sh, sh))

        # ---------------------- Acoustic side dictionaries ---------------------
        elif key in domain_names_f:
            Bii_dict_f[key] = Bii_vect_f[component_keys[key]]
            B_eq_dict_f[key] = B_eq_vect_f[component_keys[key]]

            B_eq_list_dict_f[key] = {
                int_key: B_eq_list_f[int_keys_bb_f[int_key], component_keys[key]]
                for int_key in int_keys_bb_f.keys()
            }
            B_list_dict_f[key] = {
                int_key: B_list_f[int_keys_bb_f[int_key], component_keys[key]]
                for int_key in int_keys_bb_f.keys()
            }
            B_list_dict_va_f[key] = {
                int_key: B_list_va_f[int_keys_va[int_key], component_keys[key]]
                for int_key in int_keys_va.keys()
            }

            # Acoustic coupling blocks to each structural domain
            C_sf_dict_f[key] = {
                comp_key: C_list_sf[component_keys[comp_key], component_keys[key]]
                for comp_key in domain_names_s
            }

            # VA reduced-space blocks (struct->acoustic mapping in your naming)
            C_va_dict_s[key] = {
                int_key: C_list_va_s[int_keys_va[int_key], component_keys[key]]
                for int_key in int_keys_va.keys()
            }

            for int_key in int_keys_bb_f.keys():
                A_eq_dict_f.setdefault(key, {})[int_key] = None
                A_co_dict_f.setdefault(key, {})[int_key] = None

                pair_sign = -1 if paired_int_bb_f[int_key] else 1

                if len(B_eq_list_dict_f[key][int_key].data) != 0:
                    A_co_dict_f[key][int_key] = pair_sign * sp.sparse.eye(
                        B_eq_list_dict_f[key][int_key].shape[0], format='csc'
                    )
                    A_eq_dict_f[key][int_key] = 0.5 * sp.sparse.eye(
                        B_eq_list_dict_f[key][int_key].shape[0], format='csc'
                    )
                    paired_int_bb_f[int_key] = True
                else:
                    sh = B_eq_list_dict_f[key][int_key].shape[0]
                    A_co_dict_f[key][int_key] = sp.sparse.csc_matrix((sh, sh))
                    A_eq_dict_f[key][int_key] = sp.sparse.csc_matrix((sh, sh))

        # ---------------------- FIXED-interface inverse of K -------------------
        if condition == 'FIXED':
            print(f'Computing the {condition}-INTERFACE inverse of K_{key}.')

            if key in domain_names_s:
                if modal_CB_va:
                    # Build an "interior + VA-boundary" selector:
                    #   Biiva = [Bii; Bva] where Bva stacks all VA interface selectors.
                    Bva_vect_s = sp.sparse.bmat([[B_list_dict_va_s[key][intr]] for intr in B_list_dict_va_s[key].keys()])
                    Biiva_vect_s = sp.sparse.bmat([[Bii_dict_s[key]],
                                                   [Bva_vect_s]], format='csc')

                    # Extract the submatrix on that reduced set
                    Kiiva = Biiva_vect_s @ K @ Biiva_vect_s.T

                    # Remove zero columns (fully disconnected DOFs within Kiiva)
                    cmat = sp.sparse.eye(Kiiva.shape[0], format='csc')[Kiiva.getnnz(axis=0) != 0, :]

                    Kiiva = cmat @ Kiiva @ cmat.T

                    # If nothing to invert or fully disconnected, return zeros
                    if Kiiva.shape[0] == K.shape[0] or Kiiva.shape[0] == 0:
                        K_inv = cmat.T @ sp.sparse.csc_matrix(Kiiva.shape) @ cmat
                    else:
                        K_inv = cmat.T @ sp.sparse.linalg.inv(Kiiva) @ cmat
                else:
                    # Standard fixed-interface inverse: invert interior stiffness Kii
                    Kii = Bii_dict_s[key] @ K @ Bii_dict_s[key].T

                    if Kii.shape[0] == K.shape[0] or Kii.shape[0] == 0:
                        K_inv = sp.sparse.csc_matrix(Kii.shape)
                    else:
                        K_inv = inverse_sparse_matrix(Kii, method='pardiso')

            elif key in domain_names_f:
                Kii = Bii_dict_f[key] @ K @ Bii_dict_f[key].T

                if Kii.shape[0] == K.shape[0] or Kii.shape[0] == 0:
                    K_inv = sp.sparse.csc_matrix(Kii.shape)
                else:
                    K_inv = inverse_sparse_matrix(Kii, method='pardiso')

        else:
            # If not fixed-interface, K_inv is not computed here
            K_inv = []

        # Store repacked dictionaries
        K_comp[key] = K
        M_comp[key] = M
        D_comp[key] = D
        constraint_mat_comp[key] = constraint_mat
        nodes_comp[key] = nodes
        coords_comp[key] = coords
        nodal_normals_comp[key] = nodal_normals
        connectivity_comp[key] = nodal_connectivity
        ndim_comp[key] = ndim
        const_nodes_comp[key] = constraint_nodes
        shapes_comp[key] = constraint_mat.shape[1]
        K_inv_comp[key] = K_inv

    # -------------------------------------------------------------------------
    # 5) Pack final outputs
    # -------------------------------------------------------------------------
    coupling_info = {
        'Bii': {**Bii_dict_s, **Bii_dict_f},
        'Bbb': {**B_eq_list_dict_s, **B_eq_list_dict_f},
        'B': {**B_list_dict_s, **B_list_dict_f},
        'Aco': {**A_co_dict_s, **A_co_dict_f},
        'Aeq': {**A_eq_dict_s, **A_eq_dict_f},
        'Bva': {**B_list_dict_va_s, **B_list_dict_va_f},
        'Csf': {**C_sf_dict_s, **C_sf_dict_f},
        'Cva': {**C_va_dict_s, **C_va_dict_f},
        'components': domain_names,
        'interfaces': interface_keys,
        'interface_shapes': interface_shapes,
        'connectivity': connectivity
    }

    component_matrices = {
        'K': K_comp, 'M': M_comp, 'D': D_comp,
        'constraint_mat': constraint_mat_comp,
        'nodes': nodes_comp,
        'coords': coords_comp,
        'nodal_normals': nodal_normals_comp,
        'connectivity': connectivity_comp,
        'ndim': ndim_comp,
        'Kinv': K_inv_comp,
        'shapes': shapes_comp,
        'constraint_nodes': const_nodes_comp
    }

    # Optional persistence to disk
    save_dicts, folder_name, model_name = save_info
    if save_dicts is True:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        joblib.dump(component_matrices, folder_name + 'COMP_MAT_' + model_name, compress=3)
        joblib.dump(coupling_info, folder_name + 'MODEL_INFO_' + model_name, compress=3)

    return component_matrices, coupling_info


def TPA_matrices_vibroacoustic(component_matrices, coupling_info, level_components,
    level_interfaces, CMS_method=['', 0], modal_CB_va=False,  save_job_info=(False, '', ''),
    assembly_modes=[False, 0]):
    """
    PURPOSE
    -------
    Build level-wise Transfer Path Analysis (TPA) matrices for coupled vibro-acoustic systems using
    Dynamic Substructuring (DS) with optional Component Mode Synthesis (CMS) reduction.
    
    The function assembles, for each requested TPA level and for each prescribed interface at that level,
    a reduced set of equations of motion (EOM) and the corresponding prescription operators. It supports
    both structural (S) and acoustic/fluid (F) components and their DS interfaces (b0/bb) as well as
    vibro-acoustic (VA) coupling interfaces.
    
    OVERVIEW OF THE WORKFLOW
    ------------------------
    1) Unpack per-component physical matrices and mesh data from `component_matrices`.
    2) Unpack DS localization operators and coupling blocks from `coupling_info`:
         - Interior selectors (Bii),
         - Boundary/interface selectors (Bbb/B),
         - Interface equilibrium/compatibility operators (Aeq/Aco),
         - VA interface selectors (Bva) and VA coupling blocks (Csf/Cva),
         - Interface sizes and connectivity graph.
    3) Precompute modal bases (once per component) depending on CMS method:
         - FIXED-interface methods: compute fixed-interface internal modes Φ_ii and constraint modes Ψ.
         - FREE-interface methods: compute free-interface modes Φ and attachment/dual modes Π (and derived Ψ).
    4) For each TPA level:
         - Determine passive components (in the level), active components (outside), and active interfaces.
         - Assemble block-diagonal component matrices (K, M, D) and coupling matrices for that level.
         - Build global selection matrices for interior, DS interfaces, and VA interfaces.
         - Build primal or dual assembly operators and construct the reduction/assembly transform T_red.
    5) For each prescribed interface at the level:
         - Split interface DOFs into retained (k) and prescribed (j) sets.
         - Project the EOM with T_red and substitute the prescribed DOF contribution consistently.
         - Optionally compute assembly modes and further reduce the assembled matrices.
         - Store:
             * Reduced matrices (Mat_red[level][int_j]) for solving,
             * Prescription matrices (Mat_pres[level][int_j]) for imposed interface motion/loads,
             * Detailed coupling/selection metadata (Coup_info[level][int_j]) for post-processing/TPA.
    
    SUPPORTED REDUCTION / ASSEMBLY METHODS (CMS_method[0])
    ------------------------------------------------------
    The implementation follows two orthogonal choices:
      - Formulation: PRIMAL vs DUAL assembly
      - Modal basis: FIXED-interface vs FREE-interface component bases
    
    PRIMAL formulation:
      FIXED-interface:
        - 'CBM'   : Craig–Bampton (internal fixed-interface modes + constraint modes)
        - 'CCBM'  : Condensed Craig–Bampton (static condensation of retained DS DOFs)
      FREE-interface:
        - 'RM'    : Rubin method (free modes corrected by attachment modes)
        - 'MNM'   : MacNeal method (static condensation of retained DS DOFs)
    
    DUAL formulation:
      FIXED-interface:
        - 'FDCBM' : Fixed-Dual Craig–Bampton
        - 'CFDCBM': Condensed Fixed-Dual Craig–Bampton (static condensation)
      FREE-interface:
        - 'DCBM'  : Dual Craig–Bampton
        - 'CDCBM' : Condensed Dual Craig–Bampton (static condensation)
    
    IMPORTANT CONVENTIONS / DEFINITIONS
    -----------------------------------
    - Components are identified by their names: structural components start with 'S', acoustic/fluid
      components start with 'F'. Reference components created upstream (e.g. MPC) are also supported.
    - DS interfaces ("bb"/"b0") are those handled by Bbb/Aeq/Aco; VA interfaces are those handled by Bva
      (and Csf/Cva blocks).
    - For each level, "active components" are components not included in the level that are connected to
      the level through the connectivity graph; interfaces connecting to those components are treated as
      active interfaces and excluded from the retained/prescribed split used for TPA.
    - The function returns both assembled/reduced matrices and detailed operator dictionaries so the same
      assembly can be reused for response synthesis, TPA metrics, and visualization.
    
    INPUTS (summary)
    ---------------
    component_matrices : dict
        Output of `build_ds_models_vibroacoustic(...)`, containing per-component K/M/D, constraint matrices,
        mesh information, and other metadata.
    coupling_info : dict
        Output of `build_ds_models_vibroacoustic(...)`, containing DS/VA localization operators and coupling
        blocks (Bii, Bbb, B, Aco, Aeq, Bva, Csf, Cva, interface_shapes, connectivity).
    level_components : dict
        Mapping level_name -> list of component tags participating in that level (reference components are
        automatically included if their base tag is present).
    level_interfaces : dict
        Mapping level_name -> list of interface keys to be treated as "studied / prescribed" at that level.
    CMS_method : list/tuple, default ['', 0]
        (method_name, n_modes_dict) where n_modes_dict maps component_name -> number of modes used for that component
        in modal-based reductions.
    modal_CB_va : bool, optional
        Flag controlling whether structural modal bases are extended/used for vibro-acoustic coupling in methods
        where that distinction is relevant.
    save_job_info : tuple, optional
        (save_job, job_folder, job_name) to persist Mat_red/Mat_pres/Coup_info to disk via joblib.
    assembly_modes : list/tuple, optional
        (enabled, n_modes) if True, compute and apply additional assembly-level modal truncation.
    
    OUTPUTS (summary)
    ----------------
    Mat_red : dict
        Per level and prescribed interface, reduced assembled matrices {'K','M','D'}.
    Mat_pres : dict
        Per level and prescribed interface, prescription matrices {'K','M','D'} mapping prescribed DOFs into
        the reduced system.
    Coup_info : dict
        Per level and prescribed interface, detailed coupling/selection operators, reduction matrix T_red,
        mesh metadata, and indexing helpers for post-processing.
    time_RAM : dict
        Aggregated timing, peak RAM and sparsity statistics for the major stages (modal bases, T_red, EOM projection).
    """

    # -------------------------------------------------------------------------
    # 0) Unpack component and coupling dictionaries
    # -------------------------------------------------------------------------

    # Per-component matrices (dict: comp -> sparse matrix)
    K = component_matrices['K']
    M = component_matrices['M']
    D = component_matrices['D']
    Cmat = component_matrices['constraint_mat']  # DOF constraint matrices (BC reduction operators)

    # Per-component mesh/meta
    nodes = component_matrices['nodes']
    coords = component_matrices['coords']
    nodal_normals = component_matrices['nodal_normals']
    ndim = component_matrices['ndim']
    shapes = component_matrices['shapes']
    constraint_nodes = component_matrices['constraint_nodes']

    # DS localization / coupling operators
    Bii_dict = coupling_info['Bii']
    Bbb_dict = coupling_info['Bbb']
    Aco_dict = coupling_info['Aco']
    Aeq_dict = coupling_info['Aeq']
    Bva_dict = coupling_info['Bva']
    Csf_dict = coupling_info['Csf']
    Cva_dict = coupling_info['Cva']

    component_names = coupling_info['components']
    interface_shapes = coupling_info['interface_shapes']
    connectivity = coupling_info['connectivity']

    # Reduction method + modes per component
    red_method, n_modes = CMS_method

    # Component lists by physics
    all_comp_s = [comp for comp in component_names if comp[0] == 'S']
    all_comp_f = [comp for comp in component_names if comp[0] == 'F']

    # Method families
    primal_methods = ['PRIMAL', 'CBM', 'RM', 'MNM', 'CCBM']
    dual_methods = ['DUAL', 'FDCBM', 'DCBM', 'CDCBM', 'CFDCBM']
    fixed_methods = ['CBM', 'FDCBM', 'CCBM', 'CFDCBM']
    free_methods = ['RM', 'MNM', 'DCBM', 'CDCBM']

    # Outputs
    Mat_red = {}   # reduced assembled matrices per level and per studied interface
    Mat_pres = {}  # prescription matrices per level and per studied interface
    Coup_info = {} # coupling operators per level and per studied interface

    print(f"Computing TPA matrices using the {red_method} reduction method.")

    # Time/RAM bookkeeping
    time_RAM = {
        'time': {'Phi': 0, 'Psi': 0, 'Tred': 0, 'EOM': 0, 'TOT_assembly': 0},
        'RAM':  {'Phi': 0, 'Psi': 0, 'Tred': 0, 'EOM': 0, 'TOT_assembly': 0},
        'sparsity': {'ratio': 0, 'nrows': 0},
    }

    # -------------------------------------------------------------------------
    # 1) Precompute component-level modal bases and attachment/constraint modes
    #    depending on the chosen DS/CMS family.
    # -------------------------------------------------------------------------

    if red_method in fixed_methods:
        # Fixed-interface family: compute internal fixed-interface modes Phi_ii and constraint modes Psi
        Phi_ii_dict = {}
        Psi_ib0_dict = {}
        Psi_iva_dict = {}

        for comp in component_names:
            print(f"Computing FIXED-INTERFACE modes of {comp}.")
            thread_monitoring(start=True)

            # Internal (i) sub-blocks in reduced coordinates
            Kii = Bii_dict[comp] @ K[comp] @ Bii_dict[comp].T
            Mii = Bii_dict[comp] @ M[comp] @ Bii_dict[comp].T

            # Fixed-interface modal basis (internal modes)
            freq_n, Phi_ii_dict[comp], _ = compute_modal_basis(
                Kii, Mii, sparse=True, n_modes=n_modes[comp], freq0=0.0
            )
            time_Phi, RAM_Phi = thread_monitoring(start=False, print_res=False)

            print(f"Computing CONSTRAINT modes of {comp}.")
            thread_monitoring(start=True)

            # Constraint modes for DS interfaces (b0) and for vibro-acoustic (va)
            # compute_constraint_modes returns:
            #   Psi_ib0_dict[comp] : dict per interface key of constraint modes
            #   LUKii              : LU factorization caching object (reused for va)
            Psi_ib0_dict[comp], LUKii = compute_constraint_modes(
                K[comp], Bii_dict[comp], Bbb_dict[comp]
            )
            Psi_iva_dict[comp], _ = compute_constraint_modes(
                K[comp], Bii_dict[comp], Bva_dict[comp], LUKii=LUKii, va=True
            )
            time_Psi, RAM_Psi = thread_monitoring(start=False, print_res=False)

            print(f"The maximum NATURAL FREQUENCY of {comp} is {freq_n.max():.2f} Hz.\n")

            time_RAM['time']['Phi'] += time_Phi
            time_RAM['time']['Psi'] += time_Psi
            time_RAM['RAM']['Phi'] = np.max([time_RAM['RAM']['Phi'], RAM_Phi])
            time_RAM['RAM']['Psi'] = np.max([time_RAM['RAM']['Psi'], RAM_Psi])

    elif red_method in free_methods:
        # Free-interface family: compute free modes Phi and attachment modes Pi (and optionally Kr terms)
        Phi_dict = {}
        Pi_b0_dict = {}
        Pi_va_dict = {}
        Kr_bb0_dict = {}
        Kr_va_dict = {}

        for comp in component_names:
            print(f"Computing FREE-INTERFACE modes of {comp}.")
            thread_monitoring(start=True)

            freq_n, Phi_dict[comp], _ = compute_modal_basis(
                K[comp], M[comp], sparse=True, n_modes=n_modes[comp], freq0=0.0
            )
            time_Phi, RAM_Phi = thread_monitoring(start=False, print_res=False)

            print(f"Computing ATTACHMENT modes of {comp}.")
            thread_monitoring(start=True)

            # Some methods require computing Kr_bb0 terms (primal side)
            compute_Kr_bb0 = True if red_method in primal_methods else False

            # Attachment modes on DS boundaries (b0)
            Pi_b0_dict[comp], Kr_bb0_dict[comp], LUK_nnz = compute_attachment_modes(
                K[comp],
                Bbb_dict[comp],
                Phi_dict[comp].astype(float),
                (2 * np.pi * freq_n.real.astype(float)) ** 2,
                LUK_nnz=[],
                compute_Kr_bb0=compute_Kr_bb0,
            )

            # Attachment modes on VA boundaries (va)
            Pi_va_dict[comp], Kr_va_dict[comp], _ = compute_attachment_modes(
                K[comp],
                Bva_dict[comp],
                Phi_dict[comp],
                (2 * np.pi * freq_n) ** 2,
                LUK_nnz=LUK_nnz,
                compute_Kr_bb0=compute_Kr_bb0,
            )

            time_Psi, RAM_Psi = thread_monitoring(start=False, print_res=False)

            print(f"The maximum NATURAL FREQUENCY of {comp} is {freq_n.max():.2f} Hz.\n")

            time_RAM['time']['Phi'] += time_Phi
            time_RAM['time']['Psi'] += time_Psi
            time_RAM['RAM']['Phi'] = np.max([time_RAM['RAM']['Phi'], RAM_Phi])
            time_RAM['RAM']['Psi'] = np.max([time_RAM['RAM']['Psi'], RAM_Psi])

    # -------------------------------------------------------------------------
    # 2) Expand level_components to include REF components (created by MPC).
    #    The matching rule used here is: comp.split('_')[0] must be in the user list.
    # -------------------------------------------------------------------------

    for level, level_comp in level_components.items():
        new_level_comp = []
        for comp in component_names:
            if comp.split('_')[0] in level_comp:
                new_level_comp.append(comp)
        level_components[level] = new_level_comp  # update in-place

    # -------------------------------------------------------------------------
    # 3) Process each TPA level: build assembled operators, build reduction T_red,
    #    project EOM, build prescription matrices, store coupling metadata.
    # -------------------------------------------------------------------------

    for level, level_comp in level_components.items():
        thread_monitoring(start=True)  # timing for T_red assembly

        # Split level components by physics (and keep ordering S then F)
        level_comp_s = [comp for comp in level_comp if comp in all_comp_s]
        level_comp_f = [comp for comp in level_comp if comp in all_comp_f]
        level_comp = level_comp_s + level_comp_f

        # ---------------------------------------------------------------------
        # 3A) Determine which DS/VA interfaces exist within this level
        #      (only interfaces with nonzero data are retained).
        # ---------------------------------------------------------------------

        # DS boundary interfaces in structural components
        interfaces_bb_s = []
        for comp_s in level_comp_s:
            for key in Bbb_dict[comp_s].keys():
                if len(Bbb_dict[comp_s][key].data) != 0:
                    interfaces_bb_s.append(key)
        interfaces_bb_s = list(np.unique(interfaces_bb_s))

        # DS boundary interfaces in acoustic components
        interfaces_bb_f = []
        for comp_f in level_comp_f:
            for key in Bbb_dict[comp_f].keys():
                if len(Bbb_dict[comp_f][key].data) != 0:
                    interfaces_bb_f.append(key)
        interfaces_bb_f = list(np.unique(interfaces_bb_f))

        # Vibro-acoustic interfaces present in any component of this level
        interfaces_va = []
        for comp in level_comp:
            for key in Bva_dict[comp].keys():
                if len(Bva_dict[comp][key].data) != 0:
                    interfaces_va.append(key)
        interfaces_va = list(np.unique(interfaces_va))

        # All interfaces involved by this level
        interfaces = interfaces_bb_s + interfaces_bb_f + interfaces_va

        # ---------------------------------------------------------------------
        # 3B) Determine active components and active interfaces
        #      active_comp = all components present in connectivity graph but not in this level
        #      int_active  = interfaces that connect active->passive across the level boundary
        # ---------------------------------------------------------------------

        active_comp = [comp for comp in connectivity.keys() if comp not in level_comp]

        int_active = [
            np.array(connectivity[comp_ac][comp_pas])
            for comp_ac in active_comp
            for comp_pas in level_comp
            if comp_pas in connectivity[comp_ac].keys()
        ]
        int_active = list(np.concatenate(int_active)) if len(int_active) != 0 else []

        # ---------------------------------------------------------------------
        # 3C) Build selection matrices for the level:
        #      - Bii: interior selector
        #      - Bbb: DS boundary selector (stack interfaces then block-diagonal by component)
        #      - Aeq/Aco: equilibrium/compatibility assembly operators
        #      - Beq/B: assembled equilibrium/constraint matrices
        #      - Bva: VA interface selector
        # ---------------------------------------------------------------------

        # Interior selectors
        Bii_s = sp.sparse.block_diag([Bii_dict[comp] for comp in level_comp_s]) if level_comp_s else sp.sparse.csc_matrix((0, 0))
        Bii_f = sp.sparse.block_diag([Bii_dict[comp] for comp in level_comp_f]) if level_comp_f else sp.sparse.csc_matrix((0, 0))
        Bii = sp.sparse.block_diag([Bii_s, Bii_f], format='csc')

        # ---- DS boundary selector Bbb
        # 1) stack per-component interface blocks vertically (in the component’s local “Bbb_dict order”)
        Bbb_vect_s = {
            comp: sp.sparse.bmat([
                [Bbb_dict[comp][intr]] if len(Bbb_dict[comp][intr].data) != 0
                else [sp.sparse.csc_matrix((0, Bbb_dict[comp][intr].shape[1]))]
                for intr in Bbb_dict[comp].keys()
            ])
            for comp in level_comp_s
        }
        Bbb_vect_f = {
            comp: sp.sparse.bmat([
                [Bbb_dict[comp][intr]] if len(Bbb_dict[comp][intr].data) != 0
                else [sp.sparse.csc_matrix((0, Bbb_dict[comp][intr].shape[1]))]
                for intr in Bbb_dict[comp].keys()
            ])
            for comp in level_comp_f
        }

        # 2) build block-diagonal over components
        Bbb_s = sp.sparse.block_diag([Bbb_vect_s[comp] for comp in level_comp_s], format='csc') if level_comp_s else sp.sparse.csc_matrix((0, 0))
        Bbb_f = sp.sparse.block_diag([Bbb_vect_f[comp] for comp in level_comp_f], format='csc') if level_comp_f else sp.sparse.csc_matrix((0, 0))
        Bbb = sp.sparse.block_diag([Bbb_s, Bbb_f], format='csc')

        # 3) identify which interface “label” each stacked block-column corresponds to (NaN for empty blocks)
        col_ids_Bbb_s = np.concatenate([
            [intr if len(Bbb_dict[comp][intr].data) != 0 else np.nan for intr in Bbb_dict[comp].keys()]
            for comp in level_comp_s
        ]) if level_comp_s else np.zeros((0))
        col_ids_Bbb_f = np.concatenate([
            [intr if len(Bbb_dict[comp][intr].data) != 0 else np.nan for intr in Bbb_dict[comp].keys()]
            for comp in level_comp_f
        ]) if level_comp_f else np.zeros((0))

        # ---- Build Aeq and Aco for S and F separately
        # Unique set of interface ids present in each subsystem
        row_ids_Aeq_s = np.sort(np.unique(col_ids_Bbb_s[~np.isnan(col_ids_Bbb_s)]))
        row_ids_Aeq_f = np.sort(np.unique(col_ids_Bbb_f[~np.isnan(col_ids_Bbb_f)]))

        # Concatenate the per-position blocks (this keeps the same ordering as col_ids_Bbb_*)
        Aeq_vect_s = np.concatenate([[Aeq_dict[comp][intr] for intr in Aeq_dict[comp].keys()] for comp in level_comp_s]) if level_comp_s else np.zeros((0))
        Aeq_vect_f = np.concatenate([[Aeq_dict[comp][intr] for intr in Aeq_dict[comp].keys()] for comp in level_comp_f]) if level_comp_f else np.zeros((0))
        Aco_vect_s = np.concatenate([[Aco_dict[comp][intr] for intr in Aco_dict[comp].keys()] for comp in level_comp_s]) if level_comp_s else np.zeros((0))
        Aco_vect_f = np.concatenate([[Aco_dict[comp][intr] for intr in Aco_dict[comp].keys()] for comp in level_comp_f]) if level_comp_f else np.zeros((0))

        # Assemble sparse block matrices Aeq_s / Aco_s by matching interface ids
        n_pos_int, n_int = len(col_ids_Bbb_s), len(row_ids_Aeq_s)
        Aeq_s = np.array([[None] * n_pos_int] * n_int)
        Aco_s = np.array([[None] * n_pos_int] * n_int)
        for i in range(0, n_int):
            for j in range(0, n_pos_int):
                if row_ids_Aeq_s[i] == col_ids_Bbb_s[j]:
                    Aeq_s[i, j] = Aeq_vect_s[j]
                    Aco_s[i, j] = Aco_vect_s[j]
                elif np.isnan(col_ids_Bbb_s[j]):
                    # For missing interface blocks: keep correct row dimension, zero columns
                    Aeq_s[i, j] = sp.sparse.csc_matrix((Aeq_vect_s[i].shape[0], 0))
                    Aco_s[i, j] = sp.sparse.csc_matrix((Aco_vect_s[i].shape[0], 0))

        Aeq_s = sp.sparse.bmat(Aeq_s, format='csc') if len(Aeq_s) != 0 else sp.sparse.csc_matrix((0, 0))
        Aco_s = sp.sparse.bmat(Aco_s, format='csc') if len(Aco_s) != 0 else sp.sparse.csc_matrix((0, 0))

        # Assemble Aeq_f / Aco_f similarly
        n_pos_int, n_int = len(col_ids_Bbb_f), len(row_ids_Aeq_f)
        Aeq_f = np.array([[None] * n_pos_int] * n_int)
        Aco_f = np.array([[None] * n_pos_int] * n_int)
        for i in range(0, n_int):
            for j in range(0, n_pos_int):
                if row_ids_Aeq_f[i] == col_ids_Bbb_f[j]:
                    Aeq_f[i, j] = Aeq_vect_f[j]
                    Aco_f[i, j] = Aco_vect_f[j]
                elif np.isnan(col_ids_Bbb_f[j]):
                    Aeq_f[i, j] = sp.sparse.csc_matrix((Aeq_vect_f[i].shape[0], 0))
                    Aco_f[i, j] = sp.sparse.csc_matrix((Aco_vect_f[i].shape[0], 0))

        Aeq_f = sp.sparse.bmat(Aeq_f, format='csc') if len(Aeq_f) != 0 else sp.sparse.csc_matrix((0, 0))
        Aco_f = sp.sparse.bmat(Aco_f, format='csc') if len(Aco_f) != 0 else sp.sparse.csc_matrix((0, 0))

        # Global assembly operators
        Aeq = sp.sparse.block_diag([Aeq_s, Aeq_f], format='csc')
        Aco = sp.sparse.block_diag([Aco_s, Aco_f], format='csc')

        # Equilibrium and compatibility matrices in the “stacked interface DOFs” space
        Beq_s = Aeq_s @ Bbb_s
        Beq_f = Aeq_f @ Bbb_f
        Beq = sp.sparse.block_diag([Beq_s, Beq_f], format='csc')

        B_s = Aco_s @ Bbb_s
        B_f = Aco_f @ Bbb_f
        B = sp.sparse.block_diag([B_s, B_f], format='csc')

        # ---- Vibro-acoustic boundary selector Bva (stack VA interfaces in vertical, then concatenate across comps)
        Bva_vect_s = {
            comp: sp.sparse.bmat([[Bva_dict[comp][intr]] for intr in interfaces_va] if interfaces_va
                                 else [[sp.sparse.csc_matrix((0, Bbb_vect_s[comp].shape[1]))]])
            for comp in level_comp_s
        }
        Bva_vect_f = {
            comp: sp.sparse.bmat([[Bva_dict[comp][intr]] for intr in interfaces_va] if interfaces_va
                                 else [[sp.sparse.csc_matrix((0, Bbb_vect_f[comp].shape[1]))]])
            for comp in level_comp_f
        }

        # NOTE: these “horizontal bmats” replicate the original behavior: they concatenate components side-by-side.
        Bva_s = sp.sparse.bmat([[Bva_vect_s[comp] for comp in level_comp_s]], format='csc') if level_comp_s \
            else sp.sparse.csc_matrix((sum(interface_shapes[int_s][0] for int_s in interfaces_va), 0))
        Bva_f = sp.sparse.bmat([[Bva_vect_f[comp] for comp in level_comp_f]], format='csc') if level_comp_f \
            else sp.sparse.csc_matrix((sum(interface_shapes[int_f][1] for int_f in interfaces_va), 0))

        Bva = sp.sparse.block_diag([Bva_s, Bva_f], format='csc')

        # ---------------------------------------------------------------------
        # 3D) Build modal/constraint bases per level depending on method family
        # ---------------------------------------------------------------------

        if red_method in fixed_methods:
            # Fixed-interface internal modes
            Phi_ii_s = sp.sparse.block_diag([Phi_ii_dict[comp] for comp in level_comp_s]) if level_comp_s else sp.sparse.csc_matrix((0, 0))
            Phi_ii_f = sp.sparse.block_diag([Phi_ii_dict[comp] for comp in level_comp_f]) if level_comp_f else sp.sparse.csc_matrix((0, 0))
            Phi_ii = sp.sparse.block_diag([Phi_ii_s, Phi_ii_f])

            # DS constraint modes (b0)
            Psi_ib0_vect_s = {comp: sp.sparse.bmat([[Psi_ib0_dict[comp][intr] for intr in Psi_ib0_dict[comp].keys()]]) for comp in level_comp_s}
            Psi_ib0_vect_f = {comp: sp.sparse.bmat([[Psi_ib0_dict[comp][intr] for intr in Psi_ib0_dict[comp].keys()]]) for comp in level_comp_f}

            Psi_ib0_s = sp.sparse.block_diag([Psi_ib0_vect_s[comp] for comp in level_comp_s], format='csc') if level_comp_s else sp.sparse.csc_matrix((0, 0))
            Psi_ib0_f = sp.sparse.block_diag([Psi_ib0_vect_f[comp] for comp in level_comp_f], format='csc') if level_comp_f else sp.sparse.csc_matrix((0, 0))
            Psi_ib0 = sp.sparse.block_diag([Psi_ib0_s, Psi_ib0_f], format='csc')

            # VA constraint modes (iva)
            # IMPORTANT: The code builds Psi_iva_vect_* by stacking Psi_iva_dict[comp][intr].T vertically then transposing back.
            Psi_iva_vect_s = {
                comp: sp.sparse.bmat([[Psi_iva_dict[comp][intr].T] for intr in interfaces_va] if interfaces_va
                                     else [[sp.sparse.csc_matrix((0, Bii_dict[comp].shape[0]))]]).T
                for comp in level_comp_s
            }
            Psi_iva_vect_f = {
                comp: sp.sparse.bmat([[Psi_iva_dict[comp][intr].T] for intr in interfaces_va] if interfaces_va
                                     else [[sp.sparse.csc_matrix((0, Bii_dict[comp].shape[0]))]]).T
                for comp in level_comp_f
            }

            # These are concatenated vertically over components (bmat with one column of blocks)
            Psi_iva_s = sp.sparse.bmat([[Psi_iva_vect_s[comp]] for comp in level_comp_s], format='csc') if level_comp_s \
                else sp.sparse.csc_matrix((0, sum(interface_shapes[int_s][0] for int_s in interfaces_va)))
            Psi_iva_f = sp.sparse.bmat([[Psi_iva_vect_f[comp]] for comp in level_comp_f], format='csc') if level_comp_f \
                else sp.sparse.csc_matrix((0, sum(interface_shapes[int_f][1] for int_f in interfaces_va)))

            Psi_iva = sp.sparse.block_diag([Psi_iva_s, Psi_iva_f], format='csc')

        if red_method in free_methods:
            # Free-interface modes
            Phi_s = sp.sparse.block_diag([Phi_dict[comp] for comp in level_comp_s]) if level_comp_s else sp.sparse.csc_matrix((0, 0))
            Phi_f = sp.sparse.block_diag([Phi_dict[comp] for comp in level_comp_f]) if level_comp_f else sp.sparse.csc_matrix((0, 0))
            Phi = sp.sparse.block_diag([Phi_s, Phi_f], format='csc')

            # Interface traces of Phi (for Rubin/MacNeal-style projections)
            Phi_b0_s = Bbb_s @ Phi_s
            Phi_b0_f = Bbb_f @ Phi_f
            Phi_va_s = Bva_s @ Phi_s
            Phi_va_f = Bva_f @ Phi_f

            if red_method in dual_methods:
                # Pi matrices (dual methods)
                Pi_b0_vect_s = {comp: sp.sparse.bmat([[Pi_b0_dict[comp][intr] for intr in Pi_b0_dict[comp].keys()]]) for comp in level_comp_s}
                Pi_b0_vect_f = {comp: sp.sparse.bmat([[Pi_b0_dict[comp][intr] for intr in Pi_b0_dict[comp].keys()]]) for comp in level_comp_f}

                Pi_b0_s = sp.sparse.block_diag([Pi_b0_vect_s[comp] for comp in level_comp_s], format='csc') if level_comp_s else sp.sparse.csc_matrix((0, 0))
                Pi_b0_f = sp.sparse.block_diag([Pi_b0_vect_f[comp] for comp in level_comp_f], format='csc') if level_comp_f else sp.sparse.csc_matrix((0, 0))
                Pi_b0 = sp.sparse.block_diag([Pi_b0_s, Pi_b0_f], format='csc')

                Pi_va_vect_s = {
                    comp: sp.sparse.bmat([[Pi_va_dict[comp][intr]] for intr in interfaces_va] if interfaces_va
                                         else [[sp.sparse.csc_matrix((Bbb_vect_s[comp].shape[1], 0))]])
                    for comp in level_comp_s
                }
                Pi_va_vect_f = {
                    comp: sp.sparse.bmat([[Pi_va_dict[comp][intr]] for intr in interfaces_va] if interfaces_va
                                         else [[sp.sparse.csc_matrix((Bbb_vect_f[comp].shape[1], 0))]])
                    for comp in level_comp_f
                }

                Pi_va_s = sp.sparse.block_diag([Pi_va_vect_s[comp] for comp in level_comp_s], format='csc') if level_comp_s else sp.sparse.csc_matrix((0, 0))
                Pi_va_f = sp.sparse.block_diag([Pi_va_vect_f[comp] for comp in level_comp_f], format='csc') if level_comp_f else sp.sparse.csc_matrix((0, 0))
                Pi_va = sp.sparse.block_diag([Pi_va_s, Pi_va_f], format='csc')

            if red_method in primal_methods:
                # Psi matrices for Rubin/MacNeal primal viewpoint: Psi = Pi * Kr
                Psi_b0_vect_f = {
                    comp: sp.sparse.bmat([[Pi_b0_dict[comp][intr] @ Kr_bb0_dict[comp][intr] for intr in Pi_b0_dict[comp].keys()]])
                    for comp in level_comp_f
                }
                Psi_b0_vect_s = {
                    comp: sp.sparse.bmat([[Pi_b0_dict[comp][intr] @ Kr_bb0_dict[comp][intr] for intr in Pi_b0_dict[comp].keys()]])
                    for comp in level_comp_s
                }

                Psi_b0_f = sp.sparse.block_diag([Psi_b0_vect_f[comp] for comp in level_comp_f], format='csc') if level_comp_f else sp.sparse.csc_matrix((0, 0))
                Psi_b0_s = sp.sparse.block_diag([Psi_b0_vect_s[comp] for comp in level_comp_s], format='csc') if level_comp_s else sp.sparse.csc_matrix((0, 0))
                Psi_b0 = sp.sparse.block_diag([Psi_b0_s, Psi_b0_f], format='csc')

                Psi_va_vect_s = {
                    comp: sp.sparse.bmat([[Pi_va_dict[comp][intr] @ Kr_va_dict[comp][intr]] for intr in interfaces_va] if interfaces_va
                                         else [[sp.sparse.csc_matrix((Bbb_vect_s[comp].shape[1], 0))]])
                    for comp in level_comp_s
                }
                Psi_va_vect_f = {
                    comp: sp.sparse.bmat([[Pi_va_dict[comp][intr] @ Kr_va_dict[comp][intr]] for intr in interfaces_va] if interfaces_va
                                         else [[sp.sparse.csc_matrix((Bbb_vect_f[comp].shape[1], 0))]])
                    for comp in level_comp_f
                }

                Psi_va_s = sp.sparse.block_diag([Psi_va_vect_s[comp] for comp in level_comp_s], format='csc') if level_comp_s else sp.sparse.csc_matrix((0, 0))
                Psi_va_f = sp.sparse.block_diag([Psi_va_vect_f[comp] for comp in level_comp_f], format='csc') if level_comp_f else sp.sparse.csc_matrix((0, 0))
                Psi_va = sp.sparse.block_diag([Psi_va_s, Psi_va_f], format='csc')

                # Rubin “residual” modes (free modes corrected to satisfy attachments)
                Phi_RM_s = Phi_s - Psi_b0_s @ Phi_b0_s - Psi_va_s @ Phi_va_s
                Phi_RM_f = Phi_f - Psi_b0_f @ Phi_b0_f - Psi_va_f @ Phi_va_f
                Phi_RM = sp.sparse.block_diag([Phi_RM_s, Phi_RM_f], format='csc')

        # ---------------------------------------------------------------------
        # 3E) Build surface coupling matrices C_sf and VA “pre-projection” blocks
        # ---------------------------------------------------------------------

        # C_sf couples structure DOFs to acoustic DOFs at the physical interface
        if level_comp_s and level_comp_f:
            C_sf = sp.sparse.bmat(
                [[Csf_dict[comp_s][comp_f] for comp_f in level_comp_f] for comp_s in level_comp_s]
            )
        else:
            if level_comp_s and not level_comp_f:
                C_sf = sp.sparse.csc_matrix((B_s.shape[1], 0))
            elif level_comp_f and not level_comp_s:
                C_sf = sp.sparse.csc_matrix((0, B_f.shape[1]))

        # This block builds Cva_s and Cva_f before global projection
        # (addresses multiple-acoustic-domains connection issue in your comment)
        Cva_vect_s = {
            comp: sp.sparse.bmat([[Cva_dict[comp][intr]] for intr in interfaces_va] if interfaces_va
                                 else [[sp.sparse.csc_matrix((0, Bbb_vect_f[comp].shape[1]))]])
            for comp in level_comp_f
        }
        Cva_vect_f = {
            comp: sp.sparse.bmat([[Cva_dict[comp][intr]] for intr in interfaces_va] if interfaces_va
                                 else [[sp.sparse.csc_matrix((0, Bbb_vect_s[comp].shape[1]))]])
            for comp in level_comp_s
        }

        Cva_s = sp.sparse.bmat([[Cva_vect_s[comp] for comp in level_comp_f]], format='csc') if level_comp_f \
            else sp.sparse.csc_matrix((sum(interface_shapes[int_f][0] for int_f in interfaces_va), 0))
        Cva_f = sp.sparse.bmat([[Cva_vect_f[comp] for comp in level_comp_s]], format='csc') if level_comp_s \
            else sp.sparse.csc_matrix((sum(interface_shapes[int_s][1] for int_s in interfaces_va), 0))

        # ---------------------------------------------------------------------
        # 3F) Assemble block-diagonal component matrices for this level
        # ---------------------------------------------------------------------

        K_s, M_s, D_s = [
            sp.sparse.block_diag([mat[comp] for comp in level_comp_s]) if level_comp_s else sp.sparse.csc_matrix((0, 0))
            for mat in [K, M, D]
        ]
        K_f, M_f, D_f = [
            sp.sparse.block_diag([mat[comp] for comp in level_comp_f]) if level_comp_f else sp.sparse.csc_matrix((0, 0))
            for mat in [K, M, D]
        ]

        # “Primal” coupled EOM matrices (K_sf, M_sf, D_sf):
        # K has -C_sf in upper-right, M has +C_sf^T in lower-left (as in your original formulation)
        K_sf = sp.sparse.bmat([[K_s, -C_sf], [None, K_f]], format='csc')
        M_sf = sp.sparse.bmat([[M_s, None], [C_sf.T, M_f]], format='csc')
        D_sf = sp.sparse.block_diag([D_s, D_f])

        # ---------------------------------------------------------------------
        # 3G) Dual assembly re-interpretation (your “redefinition” block)
        # ---------------------------------------------------------------------

        K_diag = sp.sparse.block_diag([K_s, K_f])
        M_diag = sp.sparse.block_diag([M_s, M_f])
        D_diag = sp.sparse.block_diag([D_s, D_f])

        C_s = sp.sparse.bmat(
            [[None, Cva_s],
             [sp.sparse.csc_matrix((Cva_f.shape)), None]],
            format='csc'
        )
        C_f = sp.sparse.bmat(
            [[None, sp.sparse.csc_matrix((Cva_s.shape))],
             [-Cva_f, None]],
            format='csc'
        )

        I_sf = sp.sparse.eye(C_s.shape[0], format='csc')

        # Dual augmented matrices:
        # Unknown ordering is defined implicitly by these blocks and later by T_0/T_1 for dual methods
        K_dual = sp.sparse.bmat([[K_diag, Bva.T, B.T],
                                 [C_s,    I_sf, None],
                                 [B,      None, None]], format='csc')
        M_dual = sp.sparse.bmat([[M_diag, None, None],
                                 [C_f,    sp.sparse.csc_matrix((I_sf.shape)), None],
                                 [None,   None, sp.sparse.csc_matrix((B.shape[0], B.shape[0]))]], format='csc')
        D_dual = sp.sparse.bmat([[D_diag, None, None],
                                 [None,   sp.sparse.csc_matrix((I_sf.shape)), None],
                                 [None,   None, sp.sparse.csc_matrix((B.shape[0], B.shape[0]))]], format='csc')

        # Constraint (BC) matrix on physical DOFs, then padded for dual augmented system
        Cmat_s = sp.sparse.block_diag([Cmat[comp] for comp in level_comp_s]) if level_comp_s else sp.sparse.csc_matrix((0, 0))
        Cmat_f = sp.sparse.block_diag([Cmat[comp] for comp in level_comp_f]) if level_comp_f else sp.sparse.csc_matrix((0, 0))
        Cmat_sf = sp.sparse.block_diag([Cmat_s, Cmat_f], format='csc')

        Cmat_dual = sp.sparse.bmat([[Cmat_sf],
                                    [sp.sparse.csc_matrix((I_sf.shape[0], Cmat_sf.shape[1]))],
                                    [sp.sparse.csc_matrix((B.shape[0],   Cmat_sf.shape[1]))]], format='csc')

        # ---------------------------------------------------------------------
        # 3H) Build interface “single-use” selection operators Abb/Ava and their dual variants
        # ---------------------------------------------------------------------

        Abb, Abb_dual, Ava, Ava_dual, Abva, Abva_dual = {}, {}, {}, {}, {}, {}

        for interf in interfaces:
            # Abb picks DS interface DOFs (0.5 weighting matches your original convention)
            if interf in interfaces_bb_s:
                Abb_s = sp.sparse.bmat([[0.5 * Bbb_dict[comp][interf] for comp in level_comp_s]])
                Abb_f = sp.sparse.csc_matrix((0, Bbb_f.shape[1]))
                Abb_dual_s = sp.sparse.bmat([[
                    sp.sparse.eye(Bbb_dict[level_comp_s[0]][intf].shape[0]) if interf == intf
                    else sp.sparse.csc_matrix((Bbb_dict[level_comp_s[0]][interf].shape[0],
                                               Bbb_dict[level_comp_s[0]][intf].shape[0]))
                    for intf in interfaces_bb_s
                ]]) if len(level_comp_s) != 0 else sp.sparse.csc_matrix((0, 0))
                Abb_dual_f = sp.sparse.csc_matrix((0, B_f.shape[0]))

            elif interf in interfaces_bb_f:
                Abb_s = sp.sparse.csc_matrix((0, Bbb_s.shape[1]))
                Abb_f = sp.sparse.bmat([[0.5 * Bbb_dict[comp][interf] for comp in level_comp_f]])
                Abb_dual_s = sp.sparse.csc_matrix((0, B_s.shape[0]))
                Abb_dual_f = sp.sparse.bmat([[
                    sp.sparse.eye(Bbb_dict[level_comp_f[0]][intf].shape[0]) if interf == intf
                    else sp.sparse.csc_matrix((Bbb_dict[level_comp_f[0]][interf].shape[0],
                                               Bbb_dict[level_comp_f[0]][intf].shape[0]))
                    for intf in interfaces_bb_f
                ]]) if len(level_comp_f) != 0 else sp.sparse.csc_matrix((0, 0))

            else:
                Abb_s = sp.sparse.csc_matrix((0, Bbb_s.shape[1]))
                Abb_f = sp.sparse.csc_matrix((0, Bbb_f.shape[1]))
                Abb_dual_s = sp.sparse.csc_matrix((0, B_s.shape[0]))
                Abb_dual_f = sp.sparse.csc_matrix((0, B_f.shape[0]))

            Abb[interf] = sp.sparse.block_diag([Abb_s, Abb_f], format='csc')
            Abb_dual[interf] = sp.sparse.block_diag([Abb_dual_s, Abb_dual_f], format='csc')

            # Ava picks VA interface DOFs
            if interf in interfaces_va:
                Ava_s = sp.sparse.bmat([[Bva_dict[comp][interf] for comp in level_comp_s]]) if len(level_comp_s) != 0 else sp.sparse.csc_matrix((0, 0))
                Ava_f = sp.sparse.bmat([[Bva_dict[comp][interf] for comp in level_comp_f]]) if len(level_comp_f) != 0 else sp.sparse.csc_matrix((0, 0))

                Ava_dual_s = sp.sparse.bmat([[
                    sp.sparse.eye(Bva_dict[level_comp_s[0]][intf].shape[0]) if interf == intf
                    else sp.sparse.csc_matrix((Bva_dict[level_comp_s[0]][interf].shape[0],
                                               Bva_dict[level_comp_s[0]][intf].shape[0]))
                    for intf in interfaces_va
                ]]) if len(level_comp_s) != 0 else sp.sparse.csc_matrix((0, 0))
                Ava_dual_f = sp.sparse.bmat([[
                    sp.sparse.eye(Bva_dict[level_comp_f[0]][intf].shape[0]) if interf == intf
                    else sp.sparse.csc_matrix((Bva_dict[level_comp_f[0]][interf].shape[0],
                                               Bva_dict[level_comp_f[0]][intf].shape[0]))
                    for intf in interfaces_va
                ]]) if len(level_comp_f) != 0 else sp.sparse.csc_matrix((0, 0))

                Ava[interf] = sp.sparse.block_diag([Ava_s, Ava_f], format='csc')
                Ava_dual[interf] = sp.sparse.block_diag([Ava_dual_s, Ava_dual_f], format='csc')
            else:
                Ava[interf] = sp.sparse.csc_matrix((0, Bva.shape[1]))
                Ava_dual[interf] = sp.sparse.csc_matrix((0, Bva.shape[0]))

            # Combined “bb + va” selector
            Abva[interf] = sp.sparse.bmat([[Abb[interf]],
                                           [Ava[interf]]], format='csc')

            # Dual combined selector (kept exactly as your original “modified definition”)
            Abva_dual[interf] = sp.sparse.bmat(
                [[sp.sparse.csc_matrix((Ava_dual[interf].shape[0], Bva.shape[1])), Ava_dual[interf], None],
                 [sp.sparse.csc_matrix((Abb_dual[interf].shape[0], Bbb.shape[1])), None,             Abb_dual[interf]]],
                format='csc'
            )

        # ---------------------------------------------------------------------
        # 3I) Collect mesh/meta for this level
        # ---------------------------------------------------------------------

        coords_s = np.vstack([coords[comp] for comp in level_comp_s]) if level_comp_s else np.zeros((0, 3))
        nodes_s = np.concatenate([nodes[comp] for comp in level_comp_s]) if level_comp_s else np.zeros((0))
        shapes_s = [shapes[comp] for comp in level_comp_s]
        ndim_s = [ndim[comp] for comp in level_comp_s]
        nodal_normals_s = np.vstack([nodal_normals[comp] for comp in level_comp_s]) if level_comp_s else np.zeros((0, 3))
        constraint_nodes_s = np.concatenate([constraint_nodes[comp] for comp in level_comp_s]) if level_comp_s else np.zeros((0))

        coords_f = np.vstack([coords[comp] for comp in level_comp_f]) if level_comp_f else np.zeros((0, 3))
        nodes_f = np.concatenate([nodes[comp] for comp in level_comp_f]) if level_comp_f else np.zeros((0))
        shapes_f = [shapes[comp] for comp in level_comp_f]
        ndim_f = [ndim[comp] for comp in level_comp_f]
        nodal_normals_f = np.vstack([nodal_normals[comp] for comp in level_comp_f]) if level_comp_f else np.zeros((0, 3))
        constraint_nodes_f = np.concatenate([constraint_nodes[comp] for comp in level_comp_f]) if level_comp_f else np.zeros((0))

        coords_sf = np.vstack([coords_s, coords_f])
        nodes_sf = np.concatenate([nodes_s, nodes_f])
        shapes_sf = shapes_s + shapes_f
        ndim_sf = ndim_s + ndim_f
        nodal_normals_sf = np.vstack([nodal_normals_s, nodal_normals_f])
        constraint_nodes_sf = np.concatenate([constraint_nodes_s, constraint_nodes_f])

        # Boolean masks per component (mesh-level and dof-level)
        component_indices = {
            comp: np.concatenate([
                np.ones(coords[c].shape[0]) if c == comp else np.zeros(coords[c].shape[0])
                for c in level_comp
            ]).astype(bool)
            for comp in level_comp
        }
        component_indices_flat = {
            comp: np.concatenate([
                np.ones(shapes[c]) if c == comp else np.zeros(shapes[c])
                for c in level_comp
            ]).astype(bool)
            for comp in level_comp
        }

        # Assemble nodal connectivity for this level
        nodal_connectivity = assembly_nodal_connectivity(component_matrices, level_comp)

        # Create per-level containers
        Mat_red[level] = {}
        Mat_pres[level] = {}
        Coup_info[level] = {}

        # Studied/prescribed interfaces at this level
        level_int = level_interfaces[level]

        # ---------------------------------------------------------------------
        # 3J) For each studied interface (int_j): build retained/prescribed partitions,
        #     build T_red, project EOM, build prescription matrices, store everything.
        # ---------------------------------------------------------------------

        for int_j in level_int if len(level_int) != 0 else [0]:
            int_pres = [int_j] if len(level_int) != 0 else []
            int_ret = [key for key in interfaces if key not in int_active]

            print(f'{level}', f'Pres.: {int_pres}', f'Intern.: {int_ret}', f'Active:{int_active}')

            # Build boolean diagonals for retained/prescribed partitions (DS and VA separately, and S/F separately)
            diag_ret_bb_s = np.concatenate([
                np.ones(interface_shapes[int_s][0]) if int_s in int_ret else np.zeros(interface_shapes[int_s][0])
                for int_s in interfaces_bb_s
            ]) if interfaces_bb_s else np.zeros(0)
            diag_pres_bb_s = np.concatenate([
                np.ones(interface_shapes[int_s][0]) if int_s in int_pres else np.zeros(interface_shapes[int_s][0])
                for int_s in interfaces_bb_s
            ]) if interfaces_bb_s else np.zeros(0)

            diag_ret_va_s = np.concatenate([
                np.ones(interface_shapes[int_s][0]) if int_s in int_ret else np.zeros(interface_shapes[int_s][0])
                for int_s in interfaces_va
            ]) if interfaces_va else np.zeros(0)
            diag_pres_va_s = np.concatenate([
                np.ones(interface_shapes[int_s][0]) if int_s in int_pres else np.zeros(interface_shapes[int_s][0])
                for int_s in interfaces_va
            ]) if interfaces_va else np.zeros(0)

            diag_ret_bb_f = np.concatenate([
                np.ones(interface_shapes[int_f][0]) if int_f in int_ret else np.zeros(interface_shapes[int_f][0])
                for int_f in interfaces_bb_f
            ]) if interfaces_bb_f else np.zeros(0)
            diag_pres_bb_f = np.concatenate([
                np.ones(interface_shapes[int_f][0]) if int_f in int_pres else np.zeros(interface_shapes[int_f][0])
                for int_f in interfaces_bb_f
            ]) if interfaces_bb_f else np.zeros(0)

            diag_ret_va_f = np.concatenate([
                np.ones(interface_shapes[int_f][1]) if int_f in int_ret else np.zeros(interface_shapes[int_f][1])
                for int_f in interfaces_va
            ]) if interfaces_va else np.zeros(0)
            diag_pres_va_f = np.concatenate([
                np.ones(interface_shapes[int_f][1]) if int_f in int_pres else np.zeros(interface_shapes[int_f][1])
                for int_f in interfaces_va
            ]) if interfaces_va else np.zeros(0)

            # Selection matrices for retained/prescribed
            Bbk_s = sp.sparse.diags(diag_ret_bb_s, format='csc')[:, diag_ret_bb_s != 0]
            Bbj_s = sp.sparse.diags(diag_pres_bb_s, format='csc')[:, diag_ret_bb_s == 0]
            Bvak_s = sp.sparse.diags(diag_ret_va_s, format='csc')[:, diag_ret_va_s != 0]
            Bvaj_s = sp.sparse.diags(diag_pres_va_s, format='csc')[:, diag_ret_va_s == 0]

            Bbk_f = sp.sparse.diags(diag_ret_bb_f, format='csc')[:, diag_ret_bb_f != 0]
            Bbj_f = sp.sparse.diags(diag_pres_bb_f, format='csc')[:, diag_ret_bb_f == 0]
            Bvak_f = sp.sparse.diags(diag_ret_va_f, format='csc')[:, diag_ret_va_f != 0]
            Bvaj_f = sp.sparse.diags(diag_pres_va_f, format='csc')[:, diag_ret_va_f == 0]

            Bvak = sp.sparse.block_diag([Bvak_s, Bvak_f], format='csc')
            Bvaj = sp.sparse.block_diag([Bvaj_s, Bvaj_f], format='csc')
            Bbk = sp.sparse.block_diag([Bbk_s, Bbk_f], format='csc')
            Bbj = sp.sparse.block_diag([Bbj_s, Bbj_f], format='csc')

            Bbbva_k = sp.sparse.block_diag([Bbk, Bvak], format='csc')
            Bbbva_j = sp.sparse.block_diag([Bbj, Bvaj], format='csc')

            # -----------------------------------------------------------------
            # 3K) Build the reduction / assembly transformation T_red
            #     (The rest of the function is a large set of method-specific
            #      block transformations; the logic is preserved exactly.)
            # -----------------------------------------------------------------

            # --- PRIMAL family ---
            if red_method in primal_methods:
                # From physical dofs X -> [X_i, X_b0, X_va]
                T_0 = sp.sparse.bmat([[Bii.T, Bbb.T, Bva.T]], format='csc')

                # From [X_i, X_b0, X_va] -> [X_i, X_b, X_va] (primal assembly via Aeq)
                I_ii = sp.sparse.eye(Bii.shape[0], format='csc')
                I_va = sp.sparse.eye(Bva.shape[0], format='csc')
                T_1 = sp.sparse.block_diag([I_ii, 2 * Aeq.T, I_va], format='csc')

                # From [X_i, X_b, X_va] -> [X_i, X_bk, X_vak, X_bj, X_vaj]
                T_2 = sp.sparse.bmat([[I_ii, None, None, None, None],
                                      [None, Bbk, None, Bbj, None],
                                      [None, None, Bvak, None, Bvaj]], format='csc')

                if red_method in fixed_methods:
                    # Fixed-interface primal CMS (CBM / CCBM)
                    I_va = sp.sparse.block_diag([sp.sparse.eye(Psi_iva.shape[1])], format='csc')
                    I_bb0 = sp.sparse.eye(Bbb.shape[0])

                    # [X_i,X_b0,X_va] -> [X_m, X_b0, X_va]
                    T_3 = sp.sparse.bmat([[Phi_ii, Psi_ib0, Psi_iva],
                                          [None,   I_bb0,  None],
                                          [None,    None,   I_va]], format='csc')

                    I_mm = sp.sparse.eye(Phi_ii.shape[1], format='csc')

                    # [X_m, X_b0, X_va] -> [X_m, X_b, X_va]
                    T_4 = sp.sparse.block_diag([I_mm, 2 * Aeq.T, I_va], format='csc')

                    if red_method in ['CBM', 'CCBM']:
                        # [X_m,X_b,X_va] -> [X_m, X_bk, X_vak, X_bj, X_vaj]
                        T_5 = sp.sparse.bmat([[I_mm, None, None, None, None],
                                              [None, Bbk, None, Bbj, None],
                                              [None, None, Bvak, None, Bvaj]], format='csc')

                        T_red = T_0 @ T_3 @ T_4 @ T_5  # CBM transformation

                        if red_method == 'CCBM':
                            # Condensed CBM: static condensation on retained boundary DOFs
                            Psi_ib = 2 * Psi_ib0 @ Aeq.T
                            Kbi = Beq @ K_sf @ Bii.T
                            Kbb = Beq @ K_sf @ (2 * Beq).T

                            KbbCBM = Kbi @ Psi_ib + Kbb

                            Kki, Kkj = Bbk.T @ Kbi, Bbk.T @ Kbb @ Bbj
                            Psi_ij = Psi_ib @ Bbj

                            KkkCBM = Bbk.T @ KbbCBM @ Bbk
                            KkkCBMinv = inverse_sparse_matrix(KkkCBM, method='pardiso')

                            Psi_kmi = -KkkCBMinv @ Kki @ Phi_ii
                            Psi_kj = -KkkCBMinv @ (Kki @ Psi_ij + Kkj)

                            Ivak = sp.sparse.eye(Bvak.shape[1])
                            Ivaj = sp.sparse.eye(Bvaj.shape[1])
                            Ibj = sp.sparse.eye(Bbj.shape[1])

                            # [X_mi, X_bk, X_vak, X_bj, X_vaj] -> [X_mi, X_vak, X_bj, X_vaj]
                            T_7 = sp.sparse.bmat([[I_mm,     None,   None, None],
                                                  [Psi_kmi,  None, Psi_kj, None],
                                                  [None,     Ivak,   None, None],
                                                  [None,     None,    Ibj, None],
                                                  [None,     None,   None, Ivaj]], format='csc')

                            T_red = T_red @ T_7

                            # Substitution blocks for prescribed DOFs (j)
                            O_jm = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_mm.shape[1]))
                            I_jj = sp.sparse.eye(Bbbva_j.shape[1])
                            O_jj = sp.sparse.csc_matrix(I_jj.shape)
                            K_red_j, M_red_j, D_red_j = [O_jm, I_jj], [O_jm, O_jj], [O_jm, O_jj]
                        else:
                            O_jm = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_mm.shape[1]))
                            O_jk = sp.sparse.csc_matrix((Bbbva_j.shape[1], Bbbva_k.shape[1]))
                            I_jj = sp.sparse.eye(Bbbva_j.shape[1])
                            O_jj = sp.sparse.csc_matrix(I_jj.shape)
                            K_red_j, M_red_j, D_red_j = [O_jm, O_jk, I_jj], [O_jm, O_jk, O_jj], [O_jm, O_jk, O_jj]

                elif red_method in free_methods:
                    # Rubin/MacNeal primal branch
                    I_va = sp.sparse.block_diag([sp.sparse.eye(Bva.shape[0])], format='csc')
                    I_bb0 = sp.sparse.eye(Bbb.shape[0])

                    # [X] -> [X_m, X_b0, X_va]
                    T_6 = sp.sparse.bmat([[Phi_RM, Psi_b0, Psi_va]], format='csc')

                    I_mm = sp.sparse.eye(Phi.shape[1], format='csc')
                    T_4 = sp.sparse.block_diag([I_mm, 2 * Aeq.T, I_va], format='csc')

                    if red_method in ['RM', 'MNM']:
                        T_5 = sp.sparse.bmat([[I_mm, None, None, None, None],
                                              [None, Bbk, None, Bbj, None],
                                              [None, None, Bvak, None, Bvaj]], format='csc')

                        T_red = T_6 @ T_4 @ T_5

                        if red_method == 'MNM':
                            Psi_b = Psi_b0 @ (2 * Aeq.T)
                            KbmRM = Psi_b.T @ K_sf @ Phi_RM
                            KbbRM = Psi_b.T @ K_sf @ Psi_b

                            KkmRM = Bbk.T @ KbmRM
                            KkkRM = Bbk.T @ KbbRM @ Bbk
                            KkjRM = Bbk.T @ KbbRM @ Bbj

                            KkkRMinv = inverse_sparse_matrix(KkkRM, method='pardiso')
                            Psi_km = -KkkRMinv @ KkmRM
                            Psi_kj = -KkkRMinv @ KkjRM

                            Ivak = sp.sparse.eye(Bvak.shape[1])
                            Ivaj = sp.sparse.eye(Bvaj.shape[1])
                            Ibj = sp.sparse.eye(Bbj.shape[1])

                            T_7 = sp.sparse.bmat([[I_mm,   None,   None, None],
                                                  [Psi_km, None, Psi_kj, None],
                                                  [None,   Ivak,   None, None],
                                                  [None,   None,    Ibj, None],
                                                  [None,   None,   None, Ivaj]], format='csc')

                            T_red = T_red @ T_7

                            O_jm = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_mm.shape[1]))
                            I_jj = sp.sparse.eye(Bbbva_j.shape[1])
                            O_jj = sp.sparse.csc_matrix(I_jj.shape)
                            K_red_j, M_red_j, D_red_j = [O_jm, I_jj], [O_jm, O_jj], [O_jm, O_jj]
                        else:
                            O_jm = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_mm.shape[1]))
                            O_jk = sp.sparse.csc_matrix((Bbbva_j.shape[1], Bbbva_k.shape[1]))
                            I_jj = sp.sparse.eye(Bbbva_j.shape[1])
                            O_jj = sp.sparse.csc_matrix(I_jj.shape)
                            K_red_j, M_red_j, D_red_j = [O_jm, O_jk, I_jj], [O_jm, O_jk, O_jj], [O_jm, O_jk, O_jj]

                else:
                    # Pure primal (no CMS)
                    T_red = T_0 @ T_1 @ T_2

                    O_ji = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_ii.shape[1]))
                    O_jk = sp.sparse.csc_matrix((Bbbva_j.shape[1], Bbbva_k.shape[1]))
                    I_jj = sp.sparse.eye(Bbbva_j.shape[1])
                    O_jj = sp.sparse.csc_matrix(I_jj.shape)
                    K_red_j, M_red_j, D_red_j = [O_ji, O_jk, I_jj], [O_ji, O_jk, O_jj], [O_ji, O_jk, O_jj]

            # --- DUAL family ---
            elif red_method in dual_methods:
                I_lambda = sp.sparse.eye(B.shape[0])

                # [X,Fva,Lambda] -> [X_i,X_b0,X_va,Lambda,Fva]
                T_0 = sp.sparse.bmat([[Bii.T, Bbb.T, Bva.T, None, None],
                                      [None,  None,  None,  None, I_sf],
                                      [None,  None,  None, I_lambda, None]], format='csc')

                I_ii = sp.sparse.eye(Bii.shape[0], format='csc')
                I_bb0 = sp.sparse.eye(Bbb.shape[0], format='csc')
                I_va = sp.sparse.eye(Bva.shape[0], format='csc')

                T_1 = sp.sparse.bmat([[I_ii, None, None, None, None, None, None],
                                      [None, I_bb0, None, None, None, None, None],
                                      [None, None, I_sf, None, None, None, None],
                                      [None, None, None, Bbk, None, Bbj, None],
                                      [None, None, None, None, Bvak, None, Bvaj]], format='csc')

                if red_method in fixed_methods:
                    # Fixed-dual CB family (FDCBM / CFDCBM)
                    T_3 = sp.sparse.bmat([[Phi_ii, Psi_ib0, Psi_iva, None, None],
                                          [None,   I_bb0,  None,   None, None],
                                          [None,    None,   I_sf,  None, None],
                                          [None,    None,   None, I_lambda, None],
                                          [None,    None,   None, None, I_va]], format='csc')

                    I_mm = sp.sparse.eye(Phi_ii.shape[1], format='csc')

                    if red_method in ['FDCBM', 'CFDCBM']:
                        T_5 = sp.sparse.bmat([[I_mm,  None, None, None, None, None, None],
                                              [None, I_bb0, None, None, None, None, None],
                                              [None,  None, I_sf, None, None, None, None],
                                              [None,  None, None, Bbk, None, Bbj, None],
                                              [None,  None, None, None, Bvak, None, Bvaj]], format='csc')

                        T_red = T_0 @ T_3 @ T_5

                        if red_method == 'CFDCBM':
                            Kbi0 = Bbb @ K_diag @ Bii.T
                            Kbb0 = Bbb @ K_diag @ Bbb.T
                            Kbva0 = Bbb @ K_sf @ Bva.T

                            Kbb0FDCBM = Kbi0 @ Psi_ib0 + Kbb0
                            KbbFDCBM = Aeq @ Kbb0FDCBM @ (2 * Aeq.T)

                            KbbFDCBMinv = inverse_sparse_matrix(KbbFDCBM, method='pardiso')
                            Kbb0FDCBMinv = (2 * Aeq.T) @ KbbFDCBMinv @ Aeq

                            Psi_bmi0 = -Kbb0FDCBMinv @ Kbi0 @ Phi_ii
                            Pi_bb = -Kbb0FDCBMinv @ Aco.T
                            Psi_bva0 = -Kbb0FDCBMinv @ Kbva0
                            Pi_bk = Pi_bb @ Bbk
                            Pi_bj = Pi_bb @ Bbj

                            Ivak = sp.sparse.eye(Bvak.shape[1])
                            Ivaj = sp.sparse.eye(Bvaj.shape[1])
                            Ibk = sp.sparse.eye(Bbk.shape[1])
                            Ibj = sp.sparse.eye(Bbj.shape[1])

                            T_7 = sp.sparse.bmat([[I_mm,      None,     None, None,  None, None],
                                                  [Psi_bmi0, Psi_bva0, Pi_bk, None, Pi_bj, None],
                                                  [None,      I_sf,    None, None,  None, None],
                                                  [None,      None,     Ibk, None,  None, None],
                                                  [None,      None,    None, Ivak,  None, None],
                                                  [None,      None,    None, None,   Ibj, None],
                                                  [None,      None,    None, None,  None, Ivaj]], format='csc')

                            T_red = T_red @ T_7

                            O_jm = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_mm.shape[1]))
                            O_jva = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_sf.shape[1]))
                            I_jj = sp.sparse.eye(Bbbva_j.shape[1])
                            O_jj = sp.sparse.csc_matrix(I_jj.shape)
                            O_jk = sp.sparse.csc_matrix((Bbbva_j.shape[1], Bbbva_k.shape[1]))
                            K_red_j, M_red_j, D_red_j = [O_jm, O_jva, O_jk, I_jj], [O_jm, O_jva, O_jk, O_jj], [O_jm, O_jva, O_jk, O_jj]
                        else:
                            O_jm = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_mm.shape[1]))
                            O_jb0 = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_bb0.shape[1]))
                            O_jva = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_sf.shape[1]))
                            O_jk = sp.sparse.csc_matrix((Bbbva_j.shape[1], Bbbva_k.shape[1]))
                            I_jj = sp.sparse.eye(Bbbva_j.shape[1])
                            O_jj = sp.sparse.csc_matrix(I_jj.shape)
                            K_red_j, M_red_j, D_red_j = [O_jm, O_jb0, O_jva, O_jk, I_jj], [O_jm, O_jb0, O_jva, O_jk, O_jj], [O_jm, O_jb0, O_jva, O_jk, O_jj]

                elif red_method in free_methods:
                    # DCBM / CDCBM dual branch
                    I_bb0 = sp.sparse.eye(Bbb.shape[0])
                    I_va = sp.sparse.eye(Bva.shape[0], format='csc')
                    I = sp.sparse.eye(B.shape[1])

                    # [X,Fva,Lambda] -> [X, Lambda, Fva]
                    T_0 = sp.sparse.bmat([[I, None, None],
                                          [None, None, I_sf],
                                          [None, I_lambda, None]], format='csc')

                    Pi_b = Pi_b0 @ (Aco.T)

                    # [X, Lambda, Fva] -> [X_m, Lambda, Fva]
                    T_3 = sp.sparse.bmat([[Phi, -Pi_b, Pi_va],
                                          [None, I_lambda, None],
                                          [None, None, I_va]], format='csc')

                    I_mm = sp.sparse.eye(Phi.shape[1], format='csc')

                    if red_method in ['DCBM', 'CDCBM']:
                        T_5 = sp.sparse.bmat([[I_mm, None, None, None, None],
                                              [None, Bbk, None, Bbj, None],
                                              [None, None, Bvak, None, Bvaj]], format='csc')

                        T_red = T_0 @ T_3 @ T_5

                        if red_method == 'CDCBM':
                            KbmDCBM = B @ Phi - Pi_b.T @ K_diag @ Phi
                            KbbDCBM = Pi_b.T @ K_diag @ Pi_b - B @ Pi_b - Pi_b.T @ B.T

                            KbbDCBMinv = inverse_sparse_matrix(KbbDCBM, method='pardiso')
                            Psi_bm = -KbbDCBMinv @ KbmDCBM
                            Psi_km = Bbk.T @ Psi_bm

                            Ivak = sp.sparse.eye(Bvak.shape[1])
                            Ivaj = sp.sparse.eye(Bvaj.shape[1])
                            Ibj = sp.sparse.eye(Bbj.shape[1])

                            T_7 = sp.sparse.bmat([[I_mm,   None, None, None],
                                                  [Psi_km, None, None, None],
                                                  [None,   Ivak, None, None],
                                                  [None,   None,  Ibj, None],
                                                  [None,   None, None, Ivaj]], format='csc')

                            T_red = T_red @ T_7

                            O_jm = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_mm.shape[1]))
                            I_jj = sp.sparse.eye(Bbbva_j.shape[1])
                            O_jj = sp.sparse.csc_matrix(I_jj.shape)
                            O_jk = sp.sparse.csc_matrix((Bbbva_j.shape[1], Bvak.shape[1]))
                            K_red_j, M_red_j, D_red_j = [O_jm, O_jk, I_jj], [O_jm, O_jk, O_jj], [O_jm, O_jk, O_jj]
                        else:
                            O_jm = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_mm.shape[1]))
                            O_jk = sp.sparse.csc_matrix((Bbbva_j.shape[1], Bbbva_k.shape[1]))
                            I_jj = sp.sparse.eye(Bbbva_j.shape[1])
                            O_jj = sp.sparse.csc_matrix(I_jj.shape)
                            K_red_j, M_red_j, D_red_j = [O_jm, O_jk, I_jj], [O_jm, O_jk, O_jj], [O_jm, O_jk, O_jj]

                else:
                    # Pure dual (no CMS)
                    T_red = T_0 @ T_1

                    O_ji = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_ii.shape[1]))
                    O_jb0 = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_bb0.shape[1]))
                    O_jva = sp.sparse.csc_matrix((Bbbva_j.shape[1], I_sf.shape[1]))
                    O_jk = sp.sparse.csc_matrix((Bbbva_j.shape[1], Bbbva_k.shape[1]))
                    I_jj = sp.sparse.eye(Bbbva_j.shape[1])
                    O_jj = sp.sparse.csc_matrix(I_jj.shape)
                    K_red_j, M_red_j, D_red_j = [O_ji, O_jb0, O_jva, O_jk, I_jj], [O_ji, O_jb0, O_jva, O_jk, O_jj], [O_ji, O_jb0, O_jva, O_jk, O_jj]

            # Finish timing of T_red construction
            time_Tred, RAM_Tred = thread_monitoring(start=False, print_res=False)

            # -----------------------------------------------------------------
            # 3L) Project equations of motion onto reduced coordinates
            # -----------------------------------------------------------------
            thread_monitoring(start=True)

            if red_method in primal_methods:
                K_red, M_red, D_red = [T_red.T @ mat @ T_red for mat in [K_sf, M_sf, D_sf]]
            elif red_method in dual_methods:
                K_red, M_red, D_red = [T_red.T @ mat @ T_red for mat in [K_dual, M_dual, D_dual]]

            time_EOM, RAM_EOM = thread_monitoring(start=False, print_res=False)

            # Update time/RAM stats
            time_RAM['time']['Tred'] += time_Tred
            time_RAM['time']['EOM'] += time_EOM
            time_RAM['RAM']['Tred'] = np.max([time_RAM['RAM']['Tred'], RAM_Tred])
            time_RAM['RAM']['EOM'] = np.max([time_RAM['RAM']['EOM'], RAM_EOM])

            time_RAM['time']['TOT_assembly'] = sum(time_RAM['time'][k] for k in time_RAM['time'].keys())
            time_RAM['RAM']['TOT_assembly'] = max(time_RAM['RAM'][k] for k in time_RAM['RAM'].keys())

            time_RAM['sparsity']['ratio'] = max(Mat.data.shape[0] / (Mat.shape[0] ** 2 + 1e-8) for Mat in [K_red, D_red, M_red])
            time_RAM['sparsity']['nrows'] = max(Mat.shape[0] for Mat in [K_red, D_red, M_red])

            # Remove prescribed DOF contribution (method-specific substitution blocks)
            if red_method not in []:
                K_red = delete_DOF_j_contribution(K_red, K_red_j)
                M_red = delete_DOF_j_contribution(M_red, M_red_j)
                D_red = delete_DOF_j_contribution(D_red, D_red_j)

            # -----------------------------------------------------------------
            # 3M) Build prescription matrices (maps prescribed interface DOFs)
            # -----------------------------------------------------------------
            Kp_red = sp.sparse.bmat(
                [[sp.sparse.csc_matrix((K_red.shape[0] - Bbbva_j.shape[1], Bbbva_j.getnnz(axis=1).sum()))],
                 [Bbbva_j[Bbbva_j.getnnz(axis=1).astype(bool), :].T]]
            )
            Mp_red = sp.sparse.csc_matrix(Kp_red.shape)
            Dp_red = sp.sparse.csc_matrix(Kp_red.shape)

            # Suppress zero rows/cols consistently across (K,D,M), prescription matrices, and T_red
            mat_red, mat_pres, T_red = suppress_zero_row_cols(
                [K_red, D_red, M_red], [Kp_red, Dp_red, Mp_red], T_red
            )

            # Optional assembly modes (second reduction stage)
            if assembly_modes[0] is True:
                print(f"Computing ASSEMBLY modes of level {level}.")
                freq_n, Phi_assembly, _ = compute_modal_basis(
                    K_red, M_red, sparse=True, n_modes=assembly_modes[1], freq0=0.0
                )
                T_red = T_red @ Phi_assembly
                mat_red = [Phi_assembly.T @ mat @ Phi_assembly for mat in [K_red, D_red, M_red]]

            # Store reduced matrices (note original ordering: mat_red[0]=K, mat_red[2]=M, mat_red[1]=D)
            Mat_red[level][int_j] = {'K': mat_red[0], 'M': mat_red[2], 'D': mat_red[1]}
            Mat_pres[level][int_j] = {'K': mat_pres[0], 'M': mat_pres[2], 'D': mat_pres[1]}

            # Store coupling info used downstream
            if red_method in primal_methods:
                Coup_info[level][int_j] = {
                    'B': B, 'Bii': Bii, 'Aco': Aco, 'Aeq': Aeq, 'Beq': Beq, 'Bbb': Bbb, 'Bva': Bva,
                    'T_red': T_red, 'Cmat': Cmat_sf, 'ndim': ndim_sf, 'shapes': shapes_sf,
                    'coords': coords_sf, 'Abb': Abva, 'components': level_comp,
                    'component_indices': component_indices, 'component_indices_flat': component_indices_flat,
                    'nodes': nodes_sf.astype(int), 'connectivity': nodal_connectivity, 'nodal_normals': nodal_normals_sf,
                    'connectivity_comp': {'coords': component_matrices['coords'],
                                         'connectivity': component_matrices['connectivity'],
                                         'nodes': component_matrices['nodes']},
                    'constraint_nodes': constraint_nodes_sf,
                }
            elif red_method in dual_methods:
                Coup_info[level][int_j] = {
                    'B': B, 'Bii': Bii, 'Aco': Aco, 'Aeq': Aeq, 'Beq': Beq, 'Bbb': Bbb, 'Bva': Bva,
                    'T_red': T_red, 'Cmat': Cmat_dual, 'ndim': ndim_sf, 'shapes': shapes_sf,
                    'coords': coords_sf, 'Abb': Abva_dual, 'components': level_comp,
                    'component_indices': component_indices, 'component_indices_flat': component_indices_flat,
                    'nodes': nodes_sf.astype(int), 'connectivity': nodal_connectivity, 'nodal_normals': nodal_normals_sf,
                    'connectivity_comp': {'coords': component_matrices['coords'],
                                         'connectivity': component_matrices['connectivity'],
                                         'nodes': component_matrices['nodes']},
                    'constraint_nodes': constraint_nodes_sf,
                }

    # -------------------------------------------------------------------------
    # 4) Optional persistence
    # -------------------------------------------------------------------------

    save_job, job_folder, job_name = save_job_info
    if save_job:
        if not os.path.exists(job_folder):
            os.makedirs(job_folder)

        joblib.dump(Coup_info, job_folder + 'COUP_INFO_' + job_name, compress=3)
        joblib.dump(Mat_red, job_folder + 'MATRICES_' + job_name, compress=3)
        joblib.dump(Mat_pres, job_folder + 'MAT_PRES_' + job_name, compress=3)

    return Mat_red, Mat_pres, Coup_info, time_RAM


def TPA_CMS_vibroacoustic(
    component_matrices,
    coupling_info,
    level_components,
    level_interfaces,
    force_ext,
    analysis_settings,
    type_TPA='1L-TPA',
    CMS_method=['', 40],
    complex_analysis=True,
    modal_CB_va=False,
    transient=False,
    save_job_info=(False, '', ''),
    load_matrices_info=(False, '', '')
):
    """
    PURPOSE
    -------
    Solve a vibro-acoustic Transfer Path Analysis (TPA) using the Dynamic Substructuring (DS) / CMS
    assembly produced by `TPA_matrices_vibroacoustic(...)`.

    This routine is the *solver* stage of your pipeline:
      1) It obtains (or loads from disk) the reduced, level-wise assembled matrices (Mat_red),
         the prescription operators (Mat_pres), and the auxiliary coupling metadata (Coup_info).
      2) It solves the assembled problem for the first level (typically the “assembly” level, AP)
         under the provided external forcing `force_ext`, using the analysis type specified by
         `analysis_settings` (harmonic / transient / modal).
      3) It propagates that solution to subsequent TPA levels by prescribing interface motion/effort
         (depending on formulation) through the level-specific prescription matrices, thereby
         computing the response contribution of each studied interface “path”.
      4) It returns a structured dictionary `X_TPA` containing, for each level and each path:
         - the reduced/global solution vector (X_flat),
         - the reshaped nodal field (X) in a (n_nodes, ndim, n_steps) or similar convention,
         - and, for dual formulations, interface forces (F) reconstructed from Lagrange multipliers.

    HIGH-LEVEL IDEA (WHAT THE FUNCTION COMPUTES)
    --------------------------------------------
    - Level 0 (AP): solve the globally assembled reduced system excited by the external load.
      The solution is projected back to the “flat” assembled DOF space and then to nodal fields.
    - Level i>0: for each studied interface at that level, build an *equivalent internal forcing*
      that reproduces the AP interface state (or previous-level interface state in multi-level TPA),
      solve the level system, and store that path contribution.

    Two TPA modes are supported:
      - '1L-TPA' : single-step path extraction. Every deeper level uses the AP interface state
                  as prescription input (or the first post-AP level if idx == 1).
      - 'MTPA'   : multi-level chaining. Each level uses the interface state of the immediately
                  previous level to build longer path sequences (e.g., "intA-intB-intC").

    IMPORTANT INPUT/OUTPUT CONVENTIONS (AS USED BY THIS CODE)
    ---------------------------------------------------------
    - `Mat_red`, `Mat_pres`, `Coup_info` must follow the exact structure returned by
      `TPA_matrices_vibroacoustic(...)`. In particular:
        * Mat_red[level][interface]['K'/'M'/'D'] are already reduced/assembled matrices.
        * Mat_pres[level][interface]['K'] maps prescribed interface DOFs into the reduced system
          as an equivalent forcing (force_int = K_pres @ X_b in this implementation).
        * Coup_info[level][interface]['T_red'] is the assembly/reduction back-projection used to
          reconstruct X_flat from X_red.
        * Coup_info[level][interface]['Cmat'] maps between constrained/full coordinates consistent
          with the upstream constraint reduction.
        * Coup_info[level][interface]['Abb'][interf] extracts the interface DOFs (displacements or
          interface variables) from a flattened assembled vector (used to obtain X_b).
        * Coup_info[level][interface] also carries B, Bva, ndim, shapes, etc., used for reshaping and
          (in dual methods) reconstructing interface forces.
    - `force_ext` is assumed to be expressed in the *same constrained/full coordinate convention*
      expected by `Cmat` (because it is projected as force = T_red.T @ Cmat @ force_ext).

    ANALYSIS TYPES (analysis_settings)
    ---------------------------------
    analysis_settings[0] selects the solver type:
      - 'HARMONIC'  : (analysis_type, stamps, solver)
            `stamps` are the frequency samples (or “stamps”) used by `harmonic_response_2`.
      - 'TRANSIENT' : (analysis_type, solver, n_inc, DELTAt, al_solver)
            time integration with `transient_response`.
      - 'MODAL'     : (analysis_type, nmodes, alpha_M, hermitian)
            computes eigenpairs of the assembled system with `compute_modal_basis`.

    PRIMAL VS DUAL FORMULATIONS (CMS_method[0])
    -------------------------------------------
    This solver distinguishes between formulations because the stored state differs:
      - Primal methods: store only displacements/pressures in X.
      - Dual methods : additionally reconstruct interface forces F from the dual variables
        using the operator B_sf = [0, Bva^T, B^T] and the constraint mapping Cmat.

    JOB I/O (load/save)
    -------------------
    - If `load_matrices_info[0]` is True, the function attempts to load Mat_red, Mat_pres and Coup_info
      from joblib files. If any file is missing, it falls back to computing them.
    - If `save_job_info[0]` is True, the computed solution `X_TPA` (and optionally Coup_info) is stored.

    RETURNS
    -------
    X_TPA : dict
        Nested dict indexed as X_TPA[level][path_name], where each entry contains:
          - 'X_flat' : flattened assembled DOF solution in the assembly coordinate space,
          - 'X'      : nodal field reshaped by `modes_flat_to_3d_3(...)`,
          - 'stamps' : frequency stamps / time stamps / modal frequencies,
          - plus for dual methods: 'F' (interface forces, possibly including external force contribution).
        For level 'AP', the path_name key is '0' by convention.
    Coup_info : dict
        The coupling metadata used for post-processing and for building prescriptions.
        Either loaded from disk or returned from `TPA_matrices_vibroacoustic`.
    time_RAM : dict
        Timing/RAM bookkeeping returned/extended from the matrix-building and solution stages.

    NOTE (BEHAVIORAL CONSTRAINT)
    ----------------------------
    This docstring describes the *exact* behavior implemented here:
      - prescriptions are applied as an equivalent forcing `force_int = K_pres @ X_b`,
      - '1L-TPA' always uses AP interface state for deeper levels (except idx==1 shortcut),
      - 'MTPA' chains paths by using previous-level solutions and builds concatenated path names.
    """

    # -------------------------------------------------------------------------
    # 0) Optionally load (Mat_red, Mat_pres, Coup_info) from disk
    # -------------------------------------------------------------------------
    save_job, job_folder, job_name = save_job_info
    load_matrices, job_folder, mat_job_name = load_matrices_info

    load_flag = False
    if load_matrices:
        load_flag = True

        # Reduced assembly matrices
        if os.path.exists(job_folder + 'MATRICES_' + mat_job_name):
            print('Loading the assembly matrices.')
            Mat_red = joblib.load(job_folder + 'MATRICES_' + mat_job_name)
        else:
            load_flag *= False

        # Prescription matrices
        if os.path.exists(job_folder + 'MAT_PRES_' + mat_job_name):
            print('Loading the presciption matrices.')
            Mat_pres = joblib.load(job_folder + 'MAT_PRES_' + mat_job_name)
        else:
            load_flag *= False

        # Coupling metadata
        if os.path.exists(job_folder + 'COUP_INFO_' + mat_job_name):
            print('Loading the coupling information.')
            Coup_info = joblib.load(job_folder + 'COUP_INFO_' + mat_job_name)
        else:
            load_flag *= False

    # If loading failed or not requested: assemble matrices now
    if load_flag is False:
        Mat_red, Mat_pres, Coup_info, time_RAM = TPA_matrices_vibroacoustic(
            component_matrices,
            coupling_info,
            level_components,
            level_interfaces,
            CMS_method=CMS_method,
            modal_CB_va=modal_CB_va
        )
    else:
        # Keep same structure as your original code when matrices are loaded
        time_RAM = {'time': {}, 'RAM': {}}

    # -------------------------------------------------------------------------
    # 1) Solve the systems level by level and store path contributions
    # -------------------------------------------------------------------------

    # Start monitoring global time and RAM for the response computations
    thread_monitoring(True)

    # Storage of results for all levels and paths
    X_TPA = {}

    # Levels are processed in the order given by level_components dict
    levels = list(level_components.keys())

    # Define primal and dual CMS/DS methods (must match upstream conventions)
    dual_methods = ['DUAL', 'FDCBM', 'DCBM', 'CDCBM', 'CFDCBM']
    primal_methods = ['PRIMAL', 'CBM', 'RM', 'MNM', 'CCBM']

    # -------------------------------------------------------------------------
    # 2) Unpack analysis settings
    # -------------------------------------------------------------------------
    analysis_type = analysis_settings[0]

    if analysis_type == 'HARMONIC':
        # Harmonic response solver settings
        analysis_type, stamps, solver = analysis_settings

    elif analysis_type == 'TRANSIENT':
        # Transient response solver settings
        analysis_type, solver, n_inc, DELTAt, al_solver = analysis_settings

    elif analysis_type == 'MODAL':
        # Modal solver settings
        analysis_type, nmodes, alpha_M, hermitian = analysis_settings

    # -------------------------------------------------------------------------
    # 3) Iterate over TPA levels
    # -------------------------------------------------------------------------
    for idx, level in enumerate(levels):

        # ---------------------------------------------------------------------
        # 3A) Level 0 / AP: solve the assembled problem excited by force_ext
        # ---------------------------------------------------------------------
        if level == levels[0]:
            # Unpack reduced assembled matrices for AP.
            # Convention in your Mat_red: AP has a single entry with key 0.
            K = Mat_red[level][0]['K']
            D = Mat_red[level][0]['D']
            M = Mat_red[level][0]['M']

            # Unpack auxiliary matrices/metadata for AP (same key 0)
            T_red = Coup_info[level][0]['T_red']
            Cmat = Coup_info[level][0]['Cmat']
            ndim = Coup_info[level][0]['ndim']
            shapes = Coup_info[level][0]['shapes']
            # The following are stored but not used explicitly here:
            # coords = Coup_info[level][0]['coords']
            # components = Coup_info[level][0]['components']
            # Abb = Coup_info[level][0]['Abb']
            Bva = Coup_info[level][0]['Bva']
            B = Coup_info[level][0]['B']

            # Project the external force into the reduced assembled coordinates:
            # force_red = T_red.T @ (Cmat @ force_ext)
            force = T_red.T @ Cmat @ force_ext

            # Choose dtype depending on damping/complex loading flags
            if complex_analysis:
                # If damping is ~0 and force is essentially real: solve in real dtype
                if (np.abs(D).max() < 1e-8) and (np.abs(force.imag.max()) < 1e-8):
                    dtype = float
                else:
                    dtype = complex
            else:
                # Force purely real solve and suppress damping
                D = sp.sparse.csc_matrix(D.shape)
                dtype = float

            # Solve assembled AP system
            if analysis_type == 'TRANSIENT':
                print('\nAnalyzing the time response of AP.')
                X_red, stamps = transient_response(
                    K, D, M, force, DELTAt, n_inc,
                    sparse=True, dtype=float, solver=solver, al_solver=al_solver
                )

            elif analysis_type == 'HARMONIC':
                print('\nAnalyzing the harmonic response of AP.')
                X_red = harmonic_response_2(
                    K, D, M, force, stamps, sparse=True, dtype=dtype, solver=solver
                )

            elif analysis_type == 'MODAL':
                print('\nAnalyzing the modes of AP.')
                try:
                    stamps, X_red, _ = compute_modal_basis(
                        K, M, n_modes=nmodes, freq0=0.0, which='LM',
                        sparse=True, alpha_M=alpha_M, hermitian=hermitian
                    )
                    # Keep only eigenvalues with ~zero imaginary part
                    X_red = X_red[:, np.abs(stamps.imag) < 1]
                    stamps = stamps[np.abs(stamps.imag) < 1]
                except:
                    print("Linear algebra error during modal basis computation.")
                    stamps, X_red = np.zeros(nmodes), sp.sparse.csc_matrix((K.shape[0], nmodes))

                X_red = X_red.toarray()

            # Back-project to flattened assembled DOFs
            X_flat = T_red @ X_red

            # Map to physical constrained DOFs and reshape to nodal format
            X = modes_flat_to_3d_3(Cmat.T @ X_flat, ndim=ndim, shapes=shapes, dtype=dtype)

            # Store AP solution. AP path name is '0' by convention.
            if CMS_method[0] in primal_methods:
                X_TPA[level] = {'0': {'X_flat': X_flat, 'X': X, 'path': '', 'stamps': stamps}}

            elif CMS_method[0] in dual_methods:
                # For dual formulations, reconstruct interface forces from the dual variables.
                # B_sf = [ 0, Bva^T, B^T ] so that interface contribution is extracted from X_flat.
                B_sf = sp.sparse.bmat(
                    [[sp.sparse.csc_matrix((B.shape[1], B.shape[1])), Bva.T, B.T]],
                    format='csc'
                )

                # Interface forces (sign convention as in your original code)
                F = modes_flat_to_3d_3(
                    -Cmat[:B.shape[1], :].T @ B_sf @ X_flat,
                    ndim=ndim, shapes=shapes, dtype=complex
                )

                # External forces in nodal format (for optional addition)
                Fext = modes_flat_to_3d_3(force_ext, ndim=ndim, shapes=shapes, dtype=complex)

                if analysis_type == 'MODAL':
                    X_TPA[level] = {'0': {'X_flat': X_flat, 'X': X, 'F': F, 'path': '', 'stamps': stamps}}
                else:
                    # In harmonic/transient: store total (interface + external) in 'F'
                    X_TPA[level] = {'0': {'X_flat': X_flat, 'X': X, 'F': F + Fext, 'path': '', 'stamps': stamps}}

        # ---------------------------------------------------------------------
        # 3B) Following levels: solve per studied interface (paths)
        # ---------------------------------------------------------------------
        else:
            # For each studied active interface at this level
            for interface in level_interfaces[level]:

                # Unpack reduced matrices for this (level, interface)
                K = Mat_red[level][interface]['K']
                D = Mat_red[level][interface]['D']
                M = Mat_red[level][interface]['M']

                # Prescription matrix for this (level, interface)
                K_pres = Mat_pres[level][interface]['K']

                # Unpack auxiliary matrices/metadata for this (level, interface)
                T_red = Coup_info[level][interface]['T_red']
                Cmat = Coup_info[level][interface]['Cmat']
                ndim = Coup_info[level][interface]['ndim']
                shapes = Coup_info[level][interface]['shapes']
                # The following are stored but not used explicitly here:
                # coords = Coup_info[level][interface]['coords']
                # components = Coup_info[level][interface]['components']
                # Abb = Coup_info[level][interface]['Abb']
                Bva = Coup_info[level][interface]['Bva']
                B = Coup_info[level][interface]['B']

                # -------------------------------------------------------------
                # 3B-1) One-level TPA (or the first level after AP)
                # -------------------------------------------------------------
                if type_TPA == '1L-TPA' or idx == 1:
                    # Extract the interface DOFs from the AP solution
                    # NOTE: Access pattern is exactly as in your code (careful with VA)
                    X_b = Coup_info['AP'][0]['Abb'][interface] @ X_TPA['AP']['0']['X_flat']

                    # Equivalent internal forcing that enforces the prescribed interface state
                    force_int = K_pres @ X_b

                    # Choose dtype
                    if complex_analysis:
                        if (np.abs(D).max() < 1e-8) and (np.abs(force_int.imag.max()) == 0.0):
                            dtype = float
                        else:
                            dtype = complex
                    else:
                        D = sp.sparse.csc_matrix(D.shape)
                        dtype = float

                    # Path name is simply the interface id (string)
                    path_name = str(interface)

                    # Solve the level system
                    if analysis_type == 'TRANSIENT':
                        print('\nAnalyzing the time response on path ' + path_name + ' on ' + level)
                        X_red, stamps = transient_response(
                            K, D, M, force_int, DELTAt, n_inc,
                            sparse=True, dtype=float, solver=solver, al_solver=al_solver
                        )

                    if analysis_type == 'HARMONIC':
                        print('\nAnalyzing the harmonic response on path ' + path_name + ' on ' + level)
                        X_red = harmonic_response_2(
                            K, D, M, force_int, stamps, sparse=True, dtype=dtype, solver=solver
                        )

                    # Back-project to flattened assembled DOFs
                    X_flat = T_red @ X_red

                    # Reshape to nodal format
                    X = modes_flat_to_3d_3(Cmat.T @ X_flat, ndim=ndim, shapes=shapes, dtype=dtype)

                    # Store results
                    if CMS_method[0] in primal_methods:
                        if X_TPA.get(level):
                            X_TPA[level][path_name] = {
                                'X_flat': X_flat, 'X': X, 'path': path_name, 'stamps': stamps
                            }
                        else:
                            X_TPA[level] = {
                                path_name: {'X_flat': X_flat, 'X': X, 'path': path_name, 'stamps': stamps}
                            }

                    elif CMS_method[0] in dual_methods:
                        # Interface forces reconstruction for dual methods
                        B_sf = sp.sparse.bmat(
                            [[sp.sparse.csc_matrix((B.shape[1], B.shape[1])), Bva.T, B.T]],
                            format='csc'
                        )
                        F = modes_flat_to_3d_3(
                            -Cmat[:B.shape[1], :].T @ B_sf @ X_flat,
                            ndim=ndim, shapes=shapes, dtype=dtype
                        )

                        if X_TPA.get(level):
                            X_TPA[level][path_name] = {
                                'X_flat': X_flat, 'X': X, 'F': F, 'path': path_name, 'stamps': stamps
                            }
                        else:
                            X_TPA[level] = {
                                path_name: {'X_flat': X_flat, 'X': X, 'F': F, 'path': path_name, 'stamps': stamps}
                            }

                # -------------------------------------------------------------
                # 3B-2) Multi-level TPA (MTPA): chain paths from previous level
                # -------------------------------------------------------------
                elif type_TPA == 'MTPA' and idx > 1:
                    # Load previous level results (i-1)
                    X_TPA_prev = X_TPA[levels[idx - 1]]
                    Coup_info_prev = Coup_info[levels[idx - 1]]

                    # For each previous path, extend it with current interface
                    for interface_prev in X_TPA_prev.keys():
                        # Previous path name (stored in dict)
                        path_name_int_prev = X_TPA_prev[interface_prev]['path']

                        # Concatenate current interface to build longer path name
                        path_name = path_name_int_prev + '-' + str(interface)

                        # Project displacements from previous level onto current level interface selector
                        # NOTE: This indexing is kept identical to your implementation.
                        X_b = (
                            Coup_info_prev[int(interface_prev.split('-')[-1])]['Abb'][interface]
                            @ X_TPA_prev[interface_prev]['X_flat']
                        )

                        # Equivalent internal forcing for current interface
                        force_int = K_pres @ X_b

                        # Choose dtype
                        if complex_analysis:
                            if (np.abs(D).max() < 1e-8) and (np.abs(force_int.imag.max()) == 0.0):
                                dtype = float
                            else:
                                dtype = complex
                        else:
                            D = sp.sparse.csc_matrix(D.shape)
                            dtype = float

                        # Solve
                        if analysis_type == 'TRANSIENT':
                            print('\nAnalyzing the time response on path ' + path_name + ' on ' + level)
                            X_red, stamps = transient_response(
                                K, D, M, force_int, DELTAt, n_inc,
                                sparse=True, dtype=float, solver=solver, al_solver=al_solver
                            )

                        if analysis_type == 'HARMONIC':
                            print('\nAnalyzing the harmonic response on path ' + path_name + ' on ' + level)
                            X_red = harmonic_response_2(
                                K, D, M, force_int, stamps, sparse=True, dtype=dtype, solver=solver
                            )

                        # Back-project and reshape
                        X_flat = T_red @ X_red
                        X = modes_flat_to_3d_3(Cmat.T @ X_flat, ndim=ndim, shapes=shapes, dtype=dtype)

                        # Store results
                        if CMS_method[0] in primal_methods:
                            if X_TPA.get(level):
                                X_TPA[level][path_name] = {
                                    'X_flat': X_flat, 'X': X, 'path': path_name, 'stamps': stamps
                                }
                            else:
                                X_TPA[level] = {
                                    path_name: {'X_flat': X_flat, 'X': X, 'path': path_name, 'stamps': stamps}
                                }

                        elif CMS_method[0] in dual_methods:
                            B_sf = sp.sparse.bmat(
                                [[sp.sparse.csc_matrix((B.shape[1], B.shape[1])), Bva.T, B.T]],
                                format='csc'
                            )
                            F = modes_flat_to_3d_3(
                                -Cmat[:B.shape[1], :].T @ B_sf @ X_flat,
                                ndim=ndim, shapes=shapes, dtype=dtype
                            )

                            if X_TPA.get(level):
                                X_TPA[level][path_name] = {
                                    'X_flat': X_flat, 'X': X, 'F': F, 'path': path_name, 'stamps': stamps
                                }
                            else:
                                X_TPA[level] = {
                                    path_name: {'X_flat': X_flat, 'X': X, 'F': F, 'path': path_name, 'stamps': stamps}
                                }

    # -------------------------------------------------------------------------
    # 4) Close monitoring and store total response metrics
    # -------------------------------------------------------------------------
    computer_results = thread_monitoring(False)

    time_RAM['time']['TOT_response'] = computer_results[0]
    time_RAM['time']['TOT_response_per_it'] = computer_results[0] / stamps.shape[0]
    time_RAM['RAM']['TOT_response'] = computer_results[1]
    time_RAM['RAM']['TOT_response_per_it'] = computer_results[1]

    # -------------------------------------------------------------------------
    # 5) Optional saving of results
    # -------------------------------------------------------------------------
    if save_job:
        if not os.path.exists(job_folder):
            os.makedirs(job_folder)

        joblib.dump(X_TPA, job_folder + 'SOLUTION_' + job_name, compress=3)

        # Save coupling info only if matrices were computed here (not loaded)
        if load_matrices is False:
            joblib.dump(Coup_info, job_folder + 'COUP_INFO_' + job_name, compress=3)

    return X_TPA, Coup_info, time_RAM

#%% Load functions for analysis
def monopole_pressure(points, source, f, p_ref, r_ref, c=340.0, r_min=1e-6):
    """
    Compute the complex acoustic pressure field radiated by an ideal free-field monopole,
    using the **time convention** :math:`e^{i\\omega t}`.

    This implementation assumes a homogeneous, lossless medium and uses the classical
    spherical spreading law with outgoing-wave phase:

        p(r) = p_ref * (r_ref / r) * exp(-i k (r - r_ref))

    where:
      - r is the source-to-field distance,
      - k = 2π f / c is the acoustic wavenumber,
      - p_ref is a known complex pressure at a reference distance r_ref
        (magnitude and phase at the same frequency).

    The pressure is returned as complex amplitude (phasor) at each field point.
    A minimum radius `r_min` is enforced to prevent singularity exactly at the source.

    Parameters
    ----------
    points : (N, 3) array_like
        Field point coordinates [m]. Will be converted to a 2D array of floats.
    source : (3,) array_like
        Source coordinates [m]. Must contain exactly (x, y, z).
    f : float
        Frequency [Hz].
    p_ref : complex or float
        Complex pressure at the reference distance `r_ref` [Pa]. If float, it is treated as
        a real amplitude with zero phase.
    r_ref : float
        Reference distance from the source where `p_ref` applies [m]. Must be > 0.
    c : float, optional
        Speed of sound [m/s]. Default 340.0.
    r_min : float, optional
        Minimum allowed distance [m] to avoid division by zero. Default 1e-6.

    Returns
    -------
    p : (N,) complex ndarray
        Complex pressures at the field points [Pa].
        (Note: despite the original docstring mentioning `(p, r)`, the function **returns only p**
        in its current behavior.)
    """
    # Ensure points is a 2D float array: (N,3)
    pts = np.atleast_2d(points).astype(float)

    # Ensure source is a flat float array of length 3
    src = np.asarray(source, dtype=float).ravel()
    if src.size != 3:
        raise ValueError("source must be length-3 (x, y, z).")
    if r_ref <= 0:
        raise ValueError("r_ref must be > 0.")

    # Source-to-field distances: r_i = ||x_i - x_src||
    r = np.linalg.norm(pts - src, axis=1)

    # Avoid singularity exactly at the source (r=0)
    r = np.maximum(r, r_min)

    # Acoustic wavenumber k = ω/c = 2πf/c
    k = 2.0 * np.pi * f / c

    # Scale from known reference pressure at r_ref:
    # p(r) = p_ref * (r_ref / r) * exp(-ik (r - r_ref))
    p = p_ref * (r_ref / r) * np.exp(-1j * k * (r - r_ref))

    # IMPORTANT: Current behavior returns only p (not (p, r))
    return p

def create_load(subdomains, load_case):
    """
    Build a (possibly frequency-dependent) *global* load vector by concatenating per-subdomain
    load contributions defined in `load_case`.

    This function is a dispatcher that supports multiple load models, each of which produces
    a load vector in the **component's native DOF ordering** (node-major, then DOF components),
    and then concatenates the resulting vectors in the same order that `load_case.items()` yields.

    Supported load types (as implemented here)
    ------------------------------------------
    - 'NODAL_FORCE'
        Applies nodal forces on a set of nodes (named selection or explicit node list),
        optionally distributed across nodes, and mapped to the requested DOF components.

    - 'FRF'
        Uses an FRF-defined input spectrum: interpolates FRF values onto `target_freqs`,
        scales by `input_value`, then applies as nodal forces (distributed or not)
        across requested DOF components. Output is a (Ndof, Nfreq) complex array.

    - 'PRESSURE'
        Converts a pressure load into nodal forces using local nodal surface normals and areas:
            F = p * A * n
        Supports either:
          * constant pressure value (scalar), or
          * spatially varying pressure defined by a dict {'nodes':..., 'values':...}
            where values is (Nnodes, Nloads) and each column is a load case / frequency / stamp.

    - 'MONOPOLE_PRESSURE'
        Computes an acoustic pressure field from one or more monopole sources, then converts
        it to nodal forces via normals and areas:
            F = p(x) * A * n
        Output is (Ndof, Nfreq) complex.

    - 'ACOUSTIC_NORMAL_ACC' (under 'ACOUSTIC' family)
        Converts a prescribed **normal acceleration** (on a surface) into an acoustic load
        proportional to area:
            f = -a_n * A
        (Sign and meaning follow your current convention. No additional frequency handling
        is imposed here; it uses the provided 'values' matrix as-is.)

    - 'NONE'
        Generates a zero load vector either sized from the domain mesh (if domain key exists),
        or sized explicitly from `load_type[1]`.

    Key behavioral notes (kept exactly as in current implementation)
    ----------------------------------------------------------------
    - Node sets can be provided as:
        * a named selection string -> loaded from the component .rst
        * a list/np.ndarray of node IDs
    - DOF mapping uses `dof_map` and expands nodal scalar forces into a full DOF vector by
      stacking per-DOF blocks (X/Y/Z/rotations/P) according to the component `ndim`.
    - If any produced force vector is 2D (frequency dependent), the function returns a 2D
      stacked result; otherwise it returns a 1D concatenation.
    - The exact shape logic at the end (including `np.tile(...)` behavior) is preserved.

    Parameters
    ----------
    subdomains : dict
        Subdomain definition mapping domain key -> (folder, ndim, ..., ..., ...)
        Only folder and ndim are used here.
    load_case : dict
        Mapping domain key -> load definition tuple/list.
        The first element of each load definition is the load type string.

    Returns
    -------
    force_vectors : ndarray
        If all loads are 1D: returns a 1D concatenated force vector.
        If at least one load is 2D: returns a 2D stacked matrix (Ndof_total, Nfreq_like).
    """
    load_types = ['NODAL_FORCE', 'PRESSURE', 'MONOPOLE_PRESSURE', 'NONE', 'ACOUSTIC_NORMAL_ACC']
    dof_map = {'X': 0, 'Y': 1, 'Z': 2, 'ROTX': 3, 'ROTY': 4, 'ROTZ': 5, 'P': 0}

    force_vectors = []

    # Iterate domains in the order given by load_case
    for key, load_type in load_case.items():
        type_of_load = load_type[0]

        # ---------------------------------------------------------------------
        # 1) NODAL_FORCE: nodal forces along selected DOF components
        # ---------------------------------------------------------------------
        if type_of_load == 'NODAL_FORCE':
            folder, ndim, _, _, _, load_dict_rst = subdomains[key]
            nodes_dom, _ = import_nodal_coordinates_from_rst(folder, '', load_data=load_dict_rst)
            num_dofs = nodes_dom.shape[0]

            # Unpack: (_, node_info, force_value, dofs, distributed)
            _, node_info, force_value, dofs, distributed = load_type

            # Resolve node set
            if type(node_info) == str:
                nodes_force, _ = import_nodal_coordinates_from_rst(folder, node_info, load_data=load_dict_rst)
            elif type(node_info) == list or type(node_info) == np.ndarray:
                nodes_force = np.array(node_info)

            # Indices of selected nodes within domain nodes
            ind_force = np.nonzero(np.isin(nodes_dom, nodes_force))[0]

            # Base scalar nodal load (one value per node)
            force = np.zeros(num_dofs)

            # Distributed vs lumped
            if distributed:
                force[ind_force] = (force_value / nodes_force.shape[0]) * np.ones(nodes_force.shape[0])
            else:
                force[ind_force] = (force_value) * np.ones(nodes_force.shape[0])

            # Expand scalar nodal vector into full DOF vector by stacking ndim blocks
            dim_force = [dof_map[axis] for axis in dofs]
            force_vect = [force if i in dim_force else np.zeros(force.shape[0]) for i in range(0, ndim)]
            force = np.concatenate(force_vect)

        # ---------------------------------------------------------------------
        # 2) FRF: frequency-dependent nodal force from interpolated FRF data
        # ---------------------------------------------------------------------
        elif type_of_load == 'FRF':
            folder, ndim, _, _, _, load_dict_rst = subdomains[key]
            nodes_dom, _ = import_nodal_coordinates_from_rst(folder, '', load_data=load_dict_rst)
            num_dofs = nodes_dom.shape[0]

            # Unpack:
            # (_, node_info, frf_freqs, frf_values, input_value, target_freqs, dofs, distributed)
            _, node_info, frf_freqs, frf_values, input_value, target_freqs, dofs, distributed = load_type

            # Resolve node set
            if type(node_info) == str:
                nodes_force, _ = import_nodal_coordinates_from_rst(folder, node_info, load_data=load_dict_rst)
            elif type(node_info) == list or type(node_info) == np.ndarray:
                nodes_force = np.array(node_info)

            ind_force = np.nonzero(np.isin(nodes_dom, nodes_force))[0]

            # Interpolate FRF onto target frequencies and scale by input amplitude
            force_values = np.interp(target_freqs, frf_freqs, frf_values) * input_value
            n_freqs = target_freqs.shape[0]

            # Frequency-dependent scalar nodal load: (num_nodes, n_freqs)
            force = np.zeros((num_dofs, n_freqs), dtype=complex)

            # Apply load on selected nodes for each frequency
            for j in range(n_freqs):
                if distributed:
                    force[ind_force, j] = (force_values[j] / nodes_force.shape[0]) * np.ones(nodes_force.shape[0])
                else:
                    force[ind_force, j] = (force_values[j]) * np.ones(nodes_force.shape[0])

            # Expand into ndim DOF blocks (stack vertically)
            dim_force = [dof_map[axis] for axis in dofs]
            force_vect = [force if i in dim_force else np.zeros(force.shape[0]) for i in range(0, ndim)]
            force = np.vstack(force_vect)

        # ---------------------------------------------------------------------
        # 3) PRESSURE family: convert pressure field to nodal forces via normals/areas
        # ---------------------------------------------------------------------
        elif 'PRESSURE' in type_of_load:
            folder, ndim, _, _, _, load_dict_rst = subdomains[key]
            nodes_dom, _ = import_nodal_coordinates_from_rst(folder, '', load_data=load_dict_rst)
            num_dofs = nodes_dom.shape[0]

            node_info = load_type[1]

            # Load surface nodes + geometry (normals, areas)
            nodes_force, coords_force, normals_force, areas_force = import_nodal_coordinates_from_rst(
                folder, node_info, compute_normals=True, load_data=load_dict_rst
            )

            # Indices of these surface nodes in the global domain node array
            ind_force = np.nonzero(np.isin(nodes_dom, nodes_force))[0]

            # -----------------------------------------------------------------
            # 3A) Constant/spatially-varying pressure field defined directly
            # -----------------------------------------------------------------
            if type_of_load == 'PRESSURE':
                _, _, pressure_value = load_type

                if type(pressure_value) == dict:
                    # Pressure values provided per node as a matrix (Nnodes, Nloads)
                    load_dict = pressure_value
                    nodes_ids, load_values = load_dict['nodes'], load_dict['values']

                    # Reorder/load only nodes that are in the surface set
                    ind_ordering = np.nonzero(np.isin(nodes_force, nodes_ids))[0]
                    nodes_ids = nodes_ids[ind_ordering]
                    load_values = load_values[ind_ordering, :]

                    n_loads = load_values.shape[1]

                    # Recompute indices for nodes_ids (not necessarily same as nodes_force)
                    ind_force = np.nonzero(np.isin(nodes_dom, nodes_ids))[0]

                    # Force tensor: (num_nodes, n_loads, ndim)
                    force = np.zeros((num_dofs, n_loads, ndim), dtype=complex)

                    # Ensure areas are column-shaped for broadcasting
                    areas_force = np.reshape(areas_force, (areas_force.shape[0], 1))

                    # For each load column: convert p -> nodal force vector
                    for i in range(0, n_loads):
                        load = np.reshape(load_values[:, i], (load_values.shape[0], 1))
                        # Only fill the translational components available in normals_force
                        force[ind_force, i, :normals_force.shape[1]] = load * areas_force * normals_force

                    # Flatten into stacked DOF blocks (Ndof_total, n_loads)
                    force = np.vstack([force[:, :, i] for i in range(0, ndim)])

                else:
                    # Constant pressure scalar -> static nodal force vector
                    force = np.zeros((num_dofs, ndim))

                    areas_force = np.reshape(areas_force, (areas_force.shape[0], 1))

                    # Apply F = p * A * n (for available normal components)
                    force[ind_force, :normals_force.shape[1]] = pressure_value * areas_force * normals_force

                    # Flatten to 1D DOF vector
                    force = np.concatenate([force[:, i] for i in range(0, ndim)])

            # -----------------------------------------------------------------
            # 3B) Monopole-based pressure field
            # -----------------------------------------------------------------
            elif type_of_load == 'MONOPOLE_PRESSURE':
                _, _, sources = load_type

                pressure_value = []
                n_force = nodes_force.shape[0]

                # For each monopole: compute p(x, f) at all surface nodes
                for source_coords, p0, r0, f_array in sources:
                    pressure_source = np.zeros((n_force, f_array.shape[0]), dtype=complex)
                    for j, f in enumerate(f_array):
                        pressure_source[:, j] = monopole_pressure(coords_force, source_coords, f, p0, r0)
                    pressure_value.append(pressure_source)

                # Sum contributions from all sources (superposition)
                pressure_value = sum(pressure_value)
                n_freqs = pressure_value.shape[1]

                # Force tensor: (num_nodes, n_freqs, ndim)
                force = np.zeros((num_dofs, n_freqs, ndim), dtype=complex)

                # Compute force at the surface nodes (n_force, n_freqs, 3-like)
                force_at_nodes = np.zeros((n_force, n_freqs, normals_force.shape[1]), dtype=complex)
                for j in range(0, n_freqs):
                    force_mag = np.reshape(pressure_value[:, j] * areas_force, (areas_force.shape[0], 1))
                    force_at_nodes[:, j, :] = force_mag * normals_force

                # Assign into global node indexing
                force[ind_force, :, :normals_force.shape[1]] = force_at_nodes

                # Flatten into stacked DOF blocks (Ndof_total, n_freqs)
                force = np.vstack([force[:, :, i] for i in range(0, ndim)])

        # ---------------------------------------------------------------------
        # 4) ACOUSTIC family: loads that create an acoustic forcing from kinematics
        # ---------------------------------------------------------------------
        elif 'ACOUSTIC' in type_of_load:
            folder, ndim, _, _, _, load_dict_rst = subdomains[key]
            nodes_dom, _ = import_nodal_coordinates_from_rst(folder, '', load_data=load_dict_rst)
            num_dofs = nodes_dom.shape[0]

            node_info = load_type[1]

            # Load surface nodes + geometry (normals, areas)
            nodes_force, coords_force, normals_force, areas_force = import_nodal_coordinates_from_rst(
                folder, node_info, compute_normals=True, load_data=load_dict_rst
            )

            # Only ACOUSTIC_NORMAL_ACC is implemented in this code path
            if type_of_load == 'ACOUSTIC_NORMAL_ACC':
                _, _, load_dict = load_type

                nodes_ids, load_values = load_dict['nodes'], load_dict['values']

                # Align ordering with nodes_force
                ind_ordering = np.nonzero(np.isin(nodes_force, nodes_ids))[0]
                nodes_ids = nodes_ids[ind_ordering]
                load_values = load_values[ind_ordering, :]

                n_loads = load_values.shape[1]

                ind_force = np.nonzero(np.isin(nodes_dom, nodes_ids))[0]

                force = np.zeros((num_dofs, n_loads, ndim), dtype=complex)

                areas_force = np.reshape(areas_force, (areas_force.shape[0], 1))

                for i in range(0, n_loads):
                    load = np.reshape(load_values[:, i], (load_values.shape[0], 1))
                    # Current convention: f = -a * A
                    force[ind_force, i, :normals_force.shape[1]] = -load * areas_force

                force = np.vstack([force[:, :, i] for i in range(0, ndim)])

        # ---------------------------------------------------------------------
        # 5) NONE: explicitly no load
        # ---------------------------------------------------------------------
        elif type_of_load == 'NONE':
            if key in subdomains.keys():
                folder, ndim, _, _, _, load_dict_rst = subdomains[key]
                nodes_dom, _ = import_nodal_coordinates_from_rst(folder, '', load_data=load_dict_rst)
                num_dofs = nodes_dom.shape[0]
                force = np.zeros(num_dofs * ndim)
            else:
                # If component is not present: load_type[1] provides target length
                force = np.zeros(load_type[1])

        else:
            raise ValueError(f'The only possible load types are: {", ".join(load_types)}')

        force_vectors.append(force)

    # -------------------------------------------------------------------------
    # 6) Concatenate/stack across subdomains (preserve original behavior)
    # -------------------------------------------------------------------------
    second_dim = 0
    for force in force_vectors:
        if len(force.shape) == 2:
            second_dim = force.shape[1]

    if second_dim == 0:
        # All forces are 1D -> concatenate into a single 1D vector
        force_vectors = np.concatenate(force_vectors)
    else:
        # At least one force is 2D -> return a 2D matrix.
        # Preserve exact tiling logic: 1D forces are replicated across second_dim columns.
        force_vectors = np.vstack([
            np.tile(force, (second_dim, 1)).T if len(force.shape) != 2 else force
            for force in force_vectors
        ])

    return force_vectors


def n_octave_bands(freqs, P_complex, n, fmin=None, fmax=None, fref=1000.0):
    """
    Coherent (complex) 1/n-octave band averaging of multi-sensor spectra.

    This function performs **band integration in the complex domain** (coherent averaging),
    which preserves phase relationships across frequency within each band. This is different
    from power/energy averaging, which would average |P|^2 instead.

    The procedure is:
      1) Validate frequency grid and spectrum array.
      2) Estimate per-bin bandwidths `df` using midpoints (works for non-uniform frequency grids).
      3) Build the set of 1/n-octave band center frequencies within [fmin, fmax], anchored at `fref`.
      4) For each band, integrate complex spectrum across bins inside [fl, fh):
            p_band = (∑ P(f_k) * df_k) / BW
         where BW = fh - fl.
         This yields a band-averaged complex pressure per sensor.

    Band definitions
    ---------------
    For each band center fc:
      - ratio between adjacent centers: r = 2^(1/n)
      - edges: fl = fc / 2^(1/(2n)), fh = fc * 2^(1/(2n))

    Parameters
    ----------
    freqs : (Nf,) array_like
        Frequency axis in Hz. Must be strictly increasing.
    P_complex : (Nsensors, Nf) array_like
        Complex spectrum values. Each row is a sensor; each column corresponds to `freqs`.
    n : int
        Octave division (e.g., n=3 for 1/3-octave bands).
    fmin, fmax : float, optional
        Band limits. Defaults to min/max of `freqs`.
    fref : float, optional
        Reference/anchor center frequency (default 1000 Hz).

    Returns
    -------
    fc : (Nb,) ndarray
        Band center frequencies [Hz].
    p_complex : (Nsensors, Nb) ndarray
        Complex band-averaged spectrum per sensor.
    """
    # -------------------------------------------------------------------------
    # 1) Sanitize inputs
    # -------------------------------------------------------------------------
    f = np.asarray(freqs, dtype=float).ravel()
    P = np.asarray(P_complex, dtype=complex)
    n = int(n)

    if f.ndim != 1:
        raise ValueError("freqs must be 1D.")
    if P.ndim != 2:
        raise ValueError("P_complex must be 2D (Nsensors x Nfreqs).")

    Ns, Nf = P.shape
    if Nf != f.size:
        raise ValueError(f"P_complex has {Nf} columns but freqs has {f.size} entries.")
    if np.any(np.diff(f) <= 0):
        raise ValueError("freqs must be strictly increasing.")

    if fmin is None:
        fmin = f[0]
    if fmax is None:
        fmax = f[-1]
    if fmin <= 0:
        raise ValueError("fmin must be > 0.")

    # -------------------------------------------------------------------------
    # 2) Bin bandwidths from midpoints (supports non-uniform frequency grids)
    # -------------------------------------------------------------------------
    mids = 0.5 * (f[1:] + f[:-1])
    left_edges = np.empty_like(f)
    right_edges = np.empty_like(f)

    # Internal edges are midpoints
    left_edges[1:], right_edges[:-1] = mids, mids

    # Extrapolate outer edges assuming same local spacing
    left_edges[0] = f[0] - (right_edges[0] - f[0])
    right_edges[-1] = f[-1] + (f[-1] - left_edges[-1])

    # Per-bin bandwidth
    df = right_edges - left_edges  # shape (Nf,)

    # -------------------------------------------------------------------------
    # 3) Build 1/n-octave band centers within [fmin, fmax]
    # -------------------------------------------------------------------------
    r = 2 ** (1 / n)                 # center frequency step ratio
    edge_factor = 2 ** (1 / (2 * n)) # factor for band edges around center

    # Start index k such that center is near fmin
    k = int(np.floor(np.log(fmin / fref) / np.log(r)))

    centers = []
    while True:
        fc_val = fref * (r ** k)
        if fc_val > fmax * edge_factor:
            break
        if fc_val >= fmin / edge_factor:
            centers.append(fc_val)
        k += 1

    fc = np.array(centers, dtype=float)  # (Nb,)

    # -------------------------------------------------------------------------
    # 4) Coherent integration in each band and bandwidth normalization
    # -------------------------------------------------------------------------
    Nb = fc.size
    p_complex = np.zeros((Ns, Nb), dtype=complex)

    for j, c in enumerate(fc):
        fl, fh = c / edge_factor, c * edge_factor

        # Include bins in [fl, fh)
        idx = (f >= fl) & (f < fh)
        if not np.any(idx):
            # No frequency bins in this band -> leave as zero
            continue

        BW = fh - fl

        # Coherent band average:
        # sum(P * df) / BW for each sensor
        p_complex[:, j] = (P[:, idx] * df[idx]).sum(axis=1) / BW

    return fc, p_complex


#%% Post-processing and visualization
def plot_contribution_bars(
    M, freqs=None, contributors=None, cmap_name="jet",
    group_width=0.95, ylabel="Contribution (dB)", xlabel="Frequency",
    title="Overall Path Contributions", ylim=None, in_third_octave_bands=False,
    fontsize=20, figsize=(16, 9), legend=False, savefig=''
):
    """
    Plot grouped bar charts of *path / component contributions* versus frequency.

    This routine is designed for Transfer Path Analysis (TPA) style post-processing where
    each “contributor” (path, interface, component, etc.) provides a contribution level
    (typically in dB) per frequency line, and the goal is to compare their relative impact
    across the spectrum.

    What the function does
    ----------------------
    1) Interprets `M` as a matrix of shape (n_contributors, n_freqs):
       - rows: contributors / paths
       - columns: frequency points (or any x-axis bins)

    2) Optionally aggregates the spectrum into fixed **third-octave bands** when
       `in_third_octave_bands=True`. The implemented aggregation is:
       - convert each dB value to a linear pressure amplitude (Pa) using pref=1e-5,
       - sum energies (p²) across the frequency bins that fall in each band,
       - convert back to SPL in dB using `calculate_spl(sqrt(sum(p²)))`.
       This is an energetic aggregation per contributor (row-wise).

       IMPORTANT: This “third octave” mode uses a fixed center-frequency dictionary and
       octave edges fc/√2 .. fc√2, matching the current code behavior exactly.

    3) Draws grouped bars with consistent colors and optional legend. If `legend=False`,
       it writes a vertical label on each bar and auto-shrinks the label to fit the bar width.

    Parameters
    ----------
    M : array_like, shape (n_contributors, n_freqs)
        Contribution matrix. Rows correspond to contributors, columns to frequency bins.
    freqs : array_like, optional
        Frequency labels (length n_freqs). If None, uses integer indices.
        When `in_third_octave_bands=True`, these are interpreted as the frequency axis
        used for bin selection.
    contributors : list[str], optional
        Names for each contributor (length n_contributors). If None, uses "C0..".
    cmap_name : str, optional
        Matplotlib colormap name used to color contributors.
    group_width : float, optional
        Width allocated per frequency group. Bar width is group_width/n_contributors.
    ylabel, xlabel, title : str, optional
        Axis/title labels.
    ylim : (ymin, ymax), optional
        y-axis limits.
    in_third_octave_bands : bool, optional
        If True, convert `M` into third-octave band levels before plotting.
    fontsize : float, optional
        Base fontsize used for labels and title.
    figsize : (w, h), optional
        Figure size in inches.
    legend : bool, optional
        If True, show a legend above the plot. If False, label bars directly.
    savefig : str, optional
        If non-empty, path where the figure is saved via `plt.savefig(savefig)`.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The created figure and axis.
    """
    M = np.asarray(M)
    n_contrib, n_freq = M.shape

    # -------------------------------------------------------------------------
    # Optional: third-octave band aggregation (behavior preserved as-is)
    # -------------------------------------------------------------------------
    if in_third_octave_bands:
        band_centers = {
            1: 16.0, 2: 20, 3: 25, 4: 31.5, 5: 40, 6: 50, 7: 63, 8: 80, 9: 100,
            10: 125, 11: 160, 12: 200, 13: 250, 14: 315, 15: 400, 16: 500,
            17: 630, 18: 800, 19: 1000
        }

        out = []
        freqs = freqs.astype(float)

        # Keep only bands that overlap the provided frequency range
        band_centers = {
            band: fc for band, fc in band_centers.items()
            if (freqs.min() < fc * np.sqrt(2)) * (freqs.max() > fc / np.sqrt(2))
        }

        for band, fc in band_centers.items():
            f1, f2 = fc / np.sqrt(2), fc * np.sqrt(2)  # octave edges in this implementation
            mask = (freqs >= f1) & (freqs < f2)
            if not mask.any():
                out.append(np.full(M.shape[0], np.nan))
            else:
                # Convert dB -> Pa amplitude, sum energies, convert back to dB
                pref = 1e-5
                p = pref * 10 ** (M[:, mask] / 20.0)
                p2 = np.sum(p ** 2, axis=1)
                Lp = calculate_spl(np.sqrt(p2))
                out.append(Lp)

        # NOTE: Current code uses band_centers.keys() in the label string (not fc values)
        freqs = np.array([f'{fc:.1f}' for fc in band_centers.keys()])
        M = np.vstack(out).T  # -> (n_contrib, n_bands)
        n_contrib, n_freq = M.shape

    # -------------------------------------------------------------------------
    # Defaults for labels
    # -------------------------------------------------------------------------
    if freqs is None:
        freqs = np.arange(n_freq)
    if contributors is None:
        contributors = [f"C{i}" for i in range(n_contrib)]

    x = np.arange(n_freq)
    bar_width = group_width / n_contrib
    offsets = (-group_width / 2) + (np.arange(n_contrib) + 0.5) * bar_width

    cmap = plt.cm.get_cmap(cmap_name, n_contrib)
    colors = [cmap(i) for i in range(n_contrib)]

    fig, ax = plt.subplots(figsize=figsize)

    # -------------------------------------------------------------------------
    # Draw bars
    # -------------------------------------------------------------------------
    for i in range(n_contrib):
        ax.bar(
            x + offsets[i],
            M[i, :],
            width=bar_width,
            label=contributors[i],
            color=colors[i],
            edgecolor="black",
            linewidth=0.4
        )

    # Vertical separators between groups
    for xpos in range(len(freqs) + 1):
        ax.axvline(x=xpos - 0.5, color="black", linestyle="-", linewidth=0.5)

    ax.set_xlim([-0.5, len(freqs) - 0.5])
    ax.set_xticks(x)
    ax.set_xticklabels(freqs, rotation=45)
    ax.tick_params(axis='x', labelsize=fontsize - 2)
    ax.set_xlabel(xlabel, size=fontsize - 2)
    ax.set_ylabel(ylabel, size=fontsize - 2)
    ax.tick_params(axis='y', labelsize=fontsize - 2)
    ax.axhline(0, linewidth=0.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.4)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.subplots_adjust(top=0.75)

    # -------------------------------------------------------------------------
    # Legend or direct bar labels (behavior preserved)
    # -------------------------------------------------------------------------
    if legend:
        leg = ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.05),
            ncols=min([len(contributors), 6]),
            fontsize=fontsize - 2
        )

        # Shrink legend font until it fits (approximate heuristic in current code)
        font_size = fontsize - 2
        while (leg.get_window_extent().width > 2 * ax.get_window_extent().width) and font_size > 8:
            font_size -= 0.5
            leg = ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.05),
                ncols=min([len(contributors), 6]),
                fontsize=font_size
            )
    else:
        # Render once so we can measure extents
        fig.canvas.draw()

        for i in range(n_contrib):
            for j in range(n_freq):
                height = M[i, j]
                if np.isnan(height):
                    continue

                xpos = x[j] + offsets[i]
                ypos = height

                # Offset text slightly above/below bar tip.
                offset_y = np.abs(ylim[1] - ylim[0]) / 100 if type(ylim) != None else 0

                txt = ax.text(
                    xpos,
                    ypos + offset_y if ypos >= 0 else ypos - offset_y,
                    contributors[i],
                    rotation=90,
                    ha="center",
                    va="bottom" if ypos >= 0 else 'top',
                    fontsize=fontsize,
                    clip_on=True
                )

                # Compute bar width in pixels and shrink text to fit
                x0 = ax.transData.transform((xpos - bar_width / 2, ypos))[0]
                x1 = ax.transData.transform((xpos + bar_width / 2, ypos))[0]
                bar_width_px = 1.5 * np.abs(x1 - x0)

                def autosize_text_to_bar(ax, text_obj, bar_width_pixels, min_size=6):
                    """
                    Shrink the fontsize of `text_obj` until its pixel width fits inside `bar_width_pixels`.
                    """
                    renderer = ax.figure.canvas.get_renderer()
                    fontsize_local = text_obj.get_fontsize()

                    while fontsize_local > min_size:
                        text_obj.set_fontsize(fontsize_local)
                        bbox = text_obj.get_window_extent(renderer=renderer)
                        if bbox.width <= bar_width_pixels:
                            return fontsize_local
                        fontsize_local -= 0.1

                    return fontsize_local

                autosize_text_to_bar(ax, txt, bar_width_px)

    ax.set_title(title, size=fontsize)
    plt.tight_layout()

    if savefig != '':
        plt.savefig(savefig)

    return fig, ax

def plot_contribution_lines2(
    M, freqs, contributors, cmap_name="jet",
    ylabel="Contribution (dB)", xlabel="Frequency",
    title="Overall Path Contributions", ylim=None, fontsize=20,
    figsize=(16, 9)
):
    """
    Plot contribution spectra as lines, separating constructive and destructive regions.

    This function visualizes a contribution matrix `M` (contributors × frequencies) using
    line plots:
      - positive contributions are drawn as **solid** lines,
      - negative contributions are drawn as **dashed** lines,
    while keeping the same contributor color for both signs.

    The separation is implemented by creating two masked copies of `M`:
      - `pos_M`: values < 0 set to NaN
      - `neg_M`: values >= 0 set to NaN
    which allows matplotlib to break the line where the sign changes.

    Parameters
    ----------
    M : array_like, shape (n_contributors, n_freqs)
        Contribution values.
    freqs : array_like, shape (n_freqs,)
        Frequency axis (or any x-axis values).
    contributors : list[str], length n_contributors
        Labels for each contributor line.
    cmap_name : str, optional
        Colormap name used to assign colors to contributors.
    ylabel, xlabel, title : str, optional
        Axis labels and title.
    ylim : (ymin, ymax), optional
        y-axis limits (NOTE: currently not applied in the existing code; kept unchanged).
    fontsize : float, optional
        Font size base.
    figsize : (w, h), optional
        Figure size in inches.

    Returns
    -------
    None
        (This function currently does not return fig/ax; it directly uses pyplot.)
    """
    n_contr = M.shape[0]
    cmap = plt.cm.get_cmap(cmap_name, n_contr)
    colors = cmap(np.linspace(0, 1, n_contr))

    # Separate positive and negative contributions
    pos_M = M.copy()
    neg_M = M.copy()
    pos_M[M < 0] = np.nan
    neg_M[M >= 0] = np.nan

    plt.figure(figsize=figsize)
    [plt.plot(freqs, pos_M[i], color=colors[i], linestyle='solid', label=contributors[i]) for i in range(0, n_contr)]
    [plt.plot(freqs, neg_M[i], color=colors[i], linestyle='dashed') for i in range(0, n_contr)]
    plt.ylabel(ylabel, size=fontsize)
    plt.xlabel(xlabel, size=fontsize)
    plt.title(title, size=fontsize + 2)
    plt.xlim([freqs[0], freqs[-1]])
    plt.legend(fontsize=fontsize - 1)
    plt.tick_params(axis="x", labelsize=fontsize - 2)
    plt.tick_params(axis="y", labelsize=fontsize - 2)
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid()

def plot_stacked_heat_strips(
    M, x=None, row_labels=None, cmap_name="jet",
    vmin=None, vmax=None, cbar_label="dB\nPa", xlabel="rpm",
    title=None, figsize=(12, 7), fontsize=14, strip_height=0.85,
    gap=0.12, add_vertical_line=None, line_kwargs=None, savefig=''
):
    """
    Plot a “stacked heat-strip” visualization: one horizontal heatmap per row.

    This plot style is useful for showing multiple signals/paths/components (rows) over a
    common x-axis (columns) such as frequency, time, or RPM. Each row is rendered as a
    thin 1×N image (a “strip”), and all strips share the same x-axis and colormap scaling.

    Scaling and robustness
    ----------------------
    If `vmin`/`vmax` are not provided, the function chooses defaults based on the data:
      - If any values are negative, `vmin` becomes `-abs(M).max()` and `vmax` becomes `abs(M).max()`.
      - Otherwise, `vmin` becomes `M.min()` and `vmax` becomes `abs(M).max()`.
    This preserves the current code’s symmetric behavior for signed data.

    Optional vertical marker
    ------------------------
    `add_vertical_line` can be:
      - an int: interpreted as a column index, converted to x-value,
      - a float: interpreted directly as an x-value.

    Parameters
    ----------
    M : array_like, shape (n_rows, n_x)
        Values to plot.
    x : array_like, shape (n_x,), optional
        X coordinates. If None, uses indices [0..n_x-1].
    row_labels : list[str], optional
        Label for each strip (n_rows).
    cmap_name : str, optional
        Colormap name.
    vmin, vmax : float, optional
        Colormap limits. If None, computed as described above.
    cbar_label : str, optional
        Colorbar label.
    xlabel : str, optional
        X-axis label (only shown on bottom strip).
    title : str, optional
        Figure title (suptitle).
    figsize : (w, h), optional
        Figure size in inches.
    fontsize : float, optional
        Base font size.
    strip_height : float, optional
        Relative strip thickness inside each axis (0..1 range).
    gap : float, optional
        Vertical spacing between strips.
    add_vertical_line : float|int, optional
        Adds a vertical marker line to each strip.
    line_kwargs : dict, optional
        Keyword arguments passed to `ax.axvline`.
    savefig : str, optional
        If non-empty, save the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list[matplotlib.axes.Axes]
        List of strip axes (length n_rows).
    """
    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError("M must be a 2D array of shape (n_rows, n_x).")
    n_rows, n_x = M.shape

    if x is None:
        x = np.arange(n_x)
    x = np.asarray(x)
    if x.shape[0] != n_x:
        raise ValueError("x must have length n_x (number of columns in M).")

    if row_labels is None:
        row_labels = [f"Row {i}" for i in range(n_rows)]
    if len(row_labels) != n_rows:
        raise ValueError("row_labels must have length n_rows.")

    finite_vals = M[np.isfinite(M)]
    if finite_vals.size == 0:
        raise ValueError("M contains no finite values to plot.")

    if vmin is None:
        if np.any(M < 0):
            vmin = -np.abs(M).max()
        else:
            vmin = M.min()
    if vmax is None:
        vmax = np.abs(M).max()
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    if line_kwargs is None:
        line_kwargs = dict(color="k", linewidth=1.0, alpha=0.7)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        sharex=True,
        figsize=figsize,
        gridspec_kw=dict(hspace=gap),
    )
    if n_rows == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap(cmap_name)
    xmin, xmax = x.min(), x.max()

    last_im = None
    for i, ax in enumerate(axes):
        strip = M[i:i + 1, :]
        last_im = ax.imshow(
            strip,
            aspect="auto",
            interpolation="bilinear",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[xmin, xmax, 0, 1],
        )

        ax.set_ylim(0.5 - strip_height / 2, 0.5 + strip_height / 2)
        ax.set_yticks([0.5])
        ax.set_yticklabels([row_labels[i]], fontsize=(fontsize - 2) / len(axes) * 4)
        ax.tick_params(axis="y", length=0)

        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)

        if add_vertical_line is not None:
            if isinstance(add_vertical_line, (int, np.integer)):
                idx = int(add_vertical_line)
                idx = max(0, min(n_x - 1, idx))
                xline = x[idx]
            else:
                xline = float(add_vertical_line)
            ax.axvline(x=xline, **line_kwargs)

    axes[-1].set_xlabel(xlabel, fontsize=fontsize)
    axes[-1].tick_params(axis="x", labelsize=fontsize - 2)

    if title:
        fig.suptitle(title, fontsize=fontsize + 2, y=0.98)

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.1, pad=0.00)
    cbar.set_ticks(np.linspace(vmin, vmax, max(3, 2 * (figsize[1] // 2) - 1)).astype(int))
    cbar.set_label(cbar_label, fontsize=fontsize - 2)
    cbar.ax.tick_params(labelsize=fontsize - 2)

    fig.subplots_adjust(top=0.8, bottom=0.20, right=0.8)

    if savefig != '':
        plt.savefig(savefig)

    return fig, axes

def plot_TPA_contribution_results_3(
    X, Coup_info, level, components, path_name, var, freq_set, section=[None, []], nodes_sel=[], descriptions={},
    ylabel='Contribution', cmap_name="jet", plot_title='Overall Path Contributions', xlabel='Frequency [Hz]', ylim=None,
    in_n_octave_bands=[False, 1], fontsize=20, figsize=(16, 9), legend=True, plot_type='contr_strips', savefig=''
):
    """
    Compute and visualize TPA contribution results as “paths × frequency” maps.

    This routine is a post-processor for the solution dictionary produced by your vibro-acoustic
    TPA workflow (e.g., `TPA_CMS_vibroacoustic`). It extracts responses (displacements/forces
    on the structural side and pressure/acoustic loads on the fluid side), computes an
    average *signed* contribution metric per path and per frequency, and provides multiple
    plotting options (bars, lines, heat-strips).

    Core idea of the contribution metric
    ------------------------------------
    For a given level and set of paths, the function forms:
      - a path response (or force) field: X_path
      - a total response field over all paths: X_total
    and computes a per-node scalar product in the complex domain, reduced to a real signed
    quantity by combining real and imaginary parts (current implementation preserves your
    exact formulas).

    The node-wise contributions are then spatially averaged (mean over nodes), optionally
    restricted to a geometric section.

    Variables and domains
    ---------------------
    - Displacement-like variables use the structural indices (comp_s).
    - Pressure-like variables use the acoustic indices (comp_f).
    - If `'FORCE' in var`, the function reads `X[level][path]['F']` instead of `...['X']`.

    Band aggregation
    ---------------
    If `in_n_octave_bands[0]` is True, contributions are aggregated into 1/n-octave bands
    using your `n_octave_bands` routine. The band aggregation is applied to the *already
    spatially-averaged* contribution matrix.

    Plotting
    --------
    - 'contr_bars'   -> `plot_contribution_bars`
    - 'contr_lines'  -> `plot_contribution_lines2`
    - 'contr_strips' -> `plot_stacked_heat_strips` of signed contributions
    - 'mag_strips'   -> `plot_stacked_heat_strips` of amplitudes

    Parameters
    ----------
    X : dict
        TPA results dict (levels -> paths -> fields), as produced by your solvers.
    Coup_info : dict
        Coupling/assembly metadata dict (levels -> interfaces -> indices/coords/etc.).
    level : str
        Level name key in X/Coup_info (e.g., 'AP', 'P1', ...).
    components : (list[str], list[str])
        Tuple of (structural_components, acoustic_components) used to filter indices.
    path_name : list[str]
        Paths to process. Each entry can be:
          - a single path key like '3' or '2-7'
          - a sum 'a+b+c' (superposition)
          - 'ALL' to include all paths at the selected level
    var : str
        Requested variable key (e.g., 'P', 'SPL', 'UNORM', 'UX', 'FORCE-UNORM', ...).
    freq_set : array_like
        Indices of stamps/frequency bins to process. (If octave-banding is enabled,
        the function first processes all bins and then bands them.)
    section : [axis, [min, max]], optional
        Spatial filtering: axis in {'X','Y','Z'} and range [min,max].
        If section[0] is None, no spatial filtering is applied.
    descriptions : dict, optional
        Map path keys to human-readable strings for labels.
    ylabel, cmap_name, plot_title, xlabel : plot formatting
    ylim : tuple, optional
        y-limits used by the plotting function(s).
    in_n_octave_bands : [bool, n, (optional) (fmin,fmax)]
        If enabled, groups results into 1/n-octave bands. Optional third element defines
        a frequency window.
    fontsize, figsize : plot formatting (can be scalar or sequences as used in your code)
    legend : bool
        Legend behavior for bar plots.
    plot_type : str
        'contr_bars', 'contr_lines', 'contr_strips', or 'mag_strips'.
    savefig : str
        If non-empty, path to save the plot.

    Returns
    -------
    contribution_result : (n_paths, n_freq_or_bands) ndarray
        Signed contribution metric, post-processed into the requested units.
    amp_phases_result : (n_paths, n_freq_or_bands, 2) ndarray
        Mean amplitude and mean phase (deg) per path and bin.
    path_name_title : (n_paths,) ndarray
        Rendered labels.
    freq_range_names : (n_freq_or_bands,) ndarray
        String labels for x-axis.
    """
    if in_n_octave_bands[0]:
        freq_set = np.arange(0, X[level][list(X[level].keys())[0]]['stamps'].shape[0], 1)

    n_rows = len(path_name)
    n_cols = len(freq_set)

    comp_s, comp_f = components

    contribution_result = np.zeros((n_rows, n_cols))
    amp_phases_result = np.zeros((n_rows, n_cols, 2))
    path_name_title = np.array([None] * n_rows)

    displacement_vars = [
        'UNORM', 'ULOG', 'UVECT', 'UX', 'UY', 'UZ', 'ROT-X', 'ROT-Y', 'ROT-Z', 'NNORM-U', 'CONST-U',
        'FORCE-UNORM', 'FORCE-UX', 'FORCE-UY', 'FORCE-UZ', 'FORCE-ROTX', 'FORCE-ROTY', 'FORCE-ROTZ',
        'FORCE-UVECT', 'FORCE-ROTVECT'
    ]

    for row, path_i in enumerate(path_name):
        paths = path_i.split('+')
        path_init = paths[0]
        if paths[0] == 'ALL':
            paths = list(X[level].keys())
            if '0' in paths and level != 'AP':
                paths.remove('0')
        path0 = paths[0]

        intf = int(path0.split('-')[-1])
        stamps = X[level][path0]['stamps']

        component_indices = Coup_info[level][intf]['component_indices']
        indices_s = sum(component_indices[comp] for comp in comp_s if comp in component_indices)
        indices_f = sum(component_indices[comp] for comp in comp_f if comp in component_indices)

        indices_s = indices_s.astype(bool) if isinstance(indices_s, np.ndarray) else np.zeros(X[level][path0]['X'].shape[0], dtype=bool)
        indices_f = indices_f.astype(bool) if isinstance(indices_f, np.ndarray) else np.zeros(X[level][path0]['X'].shape[0], dtype=bool)

        if 'FORCE' in var:
            u_Pi = sum(X[level][path]['F'][indices_s] for path in paths)
        else:
            u_Pi = sum(X[level][path]['X'][indices_s] for path in paths)
            
        nodes_s = Coup_info[level][intf]['nodes'][indices_s]
        coords_s = Coup_info[level][intf]['coords'][indices_s]

        if 'FORCE' in var:
            p_Pi = sum(X[level][path]['F'][indices_f] for path in paths)
        else:
            p_Pi = sum(X[level][path]['X'][indices_f] for path in paths)

        coords_f = Coup_info[level][intf]['coords'][indices_f]
        nodes_f = Coup_info[level][intf]['nodes'][indices_f]
        
        if row == 0:
            u_Pi_total = sum(X[level][path]['X'][indices_s] for path in X[level].keys())
            p_Pi_total = sum(X[level][path]['X'][indices_f] for path in X[level].keys())

        # Build path labels
        if len(descriptions) != 0:
            if path_init == 'ALL':
                path_name_title[row] = 'ALL'
            else:
                if '+'.join(paths) in descriptions.keys():
                    path_name_title[row] = descriptions['+'.join(paths)]
                else:
                    path_name_title[row] = ' +\n'.join([descriptions[path] for path in paths])
        else:
            path_name_title[row] = '+'.join([path for path in paths])

        for col, freq in enumerate(freq_set):
            max_dim, min_dim = 3, 3
            dict_index = {'UX': 0, 'UY': 1, 'UZ': 2, 'ROTX': 3, 'ROTY': 4, 'ROTZ': 5, 'P':0}
            if var in ['P', 'PABS', 'SPL', 'SPL-A', 'SPL-SIGNED', 'SPL-A-SIGNED', 'FORCE-P', 'FORCE-PABS', 'FORCE-SPL']:
                contribution = (p_Pi[:, freq, dict_index['P']].real * p_Pi_total[:, freq, dict_index['P']].real) + (p_Pi[:, freq, dict_index['P']].imag * p_Pi_total[:, freq, dict_index['P']].imag)
                amp_phase = p_Pi[:, freq, dict_index['P']]

            elif var in [
                'UNORM', 'UVECT', 'ULOG', 'UX', 'UY', 'UZ', 'ROT-X', 'ROT-Y', 'ROT-Z',
                'FORCE-UVECT', 'FORCE-ROTVECT', 'FORCE-UNORM', 'FORCE-UX', 'FORCE-UY', 'FORCE-UZ',
                'FORCE-ROTX', 'FORCE-ROTY', 'FORCE-ROTZ'
            ]:
                

                if var in ['UNORM', 'UVECT', 'ULOG', 'FORCE-UNORM', 'FORCE-UVECT']:
                    contribution = ((u_Pi[:, freq, :max_dim].real * u_Pi_total[:, freq, :max_dim].real) + (u_Pi[:, freq, :max_dim].imag * u_Pi_total[:, freq, :max_dim].imag)).sum(axis=1)
                    amp_phase = np.linalg.norm(u_Pi[:, freq, [dict_index['UX'],dict_index['UY'],dict_index['UZ']]],axis=1)
                    
                elif var in ['FORCE-ROTVECT']:
                    contribution = ((u_Pi[:, freq, min_dim:].real * u_Pi_total[:, freq, min_dim:].real) + (u_Pi[:, freq, min_dim:].imag * u_Pi_total[:, freq, min_dim:].imag)).sum(axis=1)
                    amp_phase = np.linalg.norm(u_Pi[:, freq, [dict_index['ROT-X'],dict_index['ROT-Y'],dict_index['ROT-Z']]],axis=1)
                else:
                    pos = dict_index[var.split('-')[-1]]
                    contribution = (u_Pi[:, freq, pos].real * u_Pi_total[:, freq, pos].real) + (u_Pi[:, freq, pos].imag * u_Pi_total[:, freq, pos].imag)
                    amp_phase = u_Pi[:, freq, pos]
            else:
                raise ValueError(f"Unknown variable: {var}")

            if var in displacement_vars:
                coords = coords_s.copy()
                nodes = nodes_s.copy()
            else:
                coords = coords_f.copy()
                nodes = nodes_f.copy()
                
            if section[0] != None:
                variable = section[0]
                min_lim, max_lim = section[1]
                dict_variables = {'X': 0, 'Y': 1, 'Z': 2}
                flag_coords = ((coords[:, dict_variables[variable]] >= min_lim) * (coords[:, dict_variables[variable]] <= max_lim)).astype(bool)

                contribution = contribution[flag_coords]
                amp_phase = amp_phase[flag_coords]
                
            if len(nodes_sel) != 0:
                pos_nodes = np.sum([nodes == node for node in nodes_sel],axis=0).astype(bool) # positions in the array containing the nodes
                contribution = contribution[pos_nodes]
                amp_phase = amp_phase[pos_nodes]
                                 
                
            contribution_value = contribution.mean(axis=0)
            amp_phase_value = amp_phase.mean(axis=0)

            contribution_result[row, col] = contribution_value
            amp_phases_result[row, col, 0] = np.abs(amp_phase_value)
            amp_phases_result[row, col, 1] = np.angle(amp_phase_value, deg=True)

    if in_n_octave_bands[0]:
        if len(in_n_octave_bands) == 3:
            fmin, fmax = in_n_octave_bands[2]
            if fmin < stamps[0] and fmax > stamps[-1]:
                new_stamps = np.insert(stamps, 0, fmin)
                new_stamps = np.append(new_stamps, fmax)
                contribution_result = np.block([
                    np.zeros((contribution_result.shape[0], 1)),
                    contribution_result,
                    np.zeros((contribution_result.shape[0], 1))
                ])
            else:
                positions = (stamps >= fmin) * (stamps <= fmax)
                new_stamps = stamps[positions]
                contribution_result = contribution_result[:, positions]

            cont_stamps, contribution_result = n_octave_bands(new_stamps, contribution_result, in_n_octave_bands[1])
        else:
            cont_stamps, contribution_result = n_octave_bands(stamps, contribution_result, in_n_octave_bands[1])
    else:
        cont_stamps = stamps[freq_set]

    if var in ['SPL', 'SPL-SIGNED', 'ULOG']:
        ref_val = 2e-5 if var in ['SPL','SPL-SIGNED'] else 1e-5*np.sqrt(np.abs(contribution_result)).max()
        contribution_result = np.sign(contribution_result) * calculate_spl(np.sqrt(np.abs(contribution_result)), reference_pressure = ref_val)
        amp_phases_result[:, :, 0] = calculate_spl(amp_phases_result[:, :, 0], reference_pressure = ref_val)
        print(f'Reference value for {var}: {ref_val:.2e}')

    elif var in ['SPL-A', 'SPL-A-SIGNED']:
        contribution_result = np.sign(contribution_result) * compute_dBA(cont_stamps, np.sqrt(np.abs(contribution_result)))
        amp_phases_result[:, :, 0] = compute_dBA(stamps, amp_phases_result[:, :, 0])

    else:
        contribution_result = np.sign(contribution_result) * np.sqrt(np.abs(contribution_result))

    contribution_result = contribution_result.astype(float)
    amp_phases_result = amp_phases_result.astype(float)

    freq_range_names = np.array([f'{freq:.1f}' for freq in cont_stamps])

    if plot_type == 'contr_bars':
        plot_contribution_bars(
            contribution_result, freqs=freq_range_names, contributors=path_name_title, ylabel=ylabel, cmap_name=cmap_name,
            title=plot_title, xlabel=xlabel, ylim=ylim, fontsize=fontsize, figsize=figsize, legend=legend, savefig=savefig
        )

    elif plot_type == 'contr_lines':
        plot_contribution_lines2(
            contribution_result, freqs=cont_stamps, contributors=path_name_title, ylabel=ylabel, cmap_name=cmap_name,
            title=plot_title, xlabel=xlabel, ylim=ylim, fontsize=fontsize, figsize=figsize
        )
        
    elif plot_type == 'mag_lines':
        if ylim is not None:
            ylim = [0, ylim[1]]
        plot_contribution_lines2(
            amp_phases_result[:, :, 0], freqs=cont_stamps, contributors=path_name_title, ylabel=ylabel, cmap_name=cmap_name,
            title=plot_title, xlabel=xlabel, ylim=ylim, fontsize=fontsize, figsize=figsize
        )

    elif plot_type == 'contr_strips':
        plot_stacked_heat_strips(
            contribution_result, x=cont_stamps, row_labels=path_name_title, xlabel=xlabel, title=plot_title,
            fontsize=fontsize, figsize=figsize, cbar_label=ylabel, cmap_name=cmap_name, savefig=savefig
        )

    elif plot_type == 'mag_strips':
        plot_stacked_heat_strips(
            amp_phases_result[:, :, 0], x=cont_stamps, row_labels=path_name_title, xlabel=xlabel, title=plot_title,
            fontsize=fontsize, figsize=figsize, cbar_label=ylabel, cmap_name=cmap_name, savefig=savefig
        )
    elif plot_type ==  'mag_bars':
        if ylim is not None:
            ylim = [0, ylim[1]]
        plot_contribution_bars(
            np.abs(contribution_result), freqs=freq_range_names, contributors=path_name_title, ylabel=ylabel, cmap_name=cmap_name,
            title=plot_title, xlabel=xlabel, ylim=ylim, fontsize=fontsize, figsize=figsize, legend=legend, savefig=savefig
        )

    return contribution_result, amp_phases_result, path_name_title, freq_range_names
    

def plot_TPA_results_pyvista(
    X, Coup_info, level, components, path_name, var, freq_set, def_factor=0.0,
    show_real_imag_values='real', plot_size=(1600, 900), descriptions={},
    section=[None, [0, 1]], show_min_max=True, share_clim=True,
    orientation=(1, 1, 1), roll_angles=(0, 0, 0), plot_contributions=True,
    show_edges=True, make_zeros_transparent=False, vector_scale=True,
    parallel_projection=False, result_on_node=False, background_color_rgb=(1.0, 1.0, 1.0),
    full_screen=False, animation=[False, ''], domain='frequency'
):
    """
    Interactive 3D visualization of vibro-acoustic TPA results using PyVista.

    This routine is a *geometry-aware* post-processor that takes the hierarchical TPA result
    dictionary `X` (solutions per level / per path) together with coupling metadata `Coup_info`
    (nodes, coordinates, connectivity, component indices, normals, etc.) and produces one or
    multiple PyVista views of:
      - structural responses (displacements / rotations) and derived norms,
      - acoustic responses (pressure, SPL variants),
      - force-like quantities when `var` contains the substring `'FORCE'`,
      - diagnostic fields such as nodal normals and constraint flags,
      - optionally, the mesh itself (`var='MESH'`).

    The function supports:
      • Multiple paths (rows) and multiple frequencies/time-instants (columns).
      • Path superposition via the `"a+b+c"` syntax, or `"ALL"` to aggregate all paths on a level.
      • Optional geometric sectioning by axis-aligned coordinate thresholding.
      • Optional deformation overlay (structural variables only) by `def_factor * Re(u)`.
      • Shared or per-subplot color limits (`share_clim`) and optional extrema markers.
      • Vector visualization through glyph arrows for 3-component fields (e.g. `UVECT`).
      • Animation mode (off-screen movie export) for frequency/time sweeps.

    Data model expectations
    -----------------------
    `X[level][path]` is expected to contain, at minimum:
      - 'X': array-like (n_dofs_or_nodes, n_stamps, n_comp) complex
      - 'stamps': array-like (n_stamps,) (frequencies in Hz or time in s)
      - optionally 'F': array-like with the same indexing, for force-domain plotting

    `Coup_info[level][intf]` is expected to contain:
      - 'component_indices': dict mapping component-name -> boolean index mask
      - 'nodes', 'coords' (per node)
      - 'connectivity_comp' and 'components' for `assembly_nodal_connectivity`
      - 'nodal_normals' and 'constraint_nodes' (for NNORM/CONST plots)

    Contribution metric (display annotation)
    ---------------------------------------
    For each subplot (path, stamp), a scalar “average contribution” is computed as the mean
    of a real-valued scalar product between the selected path field and the total field
    (summed over all paths at that level). The sign is used to label the contribution as
    CONSTRUCTIVE (+) or DESTRUCTIVE (−). This is a *post-processing indicator* used for
    visualization; it is not a replacement for a full energetic power-flow derivation.

    Parameters
    ----------
    X : dict
        TPA results dictionary (levels -> paths -> fields).
    Coup_info : dict
        Coupling metadata dictionary (levels -> interfaces -> geometry/index data).
    level : str
        Level key in `X` and `Coup_info`.
    components : (list[str], list[str])
        Tuple `(comp_s, comp_f)` defining which structural vs acoustic components belong to:
          - structural side selection (`comp_s`)
          - acoustic side selection (`comp_f`)
    path_name : str | list[str]
        Path(s) to visualize. Each item can be:
          - a single path key,
          - a sum "p1+p2+..." to superpose several paths,
          - "ALL" to use all paths in `X[level]`.
    var : str
        Variable selector. Supported by current implementation:
          Structural:
            'UNORM','UVECT','ULOG','UX','UY','UZ','ROT-X','ROT-Y','ROT-Z'
          Acoustic:
            'P','PABS','SPL','SPL-A','SPL-SIGNED','SPL-A-SIGNED'
          Diagnostics:
            'NNORM-U','NNORM-P','CONST-U','CONST-P','MESH'
          Force variants (if 'FORCE' in var):
            'FORCE-UVECT','FORCE-ROTVECT','FORCE-UNORM','FORCE-UX','FORCE-UY','FORCE-UZ',
            'FORCE-ROTX','FORCE-ROTY','FORCE-ROTZ','FORCE-P','FORCE-PABS','FORCE-SPL'
    freq_set : int | array_like[int]
        Indices into `stamps` (not frequency values). If a scalar is passed, it is wrapped.
    def_factor : float, optional
        Deformation scale factor applied to structural coordinates:
            coords_deformed = coords + def_factor * Re(disp[:,:3])
        Only used when `var` is in `displacement_vars`.
    show_real_imag_values : {'real','imag'}, optional
        Select which part of complex results is plotted.
    plot_size : (int, int), optional
        PyVista window size in pixels.
    descriptions : dict, optional
        Mapping path keys -> readable labels for plot titles.
    section : [axis, [min,max]], optional
        Axis-aligned section filter: axis in {'X','Y','Z'} with range [min,max].
        If axis is None, no filter is applied.
    show_min_max : bool, optional
        If True, mark and label min/max points (scalar fields only; disabled for vectors).
    share_clim : bool | list[float,float] | None, optional
        - True: use global min/max across all subplots
        - list: explicit [vmin,vmax]
        - None: let PyVista choose per subplot
    orientation : (float,float,float), optional
        View direction for `plotter.view_vector(orientation)`.
    roll_angles : (float,float,float), optional
        Applied to camera as: `camera.roll, camera.azimuth, camera.elevation = roll_angles`.
    plot_contributions : bool, optional
        If True, adds “average contribution” text (not shown for path 'ALL').
    show_edges : bool, optional
        Show mesh edges in PyVista.
    make_zeros_transparent : bool, optional
        If True, per-point opacity is reduced for exactly zero values.
    vector_scale : bool, optional
        If True, glyph scale follows vector magnitude; if False, fixed size.
    parallel_projection : bool, optional
        Enable parallel projection.
    result_on_node : bool, optional
        If True, enable point picking and print node/value on click (interactive mode only).
    background_color_rgb : (float,float,float), optional
        Background color in RGB (0..1).
    full_screen : bool, optional
        Passed to `plotter.show(..., full_screen=full_screen)`.
    animation : [bool, str], optional
        If animation[0] is True, renders frames off-screen and writes a movie to animation[1].
        Current animation implementation renders sequential scalars on the same mesh.
    domain : {'frequency','time'}, optional
        Controls stamp units in the title.

    Returns
    -------
    contribution_result : ndarray, shape (n_paths, n_freqs)
        Average signed contributions computed for each plotted (path, stamp) cell.
    """

    # Ensure inputs are iterable
    if not isinstance(freq_set, (list, tuple, np.ndarray)):
        freq_set = [freq_set]
    if not isinstance(path_name, (list, tuple, np.ndarray)):
        path_name = [path_name]

    n_rows = len(path_name)
    n_cols = len(freq_set)

    # Components s and f
    comp_s, comp_f = components

    # Storage for subplot payloads
    result_list = np.array([[None] * n_cols] * n_rows)
    coords_list = np.array([[None] * n_cols] * n_rows)
    mesh_list = np.array([[None] * n_cols] * n_rows)
    contribution_result = np.zeros((n_rows, n_cols))
    min_val, max_val = (1e10, -1e10)

    # Displacement and pressure variables
    displacement_vars = [
        'UNORM', 'ULOG', 'UVECT', 'UX', 'UY', 'UZ', 'ROT-X', 'ROT-Y', 'ROT-Z', 'NNORM-U', 'CONST-U',
        'FORCE-UNORM', 'FORCE-UX', 'FORCE-UY', 'FORCE-UZ', 'FORCE-ROTX', 'FORCE-ROTY', 'FORCE-ROTZ',
        'FORCE-UVECT', 'FORCE-ROTVECT', 'MESH'
    ]

    for row, path_i in enumerate(path_name):
        paths = path_i.split('+')  # sum of paths
        if paths[0] == 'ALL':
            paths = list(X[level].keys())
        path0 = paths[0]

        # Extract the interface id and stamps
        intf = int(path0.split('-')[-1])
        stamps = X[level][path0]['stamps']

        component_indices = Coup_info[level][intf]['component_indices']
        indices_s = sum(component_indices[comp] for comp in comp_s if comp in component_indices)
        indices_f = sum(component_indices[comp] for comp in comp_f if comp in component_indices)

        indices_s = indices_s.astype(bool) if isinstance(indices_s, np.ndarray) else np.zeros(X[level][path0]['X'].shape[0], dtype=bool)
        indices_f = indices_f.astype(bool) if isinstance(indices_f, np.ndarray) else np.zeros(X[level][path0]['X'].shape[0], dtype=bool)

        # Select path responses (structural side)
        if 'FORCE' in var:
            u_Pi = sum(X[level][path]['F'][indices_s] for path in paths)
        else:
            u_Pi = sum(X[level][path]['X'][indices_s] for path in paths)
        nodes_s = Coup_info[level][intf]['nodes'][indices_s]
        coords_s = Coup_info[level][intf]['coords'][indices_s]

        # Select path responses (acoustic side)
        if 'FORCE' in var:
            p_Pi = sum(X[level][path]['F'][indices_f] for path in paths)
        else:
            p_Pi = sum(X[level][path]['X'][indices_f] for path in paths)
        nodes_f = Coup_info[level][intf]['nodes'][indices_f]
        coords_f = Coup_info[level][intf]['coords'][indices_f]

        # Diagnostics
        u_n_Pi = Coup_info[level][intf]['nodal_normals'][indices_s]
        p_n_Pi = Coup_info[level][intf]['nodal_normals'][indices_f]

        u_constr_Pi = Coup_info[level][intf]['constraint_nodes'][indices_s]
        p_constr_Pi = Coup_info[level][intf]['constraint_nodes'][indices_f]

        # Connectivity info (per component subset)
        el_con_s, el_type_s = assembly_nodal_connectivity(
            Coup_info[level][intf]['connectivity_comp'],
            [c for c in comp_s if c in Coup_info[level][intf]['components']]
        )
        el_con_f, el_type_f = assembly_nodal_connectivity(
            Coup_info[level][intf]['connectivity_comp'],
            [c for c in comp_f if c in Coup_info[level][intf]['components']]
        )

        # Displacements (used for deformation overlay)
        disp_Pi = sum(X[level][path]['X'][indices_s] for path in paths)

        # Replace VTK cell type 10 by 9 (tetrahedrons by quads) as in original code
        if len(el_type_s) != 0:
            el_type_s[el_type_s == 10] = 9
        if len(el_type_f) != 0:
            el_type_f[el_type_f == 10] = 9

        # Totals (for contribution computation)
        if row == 0:
            u_Pi_total = sum(X[level][path]['X'][indices_s] for path in X[level].keys())
            p_Pi_total = sum(X[level][path]['X'][indices_f] for path in X[level].keys())

        # Build all subplot payloads for this row
        for col, freq in enumerate(freq_set):
            # -------------------------
            # Select result to visualize
            # -------------------------
            if var == 'UNORM':
                max_dim = 3
                title = 'Total Displacement [m]'
                result = np.linalg.norm(u_Pi[:, freq, :max_dim], axis=1)
                contribution = (u_Pi[:, freq, :max_dim].conj() * u_Pi_total[:, freq, :max_dim]).real.sum(axis=1)

            elif var == 'MESH':
                title = 'Mesh'
                result = np.zeros(u_Pi.shape[0])
                contribution = np.zeros(u_Pi.shape[0])

            elif var == 'UVECT':
                max_dim = 3
                title = 'Total Displacement [m]'
                result = u_Pi[:, freq, :max_dim]
                contribution = (u_Pi[:, freq, :max_dim].conj() * u_Pi_total[:, freq, :max_dim]).real.sum(axis=1)

            elif var == 'ULOG':
                max_dim = 3
                title = 'Displacement Level [dB]'
                u = np.linalg.norm(u_Pi[:, freq, :max_dim], axis=1)
                u_ref = 1e-5  # 0.01mm as reference
                result = calculate_spl(u, u_ref) / 2.0
                contribution = (u_Pi[:, freq, :max_dim].conj() * u_Pi_total[:, freq, :max_dim]).real.sum(axis=1)

            elif var == 'UX':
                title = 'X Displacement [m]'
                result = u_Pi[:, freq, 0]
                contribution = (u_Pi[:, freq, 0].conj() * u_Pi_total[:, freq, 0]).real

            elif var == 'UY':
                title = 'Y Displacement [m]'
                result = u_Pi[:, freq, 1]
                contribution = (u_Pi[:, freq, 1].conj() * u_Pi_total[:, freq, 1]).real

            elif var == 'UZ':
                title = 'Z Displacement [m]'
                result = u_Pi[:, freq, 2]
                contribution = (u_Pi[:, freq, 2].conj() * u_Pi_total[:, freq, 2]).real

            elif var == 'ROT-X':
                title = 'X Rotation [rad]'
                result = u_Pi[:, freq, 3]
                contribution = (u_Pi[:, freq, 4].conj() * u_Pi_total[:, freq, 3]).real

            elif var == 'ROT-Y':
                title = 'Y Rotation [rad]'
                result = u_Pi[:, freq, 4]
                contribution = (u_Pi[:, freq, 4].conj() * u_Pi_total[:, freq, 4]).real

            elif var == 'ROT-Z':
                title = 'Z Rotation [rad]'
                result = u_Pi[:, freq, 5]
                contribution = (u_Pi[:, freq, 5].conj() * u_Pi_total[:, freq, 5]).real

            elif var == 'P':
                title = 'Pressure [Pa]'
                result = p_Pi[:, freq, 0]
                contribution = (p_Pi[:, freq, 0].conj() * p_Pi_total[:, freq, 0]).real

            elif var == 'PABS':
                title = 'Absolute Pressure [Pa]'
                result = np.abs(p_Pi[:, freq, 0])
                contribution = (p_Pi[:, freq, 0].conj() * p_Pi_total[:, freq, 0]).real

            elif var == 'SPL':
                title = 'Sound Pressure \nLevel [dB]'
                result = calculate_spl(p_Pi[:, freq, 0])
                contribution = (p_Pi[:, freq, 0].conj() * p_Pi_total[:, freq, 0]).real

            elif var == 'SPL-A':
                title = 'A-Weighted Sound \nPressure Level [dB]'
                result = compute_dBA(stamps[freq], p_Pi[:, freq, 0])
                contribution = (p_Pi[:, freq, 0].conj() * p_Pi_total[:, freq, 0]).real

            elif var == 'SPL-SIGNED':
                title = 'Signed Sound \nPressure Level [dB]'
                result = calculate_spl(p_Pi[:, freq, 0])
                result = np.sign(p_Pi[:, freq, 0]) * result
                contribution = (p_Pi[:, freq, 0].conj() * p_Pi_total[:, freq, 0]).real

            elif var == 'SPL-A-SIGNED':
                title = 'Signed A-Weighted \nSound Pressure Level [dB]'
                result = compute_dBA(stamps[freq], p_Pi[:, freq, 0])
                result = np.sign(p_Pi[:, freq, 0]) * result
                contribution = (p_Pi[:, freq, 0].conj() * p_Pi_total[:, freq, 0]).real

            elif var == 'NNORM-P':
                title = 'Nodal normals [-]'
                result = p_n_Pi
                contribution = 0 * (p_Pi[:, freq, 0].conj() * p_Pi_total[:, freq, 0]).real

            elif var == 'NNORM-U':
                title = 'Nodal normals [-]'
                result = u_n_Pi
                contribution = 0 * (u_Pi[:, freq, 0].conj() * u_Pi_total[:, freq, 0]).real

            elif var == 'CONST-P':
                title = 'Constraints [-]'
                result = p_constr_Pi
                contribution = 0 * (p_Pi[:, freq, 0].conj() * p_Pi_total[:, freq, 0]).real

            elif var == 'CONST-U':
                title = 'Constraints [-]'
                result = u_constr_Pi
                contribution = 0 * (u_Pi[:, freq, 0].conj() * u_Pi_total[:, freq, 0]).real

            elif var == 'FORCE-UVECT':
                max_dim = 3
                title = 'Total Force [N]'
                result = u_Pi[:, freq, :max_dim]
                contribution = (u_Pi[:, freq, :max_dim].conj() * u_Pi_total[:, freq, :max_dim]).real.sum(axis=1)

            elif var == 'FORCE-ROTVECT':
                min_dim = 3
                title = 'Total Force [N]'
                result = u_Pi[:, freq, min_dim:]
                contribution = (u_Pi[:, freq, min_dim:].conj() * u_Pi_total[:, freq, min_dim:]).real.sum(axis=1)

            elif var == 'FORCE-UNORM':
                max_dim = 3
                title = 'Total Force [m]'
                result = np.linalg.norm(u_Pi[:, freq, :max_dim], axis=1)
                contribution = (u_Pi[:, freq, :max_dim].conj() * u_Pi_total[:, freq, :max_dim]).real.sum(axis=1)

            elif var == 'FORCE-UX':
                title = 'X Force [N]'
                result = u_Pi[:, freq, 0]
                contribution = (u_Pi[:, freq, 0].conj() * u_Pi_total[:, freq, 0]).real

            elif var == 'FORCE-UY':
                title = 'Y Force [N]'
                result = u_Pi[:, freq, 1]
                contribution = (u_Pi[:, freq, 1].conj() * u_Pi_total[:, freq, 1]).real

            elif var == 'FORCE-UZ':
                title = 'Z Force [N]'
                result = u_Pi[:, freq, 2]
                contribution = (u_Pi[:, freq, 2].conj() * u_Pi_total[:, freq, 2]).real

            elif var == 'FORCE-ROTX':
                title = 'X Force [N]'
                result = u_Pi[:, freq, 3]
                contribution = (u_Pi[:, freq, 4].conj() * u_Pi_total[:, freq, 3]).real

            elif var == 'FORCE-ROTY':
                title = 'Y Force [N]'
                result = u_Pi[:, freq, 4]
                contribution = (u_Pi[:, freq, 4].conj() * u_Pi_total[:, freq, 4]).real

            elif var == 'FORCE-ROTZ':
                title = 'Z Force [N]'
                result = u_Pi[:, freq, 5]
                contribution = (u_Pi[:, freq, 5].conj() * u_Pi_total[:, freq, 5]).real

            elif var == 'FORCE-P':
                title = 'Acoustic Load [N/m3]'
                result = p_Pi[:, freq, 0]
                contribution = (p_Pi[:, freq, 0].conj() * p_Pi_total[:, freq, 0]).real

            elif var == 'FORCE-PABS':
                title = 'Absolute \n Acoustic Load [N/m3]'
                result = np.abs(p_Pi[:, freq, 0])
                contribution = (p_Pi[:, freq, 0].conj() * p_Pi_total[:, freq, 0]).real

            elif var == 'FORCE-SPL':
                title = 'Acoustic Load \nLevel [dB]'
                result = calculate_spl(p_Pi[:, freq, 0])
                contribution = (p_Pi[:, freq, 0].conj() * p_Pi_total[:, freq, 0]).real

            else:
                raise ValueError(f"Unknown variable: {var}")

            # -----------------------------------------
            # Choose coordinates/connectivity for plotting
            # -----------------------------------------
            if var in displacement_vars:
                coords = coords_s.copy()
                coords += def_factor * disp_Pi[:, freq, :3].real
                nodes = nodes_s.copy()
                el_con, el_type = (el_con_s, el_type_s)
            else:
                coords = coords_f.copy()
                nodes = nodes_f.copy()
                el_con, el_type = (el_con_f, el_type_f)

            # Section selection (only affects contribution + later mesh thresholding)
            if section[0] != None:
                variable = section[0]
                min_lim, max_lim = section[1]
                dict_variables = {'X': 0, 'Y': 1, 'Z': 2}
                flag_coords = ((coords[:, dict_variables[variable]] >= min_lim) * (coords[:, dict_variables[variable]] <= max_lim)).astype(bool)
                contribution = contribution[flag_coords]

            # -----------------------------------------
            # Average contribution (signed)
            # -----------------------------------------
            contribution_value = contribution.mean(axis=0)
            if var == 'SPL':
                contribution_result[row, col] = np.sign(contribution_value) * calculate_spl(np.sqrt(np.abs(contribution_value)))
            elif var == 'SPL-SIGNED':
                contribution_result[row, col] = np.sign(contribution_value) * calculate_spl(np.sqrt(np.abs(contribution_value)))
            elif var == 'SPL-A':
                contribution_result[row, col] = np.sign(contribution_value) * compute_dBA(stamps[freq], np.sqrt(np.abs(contribution_value)))
            elif var == 'SPL-A-SIGNED':
                contribution_result[row, col] = np.sign(contribution_value) * compute_dBA(stamps[freq], np.sqrt(np.abs(contribution_value)))
            elif var == 'ULOG':
                contribution_result[row, col] = np.sign(contribution_value) * calculate_spl(np.sqrt(np.abs(contribution_value)))
            else:
                contribution_result[row, col] = np.sign(contribution_value) * np.sqrt(np.abs(contribution_value))

            # Store payload for later rendering
            coords_list[row, col] = coords
            mesh_list[row, col] = (el_con, el_type)
            result_list[row, col] = result.real if show_real_imag_values == 'real' else result.imag

            min_val = np.min([min_val, result_list[row, col].min()])
            max_val = np.max([max_val, result_list[row, col].max()])

    # =========================================================================
    # Plot using PyVista
    # =========================================================================
    if n_rows == 1:  # only one path -> lay out frequencies in a near-square grid
        new_n_rows = np.ceil(np.sqrt(n_cols)).astype(int)
        new_n_cols = np.ceil(n_cols / new_n_rows).astype(int)
        plotter = pv.Plotter(shape=(new_n_rows, new_n_cols), window_size=plot_size)
        row_col = [[row, col] for col in range(0, new_n_cols) for row in range(0, new_n_rows)]

    elif n_cols == 1:  # only one frequency -> lay out paths in a near-square grid
        new_n_rows = np.ceil(np.sqrt(n_rows)).astype(int)
        new_n_cols = np.ceil(n_rows / new_n_rows).astype(int)
        plotter = pv.Plotter(shape=(new_n_rows, new_n_cols), window_size=plot_size)
        row_col = [[row, col] for col in range(0, new_n_cols) for row in range(0, new_n_rows)]

    else:
        plotter = pv.Plotter(shape=(n_rows, n_cols), window_size=plot_size)
        row_col = [[row, col] for row in range(0, n_rows) for col in range(0, n_cols)]

    # -------------------------
    # Animation (off-screen movie)
    # -------------------------
    if animation[0]:
        _, filename_ani = animation

        plotter = pv.Plotter(shape=(1, 1), window_size=plot_size, off_screen=True)
        plotter.set_background(background_color_rgb)
        plotter.open_movie(f"{filename_ani}", framerate=4)

        plotter.show(auto_close=False)

        # NOTE: uses `coords` as last computed coords in the loops above (preserved behavior)
        distance = 2 * (coords.max(axis=0) - coords.min(axis=0)).max()
        direction = (orientation / np.linalg.norm(orientation)) * distance
        plotter.camera_position = [tuple(direction), coords.mean(axis=0), (1, 1, 1)]

        plotter.show_axes()

        if parallel_projection:
            plotter.camera.ParallelProjectionOn()
            plotter.camera.parallel_scale = 0.6 * (coords.max(axis=0) - coords.min(axis=0)).max()

        point_cloud = pv.UnstructuredGrid(mesh_list[0, 0][0], mesh_list[0, 0][1], coords)

        for col in range(0, len(freq_set)):
            var_name = var + '-' + str(col)
            point_cloud[var_name] = result_list[0, col]
            _ = plotter.add_mesh(
                point_cloud,
                name=var_name,
                scalars=var_name,
                color='grey',
                show_edges=show_edges,
                show_scalar_bar=False,
                render_points_as_spheres=True,
                point_size=1,
            )
            plotter.render()
            plotter.write_frame()

        plotter.close()  # Finalize movie

        """Need to update pyvista."""
        # (kept as-is from original code)

    # -------------------------
    # Interactive multi-subplot rendering
    # -------------------------
    else:
        plotter.set_background(background_color_rgb)

        initial_camera_position = np.array([[None] * n_cols] * n_rows)
        index = 0

        for row, path_i in enumerate(path_name):
            paths = path_i.split('+')
            path0 = paths[0]

            for col, freq in enumerate(freq_set):
                var_name = var + '-' + str(row) + '-' + str(col)

                # Build mesh for this subplot
                point_cloud = pv.UnstructuredGrid(mesh_list[row, col][0], mesh_list[row, col][1], coords_list[row, col])
                point_cloud[var_name] = result_list[row, col]

                # Optional sectioning (threshold in PyVista)
                if section[0] != None:
                    variable = section[0]
                    min_lim, max_lim = section[1]
                    dict_variables = {'X': 0, 'Y': 1, 'Z': 2}
                    flag_coords = ((coords[:, dict_variables[variable]] >= min_lim) * (coords[:, dict_variables[variable]] <= max_lim)).astype(bool)

                    # NOTE: uses `coords` from the outer scope (preserved behavior)
                    coords = coords[flag_coords, :]
                    result_list[row, col] = result_list[row, col][flag_coords]

                    point_cloud[variable] = point_cloud.points[:, dict_variables[variable]]
                    point_cloud = point_cloud.threshold([min_lim, max_lim], scalars=variable)

                # Only external surface
                point_cloud = point_cloud.extract_surface()

                # Title with stamp
                if domain == 'time':
                    title_full = f"{title}\n@ {stamps[freq]:.2f} s"
                elif domain == 'frequency':
                    title_full = f"{title}\n@ {stamps[freq]:.2f} Hz"

                # Path title
                if len(descriptions) != 0:
                    if path0 == 'ALL':
                        path_name_title = 'ALL in ' + level
                    elif descriptions.get(path_i) is not None:
                        path_name_title = descriptions[path_i]
                    else:
                        path_name_title = ' +\n'.join([descriptions[path] for path in paths])
                    title_full = f"{path_name_title}\n{title_full}"
                else:
                    path_name_title = '+'.join([path for path in paths])
                    title_full = f"{path_name_title} - {title_full}"

                # Contribution text
                type_contr = 'DESTRUCTIVE' if np.sign(contribution_result[row, col]) < 0 else 'CONSTRUCTIVE'
                contr_text_value = (
                    f"{contribution_result[row, col]:.2f}"
                    if np.abs(np.log10(np.abs(contribution_result[row, col]))) < 2
                    else f"{contribution_result[row, col]:.2e}"
                )
                contr_text = f"Average contribution: {contr_text_value} \n{type_contr}"

                # Select subplot slot
                row_i, col_i = row_col[index]
                plotter.subplot(row_i, col_i)
                index += 1

                # Shared color limits
                if share_clim is None:
                    clim = None
                elif share_clim is True:
                    clim = [min_val, max_val]
                elif type(share_clim) is list:
                    clim = [share_clim[0], share_clim[1]]

                # Vector vs scalar rendering
                is_vector = result_list[row, col].ndim == 2 and result_list[row, col].shape[1] == 3
                if is_vector:
                    _ = plotter.add_mesh(
                        point_cloud,
                        color='grey',
                        opacity=0.5,
                        show_edges=show_edges,
                        render_points_as_spheres=True,
                        point_size=1
                    )

                    vectors = point_cloud[var_name]
                    magnitudes = np.linalg.norm(vectors, axis=1)
                    max_magnitude = magnitudes.max()

                    bounds = point_cloud.bounds
                    model_size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
                    if vector_scale:
                        fixed_max_arrow_size = 0.1 * model_size / max_magnitude
                    else:
                        fixed_max_arrow_size = 0.05 * model_size

                    glyphs = point_cloud.glyph(orient=var_name, scale=vector_scale, factor=fixed_max_arrow_size)
                    _ = plotter.add_mesh(
                        glyphs,
                        scalars=None,
                        cmap='turbo',
                        show_scalar_bar=False,
                        name=var_name
                    )

                    show_min_max = False

                else:
                    if make_zeros_transparent:
                        opacity_array = np.where(point_cloud[var_name] == 0, 0.1, 1.0)
                    else:
                        opacity_array = np.ones(point_cloud[var_name].shape[0])

                    if var != 'MESH':
                        _ = plotter.add_mesh(
                            point_cloud,
                            name=var_name,
                            scalars=var_name,
                            cmap='turbo',
                            show_edges=show_edges,
                            show_scalar_bar=False,
                            clim=clim,
                            render_points_as_spheres=True,
                            point_size=1,
                            opacity=opacity_array
                        )
                    else:
                        _ = plotter.add_mesh(
                            point_cloud,
                            name=var_name,
                            color='grey',
                            show_edges=show_edges,
                            show_scalar_bar=False,
                            clim=clim,
                            render_points_as_spheres=True,
                            point_size=1,
                            opacity=opacity_array
                        )

                # Titles and annotations
                plotter.add_text(title_full, position='upper_left', font_size=11, color='black')

                if parallel_projection:
                    plotter.camera.ParallelProjectionOn()

                if plot_contributions and path0 != 'ALL':
                    plotter.add_text(
                        contr_text,
                        position='lower_left',
                        font_size=11,
                        color='green' if type_contr == 'DESTRUCTIVE' else 'red'
                    )

                plotter.add_scalar_bar(
                    title=var_name,
                    title_font_size=-10,
                    label_font_size=12,
                    width=0.04,
                    height=0.6,
                    vertical=True,
                    color="black",
                )

                # Min/max markers (scalar fields only)
                if show_min_max:
                    max_idx = np.argmax(result_list[row, col])
                    max_point = coords_list[row, col][max_idx]
                    max_marker = pv.PolyData([max_point])

                    min_idx = np.argmin(result_list[row, col])
                    min_point = coords_list[row, col][min_idx]
                    min_marker = pv.PolyData([min_point])

                    _ = plotter.add_mesh(
                        max_marker,
                        color='white',
                        opacity=0.5,
                        point_size=15,
                        render_points_as_spheres=True,
                        name=f"max-{var_name}"
                    )

                    val_max_plot = (
                        f"{result_list[row, col][max_idx]:.2f}"
                        if np.abs(np.log10(np.abs(result_list[row, col][max_idx]))) < 2
                        else f"{result_list[row, col][max_idx]:.2e}"
                    )
                    plotter.add_point_labels(
                        max_marker,
                        [f"Max: {val_max_plot}"],
                        font_size=15,
                        point_color='gray',
                        text_color='black',
                        shape=None,
                        always_visible=True
                    )

                    if result_list[row, col][min_idx] != 0:
                        _ = plotter.add_mesh(
                            min_marker,
                            color='white',
                            opacity=0.5,
                            point_size=15,
                            render_points_as_spheres=True,
                            name=f"min-{var_name}"
                        )

                        val_min_plot = (
                            f"{result_list[row, col][min_idx]:.2f}"
                            if np.abs(np.log10(np.abs(result_list[row, col][min_idx]))) < 2
                            else f"{result_list[row, col][min_idx]:.2e}"
                        )
                        plotter.add_point_labels(
                            min_marker,
                            [f"Min: {val_min_plot}"],
                            font_size=15,
                            point_color='gray',
                            text_color='black',
                            shape=None,
                            always_visible=True
                        )

                # Camera controls
                plotter.view_vector(orientation)
                # plotter.camera.roll, plotter.camera.azimuth, plotter.camera.elevation = roll_angles
                plotter.camera.roll, plotter.camera.elevation, plotter.camera.azimuth = roll_angles
                plotter.show_axes()
        
                initial_camera_position[row, col] = plotter.camera_position

        # Optional point picking
        if result_on_node:
            def on_pick(picked_mesh, idx):
                xyz = picked_mesh.points[idx]
                val = picked_mesh[var_name][idx]
                node = nodes[idx]
                print(f"Picked node {node}: {xyz}, {var_name}={val}")

                plotter.add_point_labels(
                    xyz.reshape(1, 3),
                    [f"NODE={node}\n{var_name}={val}"],
                    point_size=0, font_size=12, shape=None
                )
                plotter.render()

            _ = plotter.enable_point_picking(
                callback=on_pick,
                use_mesh=True,
                show_message=False,
                show_point=True,
                left_clicking=True,
            )

        plotter.show(interactive=True, full_screen=full_screen)

    return contribution_result

def calculate_spl(pressure, reference_pressure=2e-5, negative_values=False):
    """
    Convert pressure amplitude(s) to Sound Pressure Level (SPL) in dB.

    SPL is computed as:
        SPL = 20 * log10(|p| / p_ref)

    where `p_ref` is the reference pressure (default 20 µPa, standard in air acoustics).
    The function uses the magnitude `|p|` (so complex inputs are supported).

    By default, negative SPL values are clipped to 0 dB. This is a visualization/utility
    choice (not a physical constraint), preserved exactly from the current implementation.

    Parameters
    ----------
    pressure : float | complex | array_like
        Pressure amplitude(s) [Pa]. Can be scalar, list, or ndarray.
    reference_pressure : float, optional
        Reference pressure [Pa]. Default 2e-5 Pa.
    negative_values : bool, optional
        If False, values below 0 dB are clipped to 0.0 dB.

    Returns
    -------
    spl_db : ndarray
        SPL values in dB (same shape as input after numpy conversion).
    """
    if not isinstance(pressure, (list, tuple, np.ndarray)):
        pressure = [pressure]
    pressure_rms = np.abs(pressure)
    spl_db = 20 * np.log10(pressure_rms / reference_pressure)
    if negative_values == False:
        spl_db[spl_db < 0] = 0.0
    return spl_db

def a_weighting(f):
    """
    Compute the A-weighting curve (IEC-style) in dB for frequency values in Hz.

    The A-weighting approximates human loudness perception at moderate sound levels.
    This function returns the frequency-dependent correction A(f) in dB, which is
    typically added to an SPL spectrum in dB:
        SPL_A(f) = SPL(f) + A(f)

    Parameters
    ----------
    f : array_like
        Frequencies in Hz.

    Returns
    -------
    a_weight : ndarray
        A-weighting values in dB (same shape as `f`).
    """
    ra = (12194 ** 2 * f ** 4) / (
        (f ** 2 + 20.6 ** 2)
        * np.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2))
        * (f ** 2 + 12194 ** 2)
    )
    a_weight = 20 * np.log10(ra) + 2.00
    return a_weight

def compute_dBA(freq, pressure_values):
    """
    Apply A-weighting to pressure values and return A-weighted SPL (dB(A)).

    Current behavior (preserved):
      1) Convert `pressure_values` to SPL in dB using `calculate_spl`.
      2) Compute A-weighting correction for `freq`.
      3) Add correction: SPL_A = SPL + A(f)
      4) Clip negative results to 0 dB.

    Notes
    -----
    - Despite the docstring saying “total dBA”, this function returns the *A-weighted
      spectrum per frequency*, not a single overall level.

    Parameters
    ----------
    freq : array_like
        Frequencies in Hz. Converted to float.
    pressure_values : array_like
        Pressure amplitudes [Pa], compatible with `calculate_spl`.

    Returns
    -------
    spl_a_weighted : ndarray
        A-weighted SPL values per frequency bin (dB(A)).
    """
    spl_values = calculate_spl(pressure_values)
    a_weight = a_weighting(freq.astype(float))

    spl_a_weighted = spl_values + a_weight
    spl_a_weighted[spl_a_weighted < 0] = 0.0

    return spl_a_weighted

def plot_sparsity(matrix, labels=['matrix'], threshold=1e-12, colormap=True, abs_vals=True):
    """
    Visualize a sparse matrix as either:
      - a sparsity pattern (spy plot), or
      - a log-scaled colormap of magnitudes.

    The plot can be divided into equal-sized labeled sections, useful for block-structured
    matrices (e.g., component assemblies, partitioned DOFs). Section boundaries are drawn
    using dotted lines based on `labels`.

    Behavior preserved from current implementation
    ----------------------------------------------
    - The matrix is optionally converted to absolute values (`abs_vals=True`).
    - For `colormap=True`, the matrix is densified via `toarray()` and values below
      `threshold` are set to NaN (rendered as “under” color).
    - For `colormap=False`, the matrix is filtered in sparse form: entries below threshold
      are set to zero and eliminated, then plotted via `ax.spy`.
    - The matrix is flipped in row order in both modes (visual convention used in the code).

    Parameters
    ----------
    matrix : scipy.sparse matrix
        2D sparse matrix (any sparse format accepted; used like CSR in the docstring).
    labels : list[str], optional
        Labels defining the number of sections. Sections are equal size:
            section_size = matrix.shape[0] // len(labels)
        (This is an approximation; last remainder is not separately handled.)
    threshold : float, optional
        Minimum magnitude to be treated as non-zero / plotted.
    colormap : bool, optional
        If True, show log-scaled values; else show pattern.
    abs_vals : bool, optional
        If True, uses absolute values for plotting.

    Returns
    -------
    None
        Displays the plot via matplotlib.
    """
    if len(labels) > matrix.shape[0] or len(labels) > matrix.shape[1]:
        raise ValueError("The number of labels should not exceed matrix dimensions.")

    section_size = matrix.shape[0] // len(labels)
    tick_positions = [i * section_size for i in range(1, len(labels))]

    fig, ax = plt.subplots(figsize=(6, 6))

    if abs_vals:
        matrix = np.abs(matrix)

    if colormap:
        dense_matrix = matrix.toarray()
        masked_matrix = np.where(dense_matrix > threshold, dense_matrix, np.nan)

        cmap = plt.cm.jet
        cmap.set_under('white')

        img = ax.matshow(
            masked_matrix[np.arange(masked_matrix.shape[0] - 1, -1, -1), :],
            cmap=cmap,
            norm=mcolors.LogNorm(vmin=threshold)
        )
        fig.colorbar(img, ax=ax)
    else:
        filtered_matrix = matrix.copy()
        indices = np.arange(filtered_matrix.shape[0] - 1, -1, -1)
        filtered_matrix = filtered_matrix[indices, :]
        filtered_matrix.data = np.where(filtered_matrix.data >= threshold, filtered_matrix.data, 0)
        filtered_matrix.eliminate_zeros()

        ax.spy(filtered_matrix, markersize=1, color='black')

    ax.invert_yaxis()

    for pos in tick_positions:
        ax.axvline(x=pos, color='black', linestyle=':', linewidth=1)
        ax.axhline(y=pos, color='black', linestyle=':', linewidth=1)

    ax.set_xticks([i * section_size + section_size // 2 for i in range(len(labels))])
    ax.set_xticklabels(labels)
    ax.set_yticks([i * section_size + section_size // 2 for i in range(len(labels))])
    ax.set_yticklabels(labels)

    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(top=False, bottom=True)
    ax.tick_params(left=False)

    plt.title('Matrix Pattern (Colormap)' if colormap else 'Sparsity Pattern')
    plt.show()