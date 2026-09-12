CXX=g++ # requires g++-9 or newer / icpc (with gcc compatibility 9 or newer) / clang++ with llvm-10 or newer
CXXFLAGS = -std=c++17 -fopenmp -Wall -Wfloat-conversion # need C++17 and OpenMP

#Optional flags
DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS += -O0 -fsanitize=address,leak,undefined,pointer-compare,pointer-subtract,float-divide-by-zero,float-cast-overflow -fno-sanitize-recover=all -fstack-protector # debug build
	CXXFLAGS += -DSCTL_MEMDEBUG # Enable memory checks
else
	CXXFLAGS += -O3 -march=native -DNDEBUG # release build
endif

OS = $(shell uname -s)
ifeq "$(OS)" "Darwin"
	CXXFLAGS += -g -rdynamic -Wl,-no_pie # for stack trace (on Mac)
else
	CXXFLAGS += -gdwarf-4 -g -rdynamic # for stack trace
	CXXFLAGS += -ldl # dladdr() in stacktrace.h (libc on glibc >=2.34, libdl otherwise)
endif

# GCC `-march=native` on Sapphire Rapids and newer Intel CPUs emits AVX-512-FP16
# instructions (e.g. `vmovw`) that pre-2.38 binutils' system assembler can't
# decode. Disable just the FP16 subset; the rest of -march=native is fine. If a
# newer binutils is available (e.g. `module load binutils/2.43.1`), the user can
# remove this flag manually. macOS clang doesn't emit avx512fp16 and doesn't
# recognise the flag on Apple Silicon — skip there.
ifneq "$(OS)" "Darwin"
       CXXFLAGS += -mno-avx512fp16
endif

CXXFLAGS += -DSCTL_GLOBAL_MEM_BUFF=0 # Global memory buffer size in MB

CXXFLAGS += -DSCTL_PROFILE=5 -DSCTL_VERBOSE # Enable profiling
CXXFLAGS += -DSCTL_SIG_HANDLER # Enable SCTL stack trace

CXXFLAGS += -DSCTL_QUAD_T=__float128 # Enable quadruple precision

#CXXFLAGS += -DSCTL_MAX_DEPTH=20 # Morton/Tree depth limit (default 20); above 64/DIM-1 the code needs more than one word

#CXXFLAGS += -DSCTL_HAVE_MPI #use MPI

#CXXFLAGS += -DSCTL_COMM_PTRACER # permit direct-communication where the kernel would refuse, by opening each rank's memory to every process of the same user
#CXXFLAGS += -DSCTL_COMM_NO_DIRECT # never read peer memory: every block goes through MPI, and no Comm looks for its node group (overrides the above)

CXXFLAGS += -lblas -DSCTL_HAVE_BLAS # use BLAS
CXXFLAGS += -llapack -DSCTL_HAVE_LAPACK # use LAPACK
#CXXFLAGS += -qmkl -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK -DSCTL_HAVE_FFTW3_MKL # use MKL BLAS, LAPACK and FFTW (Intel compiler)
#CXXFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK # use MKL BLAS and LAPACK (non-Intel compiler)

CXXFLAGS += -lfftw3_omp -DSCTL_FFTW_THREADS
CXXFLAGS += -lfftw3 -DSCTL_HAVE_FFTW
CXXFLAGS += -lfftw3f -DSCTL_HAVE_FFTWF
CXXFLAGS += -lfftw3l -DSCTL_HAVE_FFTWL

#CXXFLAGS += -lmvec -lm -DSCTL_HAVE_LIBMVEC
#CXXFLAGS += -DSCTL_HAVE_SVML

#CXXFLAGS += -I${PETSC_DIR}/include -I${PETSC_DIR}/../include -DSCTL_HAVE_PETSC
#LDLIBS += -L${PETSC_DIR}/lib -lpetsc

#PVFMM_INC_DIR = ../include
#CXXFLAGS += -DSCTL_HAVE_PVFMM -I$(PVFMM_INC_DIR)


RM = rm -f
MKDIRS = mkdir -p

BINDIR = ./bin
SRCDIR = ./src
OBJDIR = ./obj
INCDIR = ./include

TARGET_BIN = \
       $(BINDIR)/test \
       $(BINDIR)/test-comm \
       $(BINDIR)/test-boundary_integral \
       $(BINDIR)/test-cheb_utils \
       $(BINDIR)/test-fft-fallback \
       $(BINDIR)/test-generic-kernel \
       $(BINDIR)/test-intrin-wrapper \
       $(BINDIR)/test-iterator \
       $(BINDIR)/test-kernel_functions \
       $(BINDIR)/test-lagrange-interp \
       $(BINDIR)/test-mat_utils \
       $(BINDIR)/test-math_utils \
       $(BINDIR)/test-matrix \
       $(BINDIR)/test-mem_mgr \
       $(BINDIR)/test-morton \
       $(BINDIR)/test-ompUtils \
       $(BINDIR)/test-permutation \
       $(BINDIR)/test-profile \
       $(BINDIR)/test-static-array \
       $(BINDIR)/test-vector \
       $(BINDIR)/test-vtudata \
       $(BINDIR)/test-fft \
       $(BINDIR)/test-fmm \
       $(BINDIR)/test-gmres \
       $(BINDIR)/test-linear-solver \
       $(BINDIR)/test-ode-solver \
       $(BINDIR)/test-pt-tree \
       $(BINDIR)/test-sort-scatter \
       $(BINDIR)/test-quadrule \
       $(BINDIR)/test-sph-harm \
       $(BINDIR)/test-tensor \
       $(BINDIR)/test-vec \
       $(BINDIR)/test-quad-elem \
       $(BINDIR)/test-scratch-pool \
       $(BINDIR)/test-scratch-pool-perf

.PHONY: all gpu test clean

all : $(TARGET_BIN)

$(BINDIR)/%: $(OBJDIR)/%.o
	-@$(MKDIRS) $(dir $@)
	$(CXX) $^ $(CXXFLAGS) $(LDLIBS) -o $@
ifeq "$(OS)" "Darwin"
	/usr/bin/dsymutil $@ -o $@.dSYM
endif

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $^ -o $@

# GPU tree (include/sctl/experimental): CUDA sources, compiled by nvcc with the MPI wrapper as the
# host compiler. Needs CUDA and a CUDA-aware MPI, so `gpu` is separate from `all`.
NVCC = nvcc -ccbin mpicxx
NVCCFLAGS = -x cu -std=c++17 -O3 -arch=native -rdc=true --expt-relaxed-constexpr -Xcompiler "-fopenmp" \
            -DSCTL_HAVE_MPI -DSCTL_MAX_DEPTH=20 -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
NVCCLIBS = -ldl

GPU_BIN = \
       $(BINDIR)/test-gpu-tree \
       $(BINDIR)/test-gpu-sort-scatter \
       $(BINDIR)/example-gpu-tree

gpu: $(GPU_BIN)

# The whole library is headers, so a .cu alone does not say when its binary is out of date: editing
# a header would otherwise leave `make gpu` handing back the previous build. The recipe passes `$<`,
# so listing them as prerequisites does not put them on the command line.
GPU_DEPS = $(wildcard $(INCDIR)/*.hpp $(INCDIR)/sctl/*.hpp $(INCDIR)/sctl/*.txx \
                      $(INCDIR)/sctl/experimental/*.hpp $(INCDIR)/sctl/experimental/*.txx)

$(BINDIR)/%: $(SRCDIR)/%.cu $(GPU_DEPS)
	-@$(MKDIRS) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -I$(INCDIR) $< $(NVCCLIBS) -o $@

test: $(TARGET_BIN)
	./$(BINDIR)/test
	./$(BINDIR)/test-comm
	./$(BINDIR)/test-boundary_integral
	./$(BINDIR)/test-cheb_utils
	./$(BINDIR)/test-fft-fallback
	./$(BINDIR)/test-generic-kernel
	./$(BINDIR)/test-intrin-wrapper
	./$(BINDIR)/test-iterator
	./$(BINDIR)/test-kernel_functions
	./$(BINDIR)/test-lagrange-interp
	./$(BINDIR)/test-mat_utils
	./$(BINDIR)/test-math_utils
	./$(BINDIR)/test-matrix
	./$(BINDIR)/test-mem_mgr
	./$(BINDIR)/test-morton
	./$(BINDIR)/test-ompUtils
	./$(BINDIR)/test-permutation
	./$(BINDIR)/test-profile
	./$(BINDIR)/test-static-array
	./$(BINDIR)/test-vector
	./$(BINDIR)/test-vtudata
	./$(BINDIR)/test-fft
	./$(BINDIR)/test-fmm
	./$(BINDIR)/test-gmres
	./$(BINDIR)/test-linear-solver
	./$(BINDIR)/test-ode-solver
	./$(BINDIR)/test-pt-tree
	./$(BINDIR)/test-sort-scatter
	./$(BINDIR)/test-quadrule
	./$(BINDIR)/test-sph-harm
	./$(BINDIR)/test-tensor
	./$(BINDIR)/test-vec
	./$(BINDIR)/test-quad-elem
	./$(BINDIR)/test-scratch-pool

clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*
	$(RM) *~ */*~ */*/*~
