
CXX=mpic++
CXXFLAGS = -std=c++11 -fopenmp -Wall -Wfloat-conversion # need C++11 and OpenMP

#Optional flags
CXXFLAGS += -O0 # debug build
#CXXFLAGS += -O3 -march=native -DNDEBUG # release build

ifeq ($(shell uname -s),Darwin)
	CXXFLAGS += -g -rdynamic -Wl,-no_pie # for stack trace (on Mac)
else
	CXXFLAGS += -g -rdynamic # for stack trace
endif

CXXFLAGS += -DSCTL_MEMDEBUG # Enable memory checks
CXXFLAGS += -DSCTL_GLOBAL_MEM_BUFF=0 # Global memory buffer size in MB

CXXFLAGS += -DSCTL_PROFILE=5 -DSCTL_VERBOSE # Enable profiling

CXXFLAGS += -DSCTL_QUAD_T=__float128 # Enable quadruple precision

#CXXFLAGS += -DSCTL_HAVE_MPI #use MPI

CXXFLAGS += -lblas -DSCTL_HAVE_BLAS # use BLAS
CXXFLAGS += -llapack -DSCTL_HAVE_LAPACK # use LAPACK
#CXXFLAGS += -mkl -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK # use MKL BLAS and LAPACK

CXXFLAGS += -lfftw3 -DSCTL_HAVE_FFTW
CXXFLAGS += -lfftw3f -DSCTL_HAVE_FFTWF
CXXFLAGS += -lfftw3l -DSCTL_HAVE_FFTWL


RM = rm -f
MKDIRS = mkdir -p

BINDIR = ./bin
SRCDIR = ./src
OBJDIR = ./obj
INCDIR = ./include

TARGET_BIN = \
       $(BINDIR)/test

all : $(TARGET_BIN)

$(BINDIR)/%: $(OBJDIR)/%.o
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $^ -o $@

clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*
	$(RM) *~ */*~ */*/*~
