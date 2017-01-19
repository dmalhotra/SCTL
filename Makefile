
CXX=icpc
CXXFLAGS = -std=c++11 -fopenmp -O3 # need C++11 and OpenMP

# Optional flags
CXXFLAGS += -DNDEBUG # release build
CXXFLAGS += -g -rdynamic # for stack trace
CXXFLAGS += -mkl -DPVFMM_HAVE_BLAS # use BLAS
CXXFLAGS += -mkl -DPVFMM_HAVE_LAPACK # use LAPACK
#CXXFLAGS += -DPVFMM_HAVE_MPI # use MPI


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
	$(RM) *~ */*~

