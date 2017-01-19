#include "pvfmm.hpp"

void ProfileMemgr(){
  long N=1e9;
  { // Without memory manager
    pvfmm::Profile::Tic("No-Memgr");

    pvfmm::Profile::Tic("Alloc");
    auto A = new double[N];
    pvfmm::Profile::Toc();

    pvfmm::Profile::Tic("Array-Write");
#pragma omp parallel for schedule(static)
    for(long i=0;i<N;i++) A[i]=0;
    pvfmm::Profile::Toc();

    pvfmm::Profile::Tic("Free");
    delete[] A;
    pvfmm::Profile::Toc();

    pvfmm::Profile::Toc();
  }
  { // With memory manager
    pvfmm::Profile::Tic("With-Memgr");

    pvfmm::Profile::Tic("Alloc");
    auto A = pvfmm::aligned_new<double>(N);
    pvfmm::Profile::Toc();

    pvfmm::Profile::Tic("Array-Write");
#pragma omp parallel for schedule(static)
    for(long i=0;i<N;i++) A[i]=0;
    pvfmm::Profile::Toc();

    pvfmm::Profile::Tic("Free");
    pvfmm::aligned_delete(A);
    pvfmm::Profile::Toc();

    pvfmm::Profile::Toc();
  }
}

void TestMatrix(){
  pvfmm::Profile::Tic("TestMatrix");
  pvfmm::Matrix<double> M1(1000,1000);
  pvfmm::Matrix<double> M2(1000,1000);

  pvfmm::Profile::Tic("Init");
  for(long i=0;i<M1.Dim(0)*M1.Dim(1);i++) M1[0][i]=i;
  for(long i=0;i<M2.Dim(0)*M2.Dim(1);i++) M2[0][i]=i*i;
  pvfmm::Profile::Toc();

  pvfmm::Profile::Tic("GEMM");
  pvfmm::Matrix<double> M3=M1*M2;
  pvfmm::Profile::Toc();

  pvfmm::Profile::Toc();
}

int main(int argc, char** argv) {

  // Dry run (profiling disabled)
  ProfileMemgr();

  // With profiling enabled
  pvfmm::Profile::Enable(true);
  ProfileMemgr();

  TestMatrix();

  // Print profiling results
  pvfmm::Profile::print();

  { // Test out-of-bound writes
    pvfmm::Iterator<char> A = pvfmm::aligned_new<char>(10);
    A[9];
    A[10]; // Should print stack tace here (in debug mode).
    //pvfmm::aligned_delete(A); // Show memory leak warning when commented
  }

  return 0;
}
